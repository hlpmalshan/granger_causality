"""Exact tangent-linear LRVB operators for the Level-1 Hybrid VB-ARD map.

The derivative in this module is an analytic forward tangent.  Finite
differences are deliberately absent from :meth:`jvp` and are exposed only by
``directional_fd_check`` for validation experiments.

The full fixed-point state is ordered as

``[row-major lag-major A means, off-diagonal alpha means, B lags, diag(Q)]``.

The reduced state contains only the A means; alpha, B, and Q are frozen at
their converged values.  The Kalman filter and RTS smoother below reproduce
the NumPy implementation in ``src.ssm.kalman_varx_p`` and propagate a single
directional tangent alongside every primal quantity.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from numba import njit
except ImportError:  # pragma: no cover - exact NumPy fallback is intentional
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.stats.lrvb_stageB_smoother_feedback import global_index
from src.stats.matrixfree_lrvb_hybrid import HybridStateLayout, state_layout


def _sym(value):
    return 0.5 * (value + value.T)


def _rel(numerator, denominator):
    return float(np.linalg.norm(numerator) / max(np.linalg.norm(denominator), 1e-15))


def _regressors(u, nb):
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = u[:, None]
    out = np.zeros((len(u), nb * u.shape[1]))
    for t in range(len(u)):
        for lag in range(nb):
            if t - lag >= 0:
                out[t, lag * u.shape[1]:(lag + 1) * u.shape[1]] = u[t - lag]
    return out


@njit(cache=True)
def _stats_accumulate_numba(means, dmeans, covs, dcovs, lag, dlag,
                            offsets, doffsets, d, M):
    """Compiled sufficient-statistic accumulation (same algebra as NumPy map)."""
    G = np.zeros((d, d)); dG = np.zeros((d, d))
    H = np.zeros((d, M)); dH = np.zeros((d, M))
    xx = np.zeros(M); dxx = np.zeros(M)
    for t in range(1, means.shape[0]):
        z = means[t - 1, :d]; dz = dmeans[t - 1, :d]
        x = means[t, :M]; dx = dmeans[t, :M]
        G += covs[t - 1, :d, :d] + np.outer(z, z)
        dG += dcovs[t - 1, :d, :d] + np.outer(dz, z) + np.outer(z, dz)
        H += lag[t, :M, :d].T + np.outer(z, x) - np.outer(z, offsets[t])
        dH += (dlag[t, :M, :d].T + np.outer(dz, x) + np.outer(z, dx) -
               np.outer(dz, offsets[t]) - np.outer(z, doffsets[t]))
        centered = x - offsets[t]; dcentered = dx - doffsets[t]
        xx += np.diag(covs[t, :M, :M]) + centered ** 2
        dxx += np.diag(dcovs[t, :M, :M]) + 2 * centered * dcentered
    return G, dG, H, dH, xx, dxx


@dataclass
class TangentResult:
    value: np.ndarray
    tangent: np.ndarray
    rescaled_A: bool


class HybridFixedPointTangentMap:
    """Functional fixed-point map with an analytic one-direction tangent."""

    def __init__(self, model, y, u, reduced=False):
        self.model = model
        self.y = np.asarray(y, float)
        self.u = np.asarray(u, float)
        if self.u.ndim == 1:
            self.u = self.u[:, None]
        self.reduced = bool(reduced)
        self.layout: HybridStateLayout = state_layout(model)
        self.full_theta = self._pack_model()
        self.theta0 = (self.full_theta[self.layout.A_slice].copy()
                       if self.reduced else self.full_theta.copy())
        self.shape = (len(self.theta0), len(self.theta0))
        self.jvp_calls = 0
        self.map_calls = 0
        self.elapsed_jvp_seconds = 0.0
        self.phi = _regressors(self.u, model.nb)
        self.offdiag = self.layout.offdiag
        self._fixed_alpha = self.full_theta[self.layout.alpha_slice].copy()
        self._fixed_B = self.full_theta[self.layout.B_slice].copy()
        self._fixed_q = self.full_theta[self.layout.Q_slice].copy()
        self.operator = LinearOperator(self.shape, matvec=self.matvec, dtype=float)

    def _pack_model(self):
        beta = np.asarray(self.model._pack_A()).reshape(-1)
        alpha = np.asarray(self.model.alpha_mean_)[self.layout.offdiag]
        B = np.asarray(self.model.B_matrices).reshape(-1)
        q = np.diag(self.model.Q)
        return np.concatenate([beta, alpha, B, q]).astype(float)

    def _expand(self, theta, tangent):
        theta = np.asarray(theta, float)
        tangent = np.asarray(tangent, float)
        if theta.shape != self.theta0.shape or tangent.shape != theta.shape:
            raise ValueError("Fixed-point value or tangent has the wrong shape.")
        if self.reduced:
            full = self.full_theta.copy(); dfull = np.zeros_like(full)
            full[self.layout.A_slice] = theta
            dfull[self.layout.A_slice] = tangent
            return full, dfull
        return theta, tangent

    def _decode(self, theta, tangent):
        L = self.layout
        beta = theta[L.A_slice].reshape(L.n_states, L.na * L.n_states)
        dbeta = tangent[L.A_slice].reshape(beta.shape)
        alpha = np.full((L.n_states, L.n_states), np.nan)
        dalpha = np.zeros_like(alpha)
        alpha[L.offdiag] = theta[L.alpha_slice]
        dalpha[L.offdiag] = tangent[L.alpha_slice]
        B = theta[L.B_slice].reshape(L.nb, L.n_states, L.n_inputs)
        dB = tangent[L.B_slice].reshape(B.shape)
        q = theta[L.Q_slice]; dq = tangent[L.Q_slice]
        if np.any(alpha[L.offdiag] <= 0) or np.any(q <= 0):
            raise FloatingPointError("The tangent-map base point has invalid alpha or Q.")
        return beta, dbeta, alpha, dalpha, B, dB, q, dq

    def _companion(self, beta, dbeta, B, dB, q, dq):
        M, p, nb = self.layout.n_states, self.layout.na, self.layout.nb
        d = M * p
        F = np.zeros((d, d)); dF = np.zeros_like(F)
        F[:M] = beta; dF[:M] = dbeta
        if p > 1:
            F[M:, :-M] = np.eye(M * (p - 1))
        G = np.zeros((d, nb * self.layout.n_inputs)); dG = np.zeros_like(G)
        G[:M] = np.hstack(B); dG[:M] = np.hstack(dB)
        Q = np.zeros((d, d)); dQ = np.zeros_like(Q)
        Q[:M, :M] = np.diag(q); dQ[:M, :M] = np.diag(dq)
        C = np.hstack([self.model.C, np.zeros((self.model.C.shape[0], d - M))])
        return F, dF, G, dG, Q, dQ, C

    def _smooth(self, beta, dbeta, B, dB, q, dq):
        F, dF, G, dG, Q, dQ, C = self._companion(beta, dbeta, B, dB, q, dq)
        T, d, ny = len(self.y), F.shape[0], self.y.shape[1]
        xp = np.zeros((T, d)); dxp = np.zeros_like(xp)
        Pp = np.zeros((T, d, d)); dPp = np.zeros_like(Pp)
        xf = np.zeros_like(xp); dxf = np.zeros_like(xp)
        Pf = np.zeros_like(Pp); dPf = np.zeros_like(Pp)
        gains = np.zeros((T, d, ny)); dgains = np.zeros_like(gains)
        previous_mean = np.zeros(d); dprevious_mean = np.zeros(d)
        previous_cov = 10.0 * np.eye(d); dprevious_cov = np.zeros((d, d))
        identity = np.eye(d)
        loglike = 0.0
        for t in range(T):
            if t == 0:
                mean_pred = previous_mean + G @ self.phi[t]
                dmean_pred = dprevious_mean + dG @ self.phi[t]
                cov_pred = previous_cov + Q
                dcov_pred = dprevious_cov + dQ
            else:
                mean_pred = F @ previous_mean + G @ self.phi[t]
                dmean_pred = (dF @ previous_mean + F @ dprevious_mean +
                              dG @ self.phi[t])
                cov_pred = F @ previous_cov @ F.T + Q
                dcov_pred = (dF @ previous_cov @ F.T +
                             F @ dprevious_cov @ F.T +
                             F @ previous_cov @ dF.T + dQ)
            cov_pred, dcov_pred = _sym(cov_pred), _sym(dcov_pred)
            innovation = self.y[t] - C @ mean_pred
            dinnovation = -C @ dmean_pred
            rawS = C @ cov_pred @ C.T + self.model.R
            drawS = C @ dcov_pred @ C.T
            scale = np.trace(rawS) / max(ny, 1)
            if not np.isfinite(scale) or scale <= 0:
                raise FloatingPointError("Innovation covariance scale is not differentiable.")
            S = _sym(rawS) + 1e-10 * scale * np.eye(ny)
            dS = _sym(drawS) + 1e-10 * np.trace(drawS) / max(ny, 1) * np.eye(ny)
            Sinv = np.linalg.pinv(S)
            K = cov_pred @ C.T @ Sinv
            dK = dcov_pred @ C.T @ Sinv - K @ dS @ Sinv
            mean_filt = mean_pred + K @ innovation
            dmean_filt = dmean_pred + dK @ innovation + K @ dinnovation
            H = identity - K @ C; dH = -dK @ C
            cov_filt = H @ cov_pred @ H.T + K @ self.model.R @ K.T
            dcov_filt = (dH @ cov_pred @ H.T + H @ dcov_pred @ H.T +
                         H @ cov_pred @ dH.T + dK @ self.model.R @ K.T +
                         K @ self.model.R @ dK.T)
            cov_filt, dcov_filt = _sym(cov_filt), _sym(dcov_filt)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                raise FloatingPointError("Non-SPD innovation covariance.")
            loglike += -0.5 * (ny * np.log(2 * np.pi) + logdet + innovation @ Sinv @ innovation)
            xp[t], dxp[t], Pp[t], dPp[t] = mean_pred, dmean_pred, cov_pred, dcov_pred
            xf[t], dxf[t], Pf[t], dPf[t] = mean_filt, dmean_filt, cov_filt, dcov_filt
            gains[t], dgains[t] = K, dK
            previous_mean, dprevious_mean = mean_filt, dmean_filt
            previous_cov, dprevious_cov = cov_filt, dcov_filt
        xs = np.zeros_like(xf); dxs = np.zeros_like(xf)
        Ps = np.zeros_like(Pf); dPs = np.zeros_like(Pf)
        J = np.zeros((max(T - 1, 0), d, d)); dJ = np.zeros_like(J)
        xs[-1], dxs[-1], Ps[-1], dPs[-1] = xf[-1], dxf[-1], Pf[-1], dPf[-1]
        for t in range(T - 2, -1, -1):
            Pinv = np.linalg.pinv(Pp[t + 1])
            smooth_gain = Pf[t] @ F.T @ Pinv
            dsmooth_gain = ((dPf[t] @ F.T + Pf[t] @ dF.T) @ Pinv -
                            smooth_gain @ dPp[t + 1] @ Pinv)
            J[t], dJ[t] = smooth_gain, dsmooth_gain
            delta = xs[t + 1] - xp[t + 1]
            ddelta = dxs[t + 1] - dxp[t + 1]
            xs[t] = xf[t] + smooth_gain @ delta
            dxs[t] = dxf[t] + dsmooth_gain @ delta + smooth_gain @ ddelta
            middle = Ps[t + 1] - Pp[t + 1]
            dmiddle = dPs[t + 1] - dPp[t + 1]
            Ps[t] = Pf[t] + smooth_gain @ middle @ smooth_gain.T
            dPs[t] = (dPf[t] + dsmooth_gain @ middle @ smooth_gain.T +
                      smooth_gain @ dmiddle @ smooth_gain.T +
                      smooth_gain @ middle @ dsmooth_gain.T)
            Ps[t], dPs[t] = _sym(Ps[t]), _sym(dPs[t])
        Plag = np.zeros_like(Ps); dPlag = np.zeros_like(Ps)
        for t in range(1, T):
            Plag[t] = Ps[t] @ J[t - 1].T
            dPlag[t] = dPs[t] @ J[t - 1].T + Ps[t] @ dJ[t - 1].T
        return {"means": xs, "dmeans": dxs, "covs": Ps, "dcovs": dPs,
                "lag_covs": Plag, "dlag_covs": dPlag,
                "log_likelihood": float(loglike)}

    def _stats(self, smooth, B, dB):
        means, dmeans = smooth["means"], smooth["dmeans"]
        covs, dcovs = smooth["covs"], smooth["dcovs"]
        lag, dlag = smooth["lag_covs"], smooth["dlag_covs"]
        M, d = self.layout.n_states, self.layout.na * self.layout.n_states
        offsets = self.phi @ np.hstack(B).T
        doffsets = self.phi @ np.hstack(dB).T
        G, dG, H, dH, xx, dxx = _stats_accumulate_numba(
            means, dmeans, covs, dcovs, lag, dlag, offsets, doffsets, d, M)
        return {"S_zz": _sym(G), "dS_zz": _sym(dG), "S_zx": H,
                "dS_zx": dH, "S_xx": xx, "dS_xx": dxx,
                "n_transitions": max(len(means) - 1, 1)}

    def _update_A(self, stats, alpha, dalpha, q, dq, perturb, dperturb):
        M, p, d = self.layout.n_states, self.layout.na, self.layout.na * self.layout.n_states
        means = np.zeros((M, d)); dmeans = np.zeros_like(means)
        covariances, dcovariances = [], []
        candidate = getattr(self.model, "candidate_mask", np.ones((M, M), bool))
        H = stats["S_zx"].copy(); dH = stats["dS_zx"].copy()
        H += perturb.T * q[None, :]
        dH += dperturb.T * q[None, :] + perturb.T * dq[None, :]
        for target in range(M):
            sources = np.flatnonzero(candidate[target])
            allowed = np.asarray([lag * M + source for lag in range(p) for source in sources], int)
            prior = np.empty(len(allowed)); dprior = np.empty(len(allowed))
            for position, column in enumerate(allowed):
                source = column % M
                if source == target:
                    prior[position], dprior[position] = self.model.diagonal_prior_precision, 0.0
                else:
                    prior[position], dprior[position] = alpha[target, source], dalpha[target, source]
            G = stats["S_zz"][np.ix_(allowed, allowed)]
            dG = stats["dS_zz"][np.ix_(allowed, allowed)]
            precision = G / q[target] + np.diag(prior)
            precision = _sym(precision) + self.model.posterior_jitter * np.eye(len(allowed))
            dprecision = (dG / q[target] - G * dq[target] / q[target] ** 2 +
                          np.diag(dprior))
            covariance = np.linalg.solve(precision, np.eye(len(allowed)))
            covariance = _sym(covariance)
            dcovariance = -covariance @ dprecision @ covariance
            rhs = H[allowed, target] / q[target]
            drhs = dH[allowed, target] / q[target] - H[allowed, target] * dq[target] / q[target] ** 2
            restricted = covariance @ rhs
            drestricted = dcovariance @ rhs + covariance @ drhs
            means[target, allowed], dmeans[target, allowed] = restricted, drestricted
            full = np.zeros((d, d)); dfull = np.zeros_like(full)
            full[np.ix_(allowed, allowed)] = covariance
            dfull[np.ix_(allowed, allowed)] = _sym(dcovariance)
            covariances.append(full); dcovariances.append(dfull)
        return means, dmeans, covariances, dcovariances

    def _update_alpha(self, beta, dbeta, covs, dcovs):
        M, p = self.layout.n_states, self.layout.na
        alpha = np.full((M, M), np.nan); dalpha = np.zeros((M, M))
        candidate = getattr(self.model, "candidate_mask", np.ones((M, M), bool))
        shape = self.model.a0 + 0.5 * p
        for target in range(M):
            for source in range(M):
                if source == target:
                    continue
                if not candidate[target, source]:
                    alpha[target, source] = self.model.a0 / self.model.b0
                    continue
                columns = np.asarray([lag * M + source for lag in range(p)])
                m, dm = beta[target, columns], dbeta[target, columns]
                S = covs[target][np.ix_(columns, columns)]
                dS = dcovs[target][np.ix_(columns, columns)]
                rate = self.model.b0 + 0.5 * (m @ m + np.trace(S))
                drate = m @ dm + 0.5 * np.trace(dS)
                alpha[target, source] = shape / rate
                dalpha[target, source] = -shape * drate / rate ** 2
        return alpha, dalpha

    def _update_B(self, smooth, beta, dbeta):
        M = self.layout.n_states
        means, dmeans = smooth["means"], smooth["dmeans"]
        d = self.layout.na * M
        R = self.phi[1:]
        Srr = R.T @ R
        Srx = R.T @ means[1:, :M]
        dSrx = R.T @ dmeans[1:, :M]
        Srz = R.T @ means[:-1, :d]
        dSrz = R.T @ dmeans[:-1, :d]
        rhs = Srx - Srz @ beta.T
        drhs = dSrx - dSrz @ beta.T - Srz @ dbeta.T
        ridge = float(getattr(self.model, "effective_B_ridge", 0.0))
        system = Srr + ridge * np.eye(Srr.shape[0])
        gamma = np.linalg.solve(system, rhs).T
        dgamma = np.linalg.solve(system, drhs).T
        shape = (self.layout.nb, M, self.layout.n_inputs)
        B = np.stack(np.split(gamma, self.layout.nb, axis=1)).reshape(shape)
        dB = np.stack(np.split(dgamma, self.layout.nb, axis=1)).reshape(shape)
        return B, dB

    def _update_Q(self, smooth, B, dB, beta, dbeta, covs, dcovs, q, dq):
        stats = self._stats(smooth, B, dB)
        n = stats["n_transitions"]
        raw = np.zeros(self.layout.n_states); draw = np.zeros_like(raw)
        for target in range(self.layout.n_states):
            m, dm, S, dS = beta[target], dbeta[target], covs[target], dcovs[target]
            G, dG = stats["S_zz"], stats["dS_zz"]
            h, dh = stats["S_zx"][:, target], stats["dS_zx"][:, target]
            value = stats["S_xx"][target] - 2 * m @ h + m @ G @ m
            dvalue = (stats["dS_xx"][target] - 2 * (dm @ h + m @ dh) +
                      2 * dm @ G @ m + m @ dG @ m)
            if self.model.include_A_posterior_uncertainty_in_Q:
                value += np.trace(S @ G)
                dvalue += np.trace(dS @ G + S @ dG)
            raw[target], draw[target] = value / n, dvalue / n
        if np.any(raw <= 1e-10):
            raise FloatingPointError("Q floor is active; exact tangent is nondifferentiable here.")
        mode = self.model.Q_update_mode
        if mode == "diag_shrink_scalar":
            rho = self.model.Q_shrinkage_rho
            candidate = (1 - rho) * raw + rho * np.mean(raw)
            dcandidate = (1 - rho) * draw + rho * np.mean(draw)
        elif mode in ("diag_raw", "diag_floor"):
            candidate, dcandidate = raw, draw
        else:
            raise NotImplementedError(f"Exact tangent does not support Q mode {mode!r}.")
        damping = self.model.Q_update_damping
        return (1 - damping) * q + damping * candidate, (1 - damping) * dq + damping * dcandidate

    def value_and_tangent(self, theta, tangent, perturbation=None, dperturbation=None):
        full, dfull = self._expand(theta, tangent)
        beta, dbeta, alpha, dalpha, B, dB, q, dq = self._decode(full, dfull)
        M, d = self.layout.n_states, self.layout.na * self.layout.n_states
        perturbation = np.zeros((M, d)) if perturbation is None else np.asarray(perturbation, float)
        dperturbation = np.zeros((M, d)) if dperturbation is None else np.asarray(dperturbation, float)
        smooth = self._smooth(beta, dbeta, B, dB, q, dq)
        stats = self._stats(smooth, B, dB)
        new_beta, dnew_beta, covs, dcovs = self._update_A(
            stats, alpha, dalpha, q, dq, perturbation, dperturbation)
        matrices = np.stack([new_beta[:, lag * M:(lag + 1) * M]
                             for lag in range(self.layout.na)])
        radius = var_companion_spectral_radius(matrices)
        if not np.isfinite(radius):
            raise FloatingPointError("Non-finite companion radius in tangent map.")
        if radius >= 0.98:
            raise FloatingPointError(
                "A stability rescaling branch is active; the exact tangent is intentionally undefined.")
        if self.reduced:
            return TangentResult(new_beta.reshape(-1), dnew_beta.reshape(-1), False)
        new_alpha, dnew_alpha = self._update_alpha(new_beta, dnew_beta, covs, dcovs)
        if bool(getattr(self.model, "estimate_B", False)):
            new_B, dnew_B = self._update_B(smooth, new_beta, dnew_beta)
        else:
            new_B, dnew_B = B, dB
        if bool(getattr(self.model, "estimate_Q", False)):
            new_q, dnew_q = self._update_Q(
                smooth, new_B, dnew_B, new_beta, dnew_beta, covs, dcovs, q, dq)
        else:
            new_q, dnew_q = q, dq
        value = np.concatenate([new_beta.reshape(-1), new_alpha[self.offdiag],
                                new_B.reshape(-1), new_q])
        deriv = np.concatenate([dnew_beta.reshape(-1), dnew_alpha[self.offdiag],
                                dnew_B.reshape(-1), dnew_q])
        return TangentResult(value, deriv, False)

    def map(self, theta, perturbation=None):
        self.map_calls += 1
        return self.value_and_tangent(theta, np.zeros_like(theta), perturbation).value

    def jvp(self, vector):
        started = time.perf_counter(); self.jvp_calls += 1
        result = self.value_and_tangent(self.theta0, np.asarray(vector, float)).tangent
        self.elapsed_jvp_seconds += time.perf_counter() - started
        return result

    def perturbation_rhs(self, direction_index):
        target, within = divmod(int(direction_index), self.layout.na * self.layout.n_states)
        lag, source = divmod(within, self.layout.n_states)
        dh = np.zeros((self.layout.n_states, self.layout.na * self.layout.n_states))
        dh[target, lag * self.layout.n_states + source] = 1.0
        return self.value_and_tangent(
            self.theta0, np.zeros_like(self.theta0), dperturbation=dh).tangent

    def matvec(self, vector):
        vector = np.asarray(vector, float)
        return vector - self.jvp(vector)


def directional_fd_check(engine, vector, epsilon):
    """Central finite difference used only to verify the analytic JVP."""
    vector = np.asarray(vector, float)
    vector = vector / max(np.linalg.norm(vector), 1e-15)
    exact = engine.jvp(vector)
    plus = engine.map(engine.theta0 + epsilon * vector)
    minus = engine.map(engine.theta0 - epsilon * vector)
    fd = (plus - minus) / (2 * epsilon)
    denominator = max(np.linalg.norm(fd), 1e-15)
    cosine = float(exact @ fd / max(np.linalg.norm(exact) * np.linalg.norm(fd), 1e-15))
    return {"epsilon": float(epsilon), "relative_jvp_error": float(np.linalg.norm(exact - fd) / denominator),
            "cosine_similarity": cosine,
            "norm_ratio": float(np.linalg.norm(exact) / denominator)}


def solve_gmres(engine, rhs, rtol=1e-5, restart=50, maxiter=100,
                preconditioner=None, x0=None):
    residuals = []; started = time.perf_counter(); before = engine.jvp_calls
    # With callback_type='pr_norm', SciPy interprets maxiter as restart cycles.
    # Convert the documented total Krylov budget to a cycle count.
    cycle_count = max(1, int(np.ceil(int(maxiter) / max(int(restart), 1))))
    solution, info = gmres(
        engine.operator, np.asarray(rhs, float), x0=x0, M=preconditioner,
        rtol=float(rtol), atol=0.0, restart=int(restart), maxiter=cycle_count,
        callback=lambda value: residuals.append(float(value)), callback_type="pr_norm")
    residual = engine.matvec(solution) - rhs
    return solution, {"solver": "GMRES", "converged": bool(info == 0),
        "solver_info": int(info), "iterations": len(residuals),
        "final_residual_norm": float(np.linalg.norm(residual)),
        "relative_residual": _rel(residual, rhs),
        "jvp_calls": int(engine.jvp_calls - before),
        "runtime_seconds": float(time.perf_counter() - started),
        "residual_history": residuals}


def edge_response(engine, target, source, rtol=1e-5, restart=50, maxiter=100):
    indices = [global_index(target, lag, source, engine.layout.n_states, engine.layout.na)
               for lag in range(engine.layout.na)]
    columns, diagnostics = [], []
    for lag, index in enumerate(indices):
        rhs = engine.perturbation_rhs(index)
        response, diagnostic = solve_gmres(engine, rhs, rtol, restart, maxiter)
        columns.append(response[:engine.layout.A_slice.stop][indices])
        diagnostics.append({**diagnostic, "lag": lag + 1, "direction_index": index})
    return np.column_stack(columns), diagnostics


def neumann_response(engine, rhs, powers=(0, 1, 2, 4, 8, 16)):
    requested = sorted(set(int(value) for value in powers))
    term = np.asarray(rhs, float).copy(); total = term.copy(); rows = []
    for k in range(max(requested) + 1):
        if k in requested:
            rows.append((k, total.copy(), float(np.linalg.norm(term))))
        if k < max(requested):
            term = engine.jvp(term); total = total + term
            if not np.all(np.isfinite(term)) or np.linalg.norm(term) > 1e12 * max(np.linalg.norm(rhs), 1e-15):
                break
    return rows


__all__ = ["HybridFixedPointTangentMap", "TangentResult", "directional_fd_check",
           "edge_response", "neumann_response", "solve_gmres"]
