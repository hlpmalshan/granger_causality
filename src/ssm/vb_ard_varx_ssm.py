"""Hybrid Kalman + variational Bayesian group-ARD for fixed-B VARX SSMs.

This is deliberately a Level-1 hybrid method, not full structured mean-field
variational inference.  At each iteration the exact companion-form Kalman/RTS
posterior is evaluated at ``A_bar = E_q[A]`` and its moments are then held fixed
while updating ``q(A)`` and ``q(alpha)``.

The coefficient vector for target row ``i`` is lag-major::

    beta_i = [A1[i, :], A2[i, :], ..., Ap[i, :]]

Thus the columns for source ``j`` are ``j, M+j, ..., (p-1)M+j``.
"""

import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.kalman_varx_p import kalman_smooth_varx_p_companion
from src.ssm.ssm_varx_p_simulator import (
    build_companion_observation_matrix,
    build_companion_process_covariance,
    build_var_companion_matrix,
    build_varx_companion_input_matrix,
    make_exogenous_regressor,
    var_companion_spectral_radius,
)


class HybridVBARDVARXSSMFixedBKnownC:
    """Infer A with group-ARD while B, C, Q, and R remain known and fixed."""

    def __init__(self, na, nb, C, B_matrices, Q, R, max_iter=100,
                 tol_objective=1e-5, tol_A_change=1e-5, a0=1e-3, b0=1e-3,
                 diagonal_prior_precision=1e-4, posterior_jitter=1e-8,
                 init_A_matrices=None, verbose=False, random_state=None):
        self.na, self.nb = int(na), int(nb)
        self.C, self.Q, self.R = map(lambda x: np.asarray(x, dtype=float), (C, Q, R))
        self.B_matrices = [np.asarray(B, dtype=float).copy() for B in B_matrices]
        self.max_iter, self.tol_objective = int(max_iter), float(tol_objective)
        self.tol_A_change = float(tol_A_change)
        self.a0, self.b0 = float(a0), float(b0)
        self.diagonal_prior_precision = float(diagonal_prior_precision)
        self.posterior_jitter = float(posterior_jitter)
        self.init_A_matrices = init_A_matrices
        self.verbose, self.random_state = bool(verbose), random_state
        self._validate_constructor()

    def _validate_constructor(self):
        if self.na < 1 or self.nb < 1 or len(self.B_matrices) != self.nb:
            raise ValueError("na and nb must be positive and len(B_matrices) must equal nb.")
        self.n_states = self.C.shape[1]
        if self.C.ndim != 2 or self.Q.shape != (self.n_states, self.n_states):
            raise ValueError("C and Q have incompatible dimensions.")
        if self.R.shape != (self.C.shape[0], self.C.shape[0]):
            raise ValueError("R has incompatible dimensions.")
        n_inputs = self.B_matrices[0].shape[1]
        if any(B.shape != (self.n_states, n_inputs) for B in self.B_matrices):
            raise ValueError("All B matrices must have shape (n_states, n_inputs).")
        if not np.allclose(self.Q, np.diag(np.diag(self.Q))):
            raise ValueError("The Level-1 q(A) update currently requires diagonal Q.")
        if np.any(np.diag(self.Q) <= 0) or self.a0 <= 0 or self.b0 <= 0:
            raise ValueError("Q diagonal and Gamma hyperparameters must be positive.")

    def _initial_A(self):
        if self.init_A_matrices is not None:
            values = [np.asarray(A, dtype=float).copy() for A in self.init_A_matrices]
            if len(values) != self.na or any(A.shape != (self.n_states, self.n_states) for A in values):
                raise ValueError("init_A_matrices has incompatible dimensions.")
            return values
        values = [np.zeros((self.n_states, self.n_states)) for _ in range(self.na)]
        np.fill_diagonal(values[0], 0.35)
        if self.na > 1:
            np.fill_diagonal(values[1], -0.05)
        return values

    def _refresh_companion(self, A_matrices):
        self.F = build_var_companion_matrix(A_matrices)
        self.G = build_varx_companion_input_matrix(self.B_matrices, self.n_states, self.na)
        self.Q_aug = build_companion_process_covariance(self.Q, self.na)
        self.C_aug = build_companion_observation_matrix(self.C, self.na)

    def smooth(self, y, u, A_matrices):
        self._refresh_companion(A_matrices)
        return kalman_smooth_varx_p_companion(
            y=np.asarray(y, dtype=float), u=np.asarray(u, dtype=float), F=self.F,
            G=self.G, Q_aug=self.Q_aug, R=self.R, C_aug=self.C_aug,
            nb=self.nb, D=None,
        )

    def _compute_posterior_sufficient_statistics(self, smooth_result, u):
        """Return sums over t=1..T-1 using exact smoothed companion moments."""
        means = smooth_result["smoother"]["x_smooth"]
        covs = smooth_result["smoother"]["P_smooth"]
        lag_covs = smooth_result["smoother"]["P_lag_one"]
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u[:, None]
        d, m = self.na * self.n_states, self.n_states
        S_zz = np.zeros((d, d)); S_zx = np.zeros((d, m)); S_xx = np.zeros(m)
        offsets = np.zeros((len(u), m))
        B_stack = np.hstack(self.B_matrices)
        for t in range(1, len(u)):
            offsets[t] = B_stack @ make_exogenous_regressor(u, t, self.nb)
            z_mean, x_mean = means[t - 1, :d], means[t, :m]
            S_zz += covs[t - 1, :d, :d] + np.outer(z_mean, z_mean)
            cross_zx = lag_covs[t, :m, :d].T + np.outer(z_mean, x_mean)
            S_zx += cross_zx - np.outer(z_mean, offsets[t])
            centered_mean = x_mean - offsets[t]
            S_xx += np.diag(covs[t, :m, :m]) + centered_mean ** 2
        return {"S_zz": 0.5 * (S_zz + S_zz.T), "S_zx": S_zx,
                "S_xx": S_xx, "n_transitions": max(len(u) - 1, 0),
                "offsets": offsets}

    def _source_columns(self, source):
        return np.asarray([lag * self.n_states + source for lag in range(self.na)])

    def _update_q_A(self, sufficient_statistics):
        G, H = sufficient_statistics["S_zz"], sufficient_statistics["S_zx"]
        d = self.na * self.n_states
        means, covariances = np.zeros((self.n_states, d)), []
        for target in range(self.n_states):
            precision_prior = np.zeros(d)
            for source in range(self.n_states):
                precision_prior[self._source_columns(source)] = (
                    self.diagonal_prior_precision if source == target
                    else self.alpha_mean_[target, source]
                )
            precision = G / self.Q[target, target] + np.diag(precision_prior)
            precision += self.posterior_jitter * np.eye(d)
            try:
                chol = np.linalg.cholesky(0.5 * (precision + precision.T))
                covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(d)))
            except np.linalg.LinAlgError:
                covariance = np.linalg.pinv(precision)
            covariance = 0.5 * (covariance + covariance.T)
            if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
                raise FloatingPointError("Invalid q(A) posterior covariance.")
            means[target] = covariance @ (H[:, target] / self.Q[target, target])
            covariances.append(covariance)
        self._beta_means_ = means
        self.A_row_covariances_ = covariances
        self.A_mean_matrices_ = self._unpack_A(means)

    def _update_q_alpha(self):
        shape = np.full((self.n_states, self.n_states), np.nan)
        rate = np.full_like(shape, np.nan); mean = np.full_like(shape, np.nan)
        for target in range(self.n_states):
            for source in range(self.n_states):
                if source == target:
                    continue
                cols = self._source_columns(source)
                group_mean = self._beta_means_[target, cols]
                group_cov = self.A_row_covariances_[target][np.ix_(cols, cols)]
                second = float(group_mean @ group_mean + np.trace(group_cov))
                shape[target, source] = self.a0 + 0.5 * self.na
                rate[target, source] = self.b0 + 0.5 * second
                mean[target, source] = shape[target, source] / rate[target, source]
        if np.any(rate[~np.eye(self.n_states, dtype=bool)] <= 0) or not np.all(
                np.isfinite(mean[~np.eye(self.n_states, dtype=bool)])):
            raise FloatingPointError("Invalid q(alpha) update.")
        self.alpha_shape_, self.alpha_rate_, self.alpha_mean_ = shape, rate, mean
        self.alpha_log_mean_ = np.log(mean)

    def _pack_A(self, A_matrices=None):
        A_matrices = self.A_mean_matrices_ if A_matrices is None else A_matrices
        return np.stack([np.concatenate([A[target] for A in A_matrices])
                         for target in range(self.n_states)])

    def _unpack_A(self, beta_rows):
        beta_rows = np.asarray(beta_rows)
        return [beta_rows[:, lag * self.n_states:(lag + 1) * self.n_states].copy()
                for lag in range(self.na)]

    def _expected_transition_sse(self, stats):
        total = 0.0
        for target in range(self.n_states):
            m, S = self._beta_means_[target], self.A_row_covariances_[target]
            total += stats["S_xx"][target] - 2 * m @ stats["S_zx"][:, target]
            total += np.trace(stats["S_zz"] @ (S + np.outer(m, m)))
        return float(total)

    def compute_surrogate_objective(self):
        """Observed Kalman log likelihood at posterior mean A (not an ELBO)."""
        return float(self.smooth_result_["log_likelihood"])

    def _refresh_derived(self):
        variances = np.zeros((self.na, self.n_states, self.n_states))
        for target, covariance in enumerate(self.A_row_covariances_):
            for lag in range(self.na):
                sl = slice(lag * self.n_states, (lag + 1) * self.n_states)
                variances[lag, target] = np.diag(covariance)[sl]
        self.A_variance_matrices_ = variances
        self.A_std_matrices_ = np.sqrt(np.maximum(variances, 0.0))
        self.edge_scores_ = self.get_edge_scores()

    def fit(self, y, u):
        y, u = np.asarray(y, dtype=float), np.asarray(u, dtype=float)
        if u.ndim == 1: u = u[:, None]
        if len(y) != len(u) or y.shape[1] != self.C.shape[0]:
            raise ValueError("y and u lengths or y observation dimension are incompatible.")
        A = self._initial_A()
        self.alpha_mean_ = np.full((self.n_states, self.n_states), self.a0 / self.b0)
        np.fill_diagonal(self.alpha_mean_, np.nan)
        self.objective_history_, self.A_change_history_ = [], []
        self.alpha_change_history_, self.expected_transition_sse_history_ = [], []
        self.rescaling_history_, self.converged_ = [], False
        self.spectral_radius_before_rescaling_history_ = []
        self.spectral_radius_after_rescaling_history_ = []
        self.rescaling_factor_history_ = []
        previous_objective = None
        for iteration in range(self.max_iter):
            smooth_result = self.smooth(y, u, A)
            stats = self._compute_posterior_sufficient_statistics(smooth_result, u)
            old_beta, old_alpha = self._pack_A(A), self.alpha_mean_.copy()
            self._update_q_A(stats)
            radius_before = var_companion_spectral_radius(self.A_mean_matrices_)
            rescaled = radius_before >= 0.98
            radius_after, rescaling_factor = radius_before, 1.0
            if rescaled:
                self.A_mean_matrices_, radius_after, rescaling_factor = stabilize_A_matrices_if_needed(
                    self.A_mean_matrices_, target_radius=0.90)
                self._beta_means_ = self._pack_A(self.A_mean_matrices_)
            self._update_q_alpha()
            A = [value.copy() for value in self.A_mean_matrices_]
            # Re-smooth so the tracked likelihood and returned states correspond to new E[A].
            self.smooth_result_ = self.smooth(y, u, A)
            objective = float(self.smooth_result_["log_likelihood"])
            A_change = float(np.linalg.norm(self._beta_means_ - old_beta) /
                             max(np.linalg.norm(old_beta), 1e-12))
            mask = ~np.eye(self.n_states, dtype=bool)
            alpha_change = float(np.linalg.norm(self.alpha_mean_[mask] - old_alpha[mask]) /
                                 max(np.linalg.norm(old_alpha[mask]), 1e-12))
            self.objective_history_.append(objective); self.A_change_history_.append(A_change)
            self.alpha_change_history_.append(alpha_change)
            self.expected_transition_sse_history_.append(self._expected_transition_sse(stats))
            self.rescaling_history_.append(bool(rescaled))
            self.spectral_radius_before_rescaling_history_.append(float(radius_before))
            self.spectral_radius_after_rescaling_history_.append(float(radius_after))
            self.rescaling_factor_history_.append(float(rescaling_factor))
            objective_change = np.inf if previous_objective is None else abs(objective - previous_objective) / max(abs(previous_objective), 1.0)
            if self.verbose:
                print(f"hybrid_vb_ard iter={iteration+1} loglik={objective:.6g} A_change={A_change:.3g} alpha_change={alpha_change:.3g}")
            if A_change < self.tol_A_change and objective_change < self.tol_objective:
                self.converged_ = True
                break
            previous_objective = objective
        self.n_iter_ = iteration + 1
        self.filtered_state_mean_ = self.smooth_result_["filter"]["x_filt"][:, :self.n_states]
        self.smoothed_state_mean_ = self.smooth_result_["smoother"]["x_smooth"][:, :self.n_states]
        self._refresh_derived()
        self.diagnostics_ = self.get_diagnostics()
        return self

    def get_A_posterior_mean(self):
        return np.asarray(self.A_mean_matrices_).copy()

    def get_A_posterior_std(self):
        return np.asarray(self.A_std_matrices_).copy()

    def get_edge_scores(self):
        rows = []
        for target in range(self.n_states):
            for source in range(self.n_states):
                if source == target: continue
                cols = self._source_columns(source); m = self._beta_means_[target, cols]
                S = self.A_row_covariances_[target][np.ix_(cols, cols)]
                rows.append({"target": target, "source": source,
                    "posterior_mean_group_norm": float(np.linalg.norm(m)),
                    "posterior_second_moment_group_norm": float(m @ m + np.trace(S)),
                    "inverse_alpha_score": float(1.0 / self.alpha_mean_[target, source]),
                    "group_snr_score": float(np.sqrt(max(m @ np.linalg.pinv(S) @ m, 0.0))),
                    "alpha_mean": float(self.alpha_mean_[target, source]),
                    "alpha_shape": float(self.alpha_shape_[target, source]),
                    "alpha_rate": float(self.alpha_rate_[target, source])})
        return rows

    def sample_A(self, n_samples, stable_only=False, max_attempts=1000):
        rng = np.random.default_rng(self.random_state); samples = []; attempts = 0
        while len(samples) < n_samples and attempts < max_attempts:
            beta = np.stack([rng.multivariate_normal(self._beta_means_[i], self.A_row_covariances_[i])
                             for i in range(self.n_states)])
            candidate = np.asarray(self._unpack_A(beta)); attempts += 1
            if not stable_only or var_companion_spectral_radius(candidate) < 1.0:
                samples.append(candidate)
        if len(samples) < n_samples:
            raise RuntimeError(f"Obtained {len(samples)} stable samples in {max_attempts} attempts.")
        return np.asarray(samples)

    def get_diagnostics(self):
        return {"method": "hybrid_vb_ard", "level": 1, "is_full_structured_vi": False,
                "converged": bool(self.converged_), "n_iter": int(self.n_iter_),
                "kalman_log_likelihood": float(self.objective_history_[-1]),
                "surrogate_objective": float(self.objective_history_[-1]),
                "expected_transition_sse": float(self.expected_transition_sse_history_[-1]),
                "spectral_radius": var_companion_spectral_radius(self.A_mean_matrices_),
                "n_rescalings": int(sum(self.rescaling_history_)),
                "all_finite": bool(np.all(np.isfinite(self._beta_means_)) and
                                   np.all(np.isfinite(self.alpha_mean_[~np.eye(self.n_states, dtype=bool)]))),
                "todo_34c": "full structured mean-field q_x with block-banded precision and E[A^T Q^-1 A]",
                "todo_34d": "sample A from q(A) and propagate uncertainty to spectral GC"}
