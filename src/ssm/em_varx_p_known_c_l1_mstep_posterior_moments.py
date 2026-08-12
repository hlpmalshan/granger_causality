import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

from src.ssm.em_varx_p_known_c_l1_mstep_fixed_cov import (
    EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov
)
from src.ssm.em_varx_p_known_c_l1_mstep import (
    stabilize_A_matrices_if_needed
)
from src.ssm.ssm_varx_p_simulator import make_exogenous_regressor


def _covariance_coordinate_descent_core_python(
        G,
        h,
        l1_penalty,
        ridge_penalty,
        beta,
        max_iter,
        tol
):
    """Coordinate descent for a standardized covariance-form objective."""

    n_features = G.shape[0]
    converged = False

    for iteration in range(max_iter):
        max_delta = 0.0

        for j in range(n_features):
            old_value = beta[j]
            rho_j = h[j] - np.dot(G[j], beta) + G[j, j] * old_value
            denominator = G[j, j] + ridge_penalty[j]

            if denominator <= 1e-12:
                new_value = 0.0
            elif rho_j > l1_penalty[j]:
                new_value = (rho_j - l1_penalty[j]) / denominator
            elif rho_j < -l1_penalty[j]:
                new_value = (rho_j + l1_penalty[j]) / denominator
            else:
                new_value = 0.0

            beta[j] = new_value
            delta = abs(new_value - old_value)

            if delta > max_delta:
                max_delta = delta

        if max_delta < tol:
            converged = True
            break

    return beta, iteration + 1, converged


if njit is not None:
    _covariance_coordinate_descent_core = njit(
        cache=True,
        nogil=True
    )(_covariance_coordinate_descent_core_python)
else:
    _covariance_coordinate_descent_core = (
        _covariance_coordinate_descent_core_python
    )


def solve_standardized_covariance_lasso(
        G,
        h,
        l1_mask,
        lambda_fraction,
        ridge_penalty,
        initial_beta=None,
        max_iter=1000,
        tol=1e-6,
        scale_epsilon=1e-12
):
    """
    Solve a covariance-form elastic-net objective after feature scaling.

    lambda_max is computed only over entries selected by l1_mask. The
    returned effective lambda is the common standardized penalty applied to
    those entries.
    """

    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float)
    l1_mask = np.asarray(l1_mask, dtype=bool)
    ridge_penalty = np.asarray(ridge_penalty, dtype=float)

    n_features = h.shape[0]

    if G.shape != (n_features, n_features):
        raise ValueError("G must be square and match h.")

    if l1_mask.shape != (n_features,):
        raise ValueError("l1_mask must match h.")

    if ridge_penalty.shape != (n_features,):
        raise ValueError("ridge_penalty must match h.")

    if lambda_fraction < 0.0:
        raise ValueError("lambda_fraction must be nonnegative.")

    G = 0.5 * (G + G.T)
    scale = np.sqrt(np.maximum(np.diag(G), scale_epsilon))
    G_std = G / np.outer(scale, scale)
    G_std = 0.5 * (G_std + G_std.T)
    h_std = h / scale

    if np.any(l1_mask):
        lambda_max = float(np.max(np.abs(h_std[l1_mask])))
    else:
        lambda_max = 0.0

    lambda_effective = float(lambda_fraction) * lambda_max
    l1_penalty = np.zeros(n_features)
    l1_penalty[l1_mask] = lambda_effective

    if initial_beta is None:
        beta_std = np.zeros(n_features)
    else:
        initial_beta = np.asarray(initial_beta, dtype=float)

        if initial_beta.shape != (n_features,):
            raise ValueError("initial_beta must match h.")

        beta_std = initial_beta * scale

    beta_std, n_iterations, converged = (
        _covariance_coordinate_descent_core(
            np.ascontiguousarray(G_std),
            np.ascontiguousarray(h_std),
            np.ascontiguousarray(l1_penalty),
            np.ascontiguousarray(ridge_penalty),
            np.ascontiguousarray(beta_std),
            int(max_iter),
            float(tol)
        )
    )

    beta = beta_std / scale
    objective = float(
        0.5 * beta_std @ G_std @ beta_std
        - h_std @ beta_std
        + np.sum(l1_penalty * np.abs(beta_std))
        + 0.5 * np.sum(ridge_penalty * beta_std ** 2)
    )

    condition_number = float(
        np.linalg.cond(
            G_std + np.diag(ridge_penalty)
        )
    )

    return {
        "beta": beta,
        "beta_standardized": beta_std,
        "scale": scale,
        "lambda_max": lambda_max,
        "lambda_effective": lambda_effective,
        "n_iterations": int(n_iterations),
        "converged": bool(converged),
        "objective": objective,
        "condition_number": condition_number
    }


def build_posterior_moment_statistics(
        smooth_result,
        u,
        na,
        nb,
        n_sources
):
    """Build average E[phi phi'], E[phi x'], and E[x x'] moments."""

    if "smoother" in smooth_result:
        smoother = smooth_result["smoother"]
    else:
        smoother = smooth_result

    required_keys = ("x_smooth", "P_smooth", "P_lag_one")
    missing = [key for key in required_keys if key not in smoother]

    if missing:
        raise KeyError(
            "Smoother result lacks posterior moments: " + ", ".join(missing)
        )

    mu = np.asarray(smoother["x_smooth"], dtype=float)
    covariance = np.asarray(smoother["P_smooth"], dtype=float)
    lag_one_cov = np.asarray(smoother["P_lag_one"], dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        u = u[:, None]

    T = mu.shape[0]
    n_aug = n_sources * na
    n_exog = u.shape[1] * nb
    n_features = n_aug + n_exog

    if mu.shape != (T, n_aug):
        raise ValueError("Smoothed companion-state mean has unexpected shape.")

    expected_covariance_shape = (T, n_aug, n_aug)

    if covariance.shape != expected_covariance_shape:
        raise ValueError("Smoothed covariance has unexpected shape.")

    if lag_one_cov.shape != expected_covariance_shape:
        raise ValueError("Lag-one smoothed covariance has unexpected shape.")

    if u.shape[0] != T:
        raise ValueError("u and the smoother output must have equal length.")

    G = np.zeros((n_features, n_features))
    H = np.zeros((n_features, n_sources))
    E_xx = np.zeros((n_sources, n_sources))
    start = max(na, nb - 1)
    n_valid = T - start

    if n_valid <= 0:
        raise ValueError("Not enough time points for the requested lags.")

    for t in range(start, T):
        mu_prev = mu[t - 1]
        mu_x = mu[t, :n_sources]
        P_prev = covariance[t - 1]
        P_x = covariance[t, :n_sources, :n_sources]
        v_t = make_exogenous_regressor(u=u, t=t, nb=nb)

        # P_lag_one[t] is Cov(s_t, s_{t-1} | y). Transposing and
        # selecting the current-state columns gives Cov(s_{t-1}, x_t | y).
        covariance_prev_x = lag_one_cov[t].T[:, :n_sources]

        G[:n_aug, :n_aug] += P_prev + np.outer(mu_prev, mu_prev)
        G[:n_aug, n_aug:] += np.outer(mu_prev, v_t)
        G[n_aug:, :n_aug] += np.outer(v_t, mu_prev)
        G[n_aug:, n_aug:] += np.outer(v_t, v_t)

        H[:n_aug] += covariance_prev_x + np.outer(mu_prev, mu_x)
        H[n_aug:] += np.outer(v_t, mu_x)
        E_xx += P_x + np.outer(mu_x, mu_x)

    G /= n_valid
    H /= n_valid
    E_xx /= n_valid
    G = 0.5 * (G + G.T)

    return {
        "G": G,
        "H": H,
        "E_xx": E_xx,
        "start": start,
        "n_valid": n_valid
    }


class EMVARXPSSMKnownCConstrainedWarmStartL1PosteriorMomentMstepFixedCov(
        EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov
):
    """Fixed-covariance VARX EM with a posterior-moment sparse M-step."""

    def __init__(
            self,
            *args,
            lambda_A_fraction=0.0,
            estimate_Q=False,
            estimate_R=False,
            **kwargs
    ):
        if estimate_Q or estimate_R:
            raise ValueError(
                "Experiment 32H requires estimate_Q=False and estimate_R=False."
            )

        if lambda_A_fraction < 0.0:
            raise ValueError("lambda_A_fraction must be nonnegative.")

        # Absolute L1 penalties from 32G are intentionally disabled. The
        # posterior-moment solver computes a target-specific lambda_max.
        kwargs.pop("lambda_A_offdiag", None)
        kwargs.pop("lambda_A_diag", None)
        kwargs.pop("lambda_B", None)

        super().__init__(
            *args,
            lambda_A_offdiag=0.0,
            lambda_A_diag=0.0,
            lambda_B=0.0,
            estimate_Q=False,
            estimate_R=False,
            **kwargs
        )

        self.lambda_A_fraction = float(lambda_A_fraction)
        self.lambda_A_effective_history = []
        self.moment_solver_condition_number_history = []

        self.mean_lambda_A_effective = np.nan
        self.median_lambda_A_effective = np.nan
        self.min_lambda_A_effective = np.nan
        self.max_lambda_A_effective = np.nan
        self.moment_solver_condition_number_mean = np.nan
        self.moment_solver_condition_number_max = np.nan

    def _make_l1_mask_and_ridge(
            self,
            target,
            n_sources,
            n_inputs
    ):
        n_features = self.na * n_sources + self.nb * n_inputs
        l1_mask = np.zeros(n_features, dtype=bool)
        ridge = np.zeros(n_features)

        for lag_index in range(self.na):
            for source in range(n_sources):
                column = lag_index * n_sources + source

                if source == target:
                    ridge[column] = self.ridge_A_diag
                else:
                    l1_mask[column] = True
                    ridge[column] = self.ridge_A_offdiag

        offset = self.na * n_sources
        ridge[offset:] = self.ridge_B

        return l1_mask, ridge

    def _initial_beta_for_target(self, target):
        values = []

        for A in self.A_matrices:
            values.extend(A[target])

        for B in self.B_matrices:
            values.extend(B[target])

        return np.asarray(values, dtype=float)

    def _m_step(self, y, u, smooth_result):
        """Update A/B from posterior moments while leaving Q/R unchanged."""

        u = np.asarray(u, dtype=float)

        if u.ndim == 1:
            u = u[:, None]

        n_sources = self.C.shape[1]
        n_inputs = u.shape[1]
        moments = build_posterior_moment_statistics(
            smooth_result=smooth_result,
            u=u,
            na=self.na,
            nb=self.nb,
            n_sources=n_sources
        )

        G = moments["G"]
        H = moments["H"]
        beta_matrix = np.zeros((n_sources, G.shape[0]))
        converged_values = []
        iteration_values = []
        objective_values = []
        effective_lambdas = []
        condition_numbers = []

        for target in range(n_sources):
            l1_mask, ridge = self._make_l1_mask_and_ridge(
                target=target,
                n_sources=n_sources,
                n_inputs=n_inputs
            )
            fit = solve_standardized_covariance_lasso(
                G=G,
                h=H[:, target],
                l1_mask=l1_mask,
                lambda_fraction=self.lambda_A_fraction,
                ridge_penalty=ridge,
                initial_beta=self._initial_beta_for_target(target),
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol
            )

            beta_matrix[target] = fit["beta"]
            converged_values.append(fit["converged"])
            iteration_values.append(fit["n_iterations"])
            objective_values.append(fit["objective"])
            effective_lambdas.append(fit["lambda_effective"])
            condition_numbers.append(fit["condition_number"])

        self.A_matrices, self.B_matrices = self._beta_matrix_to_A_B(
            beta_matrix=beta_matrix,
            n_sources=n_sources,
            n_inputs=n_inputs
        )

        if hasattr(self, "_apply_zero_constraints_to_A_matrices"):
            self._apply_zero_constraints_to_A_matrices()

        if self.stabilize_A:
            self.A_matrices, radius, scale = stabilize_A_matrices_if_needed(
                self.A_matrices,
                target_radius=self.target_radius
            )
        else:
            radius = self.spectral_radius()
            scale = 1.0

        effective_lambdas = np.asarray(effective_lambdas, dtype=float)
        condition_numbers = np.asarray(condition_numbers, dtype=float)

        self.lasso_converged_history.append(bool(np.all(converged_values)))
        self.lasso_iterations_history.append(float(np.mean(iteration_values)))
        self.lasso_objective_history.append(float(np.mean(objective_values)))
        self.A_sparse_scale_history.append(float(scale))
        self.A_radius_after_sparse_history.append(float(radius))
        self.A_nonzero_offdiag_history.append(self.count_nonzero_offdiag_A())
        self.lambda_A_effective_history.append(effective_lambdas.copy())
        self.moment_solver_condition_number_history.append(
            condition_numbers.copy()
        )

        self.mean_lambda_A_effective = float(np.mean(effective_lambdas))
        self.median_lambda_A_effective = float(np.median(effective_lambdas))
        self.min_lambda_A_effective = float(np.min(effective_lambdas))
        self.max_lambda_A_effective = float(np.max(effective_lambdas))
        self.moment_solver_condition_number_mean = float(
            np.mean(condition_numbers)
        )
        self.moment_solver_condition_number_max = float(
            np.max(condition_numbers)
        )

        # Q and R deliberately remain at their fixed initialized values.
        self._refresh_companion_matrices()
