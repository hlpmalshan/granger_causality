import numpy as np

try:
    from numba import njit
except ImportError:  # Preserve the reference implementation without Numba.
    njit = None

NUMBA_AVAILABLE = njit is not None

from src.ssm.em_varx_p_known_c_shrinkage import (
    EMVARXPSSMKnownCConstrainedWarmStartShrinkage,
    shrink_covariance_matrix
)


def soft_threshold(
        value,
        threshold
):
    if value > threshold:
        return value - threshold

    if value < -threshold:
        return value + threshold

    return 0.0


def _coordinate_descent_core_python(
        X,
        y,
        l1_penalty,
        ridge_penalty,
        beta,
        column_norms,
        max_iter,
        tol
):
    """Numerical core shared by the Python and Numba implementations."""

    n_samples, n_features = X.shape
    residual = y - X @ beta
    converged = False

    for iteration in range(max_iter):
        max_delta = 0.0

        for j in range(n_features):
            old_value = beta[j]
            residual += X[:, j] * old_value
            rho_j = np.dot(X[:, j], residual) / n_samples
            z_j = column_norms[j] + ridge_penalty[j]

            if z_j <= 1e-12:
                new_value = 0.0
            elif rho_j > l1_penalty[j]:
                new_value = (rho_j - l1_penalty[j]) / z_j
            elif rho_j < -l1_penalty[j]:
                new_value = (rho_j + l1_penalty[j]) / z_j
            else:
                new_value = 0.0

            beta[j] = new_value
            residual -= X[:, j] * new_value
            delta = abs(new_value - old_value)

            if delta > max_delta:
                max_delta = delta

        if max_delta < tol:
            converged = True
            break

    return beta, residual, iteration + 1, converged


if njit is not None:
    _coordinate_descent_core = njit(
        cache=True,
        nogil=True
    )(_coordinate_descent_core_python)
else:
    _coordinate_descent_core = _coordinate_descent_core_python


def companion_spectral_radius_from_A(
        A_matrices
):
    A_matrices = [
        np.asarray(A, dtype=float)
        for A in A_matrices
    ]

    p = len(A_matrices)
    n = A_matrices[0].shape[0]

    companion = np.zeros(
        (n * p, n * p)
    )

    companion[:n, :n * p] = np.hstack(
        A_matrices
    )

    if p > 1:
        companion[n:, :-n] = np.eye(
            n * (p - 1)
        )

    eigvals = np.linalg.eigvals(
        companion
    )

    return float(
        np.max(
            np.abs(
                eigvals
            )
        )
    )


def stabilize_A_matrices_if_needed(
        A_matrices,
        target_radius=0.98
):
    radius = companion_spectral_radius_from_A(
        A_matrices
    )

    if radius <= target_radius:
        return A_matrices, radius, 1.0

    scale = target_radius / (
        radius + 1e-12
    )

    A_stable = [
        scale * A
        for A in A_matrices
    ]

    new_radius = companion_spectral_radius_from_A(
        A_stable
    )

    return A_stable, new_radius, scale


def extract_smoothed_x(
        smooth_result,
        n_sources
):
    """
    Robustly extract the first n_sources dimensions of the smoothed state.
    This allows compatibility with different Kalman smoother output names.
    """

    candidate_keys = [
        "x_smooth",
        "x_smoothed",
        "smoothed_x",
        "x_smooth_mean",
        "smoothed_mean",
        "mean_smooth",
        "x_mean",
        "state_smooth",
        "smoothed_state"
    ]

    # kalman_smooth_varx_p_companion returns the RTS output under
    # smooth_result["smoother"].  Unwrap that container while retaining
    # compatibility with callers that pass the smoother output directly.
    if (
        isinstance(smooth_result, dict)
        and "smoother" in smooth_result
    ):
        smooth_result = smooth_result["smoother"]

    if isinstance(smooth_result, dict):
        
        for key in candidate_keys:

            if key in smooth_result:

                arr = np.asarray(
                    smooth_result[key],
                    dtype=float
                )
                
                if arr.ndim == 2 and arr.shape[1] >= n_sources:
                    return arr[:, :n_sources]

    for key in candidate_keys:

        if hasattr(smooth_result, key):

            arr = np.asarray(
                getattr(smooth_result, key),
                dtype=float
            )

            if arr.ndim == 2 and arr.shape[1] >= n_sources:
                return arr[:, :n_sources]

    raise ValueError(
        "Could not extract smoothed latent state from smooth_result. "
        "Please inspect smooth_result keys/attributes."
    )


def build_varx_design_from_x_u(
        x,
        u,
        na,
        nb
):
    """
    Build common VARX design matrix.

    Model:
        x_t = A1 x_{t-1} + ... + Ap x_{t-p}
              + B0 u_t + ... + Bq u_{t-q}
              + w_t

    Returns:
        Z:
            design matrix

        X_current:
            current latent states x_t

        start:
            first valid time index
    """

    x = np.asarray(
        x,
        dtype=float
    )

    u = np.asarray(
        u,
        dtype=float
    )

    if u.ndim == 1:
        u = u[:, None]

    T, n_sources = x.shape
    _, n_inputs = u.shape

    start = max(
        na,
        nb - 1
    )

    rows = []
    current = []

    for t in range(start, T):

        row = []

        for lag in range(1, na + 1):
            row.extend(
                x[t - lag, :]
            )

        for lag in range(0, nb):
            row.extend(
                u[t - lag, :]
            )

        rows.append(
            row
        )

        current.append(
            x[t, :]
        )

    Z = np.asarray(
        rows,
        dtype=float
    )

    X_current = np.asarray(
        current,
        dtype=float
    )

    return Z, X_current, start


def coordinate_descent_lasso_ridge(
        X,
        y,
        l1_penalty,
        ridge_penalty,
        max_iter=1000,
        tol=1e-6,
        initial_beta=None,
        standardize=True
):
    """
    Solve:

        min_beta 1/(2n)||y - X beta||^2
                 + sum_j l1_j |beta_j|
                 + 1/2 sum_j ridge_j beta_j^2

    The optimization is performed on standardized columns, so lambda_A
    is more comparable across sources/lags.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    l1_penalty = np.asarray(
        l1_penalty,
        dtype=float
    )

    ridge_penalty = np.asarray(
        ridge_penalty,
        dtype=float
    )

    n_samples, n_features = X.shape

    if standardize:

        column_scale = np.sqrt(
            np.mean(
                X ** 2,
                axis=0
            )
        )

        column_scale = np.where(
            column_scale > 1e-12,
            column_scale,
            1.0
        )

        X_work = X / column_scale

    else:

        column_scale = np.ones(
            n_features
        )

        X_work = X.copy()

    if initial_beta is None:

        beta_work = np.zeros(
            n_features
        )

    else:

        beta_original = np.asarray(
            initial_beta,
            dtype=float
        )

        beta_work = beta_original * column_scale

    column_norms = np.mean(
        X_work ** 2,
        axis=0
    )

    beta_work, residual, n_iterations, converged = (
        _coordinate_descent_core(
            # Coordinate descent repeatedly visits columns, so Fortran order
            # makes each X[:, j] contiguous for both Numba and BLAS.
            np.asfortranarray(X_work),
            np.ascontiguousarray(y),
            np.ascontiguousarray(l1_penalty),
            np.ascontiguousarray(ridge_penalty),
            np.ascontiguousarray(beta_work),
            np.ascontiguousarray(column_norms),
            int(max_iter),
            float(tol)
        )
    )

    beta_original = beta_work / column_scale

    final_residual = y - X @ beta_original

    objective = float(
        0.5 * np.mean(
            final_residual ** 2
        )
        + np.sum(
            l1_penalty * np.abs(
                beta_work
            )
        )
        + 0.5 * np.sum(
            ridge_penalty * beta_work ** 2
        )
    )

    return {
        "beta": beta_original,
        "residual": final_residual,
        "objective": objective,
        "n_iterations": n_iterations,
        "converged": converged
    }


class EMVARXPSSMKnownCConstrainedWarmStartL1Mstep(
        EMVARXPSSMKnownCConstrainedWarmStartShrinkage
):
    """
    VARX state-space EM with a row-wise L1-penalized M-step.

    This replaces the 32E post-M-step soft-thresholding with a direct
    coordinate-descent solve for each target source.

    Penalty design:
        off-diagonal A:
            L1 penalty lambda_A_offdiag

        diagonal A:
            no L1 by default; optional weak L1

        B:
            no L1 by default; weak ridge only
    """

    def __init__(
            self,
            *args,
            lambda_A_offdiag=0.0,
            lambda_A_diag=0.0,
            lambda_B=0.0,
            ridge_A_offdiag=1e-4,
            ridge_A_diag=1e-4,
            ridge_B=1e-4,
            lambda_A_lag_decay=1.0,
            lasso_max_iter=1000,
            lasso_tol=1e-6,
            stabilize_A=True,
            target_radius=0.98,
            estimate_Q=True,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.lambda_A_offdiag = float(
            lambda_A_offdiag
        )

        self.lambda_A_diag = float(
            lambda_A_diag
        )

        self.lambda_B = float(
            lambda_B
        )

        self.ridge_A_offdiag = float(
            ridge_A_offdiag
        )

        self.ridge_A_diag = float(
            ridge_A_diag
        )

        self.ridge_B = float(
            ridge_B
        )

        self.lambda_A_lag_decay = float(
            lambda_A_lag_decay
        )

        self.lasso_max_iter = int(
            lasso_max_iter
        )

        self.lasso_tol = float(
            lasso_tol
        )

        self.stabilize_A = bool(
            stabilize_A
        )

        self.target_radius = float(
            target_radius
        )

        self.estimate_Q = bool(
            estimate_Q
        )

        self.lasso_converged_history = []
        self.lasso_iterations_history = []
        self.lasso_objective_history = []
        self.A_sparse_scale_history = []
        self.A_radius_after_sparse_history = []
        self.A_nonzero_offdiag_history = []

    def _make_penalty_vectors(
            self,
            target,
            n_sources,
            n_inputs
    ):
        n_A_features = self.na * n_sources
        n_B_features = self.nb * n_inputs
        n_features = n_A_features + n_B_features

        l1 = np.zeros(
            n_features
        )

        ridge = np.zeros(
            n_features
        )

        # A features
        for lag_index in range(self.na):

            lag_scale = self.lambda_A_lag_decay ** lag_index

            for source in range(n_sources):

                col = lag_index * n_sources + source

                if source == target:

                    l1[col] = self.lambda_A_diag * lag_scale
                    ridge[col] = self.ridge_A_diag

                else:

                    l1[col] = self.lambda_A_offdiag * lag_scale
                    ridge[col] = self.ridge_A_offdiag

        # B features
        offset = n_A_features

        for lag_index in range(self.nb):

            for input_index in range(n_inputs):

                col = offset + lag_index * n_inputs + input_index

                l1[col] = self.lambda_B
                ridge[col] = self.ridge_B

        return l1, ridge

    def _beta_matrix_to_A_B(
            self,
            beta_matrix,
            n_sources,
            n_inputs
    ):
        A_matrices = [
            np.zeros(
                (n_sources, n_sources)
            )
            for _ in range(self.na)
        ]

        B_matrices = [
            np.zeros(
                (n_sources, n_inputs)
            )
            for _ in range(self.nb)
        ]

        for target in range(n_sources):

            beta = beta_matrix[target]

            for lag_index in range(self.na):

                for source in range(n_sources):

                    col = lag_index * n_sources + source

                    A_matrices[lag_index][target, source] = beta[col]

            offset = self.na * n_sources

            for lag_index in range(self.nb):

                for input_index in range(n_inputs):

                    col = offset + lag_index * n_inputs + input_index

                    B_matrices[lag_index][target, input_index] = beta[col]

        return A_matrices, B_matrices

    def count_nonzero_offdiag_A(
            self,
            threshold=1e-8
    ):
        count = 0

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source and abs(A[target, source]) > threshold:
                        count += 1

        return int(
            count
        )

    def count_nonzero_total_A(
            self,
            threshold=1e-8
    ):
        count = 0

        for A in self.A_matrices:

            count += int(
                np.sum(
                    np.abs(A) > threshold
                )
            )

        return int(
            count
        )

    def mean_abs_offdiag_A(
            self
    ):
        values = []

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source:
                        values.append(
                            abs(A[target, source])
                        )

        return float(
            np.mean(
                values
            )
        )

    def max_abs_offdiag_A(
            self
    ):
        values = []

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source:
                        values.append(
                            abs(A[target, source])
                        )

        return float(
            np.max(
                values
            )
        )

    def _m_step(
            self,
            y,
            u,
            smooth_result
    ):
        """
        Row-wise L1-penalized VARX M-step using smoothed posterior mean.

        This is a direct penalized regression solve, not post-hoc
        thresholding of a dense M-step.
        """

        y = np.asarray(
            y,
            dtype=float
        )

        u = np.asarray(
            u,
            dtype=float
        )

        if u.ndim == 1:
            u = u[:, None]

        n_sources = self.C.shape[1]
        n_inputs = u.shape[1]

        x_smooth = extract_smoothed_x(
            smooth_result=smooth_result,
            n_sources=n_sources
        )

        Z, X_current, start = build_varx_design_from_x_u(
            x=x_smooth,
            u=u,
            na=self.na,
            nb=self.nb
        )

        n_targets = n_sources
        n_features = Z.shape[1]

        beta_matrix = np.zeros(
            (n_targets, n_features)
        )

        residual_matrix = np.zeros_like(
            X_current
        )

        lasso_converged = []
        lasso_iterations = []
        lasso_objectives = []

        for target in range(n_targets):

            target_y = X_current[:, target]

            l1, ridge = self._make_penalty_vectors(
                target=target,
                n_sources=n_sources,
                n_inputs=n_inputs
            )

            # Warm-start from previous A/B if available.
            initial_beta = None

            try:

                beta_init = []

                for lag_index in range(self.na):
                    beta_init.extend(
                        self.A_matrices[lag_index][target, :]
                    )

                for lag_index in range(self.nb):
                    beta_init.extend(
                        self.B_matrices[lag_index][target, :]
                    )

                initial_beta = np.asarray(
                    beta_init,
                    dtype=float
                )

            except Exception:

                initial_beta = None

            fit = coordinate_descent_lasso_ridge(
                X=Z,
                y=target_y,
                l1_penalty=l1,
                ridge_penalty=ridge,
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol,
                initial_beta=initial_beta,
                standardize=True
            )

            beta_matrix[target] = fit["beta"]
            residual_matrix[:, target] = fit["residual"]

            lasso_converged.append(
                fit["converged"]
            )

            lasso_iterations.append(
                fit["n_iterations"]
            )

            lasso_objectives.append(
                fit["objective"]
            )

        A_matrices, B_matrices = self._beta_matrix_to_A_B(
            beta_matrix=beta_matrix,
            n_sources=n_sources,
            n_inputs=n_inputs
        )

        self.A_matrices = A_matrices
        self.B_matrices = B_matrices

        if hasattr(
                self,
                "_apply_zero_constraints_to_A_matrices"
        ):
            self._apply_zero_constraints_to_A_matrices()

        if self.stabilize_A:

            self.A_matrices, radius, scale = stabilize_A_matrices_if_needed(
                self.A_matrices,
                target_radius=self.target_radius
            )

        else:

            radius = companion_spectral_radius_from_A(
                self.A_matrices
            )

            scale = 1.0

        # Optional Q update from state residuals.
        if self.estimate_Q:

            Q_new = (
                residual_matrix.T
                @ residual_matrix
                / max(
                    residual_matrix.shape[0],
                    1
                )
            )

            Q_new = 0.5 * (
                Q_new + Q_new.T
            )

            self.Q = shrink_covariance_matrix(
                S=Q_new,
                alpha=self.alpha_Q,
                target=self.shrinkage_target_Q,
                covariance_floor=self.covariance_floor
            )

        # R update from observation residuals.
        if self.estimate_R:

            y_hat = x_smooth @ self.C.T

            D = getattr(
                self,
                "D",
                None
            )

            if D is not None:

                D = np.asarray(
                    D,
                    dtype=float
                )

                y_hat = y_hat + u @ D.T

            obs_residual = y - y_hat

            R_new = (
                obs_residual.T
                @ obs_residual
                / max(
                    obs_residual.shape[0],
                    1
                )
            )

            R_new = 0.5 * (
                R_new + R_new.T
            )

            self.R = shrink_covariance_matrix(
                S=R_new,
                alpha=self.alpha_R,
                target=self.shrinkage_target_R,
                covariance_floor=self.R_floor
            )

        self.lasso_converged_history.append(
            bool(
                np.all(
                    lasso_converged
                )
            )
        )

        self.lasso_iterations_history.append(
            float(
                np.mean(
                    lasso_iterations
                )
            )
        )

        self.lasso_objective_history.append(
            float(
                np.mean(
                    lasso_objectives
                )
            )
        )

        self.A_sparse_scale_history.append(
            scale
        )

        self.A_radius_after_sparse_history.append(
            radius
        )

        self.A_nonzero_offdiag_history.append(
            self.count_nonzero_offdiag_A()
        )

        self._refresh_companion_matrices()
