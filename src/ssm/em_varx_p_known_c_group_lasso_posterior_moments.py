import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

from src.ssm.em_varx_p_known_c_l1_mstep import (
    stabilize_A_matrices_if_needed
)
from src.ssm.em_varx_p_known_c_l1_mstep_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartL1PosteriorMomentMstepFixedCov,
    build_posterior_moment_statistics
)


def make_group_zero_constraints(source, target, na):
    """Constrain one directed A lag group to zero for every VAR lag."""

    if na < 1:
        raise ValueError("na must be positive.")

    return [
        {
            "source": int(source),
            "target": int(target),
            "lags": list(range(1, int(na) + 1))
        }
    ]


def _group_proximal_gradient_core_python(
        G,
        h,
        ridge,
        group_indices,
        group_sizes,
        lambda_effective,
        beta,
        step_size,
        max_iter,
        tol
):
    """Full-covariance proximal gradient with disjoint group-L2 prox."""

    converged = False
    extrapolated = beta.copy()
    momentum = 1.0

    for iteration in range(max_iter):
        gradient = G @ extrapolated - h + ridge * extrapolated
        candidate = extrapolated - step_size * gradient

        for group_index in range(group_indices.shape[0]):
            group_size = group_sizes[group_index]
            norm_squared = 0.0

            for position in range(group_size):
                column = group_indices[group_index, position]
                norm_squared += candidate[column] ** 2

            group_norm = np.sqrt(norm_squared)

            if group_norm <= step_size * lambda_effective:
                shrinkage = 0.0
            else:
                shrinkage = 1.0 - (
                    step_size * lambda_effective / group_norm
                )

            for position in range(group_size):
                column = group_indices[group_index, position]
                candidate[column] *= shrinkage

        max_delta = np.max(np.abs(candidate - beta))
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)
        )
        extrapolated = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - beta)
        beta = candidate
        momentum = next_momentum

        if max_delta < tol:
            converged = True
            break

    return beta, iteration + 1, converged


if njit is not None:
    _group_proximal_gradient_core = njit(
        cache=True,
        nogil=True
    )(_group_proximal_gradient_core_python)
else:
    _group_proximal_gradient_core = _group_proximal_gradient_core_python


def solve_standardized_covariance_group_lasso(
        G,
        h,
        groups,
        lambda_fraction,
        ridge_penalty,
        initial_beta=None,
        max_iter=5000,
        tol=1e-7,
        scale_epsilon=1e-12
):
    """Solve the standardized covariance-form group-lasso objective."""

    G = np.asarray(G, dtype=float)
    h = np.asarray(h, dtype=float)
    ridge_penalty = np.asarray(ridge_penalty, dtype=float)
    n_features = h.shape[0]

    if G.shape != (n_features, n_features):
        raise ValueError("G must be square and match h.")

    if ridge_penalty.shape != (n_features,):
        raise ValueError("ridge_penalty must match h.")

    if lambda_fraction < 0.0:
        raise ValueError("lambda_fraction must be nonnegative.")

    normalized_groups = []
    used_columns = set()

    for group in groups:
        group = np.asarray(group, dtype=np.int64)

        if group.ndim != 1 or group.size == 0:
            raise ValueError("Each group must be a nonempty index vector.")

        if np.any(group < 0) or np.any(group >= n_features):
            raise ValueError("Group index is outside the coefficient vector.")

        for column in group:
            column = int(column)

            if column in used_columns:
                raise ValueError("Penalized groups must be disjoint.")

            used_columns.add(column)

        normalized_groups.append(group)

    G = 0.5 * (G + G.T)
    scale = np.sqrt(np.maximum(np.diag(G), scale_epsilon))
    G_std = G / np.outer(scale, scale)
    G_std = 0.5 * (G_std + G_std.T)
    h_std = h / scale

    if normalized_groups:
        lambda_max = float(
            max(np.linalg.norm(h_std[group]) for group in normalized_groups)
        )
    else:
        lambda_max = 0.0

    lambda_effective = float(lambda_fraction) * lambda_max

    if initial_beta is None:
        beta_std = np.zeros(n_features)
    else:
        initial_beta = np.asarray(initial_beta, dtype=float)

        if initial_beta.shape != (n_features,):
            raise ValueError("initial_beta must match h.")

        beta_std = initial_beta * scale

    maximum_group_size = max(
        (group.size for group in normalized_groups),
        default=1
    )
    group_indices = np.zeros(
        (len(normalized_groups), maximum_group_size),
        dtype=np.int64
    )
    group_sizes = np.zeros(len(normalized_groups), dtype=np.int64)

    for group_index, group in enumerate(normalized_groups):
        group_indices[group_index, :group.size] = group
        group_sizes[group_index] = group.size

    smooth_hessian = G_std + np.diag(ridge_penalty)
    largest_eigenvalue = float(np.max(np.linalg.eigvalsh(smooth_hessian)))
    step_size = 1.0 / max(largest_eigenvalue, scale_epsilon)

    beta_std, n_iterations, converged = _group_proximal_gradient_core(
        np.ascontiguousarray(G_std),
        np.ascontiguousarray(h_std),
        np.ascontiguousarray(ridge_penalty),
        np.ascontiguousarray(group_indices),
        np.ascontiguousarray(group_sizes),
        float(lambda_effective),
        np.ascontiguousarray(beta_std),
        float(step_size),
        int(max_iter),
        float(tol)
    )

    beta = beta_std / scale
    group_penalty = sum(
        np.linalg.norm(beta_std[group])
        for group in normalized_groups
    )
    objective = float(
        0.5 * beta_std @ G_std @ beta_std
        - h_std @ beta_std
        + lambda_effective * group_penalty
        + 0.5 * np.sum(ridge_penalty * beta_std ** 2)
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
        "condition_number": float(np.linalg.cond(smooth_hessian)),
        "step_size": step_size
    }


class EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
        EMVARXPSSMKnownCConstrainedWarmStartL1PosteriorMomentMstepFixedCov
):
    """Posterior-moment VARX EM with source-to-target lag group sparsity."""

    def __init__(
            self,
            *args,
            lambda_A_group_fraction=0.0,
            group_solver_max_iter=5000,
            group_solver_tol=1e-7,
            estimate_Q=False,
            estimate_R=False,
            **kwargs
    ):
        if estimate_Q or estimate_R:
            raise ValueError(
                "Experiment 32I requires estimate_Q=False and estimate_R=False."
            )

        if lambda_A_group_fraction < 0.0:
            raise ValueError("lambda_A_group_fraction must be nonnegative.")

        kwargs.pop("lambda_A_fraction", None)

        super().__init__(
            *args,
            lambda_A_fraction=0.0,
            estimate_Q=False,
            estimate_R=False,
            **kwargs
        )

        self.lambda_A_group_fraction = float(lambda_A_group_fraction)
        self.group_solver_max_iter = int(group_solver_max_iter)
        self.group_solver_tol = float(group_solver_tol)

        self.lambda_A_group_effective_history = []
        self.group_solver_converged_history = []
        self.group_solver_iterations_history = []
        self.group_solver_objective_history = []

        self.mean_lambda_A_group_effective = np.nan
        self.median_lambda_A_group_effective = np.nan
        self.min_lambda_A_group_effective = np.nan
        self.max_lambda_A_group_effective = np.nan

    def _make_groups_and_ridge(self, target, n_sources, n_inputs):
        groups = []
        n_features = self.na * n_sources + self.nb * n_inputs
        ridge = np.zeros(n_features)

        for source in range(n_sources):
            columns = np.asarray(
                [lag * n_sources + source for lag in range(self.na)],
                dtype=np.int64
            )

            if source == target:
                ridge[columns] = self.ridge_A_diag
            else:
                groups.append(columns)
                ridge[columns] = self.ridge_A_offdiag

        ridge[self.na * n_sources:] = self.ridge_B
        return groups, ridge

    def _A_group_norms(self, include_diagonal):
        n_sources = self.A_matrices[0].shape[0]
        norms = []

        for target in range(n_sources):
            for source in range(n_sources):
                if not include_diagonal and target == source:
                    continue

                norms.append(
                    np.sqrt(
                        sum(
                            float(A[target, source]) ** 2
                            for A in self.A_matrices
                        )
                    )
                )

        return np.asarray(norms, dtype=float)

    def count_nonzero_offdiag_A_coefficients(self, threshold=1e-8):
        return self.count_nonzero_offdiag_A(threshold=threshold)

    def count_nonzero_total_A_coefficients(self, threshold=1e-8):
        return self.count_nonzero_total_A(threshold=threshold)

    def count_nonzero_offdiag_A_groups(self, threshold=1e-8):
        return int(np.sum(self._A_group_norms(False) > threshold))

    def count_nonzero_total_A_groups(self, threshold=1e-8):
        return int(np.sum(self._A_group_norms(True) > threshold))

    def mean_offdiag_A_group_norm(self):
        return float(np.mean(self._A_group_norms(False)))

    def median_offdiag_A_group_norm(self):
        return float(np.median(self._A_group_norms(False)))

    def max_offdiag_A_group_norm(self):
        return float(np.max(self._A_group_norms(False)))

    def _m_step(self, y, u, smooth_result):
        """Update A/B with grouped lag penalties; keep Q and R fixed."""

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
            groups, ridge = self._make_groups_and_ridge(
                target=target,
                n_sources=n_sources,
                n_inputs=n_inputs
            )
            forbidden = set(
                self._constraint_columns_for_target(
                    target=target,
                    n_latent=n_sources
                )
            )
            allowed = np.asarray(
                [
                    column
                    for column in range(G.shape[0])
                    if column not in forbidden
                ],
                dtype=np.int64
            )
            allowed_lookup = {
                int(column): position
                for position, column in enumerate(allowed)
            }
            restricted_groups = []

            for group in groups:
                restricted_group = [
                    allowed_lookup[int(column)]
                    for column in group
                    if int(column) in allowed_lookup
                ]

                if restricted_group:
                    restricted_groups.append(
                        np.asarray(restricted_group, dtype=np.int64)
                    )

            initial_beta = self._initial_beta_for_target(target)
            fit = solve_standardized_covariance_group_lasso(
                G=G[np.ix_(allowed, allowed)],
                h=H[allowed, target],
                groups=restricted_groups,
                lambda_fraction=self.lambda_A_group_fraction,
                ridge_penalty=ridge[allowed],
                initial_beta=initial_beta[allowed],
                max_iter=self.group_solver_max_iter,
                tol=self.group_solver_tol
            )

            beta_matrix[target, allowed] = fit["beta"]
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

        self.lambda_A_group_effective_history.append(
            effective_lambdas.copy()
        )
        self.group_solver_converged_history.append(
            bool(np.all(converged_values))
        )
        self.group_solver_iterations_history.append(
            float(np.mean(iteration_values))
        )
        self.group_solver_objective_history.append(
            float(np.mean(objective_values))
        )
        self.A_sparse_scale_history.append(float(scale))
        self.A_radius_after_sparse_history.append(float(radius))
        self.A_nonzero_offdiag_history.append(
            self.count_nonzero_offdiag_A_coefficients()
        )

        self.mean_lambda_A_group_effective = float(
            np.mean(effective_lambdas)
        )
        self.median_lambda_A_group_effective = float(
            np.median(effective_lambdas)
        )
        self.min_lambda_A_group_effective = float(
            np.min(effective_lambdas)
        )
        self.max_lambda_A_group_effective = float(
            np.max(effective_lambdas)
        )
        self.moment_solver_condition_number_mean = float(
            np.mean(condition_numbers)
        )
        self.moment_solver_condition_number_max = float(
            np.max(condition_numbers)
        )

        self._refresh_companion_matrices()
