"""Controlled fixed-parameter variants of the posterior-moment VARX EM.

These classes are diagnostic tools.  They use the same posterior moments and
group-lasso objective as Experiment 32I, while holding either A or B at a
supplied value throughout initialization and every M-step.
"""

import numpy as np

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
    solve_standardized_covariance_group_lasso,
)
from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.em_varx_p_known_c_l1_mstep_posterior_moments import (
    build_posterior_moment_statistics,
)


def _copy_matrices(matrices, expected_length, name):
    copied = [np.asarray(matrix, dtype=float).copy() for matrix in matrices]
    if len(copied) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} lag matrices.")
    return copied


class _ControlledGroupLassoBase(
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov
):
    """Shared history bookkeeping for exact controlled M-steps."""

    def _record_controlled_solver(self, fits, radius, scale):
        effective = np.asarray([fit["lambda_effective"] for fit in fits], dtype=float)
        conditions = np.asarray([fit["condition_number"] for fit in fits], dtype=float)
        self.lambda_A_group_effective_history.append(effective.copy())
        self.group_solver_converged_history.append(
            bool(np.all([fit["converged"] for fit in fits]))
        )
        self.group_solver_iterations_history.append(
            float(np.mean([fit["n_iterations"] for fit in fits]))
        )
        self.group_solver_objective_history.append(
            float(np.mean([fit["objective"] for fit in fits]))
        )
        self.A_sparse_scale_history.append(float(scale))
        self.A_radius_after_sparse_history.append(float(radius))
        self.A_nonzero_offdiag_history.append(
            self.count_nonzero_offdiag_A_coefficients()
        )
        self.mean_lambda_A_group_effective = float(np.mean(effective))
        self.median_lambda_A_group_effective = float(np.median(effective))
        self.min_lambda_A_group_effective = float(np.min(effective))
        self.max_lambda_A_group_effective = float(np.max(effective))
        self.moment_solver_condition_number_mean = float(np.mean(conditions))
        self.moment_solver_condition_number_max = float(np.max(conditions))


class EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB(_ControlledGroupLassoBase):
    """Estimate A while holding all B lag matrices fixed exactly."""

    def __init__(self, *args, fixed_B_matrices, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_B_matrices = _copy_matrices(fixed_B_matrices, self.nb, "fixed_B_matrices")

    def _initialize_from_proxy(self, y, u):
        super()._initialize_from_proxy(y, u)
        self.B_matrices = _copy_matrices(self.fixed_B_matrices, self.nb, "fixed_B_matrices")
        self._refresh_companion_matrices()

    def _m_step(self, y, u, smooth_result):
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u[:, None]
        n_sources = self.C.shape[1]
        n_inputs = u.shape[1]
        moments = build_posterior_moment_statistics(
            smooth_result, u, self.na, self.nb, n_sources
        )
        G, H = moments["G"], moments["H"]
        n_A = self.na * n_sources
        A_columns = np.arange(n_A, dtype=np.int64)
        B_columns = np.arange(n_A, G.shape[0], dtype=np.int64)
        beta_matrix = np.zeros((n_sources, G.shape[0]))
        fits = []

        for target in range(n_sources):
            fixed_B = np.concatenate([B[target] for B in self.fixed_B_matrices])
            groups, ridge = self._make_groups_and_ridge(target, n_sources, n_inputs)
            forbidden = set(self._constraint_columns_for_target(target, n_sources))
            allowed_A = np.asarray([c for c in A_columns if int(c) not in forbidden], dtype=np.int64)
            lookup = {int(c): i for i, c in enumerate(allowed_A)}
            restricted_groups = [
                np.asarray([lookup[int(c)] for c in group if int(c) in lookup], dtype=np.int64)
                for group in groups
            ]
            restricted_groups = [group for group in restricted_groups if len(group)]
            conditional_h = H[allowed_A, target] - G[np.ix_(allowed_A, B_columns)] @ fixed_B
            fit = solve_standardized_covariance_group_lasso(
                G=G[np.ix_(allowed_A, allowed_A)], h=conditional_h,
                groups=restricted_groups,
                lambda_fraction=self.lambda_A_group_fraction,
                ridge_penalty=ridge[allowed_A],
                initial_beta=self._initial_beta_for_target(target)[allowed_A],
                max_iter=self.group_solver_max_iter, tol=self.group_solver_tol,
            )
            beta_matrix[target, allowed_A] = fit["beta"]
            beta_matrix[target, B_columns] = fixed_B
            fits.append(fit)

        self.A_matrices, self.B_matrices = self._beta_matrix_to_A_B(
            beta_matrix, n_sources, n_inputs
        )
        self._apply_zero_constraints_to_A_matrices()
        if self.stabilize_A:
            self.A_matrices, radius, scale = stabilize_A_matrices_if_needed(
                self.A_matrices, self.target_radius
            )
        else:
            radius, scale = self.spectral_radius(), 1.0
        self.B_matrices = _copy_matrices(self.fixed_B_matrices, self.nb, "fixed_B_matrices")
        self._record_controlled_solver(fits, radius, scale)
        self._refresh_companion_matrices()


class EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedA(_ControlledGroupLassoBase):
    """Estimate B while holding all A lag matrices fixed exactly."""

    def __init__(self, *args, fixed_A_matrices, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_A_matrices = _copy_matrices(fixed_A_matrices, self.na, "fixed_A_matrices")

    def _initialize_from_proxy(self, y, u):
        super()._initialize_from_proxy(y, u)
        self.A_matrices = _copy_matrices(self.fixed_A_matrices, self.na, "fixed_A_matrices")
        self._refresh_companion_matrices()

    def _m_step(self, y, u, smooth_result):
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u[:, None]
        n_sources = self.C.shape[1]
        n_inputs = u.shape[1]
        moments = build_posterior_moment_statistics(
            smooth_result, u, self.na, self.nb, n_sources
        )
        G, H = moments["G"], moments["H"]
        n_A = self.na * n_sources
        A_columns = np.arange(n_A, dtype=np.int64)
        B_columns = np.arange(n_A, G.shape[0], dtype=np.int64)
        beta_matrix = np.zeros((n_sources, G.shape[0]))
        fits = []

        for target in range(n_sources):
            fixed_A = np.concatenate([A[target] for A in self.fixed_A_matrices])
            conditional_h = H[B_columns, target] - G[np.ix_(B_columns, A_columns)] @ fixed_A
            G_B = G[np.ix_(B_columns, B_columns)]
            ridge = self.ridge_B * np.ones(len(B_columns))
            smooth_G = 0.5 * (G_B + G_B.T) + np.diag(ridge)
            beta_B = np.linalg.pinv(smooth_G) @ conditional_h
            beta_matrix[target, A_columns] = fixed_A
            beta_matrix[target, B_columns] = beta_B
            residual = smooth_G @ beta_B - conditional_h
            fits.append({
                "lambda_effective": 0.0, "condition_number": float(np.linalg.cond(smooth_G)),
                "converged": True, "n_iterations": 1,
                "objective": float(0.5 * beta_B @ smooth_G @ beta_B - conditional_h @ beta_B),
            })

        self.A_matrices, self.B_matrices = self._beta_matrix_to_A_B(
            beta_matrix, n_sources, n_inputs
        )
        self.A_matrices = _copy_matrices(self.fixed_A_matrices, self.na, "fixed_A_matrices")
        radius, scale = self.spectral_radius(), 1.0
        self._record_controlled_solver(fits, radius, scale)
        self._refresh_companion_matrices()
