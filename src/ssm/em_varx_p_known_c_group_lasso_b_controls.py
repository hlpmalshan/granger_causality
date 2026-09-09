"""B-control variants for posterior-moment group-lasso VARX EM."""

import numpy as np

from src.ssm.em_varx_p_known_c_group_lasso_controlled import (
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB,
    _ControlledGroupLassoBase,
)
from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    solve_standardized_covariance_group_lasso,
)
from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.em_varx_p_known_c_l1_mstep_posterior_moments import (
    build_posterior_moment_statistics,
)


class EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge(_ControlledGroupLassoBase):
    """Semantic alias for the standard model with an explicit B-ridge multiplier."""

    def __init__(self, *args, ridge_B_base=1e-4, ridge_B_multiplier=1.0, **kwargs):
        if ridge_B_multiplier < 0:
            raise ValueError("ridge_B_multiplier must be nonnegative.")
        self.ridge_B_base = float(ridge_B_base)
        self.ridge_B_multiplier = float(ridge_B_multiplier)
        kwargs["ridge_B"] = self.ridge_B_base * self.ridge_B_multiplier
        super().__init__(*args, **kwargs)


class EMVARXPSSMKnownCGroupLassoPosteriorMomentBBasis(_ControlledGroupLassoBase):
    """Estimate B in a fixed lag-basis while retaining free/group-sparse A."""

    def __init__(self, *args, B_basis_matrix, **kwargs):
        basis = np.asarray(B_basis_matrix, dtype=float)
        if basis.ndim != 2:
            raise ValueError("B_basis_matrix must have shape (nb, n_B_basis).")
        self.B_basis_matrix = basis.copy()
        self.n_B_basis = basis.shape[1]
        self.B_basis_coefficients = None
        super().__init__(*args, **kwargs)
        if basis.shape[0] != self.nb:
            raise ValueError("B_basis_matrix first dimension must equal model nb.")

    def _basis_projection(self, n_sources, n_inputs):
        n_A = self.na * n_sources
        full_features = n_A + self.nb * n_inputs
        reduced_features = n_A + self.n_B_basis * n_inputs
        projection = np.zeros((full_features, reduced_features))
        projection[:n_A, :n_A] = np.eye(n_A)
        for lag in range(self.nb):
            for basis_index in range(self.n_B_basis):
                for input_index in range(n_inputs):
                    projection[
                        n_A + lag * n_inputs + input_index,
                        n_A + basis_index * n_inputs + input_index,
                    ] = self.B_basis_matrix[lag, basis_index]
        return projection

    def _initial_reduced_beta(self, target, n_sources, n_inputs):
        A_values = np.concatenate([A[target] for A in self.A_matrices])
        B_lags = np.stack([B[target] for B in self.B_matrices], axis=0)
        coefficients = np.linalg.pinv(self.B_basis_matrix) @ B_lags
        return np.concatenate([A_values, coefficients.reshape(-1)])

    def _m_step(self, y, u, smooth_result):
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u[:, None]
        n_sources = self.C.shape[1]
        n_inputs = u.shape[1]
        moments = build_posterior_moment_statistics(
            smooth_result=smooth_result, u=u, na=self.na, nb=self.nb,
            n_sources=n_sources,
        )
        projection = self._basis_projection(n_sources, n_inputs)
        G = projection.T @ moments["G"] @ projection
        H = projection.T @ moments["H"]
        n_A = self.na * n_sources
        beta_reduced = np.zeros((n_sources, G.shape[0]))
        fits = []
        for target in range(n_sources):
            groups = [
                np.asarray([lag * n_sources + source for lag in range(self.na)], dtype=np.int64)
                for source in range(n_sources) if source != target
            ]
            ridge = np.zeros(G.shape[0])
            for source in range(n_sources):
                columns = [lag * n_sources + source for lag in range(self.na)]
                ridge[columns] = self.ridge_A_diag if source == target else self.ridge_A_offdiag
            ridge[n_A:] = self.ridge_B
            forbidden = set(self._constraint_columns_for_target(target, n_sources))
            allowed = np.asarray([c for c in range(G.shape[0]) if c not in forbidden], dtype=np.int64)
            lookup = {int(c): i for i, c in enumerate(allowed)}
            restricted_groups = [
                np.asarray([lookup[int(c)] for c in group if int(c) in lookup], dtype=np.int64)
                for group in groups
            ]
            restricted_groups = [group for group in restricted_groups if len(group)]
            fit = solve_standardized_covariance_group_lasso(
                G=G[np.ix_(allowed, allowed)], h=H[allowed, target],
                groups=restricted_groups,
                lambda_fraction=self.lambda_A_group_fraction,
                ridge_penalty=ridge[allowed],
                initial_beta=self._initial_reduced_beta(target, n_sources, n_inputs)[allowed],
                max_iter=self.group_solver_max_iter, tol=self.group_solver_tol,
            )
            beta_reduced[target, allowed] = fit["beta"]
            fits.append(fit)

        beta_full = beta_reduced @ projection.T
        self.A_matrices, self.B_matrices = self._beta_matrix_to_A_B(
            beta_full, n_sources, n_inputs
        )
        self.B_basis_coefficients = beta_reduced[:, n_A:].reshape(
            n_sources, self.n_B_basis, n_inputs
        ).transpose(0, 2, 1)
        self._apply_zero_constraints_to_A_matrices()
        if self.stabilize_A:
            self.A_matrices, radius, scale = stabilize_A_matrices_if_needed(
                self.A_matrices, self.target_radius
            )
        else:
            radius, scale = self.spectral_radius(), 1.0
        self._record_controlled_solver(fits, radius, scale)
        self._refresh_companion_matrices()


__all__ = [
    "EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB",
    "EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge",
    "EMVARXPSSMKnownCGroupLassoPosteriorMomentBBasis",
]
