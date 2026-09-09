"""Point-estimated B controls for Level-1 Hybrid Kalman + VB-ARD.

The latent trajectory remains the exact companion-form Kalman/RTS posterior
evaluated at the current posterior mean of A and point estimate of B.  This is
not full structured variational inference.
"""

import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import (
    make_exogenous_regressor, var_companion_spectral_radius,
)
from src.ssm.vb_ard_varx_ssm import HybridVBARDVARXSSMFixedBKnownC


class HybridVBARDVARXSSMKnownCWithBControls(
        HybridVBARDVARXSSMFixedBKnownC):
    """Hybrid VB-ARD for A with fixed, free, or ridge point-estimated B."""

    MODES = ("fixed", "free", "ridge")

    def __init__(self, na, nb, C, Q, R, B_update_mode="fixed",
                 fixed_B_matrices=None, initial_B_matrices=None,
                 base_B_ridge=1e-6, ridge_B_multiplier=10.0,
                 B_ridge_lambda=None,
                 tol_B_change=None, tol_alpha_change=None, **kwargs):
        if B_update_mode not in self.MODES:
            raise ValueError(f"B_update_mode must be one of {self.MODES}.")
        if base_B_ridge < 0 or ridge_B_multiplier < 0:
            raise ValueError("B ridge values must be nonnegative.")
        supplied = fixed_B_matrices if B_update_mode == "fixed" else initial_B_matrices
        if supplied is None:
            # Dimensions are finalized from u during fit.  The placeholder is
            # only allowed for estimated-B modes and assumes one input, as in 34C.
            if B_update_mode == "fixed":
                raise ValueError("fixed_B_matrices is required in fixed mode.")
            supplied = [np.zeros((np.asarray(C).shape[1], 1)) for _ in range(nb)]
        self.B_update_mode = B_update_mode
        self.fixed_B_matrices = ([np.asarray(B, float).copy() for B in fixed_B_matrices]
                                 if fixed_B_matrices is not None else None)
        self.initial_B_matrices = ([np.asarray(B, float).copy() for B in initial_B_matrices]
                                   if initial_B_matrices is not None else None)
        self.base_B_ridge = float(base_B_ridge)
        self.ridge_B_multiplier = float(ridge_B_multiplier)
        if B_ridge_lambda is not None and B_ridge_lambda < 0:
            raise ValueError("B_ridge_lambda must be nonnegative.")
        self.B_ridge_lambda = (None if B_ridge_lambda is None
                               else float(B_ridge_lambda))
        super().__init__(na=na, nb=nb, C=C, B_matrices=supplied, Q=Q, R=R, **kwargs)
        self.tol_B_change = self.tol_A_change if tol_B_change is None else float(tol_B_change)
        self.tol_alpha_change = self.tol_A_change if tol_alpha_change is None else float(tol_alpha_change)

    @property
    def estimate_B(self):
        return self.B_update_mode != "fixed"

    @property
    def effective_B_ridge(self):
        if self.B_ridge_lambda is not None:
            return self.B_ridge_lambda
        multiplier = 1.0 if self.B_update_mode == "free" else self.ridge_B_multiplier
        return self.base_B_ridge * multiplier

    def _proxy_joint_ridge_B(self, y, u):
        """Initialize B without truth using joint A/B ridge on C+ y."""
        proxy = np.asarray(y, float) @ np.linalg.pinv(self.C).T
        u = np.asarray(u, float)
        if u.ndim == 1:
            u = u[:, None]
        rows, responses = [], []
        start = max(self.na, self.nb - 1)
        for t in range(start, len(u)):
            z = np.concatenate([proxy[t-lag] for lag in range(1, self.na+1)])
            r = make_exogenous_regressor(u, t, self.nb)
            rows.append(np.concatenate([z, r])); responses.append(proxy[t])
        if not rows:
            raise ValueError("Too few observations for proxy joint-ridge B initialization.")
        X, Y = np.asarray(rows), np.asarray(responses)
        ridge = max(self.base_B_ridge, 1e-6)
        coefficients = np.linalg.solve(X.T @ X + ridge*np.eye(X.shape[1]), X.T @ Y)
        gamma = coefficients[self.na*self.n_states:].T
        n_inputs = u.shape[1]
        return [gamma[:, lag*n_inputs:(lag+1)*n_inputs].copy()
                for lag in range(self.nb)]

    def _initialize_B(self, y, u):
        if self.B_update_mode == "fixed":
            self.B_initialization_mode_ = "fixed_true_B"
            return [B.copy() for B in self.fixed_B_matrices]
        if self.initial_B_matrices is not None:
            self.B_initialization_mode_ = "user_supplied_nontruth_initial_B"
            return [B.copy() for B in self.initial_B_matrices]
        self.B_initialization_mode_ = "pinv_proxy_joint_ridge"
        return self._proxy_joint_ridge_B(y, u)

    def _B_sufficient_statistics(self, smooth_result, u):
        means = smooth_result["smoother"]["x_smooth"]
        u = np.asarray(u, float)
        if u.ndim == 1:
            u = u[:, None]
        n_r = self.nb * u.shape[1]
        S_rr = np.zeros((n_r, n_r))
        S_rx = np.zeros((n_r, self.n_states))
        S_rz = np.zeros((n_r, self.na*self.n_states))
        for t in range(1, len(u)):
            r = make_exogenous_regressor(u, t, self.nb)
            z_mean = means[t-1, :self.na*self.n_states]
            x_mean = means[t, :self.n_states]
            S_rr += np.outer(r, r)
            S_rx += np.outer(r, x_mean)
            S_rz += np.outer(r, z_mean)
        return S_rr, S_rx, S_rz, u.shape[1]

    def _update_B(self, smooth_result, u):
        if not self.estimate_B:
            return
        S_rr, S_rx, S_rz, n_inputs = self._B_sufficient_statistics(smooth_result, u)
        # For row i: sum r E[x_i - E[beta_i]^T z] = S_rx - S_rz E[beta_i].
        rhs = S_rx - S_rz @ self._beta_means_.T
        system = S_rr + self.effective_B_ridge * np.eye(S_rr.shape[0])
        gamma = np.linalg.solve(system, rhs).T
        self.B_matrices = [gamma[:, lag*n_inputs:(lag+1)*n_inputs].copy()
                           for lag in range(self.nb)]
        if not all(np.all(np.isfinite(B)) for B in self.B_matrices):
            raise FloatingPointError("Non-finite point estimate in B update.")

    def fit(self, y, u):
        y, u = np.asarray(y, float), np.asarray(u, float)
        if u.ndim == 1:
            u = u[:, None]
        if len(y) != len(u) or y.shape[1] != self.C.shape[0]:
            raise ValueError("y and u dimensions are incompatible.")
        self.B_matrices = self._initialize_B(y, u)
        self.B_initial_matrices_ = [B.copy() for B in self.B_matrices]
        A = self._initial_A()
        self.alpha_mean_ = np.full((self.n_states, self.n_states), self.a0/self.b0)
        np.fill_diagonal(self.alpha_mean_, np.nan)
        self.objective_history_, self.A_change_history_ = [], []
        self.B_change_history_, self.alpha_change_history_ = [], []
        self.A_absolute_change_history_, self.B_absolute_change_history_ = [], []
        self.alpha_absolute_change_history_ = []
        self.loglike_absolute_change_history_, self.loglike_relative_change_history_ = [], []
        self.A_mean_history_, self.B_matrices_history_, self.alpha_mean_history_ = [], [], []
        self.spectral_radius_history_ = []
        self.expected_transition_sse_history_, self.rescaling_history_ = [], []
        self.converged_ = False; previous_objective = None
        offdiag = ~np.eye(self.n_states, dtype=bool)
        for iteration in range(self.max_iter):
            smooth_result = self.smooth(y, u, A)
            stats = self._compute_posterior_sufficient_statistics(smooth_result, u)
            old_beta, old_alpha = self._pack_A(A), self.alpha_mean_.copy()
            old_B = np.asarray(self.B_matrices).copy()
            self._update_q_A(stats)
            radius_before = var_companion_spectral_radius(self.A_mean_matrices_)
            rescaled = radius_before >= 0.98
            if rescaled:
                self.A_mean_matrices_, _, _ = stabilize_A_matrices_if_needed(
                    self.A_mean_matrices_, target_radius=0.90)
                self._beta_means_ = self._pack_A(self.A_mean_matrices_)
            self._update_q_alpha()
            self._update_B(smooth_result, u)
            A = [matrix.copy() for matrix in self.A_mean_matrices_]
            self.smooth_result_ = self.smooth(y, u, A)
            objective = float(self.smooth_result_["log_likelihood"])
            A_absolute_change = float(np.linalg.norm(self._beta_means_-old_beta))
            A_change = float(A_absolute_change / max(np.linalg.norm(old_beta), 1e-12))
            new_B = np.asarray(self.B_matrices)
            B_absolute_change = float(np.linalg.norm(new_B-old_B))
            B_change = float(B_absolute_change / max(np.linalg.norm(old_B), 1e-12))
            alpha_absolute_change = float(np.linalg.norm(
                self.alpha_mean_[offdiag]-old_alpha[offdiag]))
            alpha_change = float(alpha_absolute_change /
                                 max(np.linalg.norm(old_alpha[offdiag]), 1e-12))
            loglike_absolute_change = (np.inf if previous_objective is None else
                                       abs(objective-previous_objective))
            objective_change = (np.inf if previous_objective is None else
                loglike_absolute_change/max(abs(previous_objective),1.0))
            self.objective_history_.append(objective)
            self.A_change_history_.append(A_change); self.B_change_history_.append(B_change)
            self.alpha_change_history_.append(alpha_change)
            self.A_absolute_change_history_.append(A_absolute_change)
            self.B_absolute_change_history_.append(B_absolute_change)
            self.alpha_absolute_change_history_.append(alpha_absolute_change)
            self.loglike_absolute_change_history_.append(loglike_absolute_change)
            self.loglike_relative_change_history_.append(objective_change)
            self.A_mean_history_.append(np.asarray(A).copy())
            self.B_matrices_history_.append(new_B.copy())
            self.alpha_mean_history_.append(self.alpha_mean_.copy())
            self.spectral_radius_history_.append(
                var_companion_spectral_radius(self.A_mean_matrices_))
            self.expected_transition_sse_history_.append(self._expected_transition_sse(stats))
            self.rescaling_history_.append(bool(rescaled))
            if self.verbose:
                print(f"hybrid_vb_ard_{self.B_update_mode}_B iter={iteration+1} "
                      f"loglik={objective:.6g} A={A_change:.3g} B={B_change:.3g} alpha={alpha_change:.3g}")
            if (A_change < self.tol_A_change and B_change < self.tol_B_change and
                    alpha_change < self.tol_alpha_change and
                    objective_change < self.tol_objective):
                self.converged_ = True
                break
            previous_objective = objective
        self.n_iter_ = iteration + 1
        self.filtered_state_mean_ = self.smooth_result_["filter"]["x_filt"][:, :self.n_states]
        self.smoothed_state_mean_ = self.smooth_result_["smoother"]["x_smooth"][:, :self.n_states]
        self._refresh_derived()
        self.diagnostics_ = self.get_diagnostics()
        self.diagnostics_.update({"B_update_mode":self.B_update_mode,
            "B_initialization_mode":self.B_initialization_mode_,
            "effective_B_ridge":self.effective_B_ridge,
            "final_B_change_norm":self.B_change_history_[-1]})
        return self


__all__ = ["HybridVBARDVARXSSMKnownCWithBControls"]
