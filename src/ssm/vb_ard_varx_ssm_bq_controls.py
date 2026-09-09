"""Level-1 Hybrid Kalman + VB-ARD with point B and diagonal Q controls."""

import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.ssm.vb_ard_varx_ssm_b_controls import (
    HybridVBARDVARXSSMKnownCWithBControls,
)


class HybridVBARDVARXSSMKnownCWithBQControls(
        HybridVBARDVARXSSMKnownCWithBControls):
    """Estimate A variationally, B pointwise, and optionally diagonal Q."""

    Q_MODES = ("fixed_true", "diag_raw", "diag_floor",
               "diag_shrink_scalar", "diag_shrink_auto")
    FLOOR_MODES = ("none", "relative_to_mean_diag", "absolute")

    def __init__(self, *args, estimate_Q=False, Q_update_mode="fixed_true",
                 Q_floor_mode="none", Q_floor_value=0.0,
                 Q_shrinkage_rho=0.0, Q_update_damping=0.5,
                 include_A_posterior_uncertainty_in_Q=True,
                 tol_Q_change=None, initial_Q=None,
                 initial_alpha_mean=None, **kwargs):
        if Q_update_mode not in self.Q_MODES:
            raise ValueError(f"Q_update_mode must be one of {self.Q_MODES}.")
        if Q_floor_mode not in self.FLOOR_MODES:
            raise ValueError(f"Q_floor_mode must be one of {self.FLOOR_MODES}.")
        if not 0.0 <= Q_shrinkage_rho <= 1.0:
            raise ValueError("Q_shrinkage_rho must be in [0, 1].")
        if not 0.0 < Q_update_damping <= 1.0:
            raise ValueError("Q_update_damping must be in (0, 1].")
        if Q_floor_value < 0:
            raise ValueError("Q_floor_value must be nonnegative.")
        if estimate_Q != (Q_update_mode != "fixed_true"):
            raise ValueError("estimate_Q and Q_update_mode are inconsistent.")
        self.estimate_Q = bool(estimate_Q)
        self.Q_update_mode = Q_update_mode
        self.Q_floor_mode = Q_floor_mode
        self.Q_floor_value = float(Q_floor_value)
        self.Q_shrinkage_rho = float(Q_shrinkage_rho)
        self.Q_update_damping = float(Q_update_damping)
        self.include_A_posterior_uncertainty_in_Q = bool(
            include_A_posterior_uncertainty_in_Q)
        self.initial_Q = None if initial_Q is None else np.asarray(initial_Q, float).copy()
        self.initial_alpha_mean = (None if initial_alpha_mean is None else
                                   np.asarray(initial_alpha_mean, float).copy())
        super().__init__(*args, **kwargs)
        self.tol_Q_change = (self.tol_A_change if tol_Q_change is None
                             else float(tol_Q_change))

    def _proxy_residual_Q(self, y, u, A, B):
        proxy = np.asarray(y, float) @ np.linalg.pinv(self.C).T
        u = np.asarray(u, float)
        if u.ndim == 1:
            u = u[:, None]
        residuals = []
        for t in range(max(self.na, self.nb-1), len(u)):
            prediction = sum(A[lag-1] @ proxy[t-lag]
                             for lag in range(1, self.na+1))
            prediction += sum(B[lag] @ u[t-lag]
                              for lag in range(self.nb))
            residuals.append(proxy[t]-prediction)
        if not residuals:
            raise ValueError("Too few observations for proxy residual Q initialization.")
        diagonal = np.mean(np.asarray(residuals)**2, axis=0)
        diagonal = np.maximum(diagonal, 1e-6)
        return np.diag(diagonal)

    def _stabilize_Q_candidate(self, raw_diagonal):
        diagonal = np.asarray(raw_diagonal, float).copy()
        if np.any(~np.isfinite(diagonal)):
            raise FloatingPointError("Non-finite raw Q diagonal.")
        diagonal = np.maximum(diagonal, 1e-10)
        mean_diagonal = float(np.mean(diagonal))
        if self.Q_floor_mode == "relative_to_mean_diag":
            diagonal = np.maximum(diagonal, self.Q_floor_value*mean_diagonal)
        elif self.Q_floor_mode == "absolute":
            diagonal = np.maximum(diagonal, self.Q_floor_value)
        if self.Q_update_mode == "diag_shrink_scalar":
            diagonal = ((1.0-self.Q_shrinkage_rho)*diagonal +
                        self.Q_shrinkage_rho*np.mean(diagonal))
        elif self.Q_update_mode == "diag_shrink_auto":
            # Deterministic dispersion-based shrinkage, bounded in [0, 1].
            variance = float(np.var(diagonal, ddof=1)) if len(diagonal)>1 else 0.0
            rho = variance/(variance+mean_diagonal**2+1e-12)
            diagonal = (1.0-rho)*diagonal + rho*np.mean(diagonal)
            self.Q_shrinkage_rho_last_ = float(rho)
        return np.diag(np.maximum(diagonal, 1e-10))

    def _Q_candidate(self, smooth_result, u):
        stats = self._compute_posterior_sufficient_statistics(smooth_result, u)
        n = max(stats["n_transitions"], 1)
        values = np.zeros(self.n_states)
        for target in range(self.n_states):
            beta = self._beta_means_[target]
            value = stats["S_xx"][target]
            value -= 2.0*beta @ stats["S_zx"][:, target]
            value += beta @ stats["S_zz"] @ beta
            if self.include_A_posterior_uncertainty_in_Q:
                value += np.trace(self.A_row_covariances_[target] @ stats["S_zz"])
            values[target] = value/n
        return self._stabilize_Q_candidate(values)

    def _update_Q(self, smooth_result, u):
        if not self.estimate_Q:
            return self.Q.copy(), self.Q.copy()
        candidate = self._Q_candidate(smooth_result, u)
        updated = ((1.0-self.Q_update_damping)*self.Q +
                   self.Q_update_damping*candidate)
        updated = np.diag(np.maximum(np.diag(updated), 1e-10))
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError("Non-finite damped Q update.")
        self.Q = updated
        return candidate, updated

    def fit(self, y, u):
        y, u = np.asarray(y,float), np.asarray(u,float)
        if u.ndim == 1: u = u[:,None]
        self.B_matrices = self._initialize_B(y,u)
        self.B_initial_matrices_ = [B.copy() for B in self.B_matrices]
        A = self._initial_A()
        if self.initial_Q is not None:
            if self.initial_Q.shape != self.Q.shape or np.any(np.diag(self.initial_Q) <= 0):
                raise ValueError("initial_Q has incompatible dimensions or nonpositive diagonal.")
            self.Q = np.diag(np.diag(self.initial_Q))
            self.Q_initialization_mode_ = "warm_start_previous_window"
        elif self.estimate_Q:
            self.Q = self._proxy_residual_Q(y,u,A,self.B_matrices)
            self.Q_initialization_mode_ = "pinv_proxy_transition_residual_diag"
        else:
            self.Q_initialization_mode_ = "fixed_true_Q"
        self.Q_initial_ = self.Q.copy()
        if self.initial_alpha_mean is None:
            self.alpha_mean_=np.full((self.n_states,self.n_states),self.a0/self.b0)
            np.fill_diagonal(self.alpha_mean_,np.nan)
        else:
            if self.initial_alpha_mean.shape != (self.n_states,self.n_states):
                raise ValueError("initial_alpha_mean has incompatible dimensions.")
            self.alpha_mean_=self.initial_alpha_mean.copy()
            offdiag=~np.eye(self.n_states,dtype=bool)
            if np.any(~np.isfinite(self.alpha_mean_[offdiag])) or np.any(self.alpha_mean_[offdiag] <= 0):
                raise ValueError("initial_alpha_mean off-diagonal values must be finite and positive.")
            np.fill_diagonal(self.alpha_mean_,np.nan)
        names=("objective_history_","A_change_history_","B_change_history_",
          "Q_change_history_","alpha_change_history_","A_absolute_change_history_",
          "B_absolute_change_history_","Q_absolute_change_history_",
          "alpha_absolute_change_history_","loglike_absolute_change_history_",
          "loglike_relative_change_history_","A_mean_history_","B_matrices_history_",
          "Q_candidate_history_","Q_history_","alpha_mean_history_",
          "spectral_radius_history_","expected_transition_sse_history_",
          "rescaling_history_","spectral_radius_before_rescaling_history_",
          "spectral_radius_after_rescaling_history_","rescaling_factor_history_")
        for name in names:setattr(self,name,[])
        self.converged_=False;previous_objective=None
        offdiag=~np.eye(self.n_states,dtype=bool)
        for iteration in range(self.max_iter):
            smooth_result=self.smooth(y,u,A)
            stats=self._compute_posterior_sufficient_statistics(smooth_result,u)
            old_beta=self._pack_A(A);old_B=np.asarray(self.B_matrices).copy()
            old_Q=self.Q.copy();old_alpha=self.alpha_mean_.copy()
            self._update_q_A(stats)
            radius_before=var_companion_spectral_radius(self.A_mean_matrices_)
            rescaled=radius_before>=.98
            radius_after,scale=radius_before,1.
            if rescaled:
                self.A_mean_matrices_,radius_after,scale=stabilize_A_matrices_if_needed(
                    self.A_mean_matrices_,target_radius=.90)
                self._beta_means_=self._pack_A(self.A_mean_matrices_)
            self._update_q_alpha();self._update_B(smooth_result,u)
            # Recompute centered sufficient statistics with newly updated B.
            candidate,Q_after=self._update_Q(smooth_result,u)
            A=[matrix.copy() for matrix in self.A_mean_matrices_]
            self.smooth_result_=self.smooth(y,u,A)
            objective=float(self.smooth_result_["log_likelihood"])
            A_abs=float(np.linalg.norm(self._beta_means_-old_beta))
            B_abs=float(np.linalg.norm(np.asarray(self.B_matrices)-old_B))
            Q_abs=float(np.linalg.norm(self.Q-old_Q))
            alpha_abs=float(np.linalg.norm(self.alpha_mean_[offdiag]-old_alpha[offdiag]))
            changes=(A_abs/max(np.linalg.norm(old_beta),1e-12),
                     B_abs/max(np.linalg.norm(old_B),1e-12),
                     Q_abs/max(np.linalg.norm(old_Q),1e-12),
                     alpha_abs/max(np.linalg.norm(old_alpha[offdiag]),1e-12))
            ll_abs=np.inf if previous_objective is None else abs(objective-previous_objective)
            ll_rel=np.inf if previous_objective is None else ll_abs/max(abs(previous_objective),1.)
            self.objective_history_.append(objective)
            for name,value in zip(("A_change_history_","B_change_history_","Q_change_history_","alpha_change_history_"),changes):getattr(self,name).append(float(value))
            for name,value in zip(("A_absolute_change_history_","B_absolute_change_history_","Q_absolute_change_history_","alpha_absolute_change_history_"),(A_abs,B_abs,Q_abs,alpha_abs)):getattr(self,name).append(value)
            self.loglike_absolute_change_history_.append(ll_abs);self.loglike_relative_change_history_.append(ll_rel)
            self.A_mean_history_.append(np.asarray(A).copy());self.B_matrices_history_.append(np.asarray(self.B_matrices).copy())
            self.Q_candidate_history_.append(candidate.copy());self.Q_history_.append(Q_after.copy());self.alpha_mean_history_.append(self.alpha_mean_.copy())
            self.spectral_radius_history_.append(var_companion_spectral_radius(A));self.rescaling_history_.append(bool(rescaled))
            self.spectral_radius_before_rescaling_history_.append(float(radius_before))
            self.spectral_radius_after_rescaling_history_.append(float(radius_after))
            self.rescaling_factor_history_.append(float(scale))
            self.expected_transition_sse_history_.append(self._expected_transition_sse(stats))
            Q_ok=True if not self.estimate_Q else changes[2]<self.tol_Q_change
            if changes[0]<self.tol_A_change and changes[1]<self.tol_B_change and Q_ok and changes[3]<self.tol_alpha_change and ll_rel<self.tol_objective:
                self.converged_=True;break
            previous_objective=objective
        self.n_iter_=iteration+1
        self.filtered_state_mean_=self.smooth_result_["filter"]["x_filt"][:,:self.n_states]
        self.smoothed_state_mean_=self.smooth_result_["smoother"]["x_smooth"][:,:self.n_states]
        self._refresh_derived();self.diagnostics_=self.get_diagnostics()
        self.diagnostics_.update({"Q_update_mode":self.Q_update_mode,
          "Q_initialization_mode":self.Q_initialization_mode_,
          "final_Q_change_norm":self.Q_change_history_[-1],
          "Q_update_damping":self.Q_update_damping,
          "include_A_posterior_uncertainty_in_Q":self.include_A_posterior_uncertainty_in_Q})
        return self


__all__=["HybridVBARDVARXSSMKnownCWithBQControls"]
