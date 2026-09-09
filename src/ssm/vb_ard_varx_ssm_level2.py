"""Pilot Level-2 Hybrid VB-ARD with time-averaged A-uncertainty Q inflation."""
import numpy as np
from numba import njit,prange
from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.ssm.vb_ard_varx_ssm import HybridVBARDVARXSSMFixedBKnownC

@njit(cache=True,parallel=True)
def delta_q_diag_numba(row_covariances,mean_zz):
 n=row_covariances.shape[0];d=row_covariances.shape[1];out=np.zeros(n)
 for i in prange(n):
  value=0.
  for a in range(d):
   for b in range(d):value+=row_covariances[i,a,b]*mean_zz[b,a]
  out[i]=max(value,0.)
 return out

class HybridVBARDVARXSSMLevel2QeffDiag(HybridVBARDVARXSSMFixedBKnownC):
 """Approximate Level-2 q(x) update using time-averaged diagonal Qeff."""
 def __init__(self,*args,rho_A_uncertainty=.5,level2_max_outer_iter=10,level2_tol=1e-3,**kwargs):
  super().__init__(*args,**kwargs);self.rho_A_uncertainty=float(rho_A_uncertainty);self.level2_max_outer_iter=int(level2_max_outer_iter);self.level2_tol=float(level2_tol)
  if self.rho_A_uncertainty<0:raise ValueError("rho_A_uncertainty must be nonnegative.")
 def fit_level2(self,y,u,level1_model):
  self.Q_base_=self.Q.copy();self.A_mean_matrices_=[x.copy() for x in level1_model.A_mean_matrices_];self._beta_means_=level1_model._beta_means_.copy();self.A_row_covariances_=[x.copy() for x in level1_model.A_row_covariances_];self.alpha_mean_=level1_model.alpha_mean_.copy();self.alpha_shape_=level1_model.alpha_shape_.copy();self.alpha_rate_=level1_model.alpha_rate_.copy();self.level2_iteration_history_=[];previous_delta=None;self.converged_level2_=False
  A=[x.copy() for x in self.A_mean_matrices_]
  for iteration in range(self.level2_max_outer_iter):
   old_beta=self._pack_A(A);old_trace=sum(np.trace(x) for x in self.A_row_covariances_)
   self.Q=self.Q_base_;base_smooth=self.smooth(y,u,A);base_stats=self._compute_posterior_sufficient_statistics(base_smooth,u);mean_zz=base_stats["S_zz"]/max(base_stats["n_transitions"],1);delta=delta_q_diag_numba(np.asarray(self.A_row_covariances_),mean_zz);Qeff=self.Q_base_+self.rho_A_uncertainty*np.diag(delta)
   self.Q=Qeff;smooth=self.smooth(y,u,A);stats=self._compute_posterior_sufficient_statistics(smooth,u);self.Q=self.Q_base_;self._update_q_A(stats);radius=var_companion_spectral_radius(self.A_mean_matrices_);rescaled=radius>=.98
   if rescaled:self.A_mean_matrices_,_,_=stabilize_A_matrices_if_needed(self.A_mean_matrices_,target_radius=.90);self._beta_means_=self._pack_A(self.A_mean_matrices_)
   self._update_q_alpha();A=[x.copy() for x in self.A_mean_matrices_];new_trace=sum(np.trace(x) for x in self.A_row_covariances_);Achange=np.linalg.norm(self._beta_means_-old_beta)/max(np.linalg.norm(old_beta),1e-12);Cchange=abs(new_trace-old_trace)/max(abs(old_trace),1e-12);Dchange=np.inf if previous_delta is None else np.linalg.norm(delta-previous_delta)/max(np.linalg.norm(previous_delta),1e-12)
   self.level2_iteration_history_.append({"level2_outer_iter":iteration+1,"relative_A_mean_change":Achange,"relative_A_cov_trace_change":Cchange,"relative_Delta_Q_change":Dchange,"Delta_Q_A_trace":delta.sum(),"Delta_Q_A_trace_ratio_to_Q":delta.sum()/max(np.trace(self.Q_base_),1e-12),"Qeff_trace_ratio_to_Q":np.trace(Qeff)/np.trace(self.Q_base_),"spectral_radius_A":var_companion_spectral_radius(A),"rescaled_A_flag":rescaled})
   previous_delta=delta.copy()
   if max(Achange,Cchange,Dchange)<self.level2_tol:self.converged_level2_=True;break
  self.n_level2_outer_iters_=len(self.level2_iteration_history_);self.Delta_Q_A_diag_=previous_delta;self.Q_eff_=self.Q_base_+self.rho_A_uncertainty*np.diag(previous_delta);self.Q=self.Q_eff_;self.smooth_result_=self.smooth(y,u,A);self.Q=self.Q_base_;self.filtered_state_mean_=self.smooth_result_["filter"]["x_filt"][:,:self.n_states];self.smoothed_state_mean_=self.smooth_result_["smoother"]["x_smooth"][:,:self.n_states];self._refresh_derived();return self

__all__=["HybridVBARDVARXSSMLevel2QeffDiag","delta_q_diag_numba"]
