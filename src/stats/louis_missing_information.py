"""Monte-Carlo Louis missing-information diagnostics for latent VARX A rows."""
import numpy as np
from numba import njit,prange
from src.ssm.ssm_varx_p_simulator import make_exogenous_regressor

SAMPLER_LABEL="ffbs_joint_smoother_sample"

def _sample_gaussian(mean,cov,rng,eigen_floor=1e-12):
 cov=.5*(np.asarray(cov)+np.asarray(cov).T);values,vectors=np.linalg.eigh(cov);scale=max(float(np.max(np.abs(values))),1.);values=np.maximum(values,eigen_floor*scale)
 return np.asarray(mean)+vectors@(np.sqrt(values)*rng.standard_normal(len(values)))

def sample_companion_trajectories_ffbs(smooth_result,F,n_samples,random_state=None):
 """Draw joint companion trajectories using filtering backward sampling."""
 filt=smooth_result["filter"];smooth=smooth_result["smoother"];rng=np.random.default_rng(random_state)
 mf,Pf,mp,Pp=filt["x_filt"],filt["P_filt"],filt["x_pred"],filt["P_pred"];T,d=mf.shape;draws=np.empty((int(n_samples),T,d))
 gains=smooth.get("smoother_gains")
 for s in range(int(n_samples)):
  draws[s,-1]=_sample_gaussian(smooth["x_smooth"][-1],smooth["P_smooth"][-1],rng)
  for t in range(T-2,-1,-1):
   J=gains[t] if gains is not None else Pf[t]@F.T@np.linalg.pinv(Pp[t+1]);mean=mf[t]+J@(draws[s,t+1]-mp[t+1]);cov=Pf[t]-J@Pp[t+1]@J.T
   draws[s,t]=_sample_gaussian(mean,cov,rng)
 return draws

def prior_precision_for_row(n_states,na,target,alpha_mean,diagonal_prior_precision):
 values=np.empty(na*n_states)
 for source in range(n_states):
  values[[lag*n_states+source for lag in range(na)]]=diagonal_prior_precision if source==target else alpha_mean[target,source]
 return np.diag(values)

def complete_data_transition_score(companion_trajectory,u,beta,B_matrices,Q,target,na,nb,prior_precision=None):
 trajectory=np.asarray(companion_trajectory);u=np.asarray(u);u=u[:,None] if u.ndim==1 else u;M=Q.shape[0];B=np.hstack(B_matrices);score=np.zeros(na*M)
 for t in range(1,len(u)):
  z=trajectory[t-1,:na*M];offset=B@make_exogenous_regressor(u,t,nb);residual=trajectory[t,target]-offset[target]-beta@z;score+=z*residual/Q[target,target]
 if prior_precision is not None:score-=prior_precision@beta
 return score

def estimate_missing_information(companion_samples,u,beta_rows,B_matrices,Q,na,nb,prior_precisions=None):
 samples=np.asarray(companion_samples);rows=[]
 for target,beta in enumerate(np.asarray(beta_rows)):
  prior=None if prior_precisions is None else prior_precisions[target];scores=np.asarray([complete_data_transition_score(draw,u,beta,B_matrices,Q,target,na,nb,prior) for draw in samples]);cov=np.cov(scores,rowvar=False,ddof=1);rows.append({"scores":scores,"missing_information":.5*(cov+cov.T)})
 return rows

@njit(cache=True,parallel=True)
def _score_samples_numba(samples,u,beta,Bstack,qdiag,na,nb,priors):
 S,T,dstate=samples.shape;M=qdiag.shape[0];d=na*M;out=np.zeros((M,S,d));n_inputs=u.shape[1]
 for target in prange(M):
  for s in range(S):
   for t in range(1,T):
    offset=0.
    for lag in range(nb):
     index=t-lag
     if index>=0:
      for inp in range(n_inputs):offset+=Bstack[target,lag*n_inputs+inp]*u[index,inp]
    prediction=0.
    for a in range(d):prediction+=beta[target,a]*samples[s,t-1,a]
    residual=samples[s,t,target]-offset-prediction
    for a in range(d):out[target,s,a]+=samples[s,t-1,a]*residual/qdiag[target]
   for a in range(d):
    constant=0.
    for b in range(d):constant+=priors[target,a,b]*beta[target,b]
    out[target,s,a]-=constant
 return out

def estimate_missing_information_numba(companion_samples,u,beta_rows,B_matrices,Q,na,nb,prior_precisions):
 u=np.asarray(u,float);u=u[:,None] if u.ndim==1 else u;scores=_score_samples_numba(np.asarray(companion_samples,float),u,np.asarray(beta_rows,float),np.hstack(B_matrices),np.diag(Q).copy(),int(na),int(nb),np.asarray(prior_precisions,float));rows=[]
 for target in range(scores.shape[0]):
  cov=np.cov(scores[target],rowvar=False,ddof=1);rows.append({"scores":scores[target],"missing_information":.5*(cov+cov.T)})
 return rows

def stabilize_observed_information(I_complete,I_missing,eta,eigen_floor_relative=1e-8,eigen_floor_absolute=1e-10):
 complete=.5*(I_complete+I_complete.T);missing=.5*(I_missing+I_missing.T);observed=.5*((complete-eta*missing)+(complete-eta*missing).T);values,vectors=np.linalg.eigh(observed);max_complete=max(float(np.linalg.eigvalsh(complete).max()),eigen_floor_absolute);floor=max(eigen_floor_absolute,eigen_floor_relative*max_complete);clipped=np.maximum(values,floor);stable=vectors@np.diag(clipped)@vectors.T;cov=vectors@np.diag(1/clipped)@vectors.T
 chol=np.linalg.cholesky(complete);whitened=np.linalg.solve(chol,missing)@np.linalg.inv(chol.T);whitened=.5*(whitened+whitened.T)
 diag={"min_eigenvalue_I_complete":np.linalg.eigvalsh(complete).min(),"min_eigenvalue_I_missing":np.linalg.eigvalsh(missing).min(),"min_eigenvalue_I_observed_before_clipping":values.min(),"min_eigenvalue_I_observed_after_clipping":clipped.min(),"number_negative_eigenvalues_I_observed":int(np.sum(values<0)),"number_small_eigenvalues_I_observed":int(np.sum(values<floor)),"I_complete_condition_number":np.linalg.cond(complete),"I_missing_condition_number":np.linalg.cond(missing),"I_observed_condition_number_before_clipping":np.linalg.cond(observed),"I_observed_condition_number_after_clipping":np.linalg.cond(stable),"trace_missing_over_complete":np.trace(missing)/max(np.trace(complete),1e-12),"max_generalized_missing_information_fraction":np.max(np.linalg.eigvalsh(whitened)),"eigen_floor_used":floor}
 return cov,stable,diag

def louis_correct_covariances(current_covariances,missing_information,eta_grid):
 corrected={};diagnostics=[]
 for row,(current,missing) in enumerate(zip(current_covariances,missing_information)):
  complete=np.linalg.pinv(current);corrected[row]={"current":current}
  for eta in eta_grid:
   cov,_,diag=stabilize_observed_information(complete,missing,eta);label=f"louis_eta_{eta:.2f}".replace(".","p");corrected[row][label]=cov;diagnostics.append({"target_row":row,"eta":eta,"covariance_estimator":label,"trace_current_cov":np.trace(current),"trace_corrected_cov":np.trace(cov),"trace_corrected_cov_over_current_cov":np.trace(cov)/max(np.trace(current),1e-12),"diagonal_mean_ratio_corrected_current":np.mean(np.diag(cov))/max(np.mean(np.diag(current)),1e-12),"frobenius_norm_I_missing":np.linalg.norm(missing),**diag})
 return corrected,diagnostics

def stabilized_louis_covariance(I_complete,I_missing,eta,tau,eigen_floor_relative=1e-8,eigen_floor_absolute=1e-10):
 """Louis covariance after capping whitened missing-information eigenvalues."""
 if not (0<=eta and 0<=tau and eta*tau<.98):raise ValueError("eta and tau must be nonnegative with eta*tau < 0.98.")
 complete=.5*(np.asarray(I_complete)+np.asarray(I_complete).T);missing=.5*(np.asarray(I_missing)+np.asarray(I_missing).T);d,V=np.linalg.eigh(complete);d_floor=max(eigen_floor_absolute,eigen_floor_relative*max(float(d.max()),eigen_floor_absolute));d_safe=np.maximum(d,d_floor);half=V@np.diag(np.sqrt(d_safe))@V.T;inv_half=V@np.diag(1/np.sqrt(d_safe))@V.T
 K=inv_half@missing@inv_half;K=.5*(K+K.T);raw,U=np.linalg.eigh(K);nonnegative=np.maximum(raw,0.);capped=np.minimum(nonnegative,tau);Kc=U@np.diag(capped)@U.T;middle=np.eye(len(d))-eta*Kc;observed=half@middle@half;observed=.5*(observed+observed.T);before=np.linalg.eigvalsh(observed);final_floor=max(eigen_floor_absolute,eigen_floor_relative*max(float(np.linalg.eigvalsh(complete).max()),eigen_floor_absolute));required=bool(np.any(before<final_floor));after=np.maximum(before,final_floor);W=np.linalg.eigh(observed)[1];stable=W@np.diag(after)@W.T;covariance=W@np.diag(1/after)@W.T
 diagnostics={"eta":eta,"tau":tau,"eta_tau":eta*tau,"max_eigenvalue_K_raw":raw.max(),"median_eigenvalue_K_raw":np.median(raw),"min_eigenvalue_K_raw":raw.min(),"fraction_K_eigenvalues_above_0p5":np.mean(raw>.5),"fraction_K_eigenvalues_above_0p7":np.mean(raw>.7),"fraction_K_eigenvalues_above_0p9":np.mean(raw>.9),"fraction_K_eigenvalues_above_1p0":np.mean(raw>1.),"max_eigenvalue_K_capped":capped.max(),"min_eigenvalue_K":raw.min(),"max_eigenvalue_K":raw.max(),"K_condition_number":np.linalg.cond(K),"min_eigenvalue_I_minus_eta_K_capped":np.linalg.eigvalsh(middle).min(),"number_eigenvalues_capped":int(np.sum(nonnegative>tau)),"fraction_eigenvalues_capped":np.mean(nonnegative>tau),"I_complete_condition_number":np.linalg.cond(complete),"I_missing_condition_number":np.linalg.cond(missing),"I_observed_condition_number_before_final_clipping":np.linalg.cond(observed),"I_observed_condition_number_after_final_clipping":np.linalg.cond(stable),"min_eigenvalue_I_complete":d.min(),"min_eigenvalue_I_missing":np.linalg.eigvalsh(missing).min(),"min_eigenvalue_I_observed_before_final_clipping":before.min(),"min_eigenvalue_I_observed_after_final_clipping":after.min(),"number_negative_eigenvalues_I_observed":int(np.sum(before<0)),"trace_missing_over_complete":np.trace(missing)/max(np.trace(complete),1e-12),"required_final_eigen_clipping":required,"complete_eigen_floor_used":d_floor,"final_eigen_floor_used":final_floor}
 return covariance,stable,diagnostics

__all__=["SAMPLER_LABEL","sample_companion_trajectories_ffbs","prior_precision_for_row","complete_data_transition_score","estimate_missing_information","estimate_missing_information_numba","stabilize_observed_information","louis_correct_covariances","stabilized_louis_covariance"]
