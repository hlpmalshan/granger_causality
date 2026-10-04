"""Experiment 34S: sparsity x observability decomposition under rectangular C."""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key]=os.environ.get("EXPERIMENT_34S_INNER_THREADS","1")

import json,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34i_static_spectral_varx_gc as spi
import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as qb
import experiments.experiment_34qC_practical_model_choice_squareC as qc
import experiments.experiment_34r_stage1_rectangular_C_breakpoint as r1
import experiments.experiment_34r_stage2c_validate_tau1_rectangular_C as r2c
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,relative
from src.ssm.ssm_varx_p_simulator import build_var_companion_matrix
from src.ssm.vb_ard_varx_ssm_support_mask import HybridVBARDVARXSSMKnownCWithBQSupportMask
from src.stats.louis_missing_information import estimate_missing_information_numba,prior_precision_for_row,sample_companion_trajectories_ffbs,stabilized_louis_covariance
from src.stats.lrvb_stageB_smoother_feedback import project_selected_covariance,sensitivity_column

RESULTS_DIR=os.environ.get("EXPERIMENT_34S_RESULTS_DIR","results/experiment_34s")
WORKERS=max(1,int(os.environ.get("EXPERIMENT_34S_WORKERS","2")));SMOKE=os.environ.get("EXPERIMENT_34S_SMOKE","0")=="1"
MX=5 if SMOKE else 20;MY_LIST=[4] if SMOKE else [40,15];T=200 if SMOKE else 1000
NETWORKS=1 if SMOKE else 5;REPLICATES=1 if SMOKE else 2
REGIMES=["full_candidate_network","permissive_8K","permissive_4K","permissive_2K","oracle_support"]
if SMOKE:REGIMES=["full_candidate_network","oracle_support"]
METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_gaussian_rectangular_C_known_R_with_support_mask"
VB_MAX_ITER=8 if SMOKE else 75;TOL=1e-4;N_FFBS=5 if SMOKE else 50;K_OBS=10;N_FREQS=128
RUN_OPTIONAL_STAGEB_MINI=os.environ.get("EXPERIMENT_34S_RUN_OPTIONAL_STAGEB_MINI","0")=="1"
TABLES=("run","mask","C","latent","A","Amatrix","B","Q","network","spectral","spectra","deviance","calrun","grouprun","obs","obsquart","obscorr","runtime")
CHECK={key:f"_{key}_checkpoint.csv" for key in TABLES};CHECK["run"]="run_summary_partial.csv"

class Progress:
 def __init__(self,n):self.n,self.i,self.start=max(n,0),0,time.perf_counter();print("34S: all checkpointed fits complete.") if not n else None
 def update(self,label):
  self.i+=1;f=self.i/max(self.n,1);k=round(30*f);eta=(time.perf_counter()-self.start)/self.i*(self.n-self.i)
  print(f"\r34S [{'#'*k}{'-'*(30-k)}] {self.i}/{self.n} {100*f:5.1f}% ETA {eta/3600:5.2f}h {label[:35]}",end="\n" if self.i==self.n else "",flush=True)

def metadata(my,regime,network,replicate,mask):
 return {"experiment_name":"34S_sparsity_observability_decomposition","M_x":MX,"M_y":my,"observation_ratio":my/MX,"C_family":"gaussian_isotropic","C_MODE":"gaussian_isotropic","T":T,"support_regime":regime,"support_density":float((mask.sum()-MX)/(MX*(MX-1))),"true_network_id":network,"replicate_id":replicate,"method":METHOD,"run_id":f"Mx{MX}_My{my}_{regime}_net{network}_rep{replicate}"}

def C_for(my,network,replicate):
 nseed=r1.BASE_SEED+MX*100000+network*1000;seed=nseed+my*10000+T+replicate
 return r1.make_C(my,MX,"gaussian_isotropic",seed)

def paired_masks(network,replicate):
 A,_,truth=j.fixed_network(MX,r1.BASE_SEED+MX*100000+network*1000);truth=np.asarray(truth,bool);np.fill_diagonal(truth,False)
 true_pairs=list(zip(*np.where(truth)));false=[(i,jj) for i in range(MX) for jj in range(MX) if i!=jj and not truth[i,jj]]
 source_scores=[]
 for my in MY_LIST:
  C=C_for(my,network,replicate);R=.6*np.eye(my);v=np.diag(C.T@np.linalg.solve(R,C));v=(v-v.min())/(v.max()-v.min()+1e-12);source_scores.append(v)
 score=np.mean(source_scores,axis=0);edge=np.asarray([(score[i]+score[jj])/2 for i,jj in false]);cuts=np.quantile(edge,[.25,.5,.75]);bins=np.searchsorted(cuts,edge,side="right")
 rng=np.random.default_rng(6100000+network*1000+replicate);ordered=[];quartiles=[]
 for q in range(4):
  values=[pair for pair,b in zip(false,bins) if b==q];rng.shuffle(values);quartiles.append(values)
 while any(quartiles):
  for values in quartiles:
   if values:ordered.append(values.pop())
 K=len(true_pairs);masks={}
 for regime in REGIMES:
  mask=np.eye(MX,dtype=bool)|truth
  if regime=="full_candidate_network":mask[:]=True
  elif regime!="oracle_support":
   multiplier=int(regime.split("_")[1][:-1]);needed=max(0,min(multiplier*K,MX*(MX-1))-K)
   for i,jj in ordered[:needed]:mask[i,jj]=True
  if not np.all(mask[truth]):raise RuntimeError(f"{regime} accidentally excludes a true edge")
  masks[regime]=mask
 return masks,truth

def fit_masked(data,mask,seed):
 return HybridVBARDVARXSSMKnownCWithBQSupportMask(2,3,data["C"],data["Q"],data["R"],candidate_mask=mask,
  B_update_mode="free",B_ridge_lambda=0.,base_B_ridge=0.,estimate_Q=True,Q_update_mode="diag_shrink_scalar",Q_floor_mode="none",Q_floor_value=0.,Q_shrinkage_rho=.25,Q_update_damping=.5,include_A_posterior_uncertainty_in_Q=True,max_iter=VB_MAX_ITER,tol_objective=TOL,tol_A_change=TOL,tol_B_change=TOL,tol_Q_change=TOL,tol_alpha_change=TOL,a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed).fit(data["y"],data["u"])

def masked_louis(model,data,seed):
 samples=sample_companion_trajectories_ffbs(model.smooth_result_,model.F,N_FFBS,seed);beta=model._pack_A();M=model.n_states
 pri=np.asarray([prior_precision_for_row(M,2,row,model.alpha_mean_,model.diagonal_prior_precision) for row in range(M)])
 missing=estimate_missing_information_numba(samples,data["u"],beta,model.B_matrices,model.Q,2,3,pri);result=[]
 for target,(current,item) in enumerate(zip(model.A_row_covariances_,missing)):
  allowed=model._allowed_columns(target);block=stabilized_louis_covariance(np.linalg.pinv(current[np.ix_(allowed,allowed)]),item["missing_information"][np.ix_(allowed,allowed)],.7,.9)[0]
  full=np.zeros_like(current);full[np.ix_(allowed,allowed)]=block;result.append(full)
 return result

def normalize(x):x=np.asarray(x,float);return (x-x.min())/(x.max()-x.min()+1e-12)
def dynamic_scores(A,C,R):
 F=build_var_companion_matrix(A);Caug=np.hstack([C,np.zeros_like(C)]);base=Caug.T@np.linalg.solve(R,Caug);W=np.zeros_like(base);power=np.eye(len(F))
 for _ in range(K_OBS+1):W+=power.T@base@power;power=power@F
 return normalize(np.diag(W)[:C.shape[1]])
def observation_scores(Atrue,Ahat,C,R):
 inst=normalize(np.diag(C.T@np.linalg.solve(R,C)));return inst,dynamic_scores(Atrue,C,R),dynamic_scores(Ahat,C,R)

def threshold_at_fpr(truth,score,target=.05):
 metrics,curve=j.safe_curve(truth,score);eligible=[item for item in curve if np.isfinite(item[1]["fpr"]) and item[1]["fpr"]<=target]
 threshold=max(eligible,key=lambda item:item[1]["tpr"])[0] if eligible else np.inf
 return metrics,threshold

def network_rows(model,data,louis,mask,meta):
 truth=[];scores={"model_A_group_norm":[],"ordinary_vb_group_snr":[],"stabilized_louis_group_snr":[]};pairs=[];A=np.asarray(model.A_mean_matrices_);M=model.n_states
 for target in range(M):
  for source in range(M):
   if target==source:continue
   cols=[source,M+source];m=A[:,target,source];pairs.append((target,source));truth.append(bool(data["mask"][target,source]));scores["model_A_group_norm"].append(np.linalg.norm(m))
   if mask[target,source]:
    scores["ordinary_vb_group_snr"].append(np.sqrt(max(m@np.linalg.pinv(model.A_row_covariances_[target][np.ix_(cols,cols)])@m,0)))
    scores["stabilized_louis_group_snr"].append(np.sqrt(max(m@np.linalg.pinv(louis[target][np.ix_(cols,cols)])@m,0)))
   else:scores["ordinary_vb_group_snr"].append(0.);scores["stabilized_louis_group_snr"].append(0.)
 rows=[];thresholds={}
 for scope,scope_mask in (("full_universe",np.ones(len(pairs),bool)),("candidate_conditional",np.asarray([mask[t,s] for t,s in pairs]))):
  for name,values in scores.items():
   metric,threshold=threshold_at_fpr(np.asarray(truth)[scope_mask],np.asarray(values)[scope_mask]);rows.append({**meta,"network_scope":scope,"score_type":name,**metric});thresholds[(scope,name)]=threshold
 frame=pd.DataFrame(rows);frame["threshold_at_best_youden"]=frame.get("best_youden_threshold",np.nan)
 return frame,pairs,np.asarray(truth),scores,thresholds

def calibration_rows(model,data,louis,mask,meta):
 rows=[];A=np.asarray(model.A_mean_matrices_);truth=np.asarray(data["A"]);M=model.n_states
 for method,covs in (("ordinary_vb",model.A_row_covariances_),("stabilized_louis_eta_0p70_tau_0p90",louis)):
  for target in range(M):
   for lag in range(2):
    for source in range(M):
     fixed=bool(target!=source and not mask[target,source]);true=bool(data["mask"][target,source]);index=target*(2*M)+lag*M+source
     kind="diagonal" if target==source else ("offdiag_nonzero" if true else ("offdiag_zero_allowed" if mask[target,source] else "offdiag_zero_disallowed"))
     center=A[lag,target,source];actual=truth[lag,target,source];variance=float(covs[target][lag*M+source,lag*M+source]);sd=np.sqrt(max(variance,0.));error=center-actual
     rows.append({**meta,"covariance_method":method,"coefficient_global_index":index,"lag":lag+1,"target":target,"source":source,"coefficient_type":kind,"network_scope":"allowed_coefficients" if not fixed else "fixed_by_mask","posterior_center":center,"true_value":actual,"signed_error":error,"abs_error":abs(error),"squared_error":error**2,"posterior_variance":variance,"posterior_sd":sd,"ci95_lower":center-1.96*sd,"ci95_upper":center+1.96*sd,"ci95_contains_true":bool(abs(error)<=1.96*sd),"standardized_error":error/sd if sd>0 else np.nan,"interval_width_95":3.92*sd,"fixed_by_mask":fixed})
 return pd.DataFrame(rows)

def summarize_calibration(coeff):
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","coefficient_type","network_scope"]
 return coeff.groupby(keys,dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),median_signed_error=("signed_error","median"),mean_abs_error=("abs_error","mean"),median_abs_error=("abs_error","median"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),median_posterior_sd=("posterior_sd","median"),mean_interval_width_95=("interval_width_95","mean"),median_interval_width_95=("interval_width_95","median"),mean_standardized_error=("standardized_error","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),median_abs_standardized_error=("standardized_error",lambda x:x.abs().median()),n_coefficients=("posterior_sd","size")).reset_index()

def group_calibration_rows(model,data,louis,mask,meta,edge_inst):
 rows=[];A=np.asarray(model.A_mean_matrices_);truth=np.asarray(data["A"]);M=model.n_states;true_obs=[(edge_inst[t]+edge_inst[s])/2 for t,s in zip(*np.where(data["mask"]))];median=np.median(true_obs)
 for method,covs in (("ordinary_vb",model.A_row_covariances_),("stabilized_louis_eta_0p70_tau_0p90",louis)):
  for target in range(M):
   for source in range(M):
    if target==source:continue
    allowed=bool(mask[target,source]);true=bool(data["mask"][target,source]);cols=[source,M+source];difference=A[:,target,source]-truth[:,target,source];block=covs[target][np.ix_(cols,cols)]
    if allowed:
     D2=float(difference@np.linalg.pinv(block)@difference);c95=D2<=5.991464547;c99=D2<=9.210340372;trace=float(np.trace(block))
    else:D2=c95=c99=trace=np.nan
    kind="true_edge_group" if true else ("false_edge_candidate_group" if allowed else "false_edge_disallowed_group")
    obs=(edge_inst[target]+edge_inst[source])/2
    rows.append({**meta,"covariance_method":method,"group_target":target,"group_source":source,"true_edge_group":true,"candidate_mask_includes_edge":allowed,"fixed_by_mask":not allowed,"group_type":kind,"observability_subgroup":("low_observability_true_edge_group" if obs<=median else "high_observability_true_edge_group") if true else kind,"D2":D2,"D2_below_chi2_95_df2":c95,"D2_below_chi2_99_df2":c99,"group_cov_trace":trace,"group_error_norm":np.linalg.norm(difference)})
 return pd.DataFrame(rows)

def summarize_group(group):
 base=group.copy();extra=group[group.true_edge_group].copy();extra["group_type"]=extra.observability_subgroup;allrows=pd.concat([base,extra],ignore_index=True)
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","group_type"]
 return allrows.groupby(keys,dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("group_source","size")).reset_index()

def A_metrics(model,data,mask,meta):
 row=qb.a_recovery(model,data,meta);hat=np.asarray(model.A_mean_matrices_);true=np.asarray(data["A"]);off=~np.eye(model.n_states,dtype=bool);allowed=mask&off
 row["A_allowed_offdiag_relative_frobenius_error"]=relative(hat[:,allowed],true[:,allowed]);row["A_allowed_offdiag_mean_absolute_error"]=np.mean(np.abs(hat[:,allowed]-true[:,allowed]))
 excluded=data["mask"]&~mask;row["A_disallowed_true_edge_error"]=float(np.linalg.norm(hat[:,excluded]-true[:,excluded])) if excluded.any() else 0.
 return row

def work(task):
 my,regime,network,replicate,mask=task;meta=metadata(my,regime,network,replicate,mask);start=time.perf_counter();out={key:pd.DataFrame() for key in TABLES}
 try:
  data,seed=r1.simulate(MX,my,T,"gaussian_isotropic",network,replicate)
  if not np.all(mask[data["mask"]]):raise RuntimeError("Support mask excludes at least one true edge.")
  model=fit_masked(data,mask,seed+500);louis=masked_louis(model,data,seed+700);Ahat=np.asarray(model.A_mean_matrices_)
  if np.any(np.abs(Ahat[:,~mask])>1e-14):raise RuntimeError("Masked A coefficients are not exactly zero.")
  inst,dyn_true,dyn_est=observation_scores(data["A"],Ahat,data["C"],data["R"])
  network_frame,pairs,truth,scores,thresholds=network_rows(model,data,louis,mask,meta)
  oldM=spi.M;spi.M=MX;spectra,bands,_,_,_=spi.compute_spectral_tables(model.A_mean_matrices_,model.B_matrices,model.Q,np.linspace(0,np.pi,N_FREQS),data,meta,"estimated_AQ");spectral,_=spi.metric_rows(bands);spi.M=oldM
  for key,value in meta.items():spectral[key]=value
  eligible=spectral[(spectral.score_type=="transfer_spectral_gc_diagQ")&spectral.band_name.isin(["low_band","mid_band","full_band"])]
  best_band=eligible.loc[eligible.AUPRC.idxmax(),"band_name"] if len(eligible) else np.nan
  band_scores=bands[(bands.score_type=="transfer_spectral_gc_diagQ")&(bands.band_name==best_band)].set_index(["target","source"]).integrated_score if isinstance(best_band,str) else pd.Series(dtype=float)
  spectral_values=np.asarray([band_scores.get(pair,0.) for pair in pairs]);_,spectral_threshold=threshold_at_fpr(truth,spectral_values)
  edge_rows=[]
  for position,(target,source) in enumerate(pairs):
   allowed=bool(mask[target,source]);cols=[source,MX+source];mean=Ahat[:,target,source];actual=np.asarray(data["A"])[:,target,source];block=louis[target][np.ix_(cols,cols)]
   if allowed:
    try:
     block_eig=np.linalg.eigvalsh(.5*(block+block.T))
     if not np.all(np.isfinite(block_eig)) or block_eig.min()<=0 or block_eig.max()/block_eig.min()>1e12:raise np.linalg.LinAlgError("unstable Louis edge block")
     info=np.linalg.inv(block+1e-12*np.eye(2));eig=np.linalg.eigvalsh(info);sign,logdet=np.linalg.slogdet(info+1e-12*np.eye(2));p_trace=float(np.trace(info));p_min=float(eig.min());p_log=float(logdet) if sign>0 else np.nan;info_warning=""
    except np.linalg.LinAlgError as error:p_trace=p_min=p_log=np.nan;info_warning=str(error)
    snr_vb=float(scores["ordinary_vb_group_snr"][position]);snr_l=float(scores["stabilized_louis_group_snr"][position]);alpha=float(model.alpha_mean_[target,source]);ltrace=float(np.trace(block))
    diff=mean-actual;ordinary_block=model.A_row_covariances_[target][np.ix_(cols,cols)];ordinary_cover=bool(diff@np.linalg.pinv(ordinary_block)@diff<=5.991464547);louis_cover=bool(diff@np.linalg.pinv(block)@diff<=5.991464547)
   else:p_trace=p_min=p_log=snr_vb=snr_l=alpha=ltrace=np.nan;ordinary_cover=louis_cover=np.nan;info_warning="fixed_by_mask"
   detect_model=scores["model_A_group_norm"][position]>=thresholds[("full_universe","model_A_group_norm")];detect_l=(scores["stabilized_louis_group_snr"][position]>=thresholds[("full_universe","stabilized_louis_group_snr")])
   detect_spec=spectral_values[position]>=spectral_threshold;is_true=bool(truth[position]);error=float(np.linalg.norm(mean-actual))
   edge_rows.append({**meta,"target_i":target,"source_j":source,"true_edge":is_true,"candidate_mask_includes_edge":allowed,"fixed_by_mask":not allowed,"edge_o_inst_mean":(inst[target]+inst[source])/2,"edge_o_inst_min":min(inst[target],inst[source]),"edge_o_inst_product":inst[target]*inst[source],"edge_o_dyn_mean_trueA":(dyn_true[target]+dyn_true[source])/2,"edge_o_dyn_min_trueA":min(dyn_true[target],dyn_true[source]),"edge_o_dyn_product_trueA":dyn_true[target]*dyn_true[source],"edge_o_dyn_mean_estA":(dyn_est[target]+dyn_est[source])/2,"edge_o_dyn_min_estA":min(dyn_est[target],dyn_est[source]),"edge_o_dyn_product_estA":dyn_est[target]*dyn_est[source],"parameter_information_trace":p_trace,"parameter_information_min_eigenvalue":p_min,"parameter_information_logdet":p_log,"parameter_information_warning_flag":info_warning,"estimated_group_norm":np.linalg.norm(mean),"true_group_norm":np.linalg.norm(actual),"group_estimation_error":error,"group_SNR_VB":snr_vb,"group_SNR_Louis":snr_l,"E_alpha":alpha,"Louis_group_cov_trace":ltrace,"ordinary_vb_group_coverage":ordinary_cover,"stabilized_louis_group_coverage":louis_cover,"StageB_group_cov_trace":np.nan,"StageB_Louis_variance_ratio":np.nan,"StageB_available":False,"detected_by_model_A_at_FPR_0p05":detect_model,"detected_by_Louis_SNR_at_FPR_0p05":detect_l,"detected_by_spectral_best_band_at_FPR_0p05":detect_spec,"missed_true_edge_indicator":bool(is_true and not detect_model),"false_positive_indicator":bool(not is_true and detect_model)})
  edge=pd.DataFrame(edge_rows);coeff=calibration_rows(model,data,louis,mask,meta);group=group_calibration_rows(model,data,louis,mask,meta,inst)
  calrun=summarize_calibration(coeff);calrun["true_network_id"]=network;calrun["replicate_id"]=replicate
  grouprun=summarize_group(group);grouprun["true_network_id"]=network;grouprun["replicate_id"]=replicate
  latent={**meta,**qb.latent_metrics(model,{**data,"y":r1.ridge_proxy(data)@data["C"].T})};latent["pseudo_inverse_state_MSE"]=np.mean((r1.ridge_proxy(data)-data["x"])**2);latent["pseudo_inverse_state_correlation"]=np.nanmean([correlations(r1.ridge_proxy(data)[:,i],data["x"][:,i]) for i in range(MX)])
  Am=A_metrics(model,data,mask,meta);B=qa.B_recovery(model,data,meta);Q=qb.q_recovery(model,data,meta);Q["Q_update_damping"]=Q.get("Q_damping",.5)
  dev=qc.deviance(model,data,meta);cm={**meta,**r1.c_metrics(data["C"])};candidate=int(mask.sum()-MX);K=int(data["mask"].sum());false_obs=edge.loc[(~edge.true_edge)&edge.candidate_mask_includes_edge,"edge_o_inst_mean"]
  maskrow={**meta,"number_true_edges":K,"number_candidate_edges":candidate,"candidate_density":candidate/(MX*(MX-1)),"false_candidate_multiplier":(candidate-K)/max(K,1),"all_true_edges_included":bool(np.all(mask[data["mask"]])),"diagnostic_structural_prior_mask":regime!="full_candidate_network","mean_observability_selected_false_candidates":false_obs.mean(),"std_observability_selected_false_candidates":false_obs.std()}
  Atrue_array=np.asarray(data["A"]);amat=pd.DataFrame([{**meta,"lag":lag+1,"target":target,"source":source,"A_hat":Ahat[lag,target,source],"A_true":Atrue_array[lag,target,source],"candidate_mask_includes_edge":bool(mask[target,source]),"fixed_by_mask":bool(target!=source and not mask[target,source])} for lag in range(2) for target in range(MX) for source in range(MX)])
  qvalid=bool(np.all(np.isfinite(model.Q)) and np.all(np.diag(model.Q)>0));runtime={**meta,"runtime_seconds":time.perf_counter()-start,"run_status":"success" if qvalid else "numerically_unstable","numerical_warning_flag":"" if qvalid else "invalid_Q","StageB_available":False}
  run={**meta,**maskrow,**latent,**Am,**B,**Q,**runtime,"spectral_best_band":best_band,"spectral_best_band_AUPRC":eligible.AUPRC.max() if len(eligible) else np.nan}
  out.update(run=pd.DataFrame([run]),mask=pd.DataFrame([maskrow]),C=pd.DataFrame([cm]),latent=pd.DataFrame([latent]),A=pd.DataFrame([Am]),Amatrix=amat,B=pd.DataFrame([B]),Q=pd.DataFrame([Q]),network=network_frame,spectral=spectral,spectra=bands,deviance=dev,calrun=calrun,grouprun=grouprun,obs=edge,runtime=pd.DataFrame([runtime]))
 except Exception as error:
  failure={**meta,"run_status":"failed_fit","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc(),"runtime_seconds":time.perf_counter()-start};out["run"]=pd.DataFrame([failure]);out["runtime"]=pd.DataFrame([failure])
 return out

def frames(tables):return {key:(pd.concat(value,ignore_index=True) if value else pd.DataFrame()) for key,value in tables.items()}
def load_tables():
 out={key:[] for key in TABLES}
 for key,name in CHECK.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:value=pd.read_csv(path)
   except pd.errors.EmptyDataError:value=pd.DataFrame()
   if len(value):out[key]=[value]
 return out
def save_raw(tables):
 current=frames(tables)
 for key,name in CHECK.items():atomic_csv(current[key],os.path.join(RESULTS_DIR,name))
 return current

def replicate_calibration(calrun):
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","coefficient_type","network_scope"]
 return calrun.groupby(keys,dropna=False).agg(mean_coverage_across_replicates=("empirical_coverage_95","mean"),std_coverage_across_replicates=("empirical_coverage_95","std"),mean_std_z_across_replicates=("std_standardized_error","mean"),std_std_z_across_replicates=("std_standardized_error","std"),mean_interval_width_95=("mean_interval_width_95","mean"),n_replicates=("replicate_id","size"),n_networks=("true_network_id","nunique")).reset_index().assign(sem_coverage_across_replicates=lambda x:x.std_coverage_across_replicates/np.sqrt(x.n_replicates),sem_std_z_across_replicates=lambda x:x.std_std_z_across_replicates/np.sqrt(x.n_replicates))
def aggregate_calibration(calrun):
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","coefficient_type","network_scope"]
 numeric=[c for c in ["empirical_coverage_95","mean_signed_error","median_signed_error","mean_abs_error","median_abs_error","rmse","mean_posterior_sd","median_posterior_sd","mean_interval_width_95","median_interval_width_95","mean_standardized_error","std_standardized_error","median_abs_standardized_error"] if c in calrun]
 out=calrun.groupby(keys,dropna=False)[numeric].mean().reset_index();counts=calrun.groupby(keys,dropna=False).n_coefficients.sum().reset_index();return out.merge(counts,on=keys)
def replicate_groups(grouprun):
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","group_type"]
 return grouprun.groupby(keys,dropna=False).agg(mean_coverage_across_replicates=("fraction_D2_below_chi2_95_df2","mean"),std_coverage_across_replicates=("fraction_D2_below_chi2_95_df2","std"),mean_D2_across_replicates=("mean_D2","mean"),std_D2_across_replicates=("mean_D2","std"),n_replicates=("replicate_id","size"),n_networks=("true_network_id","nunique")).reset_index().assign(sem_coverage_across_replicates=lambda x:x.std_coverage_across_replicates/np.sqrt(x.n_replicates))
def aggregate_groups(grouprun):
 keys=["M_x","M_y","observation_ratio","support_regime","support_density","covariance_method","group_type"]
 numeric=["mean_D2","median_D2","fraction_D2_below_chi2_95_df2","fraction_D2_below_chi2_99_df2","mean_group_cov_trace","median_group_cov_trace"]
 out=grouprun.groupby(keys,dropna=False)[numeric].mean().reset_index();counts=grouprun.groupby(keys,dropna=False).n_groups.sum().reset_index();return out.merge(counts,on=keys)

def observability_summaries(edge):
 true=edge[edge.true_edge.astype(bool)].copy();quartiles=[]
 dimensions=["edge_o_inst_mean","edge_o_dyn_mean_trueA","parameter_information_trace"]
 keys=["M_y","support_regime","support_density","true_network_id","replicate_id"]
 for dimension in dimensions:
  part=true.copy();part["observability_dimension"]=dimension
  part["observability_quartile"]=part.groupby(keys)[dimension].transform(lambda s:pd.qcut(s.rank(method="first"),4,labels=["Q1_low","Q2","Q3","Q4_high"]))
  quartiles.append(part)
 expanded=pd.concat(quartiles,ignore_index=True) if quartiles else pd.DataFrame()
 qkeys=["M_x","M_y","observation_ratio","support_regime","support_density","observability_dimension","observability_quartile"]
 quart=expanded.groupby(qkeys,dropna=False).agg(number_true_edges=("true_edge","size"),A_group_error_mean=("group_estimation_error","mean"),A_group_error_median=("group_estimation_error","median"),detected_by_model_A_at_FPR_0p05_rate=("detected_by_model_A_at_FPR_0p05","mean"),detected_by_spectral_best_band_at_FPR_0p05_rate=("detected_by_spectral_best_band_at_FPR_0p05","mean"),stabilized_louis_group_coverage=("stabilized_louis_group_coverage","mean"),ordinary_vb_group_coverage=("ordinary_vb_group_coverage","mean"),mean_group_SNR_Louis=("group_SNR_Louis","mean"),mean_Louis_group_cov_trace=("Louis_group_cov_trace","mean"),mean_parameter_information_trace=("parameter_information_trace","mean")).reset_index()
 rows=[]
 for values,g in true.groupby(["M_x","M_y","observation_ratio","support_regime","support_density"],dropna=False):
  row=dict(zip(["M_x","M_y","observation_ratio","support_regime","support_density"],values))
  for dimension in dimensions:
   short={"edge_o_inst_mean":"inst","edge_o_dyn_mean_trueA":"dyn_trueA","parameter_information_trace":"parameter_info"}[dimension]
   row[f"corr_{short}_group_estimation_error"]=_corr(g[dimension],g.group_estimation_error);row[f"corr_{short}_detected_true_edge"]=_corr(g[dimension],g.detected_by_model_A_at_FPR_0p05.astype(float));row[f"corr_{short}_Louis_group_cov_trace"]=_corr(g[dimension],g.Louis_group_cov_trace)
  rows.append(row)
 summary=true.groupby(["M_x","M_y","observation_ratio","support_regime","support_density"],dropna=False).agg(mean_edge_observability_detected_true_edges=("edge_o_inst_mean",lambda s:s[true.loc[s.index,"detected_by_model_A_at_FPR_0p05"]].mean()),mean_edge_observability_missed_true_edges=("edge_o_inst_mean",lambda s:s[~true.loc[s.index,"detected_by_model_A_at_FPR_0p05"]].mean()),mean_group_estimation_error=("group_estimation_error","mean"),mean_parameter_information_trace=("parameter_information_trace","mean")).reset_index()
 return summary,quart,pd.DataFrame(rows)

def _corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 return float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>1 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan

def support_summary(f,calrep):
 run=f["run"][f["run"].run_status.isin(["success","numerically_unstable"])];rows=[]
 for (my,regime,density),g in run.groupby(["M_y","support_regime","support_density"],dropna=False):
  net=f["network"][(f["network"].M_y==my)&(f["network"].support_regime==regime)];spec=f["spectral"][(f["spectral"].M_y==my)&(f["spectral"].support_regime==regime)&f["spectral"].score_type.eq("transfer_spectral_gc_diagQ")&f["spectral"].band_name.isin(["low_band","mid_band","full_band"])];means=spec.groupby("band_name").AUPRC.mean();band=means.idxmax() if len(means) else np.nan;dev=f["deviance"][(f["deviance"].M_y==my)&(f["deviance"].support_regime==regime)];dv=dev.groupby("signal_type").AUPRC.mean() if len(dev) else pd.Series(dtype=float)
  c=calrep[(calrep.M_y==my)&(calrep.support_regime==regime)].set_index(["covariance_method","coefficient_type"]);cv=lambda method,typ,col:c.loc[(method,typ),col] if (method,typ) in c.index else np.nan
  rows.append({"M_x":MX,"M_y":my,"observation_ratio":my/MX,"support_regime":regime,"support_density":density,"T":T,"n_networks":g.true_network_id.nunique(),"n_replicates":len(g),"mean_number_true_edges":g.number_true_edges.mean(),"mean_number_candidate_edges":g.number_candidate_edges.mean(),"candidate_density":g.candidate_density.mean(),"false_candidate_multiplier":g.false_candidate_multiplier.mean(),"smoothed_state_correlation_mean":g.smoothed_state_correlation.mean(),"smoothed_state_correlation_sem":g.smoothed_state_correlation.std()/np.sqrt(len(g)),"A_relative_frobenius_error_mean":g.A_relative_frobenius_error.mean(),"A_offdiag_relative_frobenius_error_mean":g.A_offdiag_relative_frobenius_error.mean(),"A_support_AUPRC_model_A_group_norm_full_universe":net[(net.network_scope=="full_universe")&net.score_type.eq("model_A_group_norm")].AUPRC.mean(),"A_support_AUPRC_model_A_group_norm_candidate_conditional":net[(net.network_scope=="candidate_conditional")&net.score_type.eq("model_A_group_norm")].AUPRC.mean(),"A_support_AUPRC_stabilized_louis_group_snr_full_universe":net[(net.network_scope=="full_universe")&net.score_type.eq("stabilized_louis_group_snr")].AUPRC.mean(),"A_support_TPR_at_FPR_0p05_full_universe":net[(net.network_scope=="full_universe")&net.score_type.eq("model_A_group_norm")].TPR_at_FPR_0p05.mean(),"spectral_best_band":band,"spectral_best_band_AUPRC_mean":means.max() if len(means) else np.nan,"spectral_best_band_TPR_at_FPR_0p05_mean":spec[spec.band_name==band].TPR_at_FPR_0p05.mean() if isinstance(band,str) else np.nan,"deviance_best_signal_type":dv.idxmax() if len(dv) else np.nan,"deviance_best_AUPRC_mean":dv.max() if len(dv) else np.nan,"B_relative_error_mean":g.B_relative_frobenius_error.mean(),"B_correlation_hat_true_mean":g.B_correlation_hat_true.mean(),"Q_relative_error_mean":g.Q_relative_frobenius_error.mean(),"Q_trace_ratio_hat_to_true_mean":g.Q_trace_ratio_hat_to_true.mean(),"ordinary_vb_offdiag_nonzero_coverage":cv("ordinary_vb","offdiag_nonzero","mean_coverage_across_replicates"),"stabilized_louis_offdiag_nonzero_coverage":cv("stabilized_louis_eta_0p70_tau_0p90","offdiag_nonzero","mean_coverage_across_replicates"),"stabilized_louis_offdiag_zero_allowed_coverage":cv("stabilized_louis_eta_0p70_tau_0p90","offdiag_zero_allowed","mean_coverage_across_replicates"),"stabilized_louis_active_std_z":cv("stabilized_louis_eta_0p70_tau_0p90","offdiag_nonzero","mean_std_z_across_replicates"),"stabilized_louis_zero_allowed_std_z":cv("stabilized_louis_eta_0p70_tau_0p90","offdiag_zero_allowed","mean_std_z_across_replicates"),"mean_interval_width_95":calrep[(calrep.M_y==my)&(calrep.support_regime==regime)&calrep.covariance_method.eq("stabilized_louis_eta_0p70_tau_0p90")].mean_interval_width_95.mean() if "mean_interval_width_95" in calrep else np.nan,"runtime_seconds_mean":g.runtime_seconds.mean()})
 out=pd.DataFrame(rows);out["rescue_score"]=0.
 for my in out.M_y.unique():
  base=out[(out.M_y==my)&out.support_regime.eq("full_candidate_network")]
  if not len(base):continue
  base=base.iloc[0]
  for idx,row in out[out.M_y==my].iterrows():
   components=[(base.A_offdiag_relative_frobenius_error_mean-row.A_offdiag_relative_frobenius_error_mean)/max(base.A_offdiag_relative_frobenius_error_mean,1e-12),row.A_support_AUPRC_model_A_group_norm_full_universe-base.A_support_AUPRC_model_A_group_norm_full_universe,row.spectral_best_band_AUPRC_mean-base.spectral_best_band_AUPRC_mean,row.smoothed_state_correlation_mean-base.smoothed_state_correlation_mean,row.stabilized_louis_offdiag_nonzero_coverage-base.stabilized_louis_offdiag_nonzero_coverage,row.B_correlation_hat_true_mean-base.B_correlation_hat_true_mean]
   components=np.clip(np.nan_to_num(components),0,1);out.loc[idx,"rescue_score"]=np.dot(components,[.2,.2,.2,.15,.15,.1])
 out["regime_label"]=np.where(out.support_regime.eq("oracle_support"),"diagnostic oracle support",np.where(out.support_regime.str.startswith("permissive"),"diagnostic structural-prior mask","deployable full candidate baseline"));out["recommended_next_step"]="pending decision summary"
 return out

def decisions(summary,obs_summary,corr):
 out=summary.merge(obs_summary,on=["M_x","M_y","observation_ratio","support_regime","support_density"],how="left");out["observability_error_correlation"]=np.nan
 if len(corr):
  key=["M_x","M_y","observation_ratio","support_regime","support_density"]
  out=out.merge(corr[key+["corr_inst_group_estimation_error"]].rename(columns={"corr_inst_group_estimation_error":"observability_error_correlation"}),on=key,how="left",suffixes=("_old",""));out=out.drop(columns=["observability_error_correlation_old"],errors="ignore")
 out["substantial_rescue_condition_met"]=False
 subset=out[out.M_y==15];base=subset[subset.support_regime.eq("full_candidate_network")]
 if len(base):
  b=base.iloc[0]
  for idx,row in subset[~subset.support_regime.eq("full_candidate_network")].iterrows():
   coverage_gain=row.stabilized_louis_offdiag_nonzero_coverage-b.stabilized_louis_offdiag_nonzero_coverage
   z_gain=abs(b.stabilized_louis_active_std_z-1)-abs(row.stabilized_louis_active_std_z-1)
   ok=((b.A_offdiag_relative_frobenius_error_mean-row.A_offdiag_relative_frobenius_error_mean)/max(b.A_offdiag_relative_frobenius_error_mean,1e-12)>=.15 and row.A_support_AUPRC_model_A_group_norm_full_universe-b.A_support_AUPRC_model_A_group_norm_full_universe>=.10 and row.spectral_best_band_AUPRC_mean-b.spectral_best_band_AUPRC_mean>=.10 and (coverage_gain>=.10 or z_gain>=.10) and row.B_correlation_hat_true_mean>=b.B_correlation_hat_true_mean-.05 and .8<=row.Q_trace_ratio_hat_to_true_mean<=1.25)
   out.loc[idx,"substantial_rescue_condition_met"]=ok
 oracle=bool(out.loc[(out.M_y==15)&out.support_regime.eq("oracle_support"),"substantial_rescue_condition_met"].any());permissive=bool(out.loc[(out.M_y==15)&out.support_regime.str.startswith("permissive"),"substantial_rescue_condition_met"].any());substantial=oracle or permissive
 if oracle and permissive:next_exp="34T: stronger sparse posterior geometry"
 elif oracle:next_exp="refined candidate-support experiment or cautious 34T"
 else:next_exp="34U: edge-level observability / information analysis"
 out["substantial_rescue_My15"]=substantial;out["oracle_rescue_My15"]=oracle;out["permissive_rescue_My15"]=permissive;out["recommended_next_experiment"]=next_exp
 out["notes"]="Permissive/oracle masks are synthetic diagnostic structural priors, not deployable known-support estimators. Observability is explanatory only. Stage-B unavailable."
 columns=["M_y","support_regime","support_density","n_networks","n_replicates","mean_number_candidate_edges","candidate_density","smoothed_state_correlation_mean","smoothed_state_correlation_sem","A_relative_frobenius_error_mean","A_offdiag_relative_frobenius_error_mean","A_support_AUPRC_model_A_group_norm_full_universe","A_support_AUPRC_model_A_group_norm_candidate_conditional","A_support_TPR_at_FPR_0p05_full_universe","spectral_best_band","spectral_best_band_AUPRC_mean","spectral_best_band_TPR_at_FPR_0p05_mean","deviance_best_AUPRC_mean","B_relative_error_mean","B_correlation_hat_true_mean","Q_relative_error_mean","Q_trace_ratio_hat_to_true_mean","ordinary_vb_offdiag_nonzero_coverage","stabilized_louis_offdiag_nonzero_coverage","stabilized_louis_offdiag_zero_allowed_coverage","stabilized_louis_active_std_z","stabilized_louis_zero_allowed_std_z","mean_edge_observability_detected_true_edges","mean_edge_observability_missed_true_edges","observability_error_correlation","rescue_score","substantial_rescue_My15","oracle_rescue_My15","permissive_rescue_My15","recommended_next_experiment","notes"]
 return out[[c for c in columns if c in out]]

def build_outputs(tables):
 f=frames(tables)
 cal=aggregate_calibration(f["calrun"]) if len(f["calrun"]) else pd.DataFrame();calrep=replicate_calibration(f["calrun"]) if len(f["calrun"]) else pd.DataFrame()
 groups=aggregate_groups(f["grouprun"]) if len(f["grouprun"]) else pd.DataFrame();greprep=replicate_groups(f["grouprun"]) if len(f["grouprun"]) else pd.DataFrame()
 obs_summary,obsquart,obscorr=observability_summaries(f["obs"]) if len(f["obs"]) else (pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
 support=support_summary(f,calrep) if len(calrep) else pd.DataFrame();decision=decisions(support,obs_summary,obscorr) if len(support) else pd.DataFrame()
 return f,{"run_summary.csv":f["run"],"support_mask_summary.csv":f["mask"],"support_regime_summary.csv":support,"C_diagnostics.csv":f["C"],"latent_state_recovery_summary.csv":f["latent"],"A_recovery_summary.csv":f["A"],"A_matrix_estimates.csv":f["Amatrix"],"B_recovery_summary.csv":f["B"],"Q_recovery_summary.csv":f["Q"],"network_recovery_summary.csv":f["network"],"spectral_gc_summary.csv":f["spectra"],"spectral_band_recovery_summary.csv":f["spectral"],"deviance_diagnostics.csv":f["deviance"],"calibration_summary.csv":cal,"calibration_replicate_summary.csv":calrep,"group_calibration_summary.csv":groups,"group_calibration_replicate_summary.csv":greprep,"observability_summary.csv":obs_summary,"observability_edge_table.csv":f["obs"],"observability_quartile_summary.csv":obsquart,"observability_correlation_summary.csv":obscorr,"runtime_summary.csv":f["runtime"],"decision_summary.csv":decision}

PARTIAL={"run_summary.csv":"run_summary_partial.csv","support_regime_summary.csv":"support_regime_summary_partial.csv","latent_state_recovery_summary.csv":"latent_state_recovery_summary_partial.csv","A_recovery_summary.csv":"A_recovery_summary_partial.csv","B_recovery_summary.csv":"B_recovery_summary_partial.csv","Q_recovery_summary.csv":"Q_recovery_summary_partial.csv","network_recovery_summary.csv":"network_recovery_summary_partial.csv","spectral_band_recovery_summary.csv":"spectral_band_recovery_summary_partial.csv","deviance_diagnostics.csv":"deviance_diagnostics_partial.csv","calibration_summary.csv":"calibration_summary_partial.csv","group_calibration_summary.csv":"group_calibration_summary_partial.csv","observability_edge_table.csv":"observability_edge_table_partial.csv","observability_quartile_summary.csv":"observability_quartile_summary_partial.csv","runtime_summary.csv":"runtime_summary_partial.csv","decision_summary.csv":"decision_summary_partial.csv"}
def save_outputs(tables,final=False):
 f,outputs=build_outputs(tables)
 if final:
  for name,value in outputs.items():atomic_csv(value,os.path.join(RESULTS_DIR,name))
 for source,target in PARTIAL.items():atomic_csv(outputs[source],os.path.join(RESULTS_DIR,target))
 return f,outputs

def _save_plot(path,name):plt.tight_layout();plt.savefig(os.path.join(path,name),dpi=150);plt.close()
def _line(summary,y,path,name):
 if not len(summary) or y not in summary:plt.text(.5,.5,"No completed data",ha="center");plt.axis("off");_save_plot(path,name);return
 for my,g in summary.groupby("M_y"):
  g=g.sort_values("support_density");plt.plot(g.support_density,g[y],marker="o",label=f"M_y={my}")
 plt.xlabel("support density");plt.ylabel(y);plt.legend();_save_plot(path,name)
def _two_metrics(summary,metrics,path,name):
 if not len(summary):plt.text(.5,.5,"No completed data",ha="center");plt.axis("off");_save_plot(path,name);return
 for my,g in summary.groupby("M_y"):
  g=g.sort_values("support_density")
  for metric in metrics:plt.plot(g.support_density,g[metric],marker="o",label=f"M_y={my} {metric}")
 plt.xlabel("support density");plt.legend(fontsize=7);_save_plot(path,name)
def plots(f,o):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True);s=o["support_regime_summary.csv"]
 _line(s,"A_offdiag_relative_frobenius_error_mean",path,"01_A_offdiag_error.png");_line(s,"smoothed_state_correlation_mean",path,"02_state_correlation.png");_line(s,"A_support_AUPRC_model_A_group_norm_full_universe",path,"03_AUPRC_full_universe.png");_line(s,"A_support_AUPRC_model_A_group_norm_candidate_conditional",path,"04_AUPRC_candidate.png");_line(s,"spectral_best_band_AUPRC_mean",path,"05_spectral_best_AUPRC.png")
 spec=f["spectral"];spec=spec[spec.band_name.isin(["low_band","mid_band","full_band"])] if len(spec) else spec
 if len(spec):
  for (my,band),g in spec.groupby(["M_y","band_name"]):q=g.groupby("support_density").AUPRC.mean().sort_index();plt.plot(q.index,q.values,marker="o",label=f"My={my} {band}")
  plt.xlabel("support density");plt.ylabel("AUPRC");plt.legend(fontsize=7)
 else:plt.text(.5,.5,"No completed data",ha="center");plt.axis("off")
 _save_plot(path,"06_spectral_bands.png");_line(s,"deviance_best_AUPRC_mean",path,"07_deviance_AUPRC.png")
 _two_metrics(s,["B_relative_error_mean","B_correlation_hat_true_mean"],path,"08_B_recovery.png");_two_metrics(s,["Q_relative_error_mean","Q_trace_ratio_hat_to_true_mean"],path,"09_Q_recovery.png");_line(s,"stabilized_louis_offdiag_nonzero_coverage",path,"10_Louis_active_coverage.png");_line(s,"stabilized_louis_offdiag_zero_allowed_coverage",path,"11_Louis_zero_coverage.png");_line(s,"stabilized_louis_active_std_z",path,"12_active_std_z.png");_line(s,"mean_interval_width_95",path,"13_interval_width.png")
 edge=f["obs"];true=edge[edge.true_edge.astype(bool)] if len(edge) else edge
 if len(true):
  true.boxplot(column="edge_o_inst_mean",by="missed_true_edge_indicator");plt.suptitle("");plt.title("Detected vs missed true edges")
 else:plt.text(.5,.5,"No completed data",ha="center");plt.axis("off")
 _save_plot(path,"14_observability_detected_missed.png")
 if len(true):plt.scatter(true.edge_o_inst_mean,true.group_estimation_error,s=5,alpha=.4);plt.xlabel("edge observability");plt.ylabel("group estimation error")
 else:plt.text(.5,.5,"No completed data",ha="center");plt.axis("off")
 _save_plot(path,"15_error_vs_observability.png");_line(s,"rescue_score",path,"16_rescue_score.png");_line(s,"runtime_seconds_mean",path,"17_runtime.png")

def optional_stageB_column(args):
 model,y,u,index=args;started=time.perf_counter();column,plus,minus=sensitivity_column(model,y,u,index,1e-4,30,1e-4,reestimate_B=True,reestimate_Q=True)
 if not (plus.converged and minus.converged):column,plus,minus=sensitivity_column(model,y,u,index,1e-4,50,1e-4,reestimate_B=True,reestimate_Q=True)
 return index,column,{"direction_index":index,"plus_converged":plus.converged,"minus_converged":minus.converged,"plus_n_iter":plus.n_iter,"minus_n_iter":minus.n_iter,"runtime_seconds":time.perf_counter()-started}
def run_optional_stageB_mini(primary_outputs):
 """Manual mini-confirmation only; never called in the primary default grid."""
 summary=primary_outputs["support_regime_summary.csv"];permissive=summary[(summary.M_y==15)&summary.support_regime.str.startswith("permissive")]
 best=permissive.loc[permissive.rescue_score.idxmax(),"support_regime"] if len(permissive) else None
 regimes=["full_candidate_network"]+([best] if best else [])+["oracle_support"];masks,_=paired_masks(0,0);coefficients=[];diagnostics=[]
 for regime in dict.fromkeys(regimes):
  data,seed=r1.simulate(MX,15,T,"gaussian_isotropic",0,0);meta=metadata(15,regime,0,0,masks[regime]);model=fit_masked(data,masks[regime],seed+500);louis=masked_louis(model,data,seed+700)
  selected,_=r2c.select_directions(model,data,louis,meta,seed+900,90);indices=selected.coefficient_global_index.astype(int).tolist();directory=os.path.join(RESULTS_DIR,"optional_stageB_mini_columns",regime);os.makedirs(directory,exist_ok=True);columns={}
  for index in indices:
   path=os.path.join(directory,f"direction_{index}.npy")
   if os.path.exists(path):columns[index]=np.load(path)
  pending=[index for index in indices if index not in columns]
  with ProcessPoolExecutor(max_workers=min(WORKERS,max(len(pending),1))) as pool:
   futures=[pool.submit(optional_stageB_column,(model,data["y"],data["u"],index)) for index in pending]
   for future in futures:
    index,column,row=future.result();columns[index]=column;temporary=os.path.join(directory,f"direction_{index}.npy.tmp")
    with open(temporary,"wb") as handle:np.save(handle,column)
    os.replace(temporary,os.path.join(directory,f"direction_{index}.npy"));diagnostics.append({**meta,**row});atomic_csv(pd.DataFrame(diagnostics),os.path.join(RESULTS_DIR,"optional_stageB_mini_perturbation_diagnostics.csv"))
  valid=[index for index in indices if index in columns and np.all(np.isfinite(columns[index]))];selected=selected.set_index("coefficient_global_index").loc[valid].reset_index();raw,psd,_=project_selected_covariance(np.column_stack([columns[index] for index in valid]),valid);ordinary=qa.o.global_covariance_block(model.A_row_covariances_)[np.ix_(valid,valid)];lc=qa.o.global_covariance_block(louis)[np.ix_(valid,valid)];records=qa.p.group_records(model,data,selected,{"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":lc,"lrvb_stageB_smoother_feedback":psd})
  weights={}
  for record in records:
   weight=1. if record["diagonal"] else logistic_weight(record["group_snr_louis"],1.,4.)
   for position in record["positions"]:weights[position]=weight
  gated=gated_covariance(lc,psd,np.asarray([weights.get(k,0.) for k in range(len(lc))]))
  for name,cov in (("ordinary_vb",ordinary),("stabilized_louis_eta_0p70_tau_0p90",lc),("lrvb_stageB_smoother_feedback_psd_projected",psd),("soft_stageB_logistic_c4_tau1p0",gated.covariance)):
   coefficients.append(qa.coefficient_rows(model,data,selected,{"covariance_estimator":name,"covariance":cov},meta))
 result=pd.concat(coefficients,ignore_index=True) if coefficients else pd.DataFrame();atomic_csv(result,os.path.join(RESULTS_DIR,"optional_stageB_mini_coefficient_results.csv"))
 if len(result):result["network_scope"]="selected_directions";mini=summarize_calibration(result.rename(columns={"covariance_estimator":"covariance_method"}))
 else:mini=result
 atomic_csv(mini,os.path.join(RESULTS_DIR,"optional_stageB_mini_calibration_summary.csv"))

def config_object():return {"experiment":"34S_sparsity_observability_decomposition","M_x":MX,"M_y_list":MY_LIST,"na":2,"nb":3,"n_inputs":1,"T":T,"support_regimes":REGIMES,"N_TRUE_NETWORKS":NETWORKS,"N_REPLICATES_PER_NETWORK":REPLICATES,"N_WORKERS":WORKERS,"inner_threads":int(os.environ.get("EXPERIMENT_34S_INNER_THREADS","1")),"a0":1e-3,"b0":1e-3,"Q_true":"0.50 I","R_true":"0.60 I","Q_SHRINKAGE_RHO":.25,"Q_UPDATE_DAMPING":.5,"VB_MAX_ITER":VB_MAX_ITER,"VB_CONVERGENCE_TOL":TOL,"PRACTICAL_CONVERGENCE_TOL":TOL,"N_FFBS_SAMPLES":N_FFBS,"K_obs":K_OBS,"N_FREQS":N_FREQS,"Louis_eta":.70,"Louis_tau":.90,"A_center":"A_VB","C_known":True,"R_fixed_true":True,"StageB_available":False,"RUN_OPTIONAL_STAGEB_MINI":RUN_OPTIONAL_STAGEB_MINI,"smoke_test":SMOKE,"mask_note":"Permissive and oracle masks are synthetic diagnostic structural priors; observability is not edge evidence."}
def write_or_validate_config(config):
 path=os.path.join(RESULTS_DIR,"experiment_config.json")
 if os.path.exists(path):
  with open(path,encoding="utf8") as handle:old=json.load(handle)
  operational={"N_WORKERS","inner_threads","RUN_OPTIONAL_STAGEB_MINI"}
  if {k:v for k,v in old.items() if k not in operational}!={k:v for k,v in config.items() if k not in operational}:raise RuntimeError("Existing 34S checkpoint uses a different numerical configuration. Use a new results directory or restore the original settings.")
  return
 temporary=path+".tmp"
 with open(temporary,"w",encoding="utf8") as handle:json.dump(config,handle,indent=2)
 os.replace(temporary,path)
def conclusion(decision):
 if not len(decision):print("34S ended without sufficient completed fits for a decision.");return
 oracle=bool(decision.oracle_rescue_My15.iloc[0]);permissive=bool(decision.permissive_rescue_My15.iloc[0]);p=decision[(decision.M_y==15)&decision.support_regime.str.startswith("permissive")];best=p.loc[p.rescue_score.idxmax(),"support_regime"] if len(p) else "not available";m40=decision[decision.M_y==40];base=m40[m40.support_regime.eq("full_candidate_network")].rescue_score.mean() if len(m40) else np.nan;benefit=bool(len(m40) and m40[~m40.support_regime.eq("full_candidate_network")].rescue_score.max()>base+.05);limited=not oracle;next_exp=decision.recommended_next_experiment.iloc[0]
 print("\n34S conclusion")
 print(f"1. Oracle support rescues M_y=15: {oracle}")
 print(f"2. Permissive support rescues M_y=15: {permissive}")
 print(f"3. Best permissive density: {best}")
 print(f"4. M_y=40 benefits from support restriction: {benefit}")
 print(f"5. M_y=15 remains fundamentally observability-limited: {limited}")
 print(f"6. Recommended next experiment: {next_exp}")
 print(f"7. Restrict future Stage-B to best support regime: {bool(permissive or oracle)}")
 print("8. Retain global lambda=0.75 as a later uncertainty baseline: True")

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);write_or_validate_config(config_object());tables=load_tables();current=frames(tables);run=current["run"]
 completed=run[run.run_status.isin(["success","numerically_unstable"])] if len(run) else pd.DataFrame();done=set(zip(completed.M_y.astype(int),completed.support_regime,completed.true_network_id.astype(int),completed.replicate_id.astype(int))) if len(completed) else set()
 tasks=[]
 for network in range(NETWORKS):
  for replicate in range(REPLICATES):
   masks,_=paired_masks(network,replicate)
   for my in MY_LIST:
    for regime in REGIMES:
     if (my,regime,network,replicate) not in done:tasks.append((my,regime,network,replicate,masks[regime]))
 progress=Progress(len(tasks));iterator=iter(tasks);active={}
 with ProcessPoolExecutor(max_workers=WORKERS) as pool:
  for _ in range(min(WORKERS,len(tasks))):
   task=next(iterator,None)
   if task is not None:active[pool.submit(work,task)]=task
  while active:
   finished,_=wait(active,return_when=FIRST_COMPLETED)
   for future in finished:
    task=active.pop(future)
    try:result=future.result()
    except Exception as error:
     my,regime,network,replicate,mask=task;meta=metadata(my,regime,network,replicate,mask);failure=pd.DataFrame([{**meta,"run_status":"failed_worker","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()}]);result={key:pd.DataFrame() for key in TABLES};result["run"]=failure;result["runtime"]=failure
    for key,value in result.items():
     if len(value):tables[key].append(value)
    save_raw(tables);save_outputs(tables,False);progress.update(f"My={task[0]} {task[1]} n={task[2]} r={task[3]}")
    following=next(iterator,None)
    if following is not None:active[pool.submit(work,following)]=following
 f,outputs=save_outputs(tables,True)
 if not SMOKE:plots(f,outputs)
 if RUN_OPTIONAL_STAGEB_MINI:run_optional_stageB_mini(outputs)
 conclusion(outputs["decision_summary.csv"])
if __name__=="__main__":main()
