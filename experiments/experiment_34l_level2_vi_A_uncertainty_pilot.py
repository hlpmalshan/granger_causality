"""Experiment 34L: Level-2 A-uncertainty-aware smoother pilot."""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]=os.environ.get("EXPERIMENT_34L_BLAS_THREADS","1")
import json,time,traceback,warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,grouped_stats,relative
from src.ssm.vb_ard_varx_ssm_level2 import HybridVBARDVARXSSMLevel2QeffDiag,delta_q_diag_numba
from src.stats.louis_missing_information import SAMPLER_LABEL,estimate_missing_information_numba,prior_precision_for_row,sample_companion_trajectories_ffbs,stabilized_louis_covariance

CONFIG_LIST=[{"label":"small_controlled_M5","M":5,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":3,"N_REPLICATES_PER_NETWORK":10},{"label":"main_controlled_M20","M":20,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":2,"N_REPLICATES_PER_NETWORK":5}]
RHO_GRID=[.25,.50,1.];LEVEL2_MAX_OUTER_ITER=10;N_MC_SMOOTHER_SAMPLES=100;BASE_SEED=5250000
SMOKE_TEST=os.environ.get("EXPERIMENT_34L_SMOKE","0")=="1"
if SMOKE_TEST:CONFIG_LIST=[{"label":"small_controlled_M5","M":5,"T_VALUES":[1000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":2}];LEVEL2_MAX_OUTER_ITER=5;N_MC_SMOOTHER_SAMPLES=20
override=os.environ.get("EXPERIMENT_34L_MC_SAMPLES")
if override:N_MC_SMOOTHER_SAMPLES=int(override)
RESULTS_DIR=os.environ.get("EXPERIMENT_34L_RESULTS_DIR","results/experiment_34l")

class ProgressBar:
 def __init__(self,total,width=30):self.total=max(int(total),0);self.width=width;self.done=0
 def update(self,label):self.done+=1;f=min(self.done/max(self.total,1),1);n=round(self.width*f);print(f"\rExperiment 34L [{'#'*n}{'-'*(self.width-n)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:60]}",end="\n" if self.done>=self.total else "",flush=True)

def level2_model(data,level1,rho,seed):
 model=HybridVBARDVARXSSMLevel2QeffDiag(j.na,j.nb,data["C"],data["B"],data["Q"],data["R"],rho_A_uncertainty=rho,level2_max_outer_iter=LEVEL2_MAX_OUTER_ITER,level2_tol=1e-3,max_iter=1,a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed);return model.fit_level2(data["y"],data["u"],level1)

def louis_covariances(model,data,seed):
 samples=sample_companion_trajectories_ffbs(model.smooth_result_,model.F,N_MC_SMOOTHER_SAMPLES,seed);beta=model._pack_A();priors=np.asarray([prior_precision_for_row(model.n_states,j.na,i,model.alpha_mean_,model.diagonal_prior_precision) for i in range(model.n_states)]);missing=[x["missing_information"] for x in estimate_missing_information_numba(samples,data["u"],beta,model.B_matrices,data["Q"],j.na,j.nb,priors)];corrected=[];diagnostics=[]
 for row,current in enumerate(model.A_row_covariances_):
  cov,_,diag=stabilized_louis_covariance(np.linalg.pinv(current),missing[row],.70,.90);corrected.append(cov);diagnostics.append({"target_row":row,**diag})
 return corrected,diagnostics

def method_frames(model,data,meta,method,rho,level2,louis_cov=None):
 covsets=[("level2_current" if level2 else "level1_current",model.A_row_covariances_)]
 if louis_cov is not None:covsets.append(("level2_stabilized_louis_eta_0p70_tau_0p90" if level2 else "level1_stabilized_louis_eta_0p70_tau_0p90",louis_cov))
 coeffs=[];groups=[];scores=[];rowcov=[];true=np.asarray(data["A"]);beta=model._pack_A();common={**meta,"method":method,"rho_A_uncertainty":rho,"Qeff_mode":"Qeff_timeaveraged" if level2 else "plugin_mean_A"}
 for estimator,covs in covsets:
  method_for_cov=("level1_vb_stabilized_louis_eta_0p70_tau_0p90" if estimator.startswith("level1_stabilized") else method);specific={**common,"method":method_for_cov};corrected={row:{estimator:covs[row]} for row in range(model.n_states)};coeff=j.coefficient_rows(model,data,corrected,specific);group,score=j.group_rows(model,data,corrected,specific);coeffs.append(coeff);groups.append(group)
  long=group.copy();long["score_type"]="group_snr";long["score"]=long.group_snr;scores.append(long)
  if estimator.endswith("current"):
   modelscore=score.copy();modelscore["covariance_estimator"]=estimator;modelscore["score_type"]="model_A_group_norm";modelscore["score"]=modelscore.posterior_mean_group_norm;scores.append(modelscore)
  for target,cov in enumerate(covs):
   truth=np.concatenate([true[k,target] for k in range(j.na)]);eta,tau=(.7,.9) if "louis" in estimator else (np.nan,np.nan);rowcov.append({**specific,"target_row":target,"covariance_estimator":estimator,"error_vector_json":json.dumps((beta[target]-truth).tolist()),"covariance_json":json.dumps(cov.tolist()),"eta":eta,"tau":tau})
 return pd.concat(coeffs,ignore_index=True),pd.concat(groups,ignore_index=True),pd.concat(scores,ignore_index=True),pd.DataFrame(rowcov)

def calibration_summary(frame):
 groups=["config_label","M","T","method","covariance_estimator","rho_A_uncertainty","Qeff_mode","coefficient_type"];rows=[]
 for keys,g in frame.groupby(groups,dropna=False):rows.append({**dict(zip(groups,keys)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"posterior_sd_abs_error_correlation":j.correlations(g.posterior_sd,g.abs_error),"n_coefficients":len(g)})
 return pd.DataFrame(rows)

def group_summary(frame):return frame.groupby(["config_label","M","T","method","covariance_estimator","rho_A_uncertainty","Qeff_mode","edge_group_type"],dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2","mean"),median_mahalanobis_D2=("mahalanobis_D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df_p","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df_p","mean"),mean_group_posterior_sd_trace=("group_posterior_sd_trace","mean"),mean_group_error_norm=("group_error_norm","mean"),n_groups=("mahalanobis_D2","size")).reset_index()

def network_summary(scores):
 rows=[]
 for keys,g in scores.groupby(["config_label","M","T","method","score_type","rho_A_uncertainty","Qeff_mode","covariance_estimator"],dropna=False):
  metrics,_=j.safe_curve(g.true_link,g.score);rows.append({**dict(zip(["config_label","M","T","method","score_type","rho_A_uncertainty","Qeff_mode","covariance_estimator"],keys)),**metrics})
 return pd.DataFrame(rows)

def empirical(rowcov):
 rows=[]
 for keys,g in rowcov.groupby(["config_label","true_network_id","M","T","target_row","method","covariance_estimator","rho_A_uncertainty","Qeff_mode"],dropna=False):
  errors=np.asarray([json.loads(x) for x in g.error_vector_json]);posterior=np.mean(np.asarray([json.loads(x) for x in g.covariance_json]),axis=0);emp=np.cov(errors,rowvar=False,ddof=1) if len(errors)>1 else np.diag(errors[0]**2);diff=posterior-emp;rows.append({**dict(zip(["config_label","true_network_id","M","T","target_row","method","covariance_estimator","rho_A_uncertainty","Qeff_mode"],keys)),"trace_posterior_cov":np.trace(posterior),"trace_empirical_cov":np.trace(emp),"trace_ratio_posterior_to_empirical":np.trace(posterior)/max(np.trace(emp),1e-12),"diagonal_mean_ratio_posterior_to_empirical":np.mean(np.diag(posterior))/max(np.mean(np.diag(emp)),1e-12),"frobenius_difference_to_empirical":np.linalg.norm(diff),"diagonal_frobenius_difference_to_empirical":np.linalg.norm(np.diag(diff)),"n_replicates":len(errors)})
 return pd.DataFrame(rows)

def decision(cal,gcal,network,arecovery,emp,delta,runtime):
 rows=[]
 for keys,g in cal.groupby(["M","T","method","rho_A_uncertainty","Qeff_mode","covariance_estimator"],dropna=False):
  values={x.coefficient_type:x for _,x in g.iterrows()};diag=values.get("diagonal");active=values.get("offdiag_nonzero");zero=values.get("offdiag_zero");net=network.loc[(network["T"]==keys[1])&(network.method==keys[2])&(network.covariance_estimator==keys[5])&(network.score_type=="group_snr")];groups=gcal.loc[(gcal["T"]==keys[1])&(gcal.method==keys[2])&(gcal.covariance_estimator==keys[5])];ar=arecovery.loc[(arecovery["T"]==keys[1])&(arecovery.method==keys[2])];em=emp.loc[(emp["T"]==keys[1])&(emp.method==keys[2])&(emp.covariance_estimator==keys[5])];dq=delta.loc[(delta["T"]==keys[1])&(delta.method==keys[2])];rt=runtime.loc[(runtime["T"]==keys[1])&(runtime.method==keys[2])];dc=diag.empirical_coverage_95 if diag is not None else np.nan;ac=active.empirical_coverage_95 if active is not None else np.nan;zc=zero.empirical_coverage_95 if zero is not None else np.nan;rows.append({"M":keys[0],"T":keys[1],"method":keys[2],"rho_A_uncertainty":keys[3],"Qeff_mode":keys[4],"covariance_estimator":keys[5],"diag_coverage":dc,"offdiag_nonzero_coverage":ac,"offdiag_zero_coverage":zc,"diag_std_z":diag.std_standardized_error if diag is not None else np.nan,"offdiag_nonzero_std_z":active.std_standardized_error if active is not None else np.nan,"offdiag_zero_std_z":zero.std_standardized_error if zero is not None else np.nan,"group_true_edge_chi2_95_coverage":groups.loc[groups.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"group_false_edge_chi2_95_coverage":groups.loc[groups.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"group_snr_AUPRC":net.AUPRC.mean(),"group_snr_TPR_at_FPR_0p05":net.TPR_at_FPR_0p05.mean(),"A_relative_frobenius_error":ar.A_relative_frobenius_error.mean(),"A_offdiag_relative_frobenius_error":ar.A_offdiag_relative_frobenius_error.mean(),"trace_ratio_posterior_to_empirical":em.trace_ratio_posterior_to_empirical.mean(),"Delta_Q_A_trace_ratio_to_Q_trace":dq.Delta_Q_A_trace_ratio_to_Q_trace.mean(),"runtime_seconds":rt.total_runtime_seconds.mean(),"recommended_for_uncertainty":bool(dc>=.85 and ac>=.85 and .93<=zc<=.985),"recommended_for_ranking":bool(net.AUPRC.mean()>=network.loc[network.method=="level1_vb_current","AUPRC"].mean()-.05),"notes":"Level-2 Qeff is an approximate structured-VI pilot"})
 return pd.DataFrame(rows)

def plots(cal,network,emp,history,runtime,coeff,delta):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 def line(frame,x,y,name,label):
  g=frame.groupby(x,dropna=False)[y].mean();fig,ax=plt.subplots();ax.plot(g.index.astype(str),g.values,marker="o");ax.set(xlabel=x,ylabel=label);ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 active=cal.loc[cal.coefficient_type.isin(["diagonal","offdiag_nonzero"])];zero=cal.loc[cal.coefficient_type=="offdiag_zero"];line(active,"covariance_estimator","empirical_coverage_95","active_coverage.png","coverage");line(zero,"covariance_estimator","empirical_coverage_95","zero_coverage.png","coverage");line(cal,"covariance_estimator","std_standardized_error","standardized_sd.png","standardized error SD")
 current=coeff.loc[coeff.covariance_estimator=="level1_current"][["config_label","true_network_id","replicate_id","M","T","target_row","lag","source","posterior_sd"]].rename(columns={"posterior_sd":"level1_sd"});l2=coeff.loc[coeff.covariance_estimator=="level2_current"].merge(current,on=["config_label","true_network_id","replicate_id","M","T","target_row","lag","source"]);fig,ax=plt.subplots();ax.scatter(l2.level1_sd,l2.posterior_sd,s=2);ax.set(xlabel="Level-1 SD",ylabel="Level-2 SD");fig.tight_layout();fig.savefig(os.path.join(path,"sd_scatter.png"));plt.close(fig)
 line(active,"method","empirical_coverage_95","level1_level2_coverage.png","active coverage");line(delta,"rho_A_uncertainty","Delta_Q_A_trace_ratio_to_Q_trace","delta_Q.png","Delta-Q/Q trace");line(network,"method","AUPRC","AUPRC.png","AUPRC");line(network,"method","TPR_at_FPR_0p05","TPR05.png","TPR <= .05");line(emp,"method","trace_ratio_posterior_to_empirical","empirical_trace.png","posterior/empirical trace");line(history,"level2_outer_iter","relative_A_mean_change","convergence.png","relative A change");line(runtime,"method","total_runtime_seconds","runtime.png","seconds")

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config=json.loads(json.dumps({"experiment":"34L","CONFIG_LIST":CONFIG_LIST,"RHO_A_UNCERTAINTY_GRID":RHO_GRID,"LEVEL2_MAX_OUTER_ITER":LEVEL2_MAX_OUTER_ITER,"coordinate_A_updates_per_outer":1,"N_MC_SMOOTHER_SAMPLES":N_MC_SMOOTHER_SAMPLES,"Qeff_mode":"Qeff_timeaveraged","smoother_sampler":SAMPLER_LABEL,"numba_kernels":["delta_q_diag_numba","complete_data_score_numba"],"smoke_test":SMOKE_TEST,"TODO":"time-varying Qeff only if promising; no spectral or rectangular C"}));cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
 if os.path.exists(cp) and os.path.exists(ledger):
  with open(cp,encoding="utf8") as h:old=json.load(h)
  for key in ("CONFIG_LIST","RHO_A_UNCERTAINTY_GRID","LEVEL2_MAX_OUTER_ITER","N_MC_SMOOTHER_SAMPLES","Qeff_mode"):
   if old.get(key)!=config.get(key):raise ValueError("Existing 34L checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as h:json.dump(config,h,indent=2)
 names={"coeff":"_coeff_partial.csv","group":"_group_partial.csv","scores":"_scores_partial.csv","rowcov":"_rowcov_partial.csv","history":"_history_partial.csv","delta":"_delta_partial.csv","rowdelta":"_rowdelta_partial.csv","arecovery":"_arecovery_partial.csv","stability":"_stability_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={k:[] for k in names}
 for k,n in names.items():
  p=os.path.join(RESULTS_DIR,n)
  if os.path.exists(p):
   try:d=pd.read_csv(p)
   except pd.errors.EmptyDataError:d=pd.DataFrame()
   if len(d):tables[k]=[d]
 old=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame();completed=set(zip(old.config_label,old["T"].astype(int),old.true_network_id.astype(int),old.replicate_id.astype(int))) if len(old) else set();total=sum((c["label"],T,n,r) not in completed for c in CONFIG_LIST for T in c["T_VALUES"] for n in range(c["N_TRUE_NETWORKS"]) for r in range(c["N_REPLICATES_PER_NETWORK"]));progress=ProgressBar(total)
 # Compile Numba kernels before timed work.
 delta_q_diag_numba(np.zeros((1,2,2)),np.eye(2))
 for cfg in CONFIG_LIST:
  M=cfg["M"]
  for network_id in range(cfg["N_TRUE_NETWORKS"]):
   network_seed=BASE_SEED+M*100000+network_id*1000;A,B,mask=j.fixed_network(M,network_seed)
   for T in cfg["T_VALUES"]:
    for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
     key=(cfg["label"],T,network_id,replicate)
     if key in completed:continue
     seed=network_seed+T*10+replicate;meta={"config_label":cfg["label"],"M":M,"T":T,"true_network_id":network_id,"replicate_id":replicate};started=time.perf_counter();status="success"
     try:
      data=j.simulate(A,B,M,T,seed);fit_start=time.perf_counter();level1=j.fit_model(data,seed+500);level1_runtime=time.perf_counter()-fit_start;louis_start=time.perf_counter();l1_louis,l1diag=louis_covariances(level1,data,seed+700);louis_runtime=time.perf_counter()-louis_start;c,g,s,r=method_frames(level1,data,meta,"level1_vb_current",np.nan,False,l1_louis);tables["coeff"].append(c);tables["group"].append(g);tables["scores"].append(s);tables["rowcov"].append(r)
      rec,_=j.recovery(np.asarray(level1.A_mean_matrices_),data,level1.filtered_state_mean_,level1.smoothed_state_mean_,meta,level1_runtime,level1.n_iter_,level1.converged_);tables["arecovery"].append({**meta,"method":"level1_vb_current","rho_A_uncertainty":np.nan,**rec});tables["arecovery"].append({**meta,"method":"level1_vb_stabilized_louis_eta_0p70_tau_0p90","rho_A_uncertainty":np.nan,**rec});tables["stability"].append({**meta,"method":"level1_vb_current","stability_rescaling_occurred":bool(any(level1.rescaling_history_)),"number_of_stability_rescaling_events":int(sum(level1.rescaling_history_)),"spectral_radius_before_rescaling_max":max(level1.spectral_radius_before_rescaling_history_),"spectral_radius_after_rescaling_max":max(level1.spectral_radius_after_rescaling_history_),"min_rescaling_factor":min(level1.rescaling_factor_history_),"mean_rescaling_factor":np.mean(level1.rescaling_factor_history_)})
      tables["runtime"].append({**meta,"method":"level1_vb_current","rho_A_uncertainty":np.nan,"level1_fit_runtime_seconds":level1_runtime,"level2_outer_loop_runtime_seconds":0.,"louis_correction_runtime_seconds":louis_runtime,"total_runtime_seconds":level1_runtime+louis_runtime,"number_level2_outer_iters_completed":0,"converged_level2":False})
      tables["runtime"].append({**meta,"method":"level1_vb_stabilized_louis_eta_0p70_tau_0p90","rho_A_uncertainty":np.nan,"level1_fit_runtime_seconds":level1_runtime,"level2_outer_loop_runtime_seconds":0.,"louis_correction_runtime_seconds":louis_runtime,"total_runtime_seconds":level1_runtime+louis_runtime,"number_level2_outer_iters_completed":0,"converged_level2":False})
      for rho in RHO_GRID:
       l2start=time.perf_counter();model=level2_model(data,level1,rho,seed+800+int(rho*100));l2runtime=time.perf_counter()-l2start;louis_start=time.perf_counter();l2_louis,l2diag=louis_covariances(model,data,seed+900+int(rho*100));l2louis=time.perf_counter()-louis_start;c,g,s,r=method_frames(model,data,meta,"level2_vi_Qeff_diag",rho,True,l2_louis);tables["coeff"].append(c);tables["group"].append(g);tables["scores"].append(s);tables["rowcov"].append(r);tables["history"].append(pd.DataFrame([{**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,"Qeff_mode":"Qeff_timeaveraged",**x} for x in model.level2_iteration_history_]))
       dq=model.Delta_Q_A_diag_;tables["delta"].append({**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,"Qeff_mode":"Qeff_timeaveraged","mean_Delta_Q_A_trace":dq.sum(),"median_Delta_Q_A_trace":dq.sum(),"Delta_Q_A_trace_ratio_to_Q_trace":dq.sum()/np.trace(data["Q"]),"mean_Delta_Q_A_diag":dq.mean(),"max_Delta_Q_A_diag":dq.max(),"min_Delta_Q_A_diag":dq.min(),"Delta_Q_A_diag_coefficient_of_variation":dq.std()/max(dq.mean(),1e-12),"Qeff_trace_ratio_to_Q":np.trace(model.Q_eff_)/np.trace(data["Q"]),"Qeff_min_diag":np.diag(model.Q_eff_).min(),"Qeff_max_diag":np.diag(model.Q_eff_).max()})
       true=np.asarray(data["A"]);estimate=np.asarray(model.A_mean_matrices_)
       for row in range(M):
        active=[lag*M+source for lag in range(j.na) for source in range(M) if source!=row and np.any(true[:,row,source]!=0)];sd=np.sqrt(np.maximum(np.diag(model.A_row_covariances_[row]),0));truth_row=np.concatenate([true[k,row] for k in range(j.na)]);estimate_row=np.concatenate([estimate[k,row] for k in range(j.na)]);coverage=float(np.mean(np.abs(estimate_row[active]-truth_row[active])<=1.96*sd[active])) if active else np.nan;tables["rowdelta"].append({**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,"Qeff_mode":"Qeff_timeaveraged","target_row":row,"row_A_cov_trace":np.trace(model.A_row_covariances_[row]),"row_mean_Delta_Q_A":dq[row],"row_active_edge_count":int(data["mask"][row].sum()),"row_error_norm":np.linalg.norm(estimate[:,row]-true[:,row]),"row_coverage_active":coverage})
       rec,_=j.recovery(estimate,data,model.filtered_state_mean_,model.smoothed_state_mean_,meta,l2runtime,model.n_level2_outer_iters_,model.converged_level2_);tables["arecovery"].append({**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,**rec});tables["runtime"].append({**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,"level1_fit_runtime_seconds":level1_runtime,"level2_outer_loop_runtime_seconds":l2runtime,"louis_correction_runtime_seconds":l2louis,"total_runtime_seconds":level1_runtime+l2runtime+l2louis,"average_runtime_per_level2_outer_iter":l2runtime/max(model.n_level2_outer_iters_,1),"LEVEL2_MAX_OUTER_ITER":LEVEL2_MAX_OUTER_ITER,"number_level2_outer_iters_completed":model.n_level2_outer_iters_,"converged_level2":model.converged_level2_});tables["stability"].append({**meta,"method":"level2_vi_Qeff_diag","rho_A_uncertainty":rho,"stability_rescaling_occurred":bool(any(x["rescaled_A_flag"] for x in model.level2_iteration_history_)),"number_of_stability_rescaling_events":sum(x["rescaled_A_flag"] for x in model.level2_iteration_history_)})
     except Exception as error:status="failed";tables["runtime"].append({**meta,"method":"failed","total_runtime_seconds":time.perf_counter()-started,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()})
     tables["run"].append({**meta,"run_status":status,"total_runtime_seconds":time.perf_counter()-started});frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()}
     for k,n in names.items():atomic_csv(frames[k],os.path.join(RESULTS_DIR,n))
     if len(frames["coeff"]):
      cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);net=network_summary(frames["scores"]);emp=empirical(frames["rowcov"]);dec=decision(cal,gcal,net,frames["arecovery"],emp,frames["delta"],frames["runtime"])
      for n,d in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",gcal),("network_recovery_summary_partial.csv",net),("delta_Q_diagnostics_partial.csv",frames["delta"]),("empirical_covariance_comparison_partial.csv",emp),("runtime_summary_partial.csv",grouped_stats(frames["runtime"],["config_label","M","T","method","rho_A_uncertainty"],["level1_fit_runtime_seconds","level2_outer_loop_runtime_seconds","louis_correction_runtime_seconds","total_runtime_seconds"])),("decision_summary_partial.csv",dec)):atomic_csv(d,os.path.join(RESULTS_DIR,n))
     progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
 frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()};cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);net=network_summary(frames["scores"]);emp=empirical(frames["rowcov"]);arecovery=grouped_stats(frames["arecovery"],["config_label","M","T","method","rho_A_uncertainty"],["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error","A_mean_absolute_error","A_offdiag_mean_absolute_error","A_diagonal_mean_absolute_error","A_group_norm_pearson_correlation","A_group_norm_spearman_correlation"]);runtime_summary=grouped_stats(frames["runtime"],["config_label","M","T","method","rho_A_uncertainty"],["level1_fit_runtime_seconds","level2_outer_loop_runtime_seconds","louis_correction_runtime_seconds","total_runtime_seconds","number_level2_outer_iters_completed","converged_level2"]);dec=decision(cal,gcal,net,frames["arecovery"],emp,frames["delta"],frames["runtime"])
 outputs={"run_summary.csv":frames["run"],"level2_iteration_history.csv":frames["history"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":gcal,"network_recovery_summary.csv":net,"A_recovery_summary.csv":arecovery,"delta_Q_diagnostics.csv":frames["delta"],"row_delta_Q_diagnostics.csv":frames["rowdelta"],"empirical_covariance_comparison.csv":emp,"stability_rescaling_diagnostics.csv":frames["stability"],"runtime_summary.csv":runtime_summary,"decision_summary.csv":dec}
 for n,d in outputs.items():atomic_csv(d,os.path.join(RESULTS_DIR,n))
 plots(cal,net,emp,frames["history"],frames["runtime"],frames["coeff"],frames["delta"])

if __name__=="__main__":main()
