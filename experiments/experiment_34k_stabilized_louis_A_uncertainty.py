"""Experiment 34K: stabilized Louis covariance calibration for M=20."""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]=os.environ.get("EXPERIMENT_34K_BLAS_THREADS","1")
import json,time,traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,grouped_stats,relative
from src.stats.louis_missing_information import SAMPLER_LABEL,estimate_missing_information,louis_correct_covariances,prior_precision_for_row,sample_companion_trajectories_ffbs,stabilized_louis_covariance

M=20;T_VALUES=[1000,2000];N_TRUE_NETWORKS=2;N_REPLICATES_PER_NETWORK=5;N_MC_SMOOTHER_SAMPLES=100
ETA_GRID=[.45,.50,.55,.60,.65,.70];TAU_GRID=[.60,.70,.80,.90];BASE_SEED=5150000
SMOKE_TEST=os.environ.get("EXPERIMENT_34K_SMOKE","0")=="1"
if SMOKE_TEST:T_VALUES=[1000];N_TRUE_NETWORKS=1;N_REPLICATES_PER_NETWORK=2;N_MC_SMOOTHER_SAMPLES=50;ETA_GRID=[.50,.60];TAU_GRID=[.70,.80]
override=os.environ.get("EXPERIMENT_34K_MC_SAMPLES")
if override:N_MC_SMOOTHER_SAMPLES=int(override)
RESULTS_DIR=os.environ.get("EXPERIMENT_34K_RESULTS_DIR","results/experiment_34k")

class ProgressBar:
 def __init__(self,total,width=30):self.total=max(int(total),0);self.width=width;self.done=0;print("Experiment 34K: all checkpointed replicates are complete.") if not self.total else None
 def update(self,label):self.done+=1;f=min(self.done/max(self.total,1),1);n=round(self.width*f);print(f"\rExperiment 34K [{'#'*n}{'-'*(self.width-n)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:62]}",end="\n" if self.done>=self.total else "",flush=True)

def label(eta,tau):return f"stabilized_eta_{eta:.2f}_tau_{tau:.2f}".replace(".","p")
def correction_specs():return [(eta,tau) for eta in ETA_GRID for tau in TAU_GRID if eta*tau<.98]
def estimator_meta(estimator):
 if estimator=="current":return np.nan,np.nan
 if estimator=="louis_eta_0p50_uncapped":return .5,np.nan
 pieces=estimator.replace("stabilized_eta_","").split("_tau_");return float(pieces[0].replace("p",".")),float(pieces[1].replace("p","."))

def corrections(current_covariances,missing):
 uncapped,uncapped_diag=louis_correct_covariances(current_covariances,missing,[.5]);result={};diagnostics=[]
 for row,current in enumerate(current_covariances):
  result[row]={"current":current,"louis_eta_0p50_uncapped":uncapped[row]["louis_eta_0p50"]};complete=np.linalg.pinv(current)
  for eta,tau in correction_specs():
   cov,_,diag=stabilized_louis_covariance(complete,missing[row],eta,tau);name=label(eta,tau);result[row][name]=cov;diagnostics.append({"target_row":row,"covariance_estimator":name,"trace_current_cov":np.trace(current),"trace_corrected_cov":np.trace(cov),"trace_corrected_cov_over_current_cov":np.trace(cov)/max(np.trace(current),1e-12),**diag})
  base=next(x for x in uncapped_diag if x["target_row"]==row);base={**base,"eta":.5,"tau":np.nan,"eta_tau":np.nan,"required_final_eigen_clipping":base["number_small_eigenvalues_I_observed"]>0,"number_eigenvalues_capped":0,"fraction_eigenvalues_capped":0,"covariance_estimator":"louis_eta_0p50_uncapped"};diagnostics.append(base)
 return result,diagnostics

def add_eta_tau(frame):
 frame=frame.copy();values=frame.covariance_estimator.map(estimator_meta);frame["eta"]=[x[0] for x in values];frame["tau"]=[x[1] for x in values];return frame

def calibration_summary(frame):
 groups=["M","T","covariance_estimator","eta","tau","coefficient_type"];rows=[]
 for keys,g in frame.groupby(groups,dropna=False):rows.append({**dict(zip(groups,keys)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"posterior_sd_abs_error_correlation":j.correlations(g.posterior_sd,g.abs_error),"n_coefficients":len(g)})
 return pd.DataFrame(rows)

def group_summary(frame):return frame.groupby(["M","T","covariance_estimator","eta","tau","edge_group_type"],dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2","mean"),median_mahalanobis_D2=("mahalanobis_D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df_p","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df_p","mean"),mean_group_posterior_sd_trace=("group_posterior_sd_trace","mean"),mean_group_error_norm=("group_error_norm","mean"),n_groups=("mahalanobis_D2","size")).reset_index()

def score_summary(scores):
 rows=[]
 for estimator in ["current","louis_eta_0p50_uncapped"]+[label(*x) for x in correction_specs()]:
  for keys,g in scores.groupby(["M","T"]):
   metrics,_=j.safe_curve(g.true_link,g["group_snr_"+estimator]);eta,tau=estimator_meta(estimator);rows.append({"M":keys[0],"T":keys[1],"covariance_estimator":estimator,"eta":eta,"tau":tau,**metrics})
 return pd.DataFrame(rows)

def empirical_covariance(rowcov):
 rows=[]
 for keys,g in rowcov.groupby(["true_network_id","M","T","target_row","covariance_estimator","eta","tau"],dropna=False):
  errors=np.asarray([json.loads(x) for x in g.error_vector_json]);posterior=np.mean(np.asarray([json.loads(x) for x in g.covariance_json]),axis=0);emp=np.cov(errors,rowvar=False,ddof=1) if len(errors)>1 else np.diag(errors[0]**2);difference=posterior-emp;rows.append({"true_network_id":keys[0],"M":keys[1],"T":keys[2],"target_row":keys[3],"covariance_estimator":keys[4],"eta":keys[5],"tau":keys[6],"trace_posterior_cov":np.trace(posterior),"trace_empirical_cov":np.trace(emp),"trace_ratio_posterior_to_empirical":np.trace(posterior)/max(np.trace(emp),1e-12),"diagonal_mean_ratio_posterior_to_empirical":np.mean(np.diag(posterior))/max(np.mean(np.diag(emp)),1e-12),"frobenius_difference_to_empirical":np.linalg.norm(difference),"diagonal_frobenius_difference_to_empirical":np.linalg.norm(np.diag(difference)),"n_replicates":len(errors)})
 return pd.DataFrame(rows)

def diagnostics_summary(diag):return diag.loc[diag.covariance_estimator.str.startswith("stabilized")].groupby(["M","T","eta","tau"],dropna=False).agg(mean_trace_corrected_over_current=("trace_corrected_cov_over_current_cov","mean"),median_trace_corrected_over_current=("trace_corrected_cov_over_current_cov","median"),mean_trace_missing_over_complete=("trace_missing_over_complete","mean"),mean_number_eigenvalues_capped=("number_eigenvalues_capped","mean"),mean_fraction_eigenvalues_capped=("fraction_eigenvalues_capped","mean"),fraction_rows_requiring_final_eigen_clipping=("required_final_eigen_clipping","mean"),fraction_rows_with_negative_Iobs_before_final_clipping=("number_negative_eigenvalues_I_observed",lambda x:np.mean(x>0)),mean_min_eigenvalue_Iobs_before_final_clipping=("min_eigenvalue_I_observed_before_final_clipping","mean"),mean_min_eigenvalue_Iobs_after_final_clipping=("min_eigenvalue_I_observed_after_final_clipping","mean")).reset_index()

def decision_summary(cal,scores,emp,diag):
 rows=[]
 for keys,g in cal.groupby(["M","T","covariance_estimator","eta","tau"],dropna=False):
  r={"M":keys[0],"T":keys[1],"covariance_estimator":keys[2],"eta":keys[3],"tau":keys[4]}
  for _,item in g.iterrows():r["empirical_coverage_95_"+item.coefficient_type]=item.empirical_coverage_95;r["std_standardized_error_"+item.coefficient_type]=item.std_standardized_error
  match=scores.loc[(scores["T"]==r["T"])&(scores.covariance_estimator==r["covariance_estimator"])];em=emp.loc[(emp["T"]==r["T"])&(emp.covariance_estimator==r["covariance_estimator"])];dg=diag.loc[(diag["T"]==r["T"])&(diag.covariance_estimator==r["covariance_estimator"])];diagcov=r.get("empirical_coverage_95_diagonal",np.nan);active=r.get("empirical_coverage_95_offdiag_nonzero",np.nan);zero=r.get("empirical_coverage_95_offdiag_zero",np.nan);recommend=bool(diagcov>=.85 and active>=.85 and .93<=zero<=.985 and (dg.required_final_eigen_clipping.mean() if len(dg) else 0)<.2);current=scores.loc[(scores["T"]==r["T"])&(scores.covariance_estimator=="current")];auprc=match.AUPRC.mean();baseline=current.AUPRC.mean();rows.append({"M":r["M"],"T":r["T"],"covariance_estimator":r["covariance_estimator"],"eta":r["eta"],"tau":r["tau"],"diag_coverage":diagcov,"offdiag_nonzero_coverage":active,"offdiag_zero_coverage":zero,"diag_std_z":r.get("std_standardized_error_diagonal"),"offdiag_nonzero_std_z":r.get("std_standardized_error_offdiag_nonzero"),"offdiag_zero_std_z":r.get("std_standardized_error_offdiag_zero"),"group_snr_AUPRC":auprc,"group_snr_TPR_at_FPR_0p05":match.TPR_at_FPR_0p05.mean(),"trace_ratio_posterior_to_empirical":em.trace_ratio_posterior_to_empirical.mean(),"fraction_rows_requiring_final_clipping":dg.required_final_eigen_clipping.mean() if len(dg) else 0,"mean_fraction_eigenvalues_capped":dg.fraction_eigenvalues_capped.mean() if len(dg) else 0,"recommended_for_uncertainty":recommend,"recommended_for_group_snr":bool(auprc>=baseline-.05),"notes":"uncertainty and ranking recommendations are intentionally separate"})
 return pd.DataFrame(rows)

def stability_summary(coeff):
 f=coeff.copy();f["rescaling_group"]=np.where(f.stability_rescaling_occurred,"any_rescaling","no_rescaling");return f.groupby(["M","T","covariance_estimator","coefficient_type","rescaling_group"]).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),std_standardized_error=("standardized_error","std"),n_fits=("replicate_id","nunique")).reset_index()

def plots(cal,diag,emp,scores,coeff,runtime):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True);stable=cal.loc[cal.covariance_estimator.str.startswith("stabilized")]
 def heat(frame,value,name,label,kind=None):
  q=frame if kind is None else frame.loc[frame.coefficient_type==kind];p=q.pivot_table(index="eta",columns="tau",values=value);fig,ax=plt.subplots();im=ax.imshow(p.values,aspect="auto",origin="lower");ax.set(xticks=range(len(p.columns)),xticklabels=p.columns,yticks=range(len(p.index)),yticklabels=p.index,xlabel="tau",ylabel="eta");ax.set_title(label);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 for kind,name in (("diagonal","coverage_diagonal.png"),("offdiag_nonzero","coverage_active.png"),("offdiag_zero","coverage_zero.png")):heat(stable,"empirical_coverage_95",name,"coverage",kind)
 heat(stable,"std_standardized_error","standardized_sd.png","standardized error SD","offdiag_nonzero");heat(emp.loc[emp.covariance_estimator.str.startswith("stabilized")],"trace_ratio_posterior_to_empirical","trace_empirical_heatmap.png","posterior/empirical trace");heat(diag.loc[diag.covariance_estimator.str.startswith("stabilized")],"trace_corrected_cov_over_current_cov","trace_current_heatmap.png","corrected/current trace")
 for frame,value,name,label in ((diag,"max_eigenvalue_K_raw","K_max_distribution.png","max K eigenvalue"),(diag,"fraction_eigenvalues_capped","fraction_capped.png","fraction capped"),(scores,"AUPRC","group_snr_AUPRC.png","AUPRC"),(scores,"TPR_at_FPR_0p05","group_snr_TPR05.png","TPR <= .05")):
  fig,ax=plt.subplots();
  if "eta" in frame:ax.scatter(frame.eta,frame[value],s=8)
  ax.set(xlabel="eta",ylabel=label);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 current=coeff.loc[coeff.covariance_estimator=="current"][["true_network_id","replicate_id","T","target_row","lag","source","posterior_sd"]].rename(columns={"posterior_sd":"current_sd"});s=coeff.loc[coeff.covariance_estimator.str.startswith("stabilized")].merge(current,on=["true_network_id","replicate_id","T","target_row","lag","source"]);fig,ax=plt.subplots();ax.scatter(s.current_sd,s.posterior_sd,s=2);ax.set(xlabel="current SD",ylabel="stabilized SD");fig.tight_layout();fig.savefig(os.path.join(path,"sd_scatter.png"));plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(emp.trace_empirical_cov,emp.trace_posterior_cov,s=4);ax.set(xlabel="empirical trace",ylabel="posterior trace");fig.tight_layout();fig.savefig(os.path.join(path,"empirical_trace.png"));plt.close(fig)
 heat(stable,"empirical_coverage_95","active_eta_tau.png","active coverage","offdiag_nonzero");heat(stable,"empirical_coverage_95","zero_eta_tau.png","zero coverage","offdiag_zero");fig,ax=plt.subplots();g=runtime.groupby("N_MC_SMOOTHER_SAMPLES").total_runtime_seconds.mean();ax.plot(g.index,g.values,marker="o");ax.set(xlabel="MC samples",ylabel="seconds");fig.tight_layout();fig.savefig(os.path.join(path,"runtime.png"));plt.close(fig)

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config=json.loads(json.dumps({"experiment":"34K","M":M,"T_VALUES":T_VALUES,"N_TRUE_NETWORKS":N_TRUE_NETWORKS,"N_REPLICATES_PER_NETWORK":N_REPLICATES_PER_NETWORK,"N_MC_SMOOTHER_SAMPLES":N_MC_SMOOTHER_SAMPLES,"ETA_GRID":ETA_GRID,"TAU_GRID":TAU_GRID,"VALID_PAIRS":correction_specs(),"smoother_sampler":SAMPLER_LABEL,"controlled_setting":"identity C, fixed true B/Q/R","smoke_test":SMOKE_TEST,"TODO":"Level-2 VI only if stabilized Louis fails; spectral branch remains separate"}));cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
 if os.path.exists(cp) and os.path.exists(ledger):
  with open(cp,encoding="utf8") as h:old=json.load(h)
  for k in ("M","T_VALUES","N_TRUE_NETWORKS","N_REPLICATES_PER_NETWORK","N_MC_SMOOTHER_SAMPLES","ETA_GRID","TAU_GRID"):
   if old.get(k)!=config.get(k):raise ValueError("Existing 34K checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as h:json.dump(config,h,indent=2)
 names={"coeff":"coefficient_calibration_results_partial.csv","group":"_group_partial.csv","diag":"stabilized_louis_diagnostics_partial.csv","scores":"_scores_partial.csv","rowcov":"_rowcov_partial.csv","fit":"_fit_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={k:[] for k in names}
 for k,n in names.items():
  p=os.path.join(RESULTS_DIR,n)
  if os.path.exists(p):
   try:d=pd.read_csv(p)
   except pd.errors.EmptyDataError:d=pd.DataFrame()
   if len(d):tables[k]=[d]
 old=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame();completed=set(zip(old["T"].astype(int),old.true_network_id.astype(int),old.replicate_id.astype(int))) if len(old) else set();total=sum((T,n,r) not in completed for T in T_VALUES for n in range(N_TRUE_NETWORKS) for r in range(N_REPLICATES_PER_NETWORK));progress=ProgressBar(total)
 for network_id in range(N_TRUE_NETWORKS):
  network_seed=BASE_SEED+network_id*1000;A,B,mask=j.fixed_network(M,network_seed)
  for T in T_VALUES:
   for replicate in range(N_REPLICATES_PER_NETWORK):
    if (T,network_id,replicate) in completed:continue
    seed=network_seed+T*10+replicate;meta={"config_label":"main_controlled_M20","M":M,"T":T,"true_network_id":network_id,"replicate_id":replicate,"N_MC_SMOOTHER_SAMPLES":N_MC_SMOOTHER_SAMPLES,"smoother_sampler":SAMPLER_LABEL};started=time.perf_counter()
    try:
     data=j.simulate(A,B,M,T,seed);fit_start=time.perf_counter();model=j.fit_model(data,seed+500);fit_runtime=time.perf_counter()-fit_start;sample_start=time.perf_counter();samples=sample_companion_trajectories_ffbs(model.smooth_result_,model.F,N_MC_SMOOTHER_SAMPLES,seed+700);sample_runtime=time.perf_counter()-sample_start;missing_start=time.perf_counter();beta=model._pack_A();priors=[prior_precision_for_row(M,j.na,i,model.alpha_mean_,model.diagonal_prior_precision) for i in range(M)];missing=[x["missing_information"] for x in estimate_missing_information(samples,data["u"],beta,model.B_matrices,model.Q,j.na,j.nb,priors)];missing_runtime=time.perf_counter()-missing_start;correction_start=time.perf_counter();corrected,diagrows=corrections(model.A_row_covariances_,missing);correction_runtime=time.perf_counter()-correction_start
     rescale=bool(any(model.rescaling_history_));common={**meta,"stability_rescaling_occurred":rescale,"number_of_stability_rescaling_events":int(sum(model.rescaling_history_)),"spectral_radius_before_rescaling_max":max(model.spectral_radius_before_rescaling_history_),"spectral_radius_after_rescaling_max":max(model.spectral_radius_after_rescaling_history_),"min_rescaling_factor":min(model.rescaling_factor_history_),"mean_rescaling_factor":np.mean(model.rescaling_factor_history_)};coeff=add_eta_tau(j.coefficient_rows(model,data,corrected,common));groups,scores=j.group_rows(model,data,corrected,common);groups=add_eta_tau(groups);diag=pd.DataFrame([{**common,**x} for x in diagrows]);true=np.asarray(data["A"]);rows=[]
     for target in range(M):
      truth=np.concatenate([true[k,target] for k in range(j.na)]);error=beta[target]-truth
      for estimator,cov in corrected[target].items():
       eta,tau=estimator_meta(estimator);rows.append({**common,"target_row":target,"covariance_estimator":estimator,"eta":eta,"tau":tau,"error_vector_json":json.dumps(error.tolist()),"covariance_json":json.dumps(cov.tolist())})
     fitrow={**common,"fit_status":"success","n_iter":model.n_iter_,"converged":model.converged_,"A_relative_frobenius_error":relative(np.asarray(model.A_mean_matrices_),true)};runtime={**common,"fit_runtime_seconds":fit_runtime,"smoother_sampling_runtime_seconds":sample_runtime,"louis_missing_information_runtime_seconds":missing_runtime,"stabilized_correction_runtime_seconds":correction_runtime,"total_runtime_seconds":time.perf_counter()-started,"number_of_rows_corrected":M,"average_row_correction_runtime_seconds":correction_runtime/M};tables["coeff"].append(coeff);tables["group"].append(groups);tables["diag"].append(diag);tables["scores"].append(scores);tables["rowcov"].append(pd.DataFrame(rows));tables["fit"].append(fitrow);tables["runtime"].append(runtime);status="success"
    except Exception as error:
     status="failed";tables["fit"].append({**meta,"fit_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()});tables["runtime"].append({**meta,"total_runtime_seconds":time.perf_counter()-started})
    tables["run"].append({**meta,"fit_status":status,"total_runtime_seconds":time.perf_counter()-started});frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()}
    for k,n in names.items():atomic_csv(frames[k],os.path.join(RESULTS_DIR,n))
    if len(frames["coeff"]):
     cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);ss=score_summary(frames["scores"]);emp=empirical_covariance(frames["rowcov"]);dec=decision_summary(cal,ss,emp,frames["diag"])
     for n,d in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",gcal),("K_spectrum_diagnostics_partial.csv",frames["diag"]),("empirical_covariance_comparison_partial.csv",emp),("corrected_group_snr_summary_partial.csv",ss),("network_recovery_summary_partial.csv",ss),("runtime_summary_partial.csv",grouped_stats(frames["runtime"],["M","T"],["fit_runtime_seconds","smoother_sampling_runtime_seconds","louis_missing_information_runtime_seconds","stabilized_correction_runtime_seconds","total_runtime_seconds"])),("decision_summary_partial.csv",dec)):atomic_csv(d,os.path.join(RESULTS_DIR,n))
    progress.update(f"M=20 T={T} net={network_id+1}/{N_TRUE_NETWORKS} rep={replicate+1}/{N_REPLICATES_PER_NETWORK} {status}")
 frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()};cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);ss=score_summary(frames["scores"]);emp=empirical_covariance(frames["rowcov"]);stability=j.stability_summary(frames["coeff"]);runtime_summary=grouped_stats(frames["runtime"],["M","T"],["fit_runtime_seconds","smoother_sampling_runtime_seconds","louis_missing_information_runtime_seconds","stabilized_correction_runtime_seconds","total_runtime_seconds","number_of_rows_corrected","average_row_correction_runtime_seconds"]);decision=decision_summary(cal,ss,emp,frames["diag"]);network=j.network_summary(frames["scores"],ss)
 outputs={"run_summary.csv":frames["run"],"fit_summary.csv":frames["fit"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":gcal,"stabilized_louis_diagnostics.csv":diagnostics_summary(frames["diag"]),"K_spectrum_diagnostics.csv":frames["diag"],"empirical_covariance_comparison.csv":emp,"corrected_group_snr_scores.csv":frames["scores"],"corrected_group_snr_summary.csv":ss,"network_recovery_summary.csv":network,"stability_rescaling_diagnostics.csv":frames["fit"],"stability_rescaling_calibration_summary.csv":stability,"runtime_summary.csv":runtime_summary,"numerical_diagnostics.csv":frames["diag"],"decision_summary.csv":decision}
 for n,d in outputs.items():atomic_csv(d,os.path.join(RESULTS_DIR,n))
 plots(cal,frames["diag"],emp,ss,frames["coeff"],frames["runtime"]);print(f"Experiment 34K complete. Outputs saved under {RESULTS_DIR}/");print("Interpretation guide: select eta/tau by active and zero coverage, standardized errors, empirical trace, capping/clipping stability, and group-SNR preservation; uncertainty and ranking recommendations may differ, and Level-2 VI remains deferred unless stabilization fails.")

if __name__=="__main__":main()
