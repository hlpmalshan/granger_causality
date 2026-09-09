"""Experiment 34J: Louis missing-information correction for VB covariance of A."""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_name]=os.environ.get("EXPERIMENT_34J_BLAS_THREADS","1")
import json,time,traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats,make_network,relative
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import recovery,safe_curve
from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.vb_ard_varx_ssm import HybridVBARDVARXSSMFixedBKnownC
from src.stats.louis_missing_information import SAMPLER_LABEL,estimate_missing_information,louis_correct_covariances,prior_precision_for_row,sample_companion_trajectories_ffbs
from src.varx.varx_generator import generate_colored_input

na,nb,BURN_IN=2,3,300
CONFIG_LIST=[{"label":"small_controlled_M5","M":5,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":3,"N_REPLICATES_PER_NETWORK":10},{"label":"main_controlled_M20","M":20,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":2,"N_REPLICATES_PER_NETWORK":5}]
ETA_GRID=[.25,.5,.75,1.];N_MC_SMOOTHER_SAMPLES=100;BASE_SEED=5050000
SMOKE_TEST=os.environ.get("EXPERIMENT_34J_SMOKE","0")=="1"
if SMOKE_TEST:CONFIG_LIST=[{"label":"small_controlled_M5","M":5,"T_VALUES":[1000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":2}];ETA_GRID=[.5,1.];N_MC_SMOOTHER_SAMPLES=20
override=os.environ.get("EXPERIMENT_34J_MC_SAMPLES")
if override:N_MC_SMOOTHER_SAMPLES=int(override)
RESULTS_DIR=os.environ.get("EXPERIMENT_34J_RESULTS_DIR","results/experiment_34j")

class ProgressBar:
 def __init__(self,total,width=30):self.total=max(int(total),0);self.width=width;self.done=0;print("Experiment 34J: all checkpointed replicates are complete.") if not self.total else None
 def update(self,label):self.done+=1;f=min(self.done/max(self.total,1),1);n=round(self.width*f);print(f"\rExperiment 34J [{'#'*n}{'-'*(self.width-n)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:62]}",end="\n" if self.done>=self.total else "",flush=True)

def fixed_network(M,seed):
 A,mask=make_network(M,seed);rng=np.random.default_rng(seed+19)
 if M==5 and mask.sum()<3:
  candidates=[(s,t) for s in range(M) for t in range(M) if s!=t and not mask[t,s]]
  for index in rng.choice(len(candidates),3-int(mask.sum()),replace=False):
   s,t=candidates[index];sign=rng.choice([-1.,1.]);A[0][t,s]=sign*rng.uniform(.08,.16);A[1][t,s]=sign*rng.uniform(.03,.09);mask[t,s]=True
  A=stabilize_A_matrices_if_needed(A,target_radius=.82)[0]
 B=[rng.normal(0,scale,(M,1)) for scale in (.45,.25,.15)];return A,B,mask

def simulate(A,B,M,T,seed):
 Q=.5*np.eye(M);R=.6*np.eye(M);C=np.eye(M);u=generate_colored_input(T+BURN_IN,.95,1.,seed+200);result=generate_ssm_varx_p_data(A,B,u,Q,R,C=C,D=None,burn_in=BURN_IN,random_seed=seed+300,return_augmented=True)
 return {"A":A,"B":B,"Q":Q,"R":R,"C":C,"mask":np.any(np.asarray(A)!=0,axis=0)&~np.eye(M,dtype=bool),"x":result["x"],"y":result["y"],"u":result["u"]}

def fit_model(data,seed):
 model=HybridVBARDVARXSSMFixedBKnownC(na,nb,data["C"],data["B"],data["Q"],data["R"],max_iter=100,tol_objective=1e-6,tol_A_change=1e-6,a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed).fit(data["y"],data["u"]);return model

def estimator_labels():return ["current"]+[f"louis_eta_{eta:.2f}".replace(".","p") for eta in ETA_GRID]

def coefficient_rows(model,data,corrected,meta):
 rows=[];true=np.asarray(data["A"]);mean=np.asarray(model.A_mean_matrices_);M=true.shape[1]
 for target in range(M):
  for estimator,cov in corrected[target].items():
   for lag in range(na):
    for source in range(M):
     column=lag*M+source;sd=np.sqrt(max(cov[column,column],0));error=mean[lag,target,source]-true[lag,target,source];kind="diagonal" if target==source else ("offdiag_nonzero" if true[lag,target,source]!=0 else "offdiag_zero")
     rows.append({**meta,"covariance_estimator":estimator,"target_row":target,"lag":lag+1,"source":source,"coefficient_type":kind,"posterior_mean":mean[lag,target,source],"posterior_sd_current":np.sqrt(model.A_row_covariances_[target][column,column]),"posterior_sd_corrected":sd,"posterior_sd":sd,"true_value":true[lag,target,source],"signed_error":error,"abs_error":abs(error),"squared_error":error**2,"standardized_error":error/max(sd,1e-15),"ci95_lower":mean[lag,target,source]-1.96*sd,"ci95_upper":mean[lag,target,source]+1.96*sd,"ci95_contains_true":abs(error)<=1.96*sd})
 return pd.DataFrame(rows)

def group_rows(model,data,corrected,meta):
 rows=[];scores=[];true=np.asarray(data["A"]);mean=np.asarray(model.A_mean_matrices_);M=true.shape[1]
 for target in range(M):
  for source in range(M):
   if target==source:continue
   cols=np.asarray([lag*M+source for lag in range(na)]);truth=true[:,target,source];estimate=mean[:,target,source];kind="true_edge_group" if np.linalg.norm(truth)>0 else "false_edge_group"
   score_row={**meta,"target":target,"source":source,"edge_group_type":kind,"true_link":kind=="true_edge_group","true_group_norm":np.linalg.norm(truth),"posterior_mean_group_norm":np.linalg.norm(estimate)}
   for estimator,cov in corrected[target].items():
    block=cov[np.ix_(cols,cols)];difference=estimate-truth;D2=float(difference@np.linalg.pinv(block)@difference);snr=np.sqrt(max(float(estimate@np.linalg.pinv(block)@estimate),0));rows.append({**meta,"covariance_estimator":estimator,"target":target,"source":source,"edge_group_type":kind,"mahalanobis_D2":D2,"D2_below_chi2_95_df_p":D2<=5.991464547,"D2_below_chi2_99_df_p":D2<=9.210340372,"group_posterior_sd_trace":np.trace(block),"group_error_norm":np.linalg.norm(difference),"group_snr":snr});score_row["group_snr_"+estimator]=snr
   scores.append(score_row)
 return pd.DataFrame(rows),pd.DataFrame(scores)

def calibration_summary(frame):
 groups=["config_label","M","T","covariance_estimator","coefficient_type"];rows=[]
 for keys,g in frame.groupby(groups,dropna=False):rows.append({**dict(zip(groups,keys)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"posterior_sd_abs_error_correlation":correlations(g.posterior_sd,g.abs_error),"n_coefficients":len(g)})
 return pd.DataFrame(rows)

def group_summary(frame):return frame.groupby(["config_label","M","T","covariance_estimator","edge_group_type"],dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2","mean"),median_mahalanobis_D2=("mahalanobis_D2","median"),fraction_D2_below_chi2_95_df_p=("D2_below_chi2_95_df_p","mean"),fraction_D2_below_chi2_99_df_p=("D2_below_chi2_99_df_p","mean"),mean_group_posterior_sd_trace=("group_posterior_sd_trace","mean"),mean_group_error_norm=("group_error_norm","mean"),n_groups=("mahalanobis_D2","size")).reset_index()

def score_summary(scores):
 rows=[]
 for estimator in estimator_labels():
  column="group_snr_"+estimator
  for keys,g in scores.groupby(["config_label","M","T"]):
   metrics,_=safe_curve(g.true_link,g[column]);rows.append({"config_label":keys[0],"M":keys[1],"T":keys[2],"covariance_estimator":estimator,**metrics})
 return pd.DataFrame(rows)

def network_summary(scores,group_snr_summary):
 rows=group_snr_summary.copy();rows["score_family"]="group_snr_"+rows.covariance_estimator.astype(str)
 extra=[]
 for keys,g in scores.groupby(["config_label","M","T"]):
  metrics,_=safe_curve(g.true_link,g.posterior_mean_group_norm);extra.append({"config_label":keys[0],"M":keys[1],"T":keys[2],"covariance_estimator":"not_applicable","score_family":"model_A_group_norm",**metrics})
 return pd.concat([rows,pd.DataFrame(extra)],ignore_index=True)

def empirical_covariance(rowcov):
 rows=[]
 for keys,g in rowcov.groupby(["config_label","true_network_id","M","T","target_row","covariance_estimator"]):
  errors=np.asarray([json.loads(x) for x in g.error_vector_json]);posterior=np.mean(np.asarray([json.loads(x) for x in g.covariance_json]),axis=0);empirical=np.cov(errors,rowvar=False,ddof=1) if len(errors)>1 else np.diag(errors[0]**2);rows.append({"config_label":keys[0],"true_network_id":keys[1],"M":keys[2],"T":keys[3],"target_row":keys[4],"covariance_estimator":keys[5],"trace_posterior_cov":np.trace(posterior),"trace_empirical_cov":np.trace(empirical),"trace_ratio_posterior_to_empirical":np.trace(posterior)/max(np.trace(empirical),1e-12),"diagonal_mean_ratio_posterior_to_empirical":np.mean(np.diag(posterior))/max(np.mean(np.diag(empirical)),1e-12),"frobenius_difference_to_empirical":np.linalg.norm(posterior-empirical),"eigenvalue_spectrum_posterior":json.dumps(np.linalg.eigvalsh(posterior).tolist()),"eigenvalue_spectrum_empirical":json.dumps(np.linalg.eigvalsh(empirical).tolist()),"n_replicates":len(errors)})
 return pd.DataFrame(rows)

def stability_summary(coeff):
 frame=coeff.copy();frame["rescaling_group"]=np.where(frame.stability_rescaling_occurred,"any_rescaling","no_rescaling");return frame.groupby(["config_label","M","T","covariance_estimator","coefficient_type","rescaling_group"]).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),std_standardized_error=("standardized_error","std"),n_fits=("replicate_id","nunique")).reset_index()

def save_plots(cal,diag,emp,scores,stability,runtime,coeff):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 def bars(frame,index,columns,value,name,label):
  p=frame.pivot_table(index=index,columns=columns,values=value);fig,ax=plt.subplots();
  if len(p):p.plot.bar(ax=ax)
  ax.set_ylabel(label);ax.tick_params(axis="x",rotation=15);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 bars(cal,"coefficient_type","covariance_estimator","empirical_coverage_95","coverage.png","coverage");bars(cal,"coefficient_type","covariance_estimator","std_standardized_error","standardized_sd.png","standardized error SD")
 fig,ax=plt.subplots();ax.scatter(cal.mean_posterior_sd,cal.rmse);ax.set(xlabel="mean posterior SD",ylabel="RMSE");fig.tight_layout();fig.savefig(os.path.join(path,"sd_vs_rmse.png"));plt.close(fig)
 keys=["config_label","M","T","true_network_id","replicate_id","target_row","lag","source"];current=coeff.loc[coeff.covariance_estimator=="current",keys+["posterior_sd"]].rename(columns={"posterior_sd":"current_sd"});corrected=coeff.loc[coeff.covariance_estimator!="current",keys+["posterior_sd"]].rename(columns={"posterior_sd":"corrected_sd"}).merge(current,on=keys);fig,ax=plt.subplots();ax.scatter(corrected.current_sd,corrected.corrected_sd,s=3);ax.set(xlabel="current SD",ylabel="corrected SD");fig.tight_layout();fig.savefig(os.path.join(path,"current_corrected_sd.png"));plt.close(fig)
 bars(emp,"covariance_estimator",None,"trace_posterior_cov","covariance_trace.png","posterior trace") if False else None
 for frame,x,y,name in ((diag,"eta","trace_corrected_cov_over_current_cov","trace_ratio_eta.png"),(diag,"target_row","trace_missing_over_complete","missing_fraction.png"),(scores,"covariance_estimator","AUPRC","group_snr_AUPRC.png"),(scores,"covariance_estimator","TPR_at_FPR_0p05","group_snr_TPR05.png"),(stability,"rescaling_group","empirical_coverage_95","coverage_rescaling.png"),(runtime,"N_MC_SMOOTHER_SAMPLES","total_runtime_seconds","runtime_mc.png")):
  fig,ax=plt.subplots();g=frame.groupby(x)[y].mean();ax.plot(g.index.astype(str),g.values,marker="o");ax.set(xlabel=x,ylabel=y);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(diag.min_eigenvalue_I_observed_before_clipping,diag.min_eigenvalue_I_observed_after_clipping,s=5);ax.set(xlabel="min eigenvalue before",ylabel="after");fig.tight_layout();fig.savefig(os.path.join(path,"observed_eigenvalues.png"));plt.close(fig)
 fig,ax=plt.subplots();ax.scatter(emp.trace_empirical_cov,emp.trace_posterior_cov,s=5);ax.set(xlabel="empirical trace",ylabel="posterior trace");fig.tight_layout();fig.savefig(os.path.join(path,"empirical_trace.png"));plt.close(fig)

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config=json.loads(json.dumps({"experiment":"34J","CONFIG_LIST":CONFIG_LIST,"ETA_GRID":ETA_GRID,"N_MC_SMOOTHER_SAMPLES":N_MC_SMOOTHER_SAMPLES,"smoother_sampler":SAMPLER_LABEL,"METHOD":"hybrid_vb_ard_fixed_B_fixed_Q_known_C_known_R","eigen_floor_relative":1e-8,"eigen_floor_absolute":1e-10,"smoke_test":SMOKE_TEST,"TODO":"Level-2 structured VI only if Louis correction fails; spectral GC remains separate"}));cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
 if os.path.exists(cp) and os.path.exists(ledger):
  with open(cp,encoding="utf8") as h:old=json.load(h)
  for key in ("CONFIG_LIST","ETA_GRID","N_MC_SMOOTHER_SAMPLES","METHOD"):
   if old.get(key)!=config.get(key):raise ValueError("Existing 34J checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as h:json.dump(config,h,indent=2)
 names={"coeff":"coefficient_calibration_results_partial.csv","group":"_group_partial.csv","diag":"louis_correction_diagnostics_partial.csv","scores":"_scores_partial.csv","rowcov":"_row_cov_partial.csv","fit":"_fit_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={k:[] for k in names}
 for k,n in names.items():
  p=os.path.join(RESULTS_DIR,n)
  if os.path.exists(p):
   try:d=pd.read_csv(p)
   except pd.errors.EmptyDataError:d=pd.DataFrame()
   if len(d):tables[k]=[d]
 old=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame();completed=set(zip(old.config_label,old["T"].astype(int),old.true_network_id.astype(int),old.replicate_id.astype(int))) if len(old) else set();total=sum((c["label"],T,n,r) not in completed for c in CONFIG_LIST for T in c["T_VALUES"] for n in range(c["N_TRUE_NETWORKS"]) for r in range(c["N_REPLICATES_PER_NETWORK"]));progress=ProgressBar(total)
 for cfg in CONFIG_LIST:
  M=cfg["M"]
  for network_id in range(cfg["N_TRUE_NETWORKS"]):
   network_seed=BASE_SEED+M*100000+network_id*1000;A,B,mask=fixed_network(M,network_seed)
   for T in cfg["T_VALUES"]:
    for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
     key=(cfg["label"],T,network_id,replicate)
     if key in completed:continue
     seed=network_seed+T*10+replicate;meta={"config_label":cfg["label"],"M":M,"T":T,"true_network_id":network_id,"replicate_id":replicate,"N_MC_SMOOTHER_SAMPLES":N_MC_SMOOTHER_SAMPLES,"smoother_sampler":SAMPLER_LABEL};started=time.perf_counter();status="fit"
     try:
      data=simulate(A,B,M,T,seed);fit_start=time.perf_counter();model=fit_model(data,seed+500);fit_runtime=time.perf_counter()-fit_start;sample_start=time.perf_counter();samples=sample_companion_trajectories_ffbs(model.smooth_result_,model.F,N_MC_SMOOTHER_SAMPLES,seed+700);sample_runtime=time.perf_counter()-sample_start;correction_start=time.perf_counter();beta=model._pack_A();priors=[prior_precision_for_row(M,na,i,model.alpha_mean_,model.diagonal_prior_precision) for i in range(M)];missing_rows=estimate_missing_information(samples,data["u"],beta,model.B_matrices,model.Q,na,nb,priors);missing=[x["missing_information"] for x in missing_rows];prior_checks=[]
      for target,item in enumerate(missing_rows):
       shifted=item["scores"]+priors[target]@beta[target];prior_checks.append(np.max(np.abs(np.cov(shifted,rowvar=False,ddof=1)-item["missing_information"])))
      corrected,diag_rows=louis_correct_covariances(model.A_row_covariances_,missing,ETA_GRID);correction_runtime=time.perf_counter()-correction_start
      rescaling=bool(any(model.rescaling_history_));common={**meta,"stability_rescaling_occurred":rescaling,"number_of_stability_rescaling_events":int(sum(model.rescaling_history_)),"spectral_radius_before_rescaling_max":max(getattr(model,"spectral_radius_before_rescaling_history_",[model.diagnostics_["spectral_radius"]])),"spectral_radius_after_rescaling_max":max(getattr(model,"spectral_radius_after_rescaling_history_",[model.diagnostics_["spectral_radius"]])),"min_rescaling_factor":min(getattr(model,"rescaling_factor_history_",[1.])),"mean_rescaling_factor":np.mean(getattr(model,"rescaling_factor_history_",[1.]))};coeff=coefficient_rows(model,data,corrected,common);groups,scores=group_rows(model,data,corrected,common);diagframe=pd.DataFrame([{**common,**row,"prior_constant_covariance_max_abs_difference":prior_checks[row["target_row"]]} for row in diag_rows]);truebeta=np.asarray(data["A"])
      rowcov=[]
      for target in range(M):
       error=np.concatenate([truebeta[k,target] for k in range(na)]);estimate=beta[target]
       for estimator,cov in corrected[target].items():rowcov.append({**common,"target_row":target,"covariance_estimator":estimator,"error_vector_json":json.dumps((estimate-error).tolist()),"covariance_json":json.dumps(cov.tolist())})
      fitrow={**common,"fit_status":"success","n_iter":model.n_iter_,"converged":model.converged_,"A_relative_frobenius_error":relative(np.asarray(model.A_mean_matrices_),truebeta),"A_offdiag_relative_frobenius_error":recovery(np.asarray(model.A_mean_matrices_),data,model.filtered_state_mean_,model.smoothed_state_mean_,meta,fit_runtime,model.n_iter_,model.converged_)[0]["A_offdiag_relative_frobenius_error"]};runtime={**common,"fit_runtime_seconds":fit_runtime,"smoother_sampling_runtime_seconds":sample_runtime,"louis_correction_runtime_seconds":correction_runtime,"total_runtime_seconds":time.perf_counter()-started,"number_of_rows_corrected":M,"average_row_correction_runtime_seconds":correction_runtime/M};tables["coeff"].append(coeff);tables["group"].append(groups);tables["diag"].append(diagframe);tables["scores"].append(scores);tables["rowcov"].append(pd.DataFrame(rowcov));tables["fit"].append(fitrow);tables["runtime"].append(runtime);fit_status="success"
     except Exception as error:
      fit_status="failed_"+status;fitrow={**meta,"fit_status":fit_status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()};tables["fit"].append(fitrow);tables["runtime"].append({**meta,"total_runtime_seconds":time.perf_counter()-started})
     tables["run"].append({**meta,"fit_status":fit_status,"total_runtime_seconds":time.perf_counter()-started});frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()}
     for k,n in names.items():atomic_csv(frames[k],os.path.join(RESULTS_DIR,n))
     if len(frames["coeff"]):
      cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);ss=score_summary(frames["scores"]);emp=empirical_covariance(frames["rowcov"])
      for n,d in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",gcal),("empirical_covariance_comparison_partial.csv",emp),("corrected_group_snr_summary_partial.csv",ss),("network_recovery_summary_partial.csv",network_summary(frames["scores"],ss)),("runtime_summary_partial.csv",grouped_stats(frames["runtime"],["config_label","M","T"],["fit_runtime_seconds","smoother_sampling_runtime_seconds","louis_correction_runtime_seconds","total_runtime_seconds"])),("decision_summary_partial.csv",ss)):atomic_csv(d,os.path.join(RESULTS_DIR,n))
     progress.update(f"{cfg['label']} M={M} T={T} net={network_id+1} rep={replicate+1} {fit_status}")
 frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()};cal=calibration_summary(frames["coeff"]);gcal=group_summary(frames["group"]);ss=score_summary(frames["scores"]);emp=empirical_covariance(frames["rowcov"]);stability=stability_summary(frames["coeff"]);runtime_summary=grouped_stats(frames["runtime"],["config_label","M","T"],["fit_runtime_seconds","smoother_sampling_runtime_seconds","louis_correction_runtime_seconds","total_runtime_seconds","number_of_rows_corrected","average_row_correction_runtime_seconds"])
 outputs={"run_summary.csv":frames["run"],"fit_summary.csv":frames["fit"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":gcal,"louis_correction_diagnostics.csv":frames["diag"],"empirical_covariance_comparison.csv":emp,"corrected_group_snr_scores.csv":frames["scores"],"corrected_group_snr_summary.csv":ss,"network_recovery_summary.csv":network_summary(frames["scores"],ss),"stability_rescaling_diagnostics.csv":frames["fit"],"stability_rescaling_calibration_summary.csv":stability,"runtime_summary.csv":runtime_summary,"numerical_diagnostics.csv":frames["diag"],"decision_summary.csv":ss}
 for n,d in outputs.items():atomic_csv(d,os.path.join(RESULTS_DIR,n))
 save_plots(cal,frames["diag"],emp,ss,stability,frames["runtime"],frames["coeff"]);print(f"Experiment 34J complete. Outputs saved under {RESULTS_DIR}/");print("Interpretation guide: compare active/zero coverage, eta, empirical covariance, standardized errors, shrinkage bias, rescaling, group-SNR preservation, and clipping stability; use corrected covariance for uncertainty only if ranking degrades, and defer Level-2 VI unless correction fails.")

if __name__=="__main__":main()
