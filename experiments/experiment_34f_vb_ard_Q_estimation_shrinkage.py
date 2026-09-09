"""Experiment 34F: diagonal Q estimation/shrinkage for practical VB-ARD."""

import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]=os.environ.get("EXPERIMENT_34F_BLAS_THREADS","1")
import json,time,traceback,warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B as scaleup
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats,relative
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import recovery,safe_curve,simulate,uncertainty_summary,vb_details
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import materialize_tables,network_scores,support_score_rows,vb_score_rows
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import B_metrics,em_diagnostics,fit_em_variant
from experiments.experiment_34d_vb_ard_B_ridge_convergence_calibration import score_summaries
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls

scaleup.LGC_LAMBDA_GRID=[0.0]
N_SOURCES,na,nb=20,2,3
T_VALUES=[1000,2000];N_OUTER_RUNS_BY_T={1000:10,2000:5}
Q_VARIANTS=["fixed_Q_true","estimate_Q_diag_raw","estimate_Q_diag_floor_rel_0p05",
 "estimate_Q_diag_shrink_scalar_rho_0p10","estimate_Q_diag_shrink_scalar_rho_0p25",
 "estimate_Q_diag_shrink_scalar_rho_0p50"]
VB_MAX_ITER=100;Q_UPDATE_DAMPING=.5;PRACTICAL_TOLERANCE=1e-3
METHODS=["hybrid_vb_ard_free_B","group_lasso_em_free_B_fixed_Q_true"]
BASE_SEED=4600000
SMOKE_TEST=os.environ.get("EXPERIMENT_34F_SMOKE","0")=="1"
if SMOKE_TEST:
 T_VALUES=[1000];N_OUTER_RUNS_BY_T={1000:1};Q_VARIANTS=["fixed_Q_true","estimate_Q_diag_raw","estimate_Q_diag_shrink_scalar_rho_0p25"];METHODS=["hybrid_vb_ard_free_B"]
override=os.environ.get("EXPERIMENT_34F_N_OUTER_RUNS")
if override is not None:N_OUTER_RUNS_BY_T={T:int(override) for T in T_VALUES}
RESULTS_DIR=os.environ.get("EXPERIMENT_34F_RESULTS_DIR","results/experiment_34f")


class ProgressBar:
 def __init__(self,total,width=30):
  self.total=max(int(total),0);self.width=width;self.completed=0
  if self.total==0:print("Experiment 34F: all checkpointed work is already complete.")
 def update(self,label=""):
  self.completed+=1;fraction=min(self.completed/max(self.total,1),1.0)
  filled=int(round(self.width*fraction));bar="#"*filled+"-"*(self.width-filled)
  text=f"\rExperiment 34F [{bar}] {self.completed}/{self.total} ({100*fraction:5.1f}%) {label[:55]}"
  print(text,end="\n" if self.completed>=self.total else "",flush=True)


def score_summaries(scores):
 groups=["n_sources","T","method","Q_VARIANT","signal_type","score_type",
         "lambda_A_group_fraction","a0","b0","lgc_lambda"]
 summaries=[];points=[];fixed=[]
 for keys,g in scores.groupby(groups,dropna=False):
  meta=dict(zip(groups,keys));metrics,curve=safe_curve(g.true_link,g.score)
  summaries.append({**meta,**metrics})
  for threshold,item in curve:points.append({**meta,"threshold":threshold,**item})
  for level in (.01,.03,.05,.10):
   eligible=[p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"]<=level]
   threshold,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
   fixed.append({**meta,"target_fpr_level":level,"threshold":threshold,"actual_fpr":item["fpr"],
                 **{key:item[key] for key in ("tpr","precision","f1","tp","fp","tn","fn")}})
 return pd.DataFrame(summaries),pd.DataFrame(points),pd.DataFrame(fixed)


def variant_options(name):
 if name=="fixed_Q_true":return dict(estimate_Q=False,Q_update_mode="fixed_true",Q_floor_mode="none",Q_floor_value=0.,Q_shrinkage_rho=0.)
 if name=="estimate_Q_diag_raw":return dict(estimate_Q=True,Q_update_mode="diag_raw",Q_floor_mode="none",Q_floor_value=0.,Q_shrinkage_rho=0.)
 if name=="estimate_Q_diag_floor_rel_0p05":return dict(estimate_Q=True,Q_update_mode="diag_floor",Q_floor_mode="relative_to_mean_diag",Q_floor_value=.05,Q_shrinkage_rho=0.)
 rho={"estimate_Q_diag_shrink_scalar_rho_0p10":.10,"estimate_Q_diag_shrink_scalar_rho_0p25":.25,"estimate_Q_diag_shrink_scalar_rho_0p50":.50}[name]
 return dict(estimate_Q=True,Q_update_mode="diag_shrink_scalar",Q_floor_mode="none",Q_floor_value=0.,Q_shrinkage_rho=rho)


def fit_vb(data,variant,seed):
 options=variant_options(variant)
 model=HybridVBARDVARXSSMKnownCWithBQControls(na,nb,data["C"],data["Q"],data["R"],
  B_update_mode="free",B_ridge_lambda=0.,estimate_Q=options["estimate_Q"],Q_update_mode=options["Q_update_mode"],
  Q_floor_mode=options["Q_floor_mode"],Q_floor_value=options["Q_floor_value"],Q_shrinkage_rho=options["Q_shrinkage_rho"],
  Q_update_damping=Q_UPDATE_DAMPING,include_A_posterior_uncertainty_in_Q=True,max_iter=VB_MAX_ITER,
  tol_objective=1e-6,tol_A_change=1e-6,tol_B_change=1e-6,tol_Q_change=1e-6,tol_alpha_change=1e-6,
  a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed).fit(data["y"],data["u"])
 return model,model.get_A_posterior_mean(),np.asarray(model.B_matrices),model.filtered_state_mean_,model.smoothed_state_mean_


def Q_metrics(Qhat,data,model=None):
 true=np.asarray(data["Q"]);hat=np.asarray(Qhat);td=np.diag(true);hd=np.diag(hat)
 energy_true=np.sum(true**2)
 return {"Q_relative_frobenius_error":relative(hat,true),"Q_diag_relative_error":relative(hd,td),
  "Q_mean_absolute_error":np.mean(abs(hat-true)),"Q_diag_mean_absolute_error":np.mean(abs(hd-td)),
  "Q_trace_true":np.trace(true),"Q_trace_hat":np.trace(hat),"Q_trace_ratio_hat_to_true":np.trace(hat)/np.trace(true),
  "Q_energy_true":energy_true,"Q_energy_hat":np.sum(hat**2),"Q_energy_ratio_hat_to_true":np.sum(hat**2)/energy_true,
  "Q_min_diag":np.min(hd),"Q_median_diag":np.median(hd),"Q_mean_diag":np.mean(hd),"Q_max_diag":np.max(hd),
  "Q_diag_std":np.std(hd),"Q_diag_coefficient_of_variation":np.std(hd)/np.mean(hd),
  "Q_condition_number":np.linalg.cond(hat),
  "Q_change_norm":0. if model is None else model.Q_absolute_change_history_[-1],
  "relative_Q_change_norm":0. if model is None else model.Q_change_history_[-1],
  "initial_Q_relative_error":relative(hat if model is None else model.Q_initial_,true),
  "final_Q_relative_error":relative(hat,true),"Q_diag_correlation_with_true":correlations(hd,td)}


def diagnostics(model):
 fixed=not model.estimate_Q
 flags={"converged_A":model.A_change_history_[-1]<1e-6,"converged_B":model.B_change_history_[-1]<1e-6,
  "converged_Q":True if fixed else model.Q_change_history_[-1]<1e-6,
  "converged_alpha":model.alpha_change_history_[-1]<1e-6,"converged_loglike":model.loglike_relative_change_history_[-1]<1e-6}
 flags["converged_all"]=all(flags.values());flags["hit_max_iter"]=model.n_iter_>=VB_MAX_ITER and not flags["converged_all"]
 practical=(model.A_change_history_[-1]<PRACTICAL_TOLERANCE and model.B_change_history_[-1]<PRACTICAL_TOLERANCE and
  (fixed or model.Q_change_history_[-1]<PRACTICAL_TOLERANCE) and model.alpha_change_history_[-1]<PRACTICAL_TOLERANCE)
 return {"n_iter":model.n_iter_,"converged":flags["converged_all"],**flags,"practically_converged":practical,
  "final_kalman_log_likelihood":model.objective_history_[-1],"surrogate_objective":model.objective_history_[-1],
  "final_A_change_norm":model.A_absolute_change_history_[-1],"final_B_change_norm":model.B_absolute_change_history_[-1],
  "final_Q_change_norm":model.Q_absolute_change_history_[-1],"final_alpha_change_norm":model.alpha_absolute_change_history_[-1],
  "final_relative_A_change_norm":model.A_change_history_[-1],"final_relative_B_change_norm":model.B_change_history_[-1],
  "final_relative_Q_change_norm":model.Q_change_history_[-1],"final_relative_alpha_change_norm":model.alpha_change_history_[-1],
  "final_loglike_change":model.loglike_relative_change_history_[-1],"spectral_radius_A":model.spectral_radius_history_[-1],
  "stability_rescaling_flag":bool(any(model.rescaling_history_)),
  "stability_rescaling_occurred":bool(any(model.rescaling_history_)),
  "number_of_stability_rescaling_events":int(sum(model.rescaling_history_)),
  "spectral_radius_before_rescaling_max":max(getattr(model,"spectral_radius_before_rescaling_history_",model.spectral_radius_history_)),
  "spectral_radius_after_rescaling_max":max(getattr(model,"spectral_radius_after_rescaling_history_",model.spectral_radius_history_)),
  "min_rescaling_factor":min(getattr(model,"rescaling_factor_history_",[1.])),
  "numerical_warning_flag":not model.diagnostics_["all_finite"]}


def iteration_rows(model,data,meta):
 rows=[];truth=np.asarray(data["Q"]);off=~np.eye(model.n_states,dtype=bool)
 for i in range(model.n_iter_):
  Q=model.Q_history_[i];candidate=model.Q_candidate_history_[i];alpha=model.alpha_mean_history_[i]
  rows.append({**meta,"iteration":i+1,"kalman_log_likelihood":model.objective_history_[i],"surrogate_objective":model.objective_history_[i],
   "A_change_norm":model.A_absolute_change_history_[i],"relative_A_change_norm":model.A_change_history_[i],
   "B_change_norm":model.B_absolute_change_history_[i],"relative_B_change_norm":model.B_change_history_[i],
   "Q_change_norm":model.Q_absolute_change_history_[i],"relative_Q_change_norm":model.Q_change_history_[i],
   "alpha_change_norm":model.alpha_absolute_change_history_[i],"relative_alpha_change_norm":model.alpha_change_history_[i],
   "loglike_relative_change":model.loglike_relative_change_history_[i],"Q_candidate_diag":json.dumps(np.diag(candidate).tolist()),
   "Q_after_damping_diag":json.dumps(np.diag(Q).tolist()),"Q_relative_error":relative(Q,truth),"Q_trace":np.trace(Q),
   "min_Q_diag":np.min(np.diag(Q)),"median_Q_diag":np.median(np.diag(Q)),"mean_Q_diag":np.mean(np.diag(Q)),
   "max_Q_diag":np.max(np.diag(Q)),"Q_condition_number":np.linalg.cond(Q),"Q_spectral_norm":np.linalg.norm(Q,2),
   "Q_energy_ratio_hat_to_true":np.sum(Q**2)/np.sum(truth**2),"spectral_radius_A":model.spectral_radius_history_[i],
   "rescaled_A_flag":model.rescaling_history_[i],
   "spectral_radius_before_rescaling":getattr(model,"spectral_radius_before_rescaling_history_",model.spectral_radius_history_)[i],
   "spectral_radius_after_rescaling":getattr(model,"spectral_radius_after_rescaling_history_",model.spectral_radius_history_)[i],
   "rescaling_factor":getattr(model,"rescaling_factor_history_",[1.]*model.n_iter_)[i],
   "min_alpha_mean":np.nanmin(alpha[off]),"median_alpha_mean":np.nanmedian(alpha[off]),
   "max_alpha_mean":np.nanmax(alpha[off]),"nan_inf_warning_flag":not np.all(np.isfinite(Q))})
 return pd.DataFrame(rows)


def failure_row(meta,runtime,status,error,messages):
 row={**meta,"runtime_seconds":runtime,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),
  "traceback":traceback.format_exc(),"warning_count":len(messages),"warning_messages":" | ".join(messages),"failure_rate":1.,
  "converged_all":False,"practically_converged":False,"hit_max_iter":False,"numerical_warning_flag":True}
 for key in PARAM_METRICS+B_METRICS+Q_METRICS:row.setdefault(key,np.nan)
 return row


GROUPS=["n_sources","T","method","Q_VARIANT","Q_update_mode","Q_shrinkage_rho","Q_update_damping","lambda_A_group_fraction","a0","b0"]
PARAM_METRICS=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error",
 "A_group_norm_pearson_correlation","A_group_norm_spearman_correlation","A_support_ROC_AUC","A_support_AUPRC",
 "A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10",
 "precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse",
 "runtime_seconds","n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]
B_METRICS=["B_relative_frobenius_error","B_lag0_relative_frobenius_error","B_lag1_relative_frobenius_error","B_lag2_relative_frobenius_error",
 "B_mean_absolute_error","B_pearson_correlation","B_spearman_correlation","B_energy_ratio_hat_to_true","B_change_norm","relative_B_change_norm",
 "initial_B_relative_error","final_B_relative_error"]
Q_METRICS=["Q_relative_frobenius_error","Q_diag_relative_error","Q_mean_absolute_error","Q_diag_mean_absolute_error","Q_trace_ratio_hat_to_true",
 "Q_energy_ratio_hat_to_true","Q_min_diag","Q_median_diag","Q_mean_diag","Q_max_diag","Q_diag_std","Q_diag_coefficient_of_variation",
 "Q_condition_number","Q_change_norm","relative_Q_change_norm","initial_Q_relative_error","final_Q_relative_error"]


def edge_summary(frame):
 rows=[];groups=["n_sources","T","Q_VARIANT","a0","b0"]
 for keys,g in frame.groupby(groups,dropna=False):
  for score in ("posterior_mean_group_norm","posterior_second_moment_group_norm","inverse_alpha_score","group_snr_score"):
   metrics,_=safe_curve(g.true_link,g[score]);rows.append({**dict(zip(groups,keys)),"vb_score_type":score,**metrics})
 return pd.DataFrame(rows)


def variant_summary(parameter):
 columns={"A_support_AUPRC":"mean_A_support_AUPRC","A_support_TPR_at_FPR_0p01":"mean_A_support_TPR_at_FPR_0p01",
  "A_support_TPR_at_FPR_0p05":"mean_A_support_TPR_at_FPR_0p05","precision_at_FPR_0p01":"mean_precision_at_FPR_0p01",
  "precision_at_FPR_0p05":"mean_precision_at_FPR_0p05","F1_at_FPR_0p05":"mean_F1_at_FPR_0p05",
  "group_snr_AUPRC":"mean_VB_group_SNR_AUPRC","group_snr_TPR_at_FPR_0p01":"mean_VB_group_SNR_TPR_at_FPR_0p01",
  "group_snr_TPR_at_FPR_0p05":"mean_VB_group_SNR_TPR_at_FPR_0p05","Q_relative_frobenius_error":"mean_Q_relative_error",
  "Q_trace_ratio_hat_to_true":"mean_Q_trace_ratio_hat_to_true","B_relative_frobenius_error":"mean_B_relative_error",
  "runtime_seconds":"mean_runtime_seconds","practically_converged":"practical_convergence_rate","failure_rate":"failure_rate",
  "offdiag_nonzero_coverage":"mean_offdiag_nonzero_coverage","offdiag_zero_coverage":"mean_offdiag_zero_coverage"}
 return parameter.groupby(["T","Q_VARIANT","Q_update_mode","Q_shrinkage_rho","Q_update_damping"],dropna=False)[list(columns)].mean().reset_index().rename(columns=columns)


def summaries(frames):
 parameter=grouped_stats(frames["parameter"],GROUPS,PARAM_METRICS);b=grouped_stats(frames["parameter"],GROUPS,B_METRICS)
 q=grouped_stats(frames["parameter"],GROUPS,Q_METRICS);support=grouped_stats(frames["parameter"],GROUPS,["A_support_ROC_AUC","A_support_AUPRC",
  "A_support_best_youden_J","A_support_best_F1","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05",
  "A_support_TPR_at_FPR_0p10","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05"])
 edge=edge_summary(frames["vb_edges"]) if len(frames["vb_edges"]) else pd.DataFrame();cal=uncertainty_summary(frames["uncertainty"]) if len(frames["uncertainty"]) else pd.DataFrame()
 roc,points,fixed=score_summaries(frames["scores"]);return parameter,b,q,support,edge,cal,roc,points,fixed,variant_summary(frames["parameter"])


def save_plots(parameter,edge,edges,calibration,qdiag):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True);success=parameter.loc[parameter.run_status=="success"]
 def bars(metric,name,label):
  pivot=success.groupby(["Q_VARIANT","T"])[metric].mean().unstack();fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel(label);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 for args in (("A_support_AUPRC","AUPRC_by_Q_T.png","A-support AUPRC"),("A_support_TPR_at_FPR_0p01","TPR01_by_Q_T.png","TPR at FPR <= .01"),
  ("A_support_TPR_at_FPR_0p05","TPR05_by_Q_T.png","TPR at FPR <= .05"),("precision_at_FPR_0p05","precision05_by_Q_T.png","precision at FPR <= .05"),
  ("A_offdiag_relative_frobenius_error","A_error_by_Q_T.png","A offdiag relative error"),("B_relative_frobenius_error","B_error_by_Q_T.png","B relative error"),
  ("Q_relative_frobenius_error","Q_error_by_Q_T.png","Q relative error"),("Q_trace_ratio_hat_to_true","Q_trace_by_Q_T.png","Q trace ratio"),
  ("runtime_seconds","runtime_by_Q_T.png","seconds"),("practically_converged","practical_convergence_by_Q_T.png","practical convergence rate")):bars(*args)
 if len(edge):
  snr=edge.loc[edge.vb_score_type=="group_snr_score"];pivot=snr.pivot_table(index="Q_VARIANT",columns="T",values="AUPRC")
  fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel("VB group-SNR AUPRC");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_AUPRC_by_Q_T.png"));plt.close(fig)
 if len(qdiag):
  fig,ax=plt.subplots();values=[g.Q_diag_value for _,g in qdiag.groupby("Q_VARIANT")];labels=[k for k,_ in qdiag.groupby("Q_VARIANT")]
  ax.boxplot(values,labels=labels);ax.tick_params(axis="x",rotation=25);ax.set_ylabel("Q diagonal");fig.tight_layout();fig.savefig(os.path.join(path,"Q_diagonal_distribution.png"));plt.close(fig)
 if len(calibration):
  pivot=calibration.loc[calibration.coefficient_type!="all"].pivot_table(index="Q_VARIANT",columns="coefficient_type",values="empirical_coverage_95")
  fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel("empirical 95% coverage");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,"coverage_by_Q_type.png"));plt.close(fig)
 if len(edges):
  selected=[v for v in ("fixed_Q_true","estimate_Q_diag_raw","estimate_Q_diag_floor_rel_0p05","estimate_Q_diag_shrink_scalar_rho_0p25") if v in set(edges.Q_VARIANT)]
  values=[];labels=[]
  for variant in selected:
   for truth,label in ((True,"true"),(False,"false")):values.append(edges.loc[(edges.Q_VARIANT==variant)&(edges.true_link==truth),"group_snr_score"]);labels.append(variant+"\n"+label)
  fig,ax=plt.subplots();ax.boxplot(values,labels=labels);ax.set_ylabel("VB group-SNR");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_true_false_selected_Q.png"));plt.close(fig)


def main():
 os.makedirs(RESULTS_DIR,exist_ok=True)
 config={"experiment":"34F","method":"Level-1 Hybrid Kalman + VB-ARD B+Q","M":N_SOURCES,"T_VALUES":T_VALUES,"N_OUTER_RUNS_BY_T":N_OUTER_RUNS_BY_T,
  "Q_VARIANTS":Q_VARIANTS,"METHODS":METHODS,"VB_MAX_ITER":VB_MAX_ITER,"B_ridge_lambda":0.,"a0":1e-3,"b0":1e-3,
  "Q_UPDATE_DAMPING":Q_UPDATE_DAMPING,"include_A_posterior_uncertainty_in_Q":True,"LGC_LAMBDA_GRID":[0.0],
  "group_lasso_estimated_Q":{"included":False,"reason":"posterior-moment group-lasso class explicitly requires estimate_Q=False"},
  "Q_initialization_mode":"pinv_proxy_transition_residual_diag for estimated Q; fixed_true_Q for control","smoke_test":SMOKE_TEST,
  "TODO":"structured q_x, posterior spectral GC, and C-mixing (34G) remain deferred"}
 config=json.loads(json.dumps(config));cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
 if os.path.exists(cp) and os.path.exists(ledger):
  with open(cp,encoding="utf-8") as h:old=json.load(h)
  keys=("M","T_VALUES","N_OUTER_RUNS_BY_T","Q_VARIANTS","METHODS","VB_MAX_ITER","Q_UPDATE_DAMPING","include_A_posterior_uncertainty_in_Q")
  if any(old.get(k)!=config.get(k) for k in keys):raise ValueError("Existing 34F checkpoint has a different numerical configuration.")
 with open(cp,"w",encoding="utf-8") as h:json.dump(config,h,indent=2)
 checkpoint={"parameter":"parameter_results_partial.csv","support":"_support_checkpoint.csv","vb_edges":"vb_edge_scores_partial.csv","uncertainty":"_uncertainty_checkpoint.csv",
  "objective":"_objective_checkpoint.csv","convergence":"convergence_diagnostics_partial.csv","qdiag":"_qdiag_checkpoint.csv","network":"_network_checkpoint.csv",
  "lgc":"_lgc_checkpoint.csv","scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
 tables={k:[] for k in checkpoint if k!="run"};run_rows=[]
 for key,name in checkpoint.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:loaded=pd.read_csv(path)
   except pd.errors.EmptyDataError:loaded=pd.DataFrame()
   if key=="run":run_rows=loaded.to_dict("records")
   elif len(loaded):tables[key]=[loaded]
 completed={(int(r["T"]),int(r["outer_run"])) for r in run_rows}
 fits_per_dataset=(len(Q_VARIANTS) if "hybrid_vb_ard_free_B" in METHODS else 0)+(
  1 if "group_lasso_em_free_B_fixed_Q_true" in METHODS else 0)
 remaining=sum(fits_per_dataset for T in T_VALUES for outer in range(N_OUTER_RUNS_BY_T[T])
               if (T,outer) not in completed)
 progress=ProgressBar(remaining)
 for T in T_VALUES:
  for outer in range(N_OUTER_RUNS_BY_T[T]):
   if (T,outer) in completed:continue
   seed=BASE_SEED+T*100+outer;data=simulate(N_SOURCES,T,"sparse_random",seed);base={"n_sources":N_SOURCES,"T":T,"outer_run":outer,"case_name":"sparse_random","C_MODE":"identity"}
   baseline={**base,"method":"dataset_baseline","Q_VARIANT":"dataset_baseline","Q_update_mode":"not_applicable","Q_shrinkage_rho":np.nan,"Q_update_damping":np.nan,
    "B_MODEL_VARIANT":"free_B","B_ridge_lambda":np.nan,"convergence_config_name":"not_applicable","lambda_A_group_fraction":np.nan,"a0":np.nan,"b0":np.nan}
   proxy=data["y"]@np.linalg.pinv(data["C"]).T
   for signal_type,signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
    try:
     network,lgcs,scores=network_scores(signal,signal_type,data,baseline,compute_lgc=signal_type=="observed_y");tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
    except Exception as error:warnings.warn(f"34F baseline metric failure T={T}, run={outer}, {signal_type}: {error}")
   specs=[]
   if "hybrid_vb_ard_free_B" in METHODS:specs.extend((f"hybrid_vb_ard_free_B_{v}",v) for v in Q_VARIANTS)
   if "group_lasso_em_free_B_fixed_Q_true" in METHODS:specs.append(("group_lasso_em_free_B_fixed_Q_true","fixed_Q_true_em"))
   statuses=[]
   for method,variant in specs:
    is_em=variant=="fixed_Q_true_em";actual_variant="fixed_Q_true" if is_em else variant;opts=variant_options(actual_variant)
    meta={**base,"method":method,"Q_VARIANT":actual_variant,"Q_update_mode":"fixed_true" if is_em else opts["Q_update_mode"],
     "Q_shrinkage_rho":0. if is_em else opts["Q_shrinkage_rho"],"Q_update_damping":0. if is_em else Q_UPDATE_DAMPING,
     "Q_floor_mode":"none" if is_em else opts["Q_floor_mode"],"Q_floor_value":0. if is_em else opts["Q_floor_value"],
     "B_MODEL_VARIANT":"free_B","B_ridge_lambda":0.,"convergence_config_name":"max100" if not is_em else "em_default",
     "lambda_A_group_fraction":.003 if is_em else np.nan,"a0":np.nan if is_em else 1e-3,"b0":np.nan if is_em else 1e-3}
    started=time.perf_counter();messages=[];phase="fit"
    try:
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      if is_em:
       model,Ah,Bh,filtered,smoothed,initialization=fit_em_variant(data,"free_B",.003,seed+500);diag=em_diagnostics(model)
       diag.update({"converged_all":diag["converged"],"practically_converged":diag["converged"],"hit_max_iter":not diag["converged"],
        "final_Q_change_norm":0.,"final_relative_Q_change_norm":0.,"final_relative_A_change_norm":diag["final_A_change_norm"],
        "final_relative_B_change_norm":diag.get("relative_B_change_norm",np.nan),"final_relative_alpha_change_norm":np.nan,"final_loglike_change":diag["final_A_change_norm"]})
      else:model,Ah,Bh,filtered,smoothed=fit_vb(data,actual_variant,seed+500);initialization=model.B_initialization_mode_;diag=diagnostics(model)
      messages=[str(w.message) for w in caught]
     for message in messages:warnings.warn(f"Recorded 34F warning T={T}, run={outer}, {method}: {message}")
     phase="metric";runtime=time.perf_counter()-started
     row,support=recovery(Ah,data,filtered,smoothed,meta,runtime,diag["n_iter"],diag["converged_all"]);row.update(diag)
     row.update(B_metrics(Bh,data,initialization,diag["final_B_change_norm"]));row.update(
      Q_metrics(model.Q,data,None if is_em else model))
     initialB=np.asarray(model.B_initial_matrices_);row.update({"initial_B_relative_error":relative(initialB,np.asarray(data["B"])),"final_B_relative_error":row["B_relative_frobenius_error"],
      "relative_B_change_norm":diag.get("final_relative_B_change_norm",diag.get("relative_B_change_norm",np.nan)),"precision_at_FPR_0p01":row.get("A_support_precision_at_FPR_0p01",np.nan),
      "precision_at_FPR_0p05":row.get("A_support_precision_at_FPR_0p05",np.nan),"F1_at_FPR_0p01":row.get("A_support_F1_at_FPR_0p01",np.nan),
      "F1_at_FPR_0p05":row.get("A_support_F1_at_FPR_0p05",np.nan),"runtime_seconds":runtime,"run_status":"success","failure_rate":0.,
      "warning_count":len(messages),"warning_messages":" | ".join(messages),"numerical_warning_flag":bool(messages) or diag.get("numerical_warning_flag",False)})
     tables["support"].append(support)
     for signal_type,signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
      network,lgcs,scores=network_scores(signal,signal_type,data,meta,compute_lgc=True);tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
     tables["scores"].append(support_score_rows(support,meta))
     if not is_em:
      coeff,edges,_=vb_details(model,data,meta);edges["true_A_group_norm"]=edges.true_group_norm;history=iteration_rows(model,data,meta)
      tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["objective"].append(history);tables["convergence"].append(history);tables["scores"].extend(vb_score_rows(edges,meta))
      qrows=[]
      for iteration,Q in enumerate(model.Q_history_,1):
       for target,value in enumerate(np.diag(Q)):qrows.append({**meta,"iteration":iteration,"target":target,"Q_diag_value":value})
      tables["qdiag"].append(pd.DataFrame(qrows));snr,_=safe_curve(edges.true_link,edges.group_snr_score)
      nz=coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean();zero=coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean()
      row.update({"group_snr_AUPRC":snr["AUPRC"],"group_snr_TPR_at_FPR_0p01":snr["TPR_at_FPR_0p01"],"group_snr_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"],
       "offdiag_nonzero_coverage":nz,"offdiag_zero_coverage":zero})
     status="success"
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as error:
     runtime=time.perf_counter()-started;status="numerical_error";row=failure_row(meta,runtime,status,error,messages)
    except Exception as error:
     runtime=time.perf_counter()-started;status="failed_metric" if phase=="metric" else "failed_fit";row=failure_row(meta,runtime,status,error,messages)
    tables["parameter"].append(row);tables["runtime"].append({**meta,"runtime_seconds":runtime,"run_status":status,"failure_rate":float(status!="success"),
     "n_iter":row.get("n_iter",np.nan),"converged_all":row.get("converged_all",False),"practically_converged":row.get("practically_converged",False),
     "hit_max_iter":row.get("hit_max_iter",False),"spectral_radius_A":row.get("spectral_radius_A",np.nan)});statuses.append(status)
    progress.update(f"T={T} run={outer+1}/{N_OUTER_RUNS_BY_T[T]} {actual_variant} {status}")
   run_rows.append({"T":T,"outer_run":outer,"n_methods":len(statuses),"n_failed":sum(s!="success" for s in statuses),"failure_rate":sum(s!="success" for s in statuses)/len(statuses),"completed":True,"timestamp":time.time()})
   frames=materialize_tables(tables);parameter,b,q,support,edge,cal,roc,_,fixed,qvariant=summaries(frames)
   partial={"parameter_results_partial.csv":frames["parameter"],"parameter_summary_partial.csv":parameter,"B_recovery_summary_partial.csv":b,"Q_recovery_summary_partial.csv":q,
    "A_support_summary_partial.csv":support,"vb_edge_scores_partial.csv":frames["vb_edges"],"vb_edge_score_summary_partial.csv":edge,"vb_uncertainty_calibration_summary_partial.csv":cal,
    "fixed_fpr_operating_points_partial.csv":fixed,"roc_summary_partial.csv":roc,"convergence_diagnostics_partial.csv":frames["convergence"],"Q_variant_summary_partial.csv":qvariant}
   for name,frame in partial.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
   for key,name in checkpoint.items():
    if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
   atomic_csv(pd.DataFrame(run_rows),ledger)
 frames=materialize_tables(tables);parameter,b,q,support,edge,cal,roc,points,fixed,qvariant=summaries(frames);decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
 outputs={"run_summary.csv":pd.DataFrame(run_rows),"runtime_summary.csv":grouped_stats(frames["runtime"],GROUPS,["runtime_seconds","n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]),
  "parameter_results.csv":frames["parameter"],"parameter_summary.csv":parameter,"B_recovery_summary.csv":b,"Q_recovery_summary.csv":q,"A_support_summary.csv":support,
  "network_results.csv":frames["network"],"link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","method","Q_VARIANT","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
  "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,"lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],
  "vb_edge_scores.csv":frames["vb_edges"],"vb_edge_score_summary.csv":edge,"vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":cal,
  "objective_history.csv":frames["objective"],"convergence_diagnostics.csv":frames["convergence"],"Q_variant_summary.csv":qvariant}
 for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
 save_plots(frames["parameter"],edge,frames["vb_edges"],cal,frames["qdiag"])
 print(f"Experiment 34F complete. Outputs saved under {RESULTS_DIR}/")
 print("\nInterpretation guide: quantify the fixed-to-estimated Q A-support loss; compare raw, floor, and scalar shrinkage for group-SNR and FPR 0.01/0.05; inspect Q/B/latent recovery, practical convergence, and coverage. Raw-Q parity supports diagonal estimation; stabilization recovery supports floor/shrinkage; universal degradation identifies Q estimation as the bottleneck. Select by A-support/FPR rather than Q error alone. Structured q_x, posterior spectral GC, and C-mixing (34G) remain deferred.")

if __name__=="__main__":main()
