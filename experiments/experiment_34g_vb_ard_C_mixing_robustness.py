"""Experiment 34G: known non-identity C robustness of practical VB-ARD."""

import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]=os.environ.get("EXPERIMENT_34G_BLAS_THREADS","1")
import json,time,traceback,warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B as scaleup
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats,make_network,relative
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import recovery,safe_curve,uncertainty_summary,vb_details
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import materialize_tables,network_scores,support_score_rows,vb_score_rows
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import B_metrics
from experiments.experiment_34f_vb_ard_Q_estimation_shrinkage import Q_METRICS,Q_metrics,diagnostics,fit_vb,iteration_rows
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.varx.varx_generator import generate_colored_input

scaleup.LGC_LAMBDA_GRID=[0.0]
N_SOURCES,na,nb,BURN_IN=20,2,3,300
C_MODE_LIST=["mild_mixing","strong_mixing"];T_VALUES=[1000,2000]
N_OUTER_RUNS_BY_CONDITION={"mild_mixing":{1000:10,2000:5},"strong_mixing":{1000:10,2000:5}}
METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25"
BASE_SEED=4700000
SMOKE_TEST=os.environ.get("EXPERIMENT_34G_SMOKE","0")=="1"
if SMOKE_TEST:
 T_VALUES=[1000];N_OUTER_RUNS_BY_CONDITION={mode:{1000:1} for mode in C_MODE_LIST}
override=os.environ.get("EXPERIMENT_34G_N_OUTER_RUNS")
if override is not None:N_OUTER_RUNS_BY_CONDITION={mode:{T:int(override) for T in T_VALUES} for mode in C_MODE_LIST}
RESULTS_DIR=os.environ.get("EXPERIMENT_34G_RESULTS_DIR","results/experiment_34g")


class ProgressBar:
 def __init__(self,total,width=30):
  self.total=max(int(total),0);self.width=width;self.completed=0
  if not self.total:print("Experiment 34G: all checkpointed work is already complete.")
 def update(self,label):
  self.completed+=1;fraction=min(self.completed/max(self.total,1),1.);filled=int(round(self.width*fraction))
  print(f"\rExperiment 34G [{'#'*filled}{'-'*(self.width-filled)}] {self.completed}/{self.total} ({100*fraction:5.1f}%) {label[:60]}",
        end="\n" if self.completed>=self.total else "",flush=True)


def make_C(n,mode,seed):
 rng=np.random.default_rng(seed+150);epsilon={"mild_mixing":.15,"strong_mixing":.45}[mode]
 mixing=rng.normal(0.,1./np.sqrt(n),(n,n));np.fill_diagonal(mixing,0.)
 C=np.eye(n)+epsilon*mixing
 # Unit row energy keeps observation scale comparable to identity-C.
 C=C/np.linalg.norm(C,axis=1,keepdims=True)
 singular=np.linalg.svd(C,compute_uv=False)
 if singular[-1]<1e-3:raise np.linalg.LinAlgError("Generated C is too close to singular.")
 return C


def simulate(n,T,mode,seed):
 A,mask=make_network(n,seed);rng=np.random.default_rng(seed+100)
 B=[rng.normal(0,scale,(n,1)) for scale in (.45,.25,.15)];C=make_C(n,mode,seed)
 u=generate_colored_input(T+BURN_IN,.95,1.,seed+200);Q=.5*np.eye(n);R=.6*np.eye(n)
 result=generate_ssm_varx_p_data(A,B,u,Q,R,C=C,D=None,burn_in=BURN_IN,random_seed=seed+300,return_augmented=True)
 return {"A":A,"B":B,"C":C,"Q":Q,"R":R,"mask":mask,"x":result["x"],"y":result["y"],"u":result["u"]}


def C_metrics(data,proxy):
 C=data["C"];singular=np.linalg.svd(C,compute_uv=False);off=C.copy();np.fill_diagonal(off,0.);diag=np.diag(np.diag(C))
 proxy_corr=float(np.nanmean([correlations(proxy[:,i],data["x"][:,i]) for i in range(C.shape[1])]))
 return {"C_condition_number":np.linalg.cond(C),"C_min_singular_value":singular[-1],"C_max_singular_value":singular[0],
  "C_spectral_norm":np.linalg.norm(C,2),"C_frobenius_norm":np.linalg.norm(C),"C_offdiag_frobenius_norm":np.linalg.norm(off),
  "C_diagonal_frobenius_norm":np.linalg.norm(diag),"C_offdiag_energy_ratio":np.sum(off**2)/np.sum(C**2),
  "pinv_C_condition_number":np.linalg.cond(np.linalg.pinv(C)),"pinv_proxy_mse":np.mean((proxy-data["x"])**2),
  "pinv_proxy_correlation":proxy_corr}


def score_summaries(scores):
 groups=["n_sources","T","C_MODE","method","Q_VARIANT","signal_type","score_type","a0","b0","lgc_lambda"]
 summaries=[];points=[];fixed=[]
 for keys,g in scores.groupby(groups,dropna=False):
  meta=dict(zip(groups,keys));metrics,curve=safe_curve(g.true_link,g.score);summaries.append({**meta,**metrics})
  for threshold,item in curve:points.append({**meta,"threshold":threshold,**item})
  for level in (.01,.03,.05,.10):
   eligible=[p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"]<=level]
   threshold,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
   fixed.append({**meta,"target_fpr_level":level,"threshold":threshold,"actual_fpr":item["fpr"],
                 **{k:item[k] for k in ("tpr","precision","f1","tp","fp","tn","fn")}})
 return pd.DataFrame(summaries),pd.DataFrame(points),pd.DataFrame(fixed)


def edge_summary(frame):
 rows=[];groups=["n_sources","T","C_MODE","Q_VARIANT","a0","b0"]
 for keys,g in frame.groupby(groups,dropna=False):
  for score in ("posterior_mean_group_norm","posterior_second_moment_group_norm","inverse_alpha_score","group_snr_score"):
   metrics,_=safe_curve(g.true_link,g[score]);rows.append({**dict(zip(groups,keys)),"vb_score_type":score,**metrics})
 return pd.DataFrame(rows)


GROUPS=["n_sources","T","C_MODE","method","Q_VARIANT","Q_shrinkage_rho","Q_update_damping","a0","b0"]
PARAM_METRICS=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error",
 "A_group_norm_pearson_correlation","A_group_norm_spearman_correlation","A_support_ROC_AUC","A_support_AUPRC",
 "A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10",
 "precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse",
 "runtime_seconds","n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]
B_METRICS=["B_relative_frobenius_error","B_lag0_relative_frobenius_error","B_lag1_relative_frobenius_error","B_lag2_relative_frobenius_error",
 "B_mean_absolute_error","B_pearson_correlation","B_spearman_correlation","B_energy_ratio_hat_to_true","B_change_norm","relative_B_change_norm",
 "initial_B_relative_error","final_B_relative_error"]
C_METRICS=["C_condition_number","C_min_singular_value","C_max_singular_value","C_spectral_norm","C_frobenius_norm",
 "C_offdiag_frobenius_norm","C_diagonal_frobenius_norm","C_offdiag_energy_ratio","pinv_C_condition_number","pinv_proxy_mse","pinv_proxy_correlation"]


def robustness_summary(parameter):
 rows=[]
 for _,r in parameter.iterrows():
  base={"C_MODE":r["C_MODE"],"T":r["T"],"score_type":"model_A","A_support_AUPRC":r.get("A_support_AUPRC"),
   "A_support_TPR_at_FPR_0p01":r.get("A_support_TPR_at_FPR_0p01"),"A_support_TPR_at_FPR_0p05":r.get("A_support_TPR_at_FPR_0p05"),
   "precision_at_FPR_0p01":r.get("precision_at_FPR_0p01"),"precision_at_FPR_0p05":r.get("precision_at_FPR_0p05"),"F1_at_FPR_0p05":r.get("F1_at_FPR_0p05"),
   "VB_group_SNR_AUPRC":np.nan,"VB_group_SNR_TPR_at_FPR_0p01":np.nan,"VB_group_SNR_TPR_at_FPR_0p05":np.nan,
   "B_relative_error":r.get("B_relative_frobenius_error"),"Q_relative_error":r.get("Q_relative_frobenius_error"),
   "smoothed_signal_mse":r.get("smoothed_signal_mse"),"smoothed_signal_correlation":r.get("smoothed_signal_correlation"),
   "pinv_proxy_mse":r.get("pinv_proxy_mse"),"pinv_proxy_correlation":r.get("pinv_proxy_correlation"),"runtime_seconds":r.get("runtime_seconds"),
   "practically_converged":r.get("practically_converged"),"failure_rate":r.get("failure_rate")};rows.append(base)
  extra=base.copy();extra.update({"score_type":"vb_group_snr","VB_group_SNR_AUPRC":r.get("group_snr_AUPRC"),
   "VB_group_SNR_TPR_at_FPR_0p01":r.get("group_snr_TPR_at_FPR_0p01"),"VB_group_SNR_TPR_at_FPR_0p05":r.get("group_snr_TPR_at_FPR_0p05")});rows.append(extra)
 frame=pd.DataFrame(rows);columns={k:"mean_"+k for k in ["A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p05",
  "precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p05","VB_group_SNR_AUPRC","VB_group_SNR_TPR_at_FPR_0p01",
  "VB_group_SNR_TPR_at_FPR_0p05","B_relative_error","Q_relative_error","smoothed_signal_mse","smoothed_signal_correlation",
  "pinv_proxy_mse","pinv_proxy_correlation","runtime_seconds"]};columns.update({"practically_converged":"practical_convergence_rate"})
 return frame.groupby(["C_MODE","T","score_type"],dropna=False)[list(columns)+["failure_rate"]].mean().reset_index().rename(columns=columns)


def summaries(frames):
 p=grouped_stats(frames["parameter"],GROUPS,PARAM_METRICS);b=grouped_stats(frames["parameter"],GROUPS,B_METRICS);q=grouped_stats(frames["parameter"],GROUPS,Q_METRICS)
 support=grouped_stats(frames["parameter"],GROUPS,["A_support_ROC_AUC","A_support_AUPRC","A_support_best_youden_J","A_support_best_F1",
  "A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10",
  "precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05"])
 c=grouped_stats(frames["parameter"],["C_MODE","T"],C_METRICS+["smoothed_signal_mse","smoothed_signal_correlation","filtered_signal_mse","filtered_signal_correlation"])
 latent=grouped_stats(frames["parameter"],["C_MODE","T"],["filtered_signal_mse","smoothed_signal_mse","filtered_signal_correlation","smoothed_signal_correlation","pinv_proxy_mse","pinv_proxy_correlation"])
 edge=edge_summary(frames["vb_edges"]);cal=uncertainty_summary(frames["uncertainty"]);roc,points,fixed=score_summaries(frames["scores"])
 return p,b,q,support,c,latent,edge,cal,roc,points,fixed,robustness_summary(frames["parameter"])


def failure_row(meta,runtime,status,error,messages):
 row={**meta,"runtime_seconds":runtime,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc(),
  "warning_count":len(messages),"warning_messages":" | ".join(messages),"failure_rate":1.,"converged_all":False,"practically_converged":False,"hit_max_iter":False}
 for key in PARAM_METRICS+B_METRICS+Q_METRICS+C_METRICS:row.setdefault(key,np.nan)
 return row


def save_plots(parameter,edge,edges):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True);success=parameter.loc[parameter.run_status=="success"]
 def bars(metric,name,label):
  pivot=success.groupby(["C_MODE","T"])[metric].mean().unstack();fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel(label);ax.tick_params(axis="x",rotation=0);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 for args in (("A_support_AUPRC","AUPRC_by_C_T.png","A-support AUPRC"),("A_support_TPR_at_FPR_0p01","TPR01_by_C_T.png","TPR at FPR <= .01"),
  ("A_support_TPR_at_FPR_0p05","TPR05_by_C_T.png","TPR at FPR <= .05"),("precision_at_FPR_0p05","precision05_by_C_T.png","precision at FPR <= .05"),
  ("A_offdiag_relative_frobenius_error","A_error_by_C_T.png","A offdiag relative error"),("B_relative_frobenius_error","B_error_by_C_T.png","B relative error"),
  ("Q_relative_frobenius_error","Q_error_by_C_T.png","Q relative error"),("runtime_seconds","runtime_by_C_T.png","seconds"),
  ("practically_converged","practical_convergence_by_C_T.png","practical convergence rate")):bars(*args)
 snr=edge.loc[edge.vb_score_type=="group_snr_score"].pivot_table(index="C_MODE",columns="T",values="AUPRC");fig,ax=plt.subplots();snr.plot.bar(ax=ax);ax.set_ylabel("VB group-SNR AUPRC");fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_AUPRC_by_C_T.png"));plt.close(fig)
 def paired(left,right,name,left_label,right_label):
  fig,ax=plt.subplots();other=ax.twinx()
  for mode,g in success.groupby("C_MODE"):
   first=g.groupby("T")[left].mean();second=g.groupby("T")[right].mean()
   ax.plot(first.index,first.values,marker="o",label=mode+" MSE")
   other.plot(second.index,second.values,marker="x",linestyle="--",label=mode+" correlation")
  ax.set(xlabel="T",ylabel=left_label);other.set_ylabel(right_label)
  lines=ax.get_lines()+other.get_lines();ax.legend(lines,[line.get_label() for line in lines],fontsize=7)
  fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 paired("smoothed_signal_mse","smoothed_signal_correlation","latent_recovery_by_C_T.png","smoothed MSE","smoothed correlation")
 paired("pinv_proxy_mse","pinv_proxy_correlation","pinv_proxy_recovery_by_C_T.png","pinv proxy MSE","pinv proxy correlation")
 fig,ax=plt.subplots();ax.scatter(success.C_condition_number,success.A_support_AUPRC);ax.set(xlabel="C condition number",ylabel="A-support AUPRC");fig.tight_layout();fig.savefig(os.path.join(path,"C_condition_vs_AUPRC.png"));plt.close(fig)
 values=[];labels=[]
 for mode in C_MODE_LIST:
  for truth,label in ((True,"true"),(False,"false")):values.append(edges.loc[(edges.C_MODE==mode)&(edges.true_link==truth),"group_snr_score"]);labels.append(mode+"\n"+label)
 fig,ax=plt.subplots();ax.boxplot(values,labels=labels);ax.set_ylabel("VB group-SNR");fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_true_false_by_C.png"));plt.close(fig)


def main():
 os.makedirs(RESULTS_DIR,exist_ok=True)
 config={"experiment":"34G","METHOD":METHOD,"C_MODE_LIST":C_MODE_LIST,"T_VALUES":T_VALUES,"N_OUTER_RUNS_BY_CONDITION":N_OUTER_RUNS_BY_CONDITION,
  "C_generation":{"mild_epsilon":.15,"strong_epsilon":.45,"normalization":"unit row L2"},"identity_reference":"Compare against Experiment 34F; identity is intentionally not rerun",
  "Q_VARIANT":"estimate_Q_diag_shrink_scalar_rho_0p25","Q_shrinkage_rho":.25,"Q_update_damping":.5,"include_A_posterior_uncertainty_in_Q":True,
  "B_ridge_lambda":0.,"VB_MAX_ITER":100,"a0":1e-3,"b0":1e-3,"LGC_LAMBDA_GRID":[0.0],"smoke_test":SMOKE_TEST,
  "TODO":"34H static frequency-domain VARX-GC; structured q_x only if mixing exposes a Level-1 limitation"}
 config=json.loads(json.dumps(config));cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
 if os.path.exists(cp) and os.path.exists(ledger):
  with open(cp,encoding="utf-8") as h:old=json.load(h)
  for key in ("C_MODE_LIST","T_VALUES","N_OUTER_RUNS_BY_CONDITION","METHOD","Q_VARIANT","Q_shrinkage_rho","Q_update_damping"):
   if old.get(key)!=config.get(key):raise ValueError("Existing 34G checkpoint has a different numerical configuration.")
 with open(cp,"w",encoding="utf-8") as h:json.dump(config,h,indent=2)
 checkpoint={"parameter":"parameter_results_partial.csv","support":"_support_checkpoint.csv","vb_edges":"vb_edge_scores_partial.csv","uncertainty":"_uncertainty_checkpoint.csv",
  "objective":"_objective_checkpoint.csv","convergence":"convergence_diagnostics_partial.csv","network":"_network_checkpoint.csv","lgc":"_lgc_checkpoint.csv",
  "scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
 tables={k:[] for k in checkpoint if k!="run"};run_rows=[]
 for key,name in checkpoint.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:loaded=pd.read_csv(path)
   except pd.errors.EmptyDataError:loaded=pd.DataFrame()
   if key=="run":run_rows=loaded.to_dict("records")
   elif len(loaded):tables[key]=[loaded]
 completed={(r["C_MODE"],int(r["T"]),int(r["outer_run"])) for r in run_rows}
 remaining=sum(1 for mode in C_MODE_LIST for T in T_VALUES for outer in range(N_OUTER_RUNS_BY_CONDITION[mode][T]) if (mode,T,outer) not in completed);progress=ProgressBar(remaining)
 for mode in C_MODE_LIST:
  for T in T_VALUES:
   for outer in range(N_OUTER_RUNS_BY_CONDITION[mode][T]):
    if (mode,T,outer) in completed:continue
    seed=BASE_SEED+(0 if mode=="mild_mixing" else 1000000)+T*100+outer;messages=[];started=time.perf_counter();phase="simulate"
    base={"n_sources":N_SOURCES,"T":T,"outer_run":outer,"case_name":mode,"C_MODE":mode,"method":METHOD,"Q_VARIANT":"estimate_Q_diag_shrink_scalar_rho_0p25",
     "Q_update_mode":"diag_shrink_scalar","Q_shrinkage_rho":.25,"Q_update_damping":.5,"a0":1e-3,"b0":1e-3,"lambda_A_group_fraction":np.nan,
     "B_MODEL_VARIANT":"free_B","B_ridge_lambda":0.,"convergence_config_name":"max100"}
    try:
     data=simulate(N_SOURCES,T,mode,seed);proxy=data["y"]@np.linalg.pinv(data["C"]).T;cmetrics=C_metrics(data,proxy)
     baseline={**base,"method":"dataset_baseline"}
     for signal_type,signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
      network,lgcs,scores=network_scores(signal,signal_type,data,baseline,compute_lgc=signal_type=="observed_y");tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
     phase="fit"
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always");model,Ah,Bh,filtered,smoothed=fit_vb(data,"estimate_Q_diag_shrink_scalar_rho_0p25",seed+500);diag=diagnostics(model);messages=[str(w.message) for w in caught]
     for message in messages:warnings.warn(f"Recorded 34G warning {mode}, T={T}, run={outer}: {message}")
     phase="metric";runtime=time.perf_counter()-started;row,support=recovery(Ah,data,filtered,smoothed,base,runtime,diag["n_iter"],diag["converged_all"])
     row.update(diag);row.update(B_metrics(Bh,data,model.B_initialization_mode_,diag["final_B_change_norm"]));row.update(Q_metrics(model.Q,data,model));row.update(cmetrics)
     initialB=np.asarray(model.B_initial_matrices_);row.update({"initial_B_relative_error":relative(initialB,np.asarray(data["B"])),"initial_B_energy_ratio_hat_to_true":np.sum(initialB**2)/np.sum(np.asarray(data["B"])**2),
      "final_B_relative_error":row["B_relative_frobenius_error"],"final_B_energy_ratio_hat_to_true":row["B_energy_ratio_hat_to_true"],"relative_B_change_norm":diag["final_relative_B_change_norm"],
      "precision_at_FPR_0p01":row.get("A_support_precision_at_FPR_0p01",np.nan),"precision_at_FPR_0p05":row.get("A_support_precision_at_FPR_0p05",np.nan),
      "F1_at_FPR_0p01":row.get("A_support_F1_at_FPR_0p01",np.nan),"F1_at_FPR_0p05":row.get("A_support_F1_at_FPR_0p05",np.nan),
      "run_status":"success","failure_rate":0.,"warning_count":len(messages),"warning_messages":" | ".join(messages),"numerical_warning_flag":bool(messages) or diag["numerical_warning_flag"]})
     tables["support"].append(support)
     for signal_type,signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
      network,lgcs,scores=network_scores(signal,signal_type,data,base,compute_lgc=True);tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
     tables["scores"].append(support_score_rows(support,base));coeff,edges,_=vb_details(model,data,base);edges["true_A_group_norm"]=edges.true_group_norm
     history=iteration_rows(model,data,base);tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["objective"].append(history);tables["convergence"].append(history);tables["scores"].extend(vb_score_rows(edges,base))
     snr,_=safe_curve(edges.true_link,edges.group_snr_score);row.update({"group_snr_AUPRC":snr["AUPRC"],"group_snr_TPR_at_FPR_0p01":snr["TPR_at_FPR_0p01"],"group_snr_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"]});status="success"
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as error:
     runtime=time.perf_counter()-started;status="numerical_error";row=failure_row(base,runtime,status,error,messages)
    except Exception as error:
     runtime=time.perf_counter()-started;status="failed_metric" if phase=="metric" else "failed_fit";row=failure_row(base,runtime,status,error,messages)
    tables["parameter"].append(row);tables["runtime"].append({**base,"runtime_seconds":runtime,"run_status":status,"failure_rate":float(status!="success"),"n_iter":row.get("n_iter",np.nan),
     "converged_all":row.get("converged_all",False),"practically_converged":row.get("practically_converged",False),"hit_max_iter":row.get("hit_max_iter",False),"spectral_radius_A":row.get("spectral_radius_A",np.nan)})
    run_rows.append({"C_MODE":mode,"T":T,"outer_run":outer,"run_status":status,"failure_rate":float(status!="success"),"completed":True,"timestamp":time.time()})
    frames=materialize_tables(tables);p,b,q,support,c,latent,edge,cal,roc,_,fixed,robust=summaries(frames)
    partial={"parameter_results_partial.csv":frames["parameter"],"parameter_summary_partial.csv":p,"B_recovery_summary_partial.csv":b,"Q_recovery_summary_partial.csv":q,
     "A_support_summary_partial.csv":support,"C_mixing_summary_partial.csv":c,"latent_recovery_summary_partial.csv":latent,"vb_edge_scores_partial.csv":frames["vb_edges"],
     "vb_edge_score_summary_partial.csv":edge,"vb_uncertainty_calibration_summary_partial.csv":cal,"fixed_fpr_operating_points_partial.csv":fixed,
     "roc_summary_partial.csv":roc,"convergence_diagnostics_partial.csv":frames["convergence"],"C_robustness_summary_partial.csv":robust}
    for name,frame in partial.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    for key,name in checkpoint.items():
     if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
    atomic_csv(pd.DataFrame(run_rows),ledger);progress.update(f"{mode} T={T} run={outer+1}/{N_OUTER_RUNS_BY_CONDITION[mode][T]} {status}")
 frames=materialize_tables(tables);p,b,q,support,c,latent,edge,cal,roc,points,fixed,robust=summaries(frames);decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
 outputs={"run_summary.csv":pd.DataFrame(run_rows),"runtime_summary.csv":grouped_stats(frames["runtime"],GROUPS,["runtime_seconds","n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]),
  "parameter_results.csv":frames["parameter"],"parameter_summary.csv":p,"B_recovery_summary.csv":b,"Q_recovery_summary.csv":q,"A_support_summary.csv":support,"C_mixing_summary.csv":c,
  "latent_recovery_summary.csv":latent,"network_results.csv":frames["network"],"link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","C_MODE","method","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
  "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,"lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],
  "vb_edge_scores.csv":frames["vb_edges"],"vb_edge_score_summary.csv":edge,"vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":cal,
  "objective_history.csv":frames["objective"],"convergence_diagnostics.csv":frames["convergence"],"C_robustness_summary.csv":robust}
 for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
 save_plots(frames["parameter"],edge,frames["vb_edges"]);print(f"Experiment 34G complete. Outputs saved under {RESULTS_DIR}/")
 print("Interpretation guide: compare mild/strong mixing against 34F identity results, group-SNR and low-FPR degradation, VB versus pinv source recovery, C conditioning, B/Q recovery, T=2000 rescue, and readiness for 34H static spectral GC. Credible intervals remain diagnostics only; structured q_x remains deferred unless mixing exposes a Level-1 limitation.")

if __name__=="__main__":main()
