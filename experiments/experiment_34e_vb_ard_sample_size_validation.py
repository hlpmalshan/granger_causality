"""Experiment 34E: sample-size validation of practical free-B VB-ARD.

This remains the Level-1 Hybrid Kalman + VB-ARD approximation.  Failures in
short, underdetermined records are recorded and checkpointed rather than hidden
or allowed to terminate the experiment.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34E_BLAS_THREADS", "1")

import json
import time
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B as scaleup
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, grouped_stats
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import (
    recovery, safe_curve, simulate, uncertainty_summary, vb_details,
)
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import (
    materialize_tables, network_scores, support_score_rows, vb_score_rows,
)
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import (
    B_metrics, em_diagnostics, fit_em_variant,
)
from experiments.experiment_34d_vb_ard_B_ridge_convergence_calibration import (
    edge_summary, fit_vb, iteration_rows, score_summaries, vb_diagnostics,
)


scaleup.LGC_LAMBDA_GRID = [0.0]
N_SOURCES, na, nb = 20, 2, 3
T_VALUES = [50, 100, 250, 500, 1000, 2000]
N_OUTER_RUNS_BY_T = {50:20,100:20,250:15,500:10,1000:10,2000:5}
METHODS = ["group_lasso_em_free_B", "hybrid_vb_ard_free_B"]
GROUP_LASSO_LAMBDA_GRID = [0.003]
VB_CONFIG = {"name":"max100","max_iter":100}
VB_HYPERPARAM = {"a0":1e-3,"b0":1e-3}
PRACTICAL_TOLERANCE = 1e-3
BASE_SEED = 4500000
SMOKE_TEST = os.environ.get("EXPERIMENT_34E_SMOKE", "0") == "1"
if SMOKE_TEST:
    T_VALUES=[50,250];N_OUTER_RUNS_BY_T={50:1,250:1}
N_OUTER_OVERRIDE=os.environ.get("EXPERIMENT_34E_N_OUTER_RUNS")
if N_OUTER_OVERRIDE is not None:
    N_OUTER_RUNS_BY_T={T:int(N_OUTER_OVERRIDE) for T in T_VALUES}
RESULTS_DIR=os.environ.get("EXPERIMENT_34E_RESULTS_DIR","results/experiment_34e")


PARAMETER_METRICS=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error",
 "A_diagonal_relative_frobenius_error","A_group_norm_pearson_correlation","A_group_norm_spearman_correlation",
 "A_support_ROC_AUC","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03",
 "A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10","precision_at_FPR_0p01","precision_at_FPR_0p05",
 "F1_at_FPR_0p01","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse","runtime_seconds",
 "n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]
B_METRICS=["B_relative_frobenius_error","B_lag0_relative_frobenius_error","B_lag1_relative_frobenius_error",
 "B_lag2_relative_frobenius_error","B_mean_absolute_error","B_pearson_correlation","B_spearman_correlation",
 "B_energy_ratio_hat_to_true","B_change_norm","relative_B_change_norm","initial_B_relative_error","final_B_relative_error"]
GROUPS=["n_sources","T","method","lambda_A_group_fraction","a0","b0"]


def failure_row(meta,runtime,status,error,warning_messages):
    row={**meta,"runtime_seconds":runtime,"run_status":status,"error_type":type(error).__name__,
      "error_message":str(error),"traceback":traceback.format_exc(),"warning_count":len(warning_messages),
      "warning_messages":" | ".join(warning_messages),"failure_rate":1.0,"numerical_warning_flag":True,
      "converged_all":False,"practically_converged":False,"performance_stable_flag":False,"hit_max_iter":False}
    for key in set(PARAMETER_METRICS+B_METRICS+["n_iter","spectral_radius_A","final_kalman_log_likelihood",
      "surrogate_objective","final_A_change_norm","final_B_change_norm","final_alpha_change_norm",
      "final_relative_A_change_norm","final_relative_B_change_norm","final_relative_alpha_change_norm",
      "final_loglike_change","filtered_signal_correlation","smoothed_signal_correlation"]):
        row.setdefault(key,np.nan)
    return row


def practical_fields(diag,row):
    A=diag.get("relative_A_change_norm",np.inf);B=diag.get("relative_B_change_norm",np.inf)
    alpha=diag.get("relative_alpha_change_norm",0.0)
    practical=bool(np.isfinite(A) and np.isfinite(B) and np.isfinite(alpha) and
                   A<PRACTICAL_TOLERANCE and B<PRACTICAL_TOLERANCE and alpha<PRACTICAL_TOLERANCE)
    return {"practically_converged":practical,
      "performance_stable_flag":bool(practical and np.isfinite(row.get("A_support_AUPRC",np.nan))),
      "final_relative_A_change_norm":A,"final_relative_B_change_norm":B,
      "final_relative_alpha_change_norm":alpha,
      "final_loglike_change":diag.get("loglike_relative_change",np.nan)}


def successful_row(model,Ah,Bh,filtered,smoothed,data,meta,runtime,diag,initialization):
    row,support=recovery(Ah,data,filtered,smoothed,meta,runtime,diag["n_iter"],diag["converged_all"])
    row.update(diag);row.update(B_metrics(Bh,data,initialization,diag["final_B_change_norm"]))
    initial=np.asarray(model.B_initial_matrices_)
    row.update({"relative_B_change_norm":diag.get("relative_B_change_norm",np.nan),
      "initial_B_relative_error":np.linalg.norm(initial-np.asarray(data["B"]))/np.linalg.norm(data["B"]),
      "initial_B_energy_ratio_hat_to_true":np.sum(initial**2)/np.sum(np.asarray(data["B"])**2),
      "final_B_relative_error":row["B_relative_frobenius_error"],
      "final_B_energy_ratio_hat_to_true":row["B_energy_ratio_hat_to_true"],
      "precision_at_FPR_0p01":row.get("A_support_precision_at_FPR_0p01",np.nan),
      "precision_at_FPR_0p05":row.get("A_support_precision_at_FPR_0p05",np.nan),
      "F1_at_FPR_0p01":row.get("A_support_F1_at_FPR_0p01",np.nan),
      "F1_at_FPR_0p05":row.get("A_support_F1_at_FPR_0p05",np.nan),
      "run_status":"success","error_type":"","error_message":"","failure_rate":0.0})
    row.update(practical_fields(diag,row))
    return row,support


def sample_size_summary(parameter):
    rows=[]
    for _,row in parameter.iterrows():
        base={"T":row["T"],"method":row["method"],"score_type":"model_A",
          "A_support_AUPRC":row.get("A_support_AUPRC"),"A_support_TPR_at_FPR_0p01":row.get("A_support_TPR_at_FPR_0p01"),
          "A_support_TPR_at_FPR_0p05":row.get("A_support_TPR_at_FPR_0p05"),"precision_at_FPR_0p05":row.get("precision_at_FPR_0p05"),
          "F1_at_FPR_0p05":row.get("F1_at_FPR_0p05"),"VB_group_SNR_AUPRC":np.nan,
          "VB_group_SNR_TPR_at_FPR_0p01":np.nan,"VB_group_SNR_TPR_at_FPR_0p05":np.nan,
          "B_relative_error":row.get("B_relative_frobenius_error"),"runtime_seconds":row.get("runtime_seconds"),
          "practically_converged":row.get("practically_converged",False),"failure_rate":row.get("failure_rate",0.)}
        rows.append(base)
        if row["method"]=="hybrid_vb_ard_free_B":
            extra=base.copy();extra.update({"score_type":"vb_group_snr","VB_group_SNR_AUPRC":row.get("group_snr_AUPRC"),
              "VB_group_SNR_TPR_at_FPR_0p01":row.get("group_snr_TPR_at_FPR_0p01"),
              "VB_group_SNR_TPR_at_FPR_0p05":row.get("group_snr_TPR_at_FPR_0p05")});rows.append(extra)
    frame=pd.DataFrame(rows)
    metrics=["A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p05","precision_at_FPR_0p05",
      "F1_at_FPR_0p05","VB_group_SNR_AUPRC","VB_group_SNR_TPR_at_FPR_0p01","VB_group_SNR_TPR_at_FPR_0p05",
      "B_relative_error","runtime_seconds","practically_converged","failure_rate"]
    return frame.groupby(["T","method","score_type"],dropna=False)[metrics].mean().reset_index().rename(
      columns={"practically_converged":"practical_convergence_rate"})


def all_summaries(frames):
    parameter=grouped_stats(frames["parameter"],GROUPS,PARAMETER_METRICS)
    b=grouped_stats(frames["parameter"],GROUPS,B_METRICS)
    support=grouped_stats(frames["parameter"],GROUPS,["A_support_ROC_AUC","A_support_AUPRC","A_support_best_youden_J",
      "A_support_best_F1","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05",
      "A_support_TPR_at_FPR_0p10","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05"])
    edge=edge_summary(frames["vb_edges"]) if len(frames["vb_edges"]) else pd.DataFrame()
    calibration=uncertainty_summary(frames["uncertainty"]) if len(frames["uncertainty"]) else pd.DataFrame()
    roc,points,fixed=score_summaries(frames["scores"])
    return parameter,b,support,edge,calibration,roc,points,fixed,sample_size_summary(frames["parameter"])


def save_plots(parameter,edge_summary_frame,edges,calibration):
    path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
    success=parameter.loc[parameter.run_status=="success"]
    def line(metric,name,label):
        fig,ax=plt.subplots()
        for method,g in success.groupby("method"):
            s=g.groupby("T")[metric].mean();ax.plot(s.index,s.values,marker="o",label=method)
        ax.set(xlabel="T",ylabel=label,xscale="log");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    items=(("A_support_AUPRC","AUPRC_vs_T.png","A-support AUPRC"),("A_support_TPR_at_FPR_0p01","TPR_FPR01_vs_T.png","TPR at FPR <= 0.01"),
      ("A_support_TPR_at_FPR_0p05","TPR_FPR05_vs_T.png","TPR at FPR <= 0.05"),("precision_at_FPR_0p05","precision_FPR05_vs_T.png","precision at FPR <= 0.05"),
      ("F1_at_FPR_0p05","F1_FPR05_vs_T.png","F1 at FPR <= 0.05"),("A_offdiag_relative_frobenius_error","A_error_vs_T.png","A off-diagonal relative error"),
      ("B_relative_frobenius_error","B_error_vs_T.png","B relative error"),("runtime_seconds","runtime_vs_T.png","seconds"),
      ("practically_converged","practical_convergence_vs_T.png","practical convergence rate"))
    for args in items:line(*args)
    if len(edge_summary_frame):
        snr=edge_summary_frame.loc[edge_summary_frame.vb_score_type=="group_snr_score"]
        fig,ax=plt.subplots();ax.plot(snr["T"],snr.AUPRC,marker="o")
        ax.set(xlabel="T",ylabel="VB group-SNR AUPRC",xscale="log");fig.tight_layout();fig.savefig(os.path.join(path,"VB_group_SNR_AUPRC_vs_T.png"));plt.close(fig)
    if len(calibration):
        selected=calibration.loc[calibration.coefficient_type!="all"]
        fig,ax=plt.subplots()
        for kind,g in selected.groupby("coefficient_type"):
            s=g.groupby("T").empirical_coverage_95.mean();ax.plot(s.index,s.values,marker="o",label=kind)
        ax.set(xlabel="T",ylabel="empirical 95% coverage",xscale="log");ax.legend();fig.tight_layout();fig.savefig(os.path.join(path,"coverage_vs_T.png"));plt.close(fig)
    selected_T=[value for value in (50,250,1000,2000) if value in set(edges["T"])] if len(edges) else []
    if selected_T:
        fig,ax=plt.subplots();values=[];labels=[]
        for T in selected_T:
            for truth,label in ((True,"true"),(False,"false")):
                values.append(edges.loc[(edges["T"]==T)&(edges.true_link==truth),"group_snr_score"]);labels.append(f"{T}\n{label}")
        ax.boxplot(values,labels=labels);ax.set_ylabel("VB group-SNR");fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_true_false_selected_T.png"));plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    config={"experiment":"34E","method":"Level-1 Hybrid Kalman + VB-ARD free B","M":N_SOURCES,"na":na,"nb":nb,
      "T_VALUES":T_VALUES,"N_OUTER_RUNS_BY_T":N_OUTER_RUNS_BY_T,"METHODS":METHODS,
      "GROUP_LASSO_LAMBDA_GRID":GROUP_LASSO_LAMBDA_GRID,"VB_MAX_ITER":VB_CONFIG["max_iter"],"B_ridge_lambda":0.0,
      "VB_HYPERPARAM":VB_HYPERPARAM,"practical_tolerance":PRACTICAL_TOLERANCE,"LGC_LAMBDA_GRID":[0.0],
      "B_initialization_mode":"pinv_proxy_joint_ridge","Q_true":"0.50 I","R_true":"0.60 I","C_MODE":"identity",
      "smoke_test":SMOKE_TEST,"TODO":"structured q_x and posterior spectral GC remain deferred"}
    config=json.loads(json.dumps(config));config_path=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
    if os.path.exists(config_path) and os.path.exists(ledger):
        with open(config_path,encoding="utf-8") as h:old=json.load(h)
        keys=("M","T_VALUES","N_OUTER_RUNS_BY_T","METHODS","GROUP_LASSO_LAMBDA_GRID","VB_MAX_ITER","B_ridge_lambda","VB_HYPERPARAM","practical_tolerance")
        if any(old.get(k)!=config.get(k) for k in keys):raise ValueError("Existing 34E checkpoint has a different numerical configuration.")
    with open(config_path,"w",encoding="utf-8") as h:json.dump(config,h,indent=2)
    checkpoint={"parameter":"parameter_results_partial.csv","support":"_support_checkpoint.csv","vb_edges":"vb_edge_scores_partial.csv",
      "uncertainty":"_uncertainty_checkpoint.csv","objective":"_objective_checkpoint.csv","convergence":"convergence_diagnostics_partial.csv",
      "network":"_network_checkpoint.csv","lgc":"_lgc_checkpoint.csv","scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
    tables={k:[] for k in checkpoint if k!="run"};run_rows=[]
    for key,name in checkpoint.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            try:loaded=pd.read_csv(path)
            except pd.errors.EmptyDataError:loaded=pd.DataFrame()
            if key=="run":run_rows=loaded.to_dict("records")
            elif len(loaded):tables[key]=[loaded]
    completed={(int(r["T"]),int(r["outer_run"])) for r in run_rows}
    for T in T_VALUES:
      for outer in range(N_OUTER_RUNS_BY_T[T]):
       if (T,outer) in completed:continue
       seed=BASE_SEED+T*100+outer;data=simulate(N_SOURCES,T,"sparse_random",seed)
       base={"n_sources":N_SOURCES,"T":T,"outer_run":outer,"case_name":"sparse_random","C_MODE":"identity"}
       baseline={**base,"method":"dataset_baseline","B_MODEL_VARIANT":"free_B","B_ridge_lambda":np.nan,
         "convergence_config_name":"not_applicable","a0":np.nan,"b0":np.nan,"lambda_A_group_fraction":np.nan}
       proxy=data["y"]@np.linalg.pinv(data["C"]).T
       for signal_type,signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
        try:
         network,lgcs,scores=network_scores(signal,signal_type,data,baseline,compute_lgc=signal_type=="observed_y")
         tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
        except Exception as error:
         warnings.warn(f"Baseline metric failure T={T}, run={outer}, signal={signal_type}: {error}")
       specs=[]
       if "group_lasso_em_free_B" in METHODS:specs.extend(("group_lasso_em_free_B",lam) for lam in GROUP_LASSO_LAMBDA_GRID)
       if "hybrid_vb_ard_free_B" in METHODS:specs.append(("hybrid_vb_ard_free_B",np.nan))
       statuses=[]
       for method,lam in specs:
        meta={**base,"method":method,"B_MODEL_VARIANT":"free_B","B_ridge_lambda":0.0,
          "convergence_config_name":"max100" if method.startswith("hybrid") else "em_default",
          "lambda_A_group_fraction":lam,"a0":1e-3 if method.startswith("hybrid") else np.nan,
          "b0":1e-3 if method.startswith("hybrid") else np.nan}
        started=time.perf_counter();warning_messages=[];phase="fit"
        try:
         with warnings.catch_warnings(record=True) as caught:
          warnings.simplefilter("always")
          if method.startswith("hybrid"):
           model,Ah,Bh,filtered,smoothed=fit_vb(data,"free",0.0,VB_CONFIG,seed+500);initialization=model.B_initialization_mode_;diag=vb_diagnostics(model,"free",VB_CONFIG["max_iter"])
          else:
           model,Ah,Bh,filtered,smoothed,initialization=fit_em_variant(data,"free_B",lam,seed+500);diag=em_diagnostics(model)
           diag.update({"converged_A":diag["converged"],"converged_B":diag["converged"],"converged_alpha":np.nan,
             "converged_loglike":diag["converged"],"converged_all":diag["converged"],"hit_max_iter":not diag["converged"],
             "relative_A_change_norm":diag["final_A_change_norm"],"relative_B_change_norm":diag.get("relative_B_change_norm",np.nan),
             "relative_alpha_change_norm":0.0,"loglike_relative_change":diag["final_A_change_norm"]})
          warning_messages=[str(item.message) for item in caught]
         for message in warning_messages:
          warnings.warn(f"Recorded {method} warning at T={T}, run={outer}: {message}")
         phase="metric"
         runtime=time.perf_counter()-started;row,support=successful_row(model,Ah,Bh,filtered,smoothed,data,meta,runtime,diag,initialization)
         row.update({"warning_count":len(warning_messages),"warning_messages":" | ".join(warning_messages),
           "numerical_warning_flag":bool(warning_messages) or diag.get("numerical_warning_flag",False)})
         tables["support"].append(support)
         for signal_type,signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
          network,lgcs,scores=network_scores(signal,signal_type,data,meta,compute_lgc=True);tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
         tables["scores"].append(support_score_rows(support,meta))
         if method.startswith("hybrid"):
          coeff,edges,_=vb_details(model,data,meta);edges["true_A_group_norm"]=edges.true_group_norm
          history=iteration_rows(model,data,meta);tables["objective"].append(history);tables["convergence"].append(history)
          tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["scores"].extend(vb_score_rows(edges,meta))
          snr,_=safe_curve(edges.true_link,edges.group_snr_score);nz=coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean();zero=coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean()
          row.update({"group_snr_AUPRC":snr["AUPRC"],"group_snr_TPR_at_FPR_0p01":snr["TPR_at_FPR_0p01"],
            "group_snr_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"],"offdiag_nonzero_coverage":nz,"offdiag_zero_coverage":zero})
          print({"mean_alpha_true":edges.loc[edges.true_link,"alpha_mean"].mean(),"mean_alpha_false":edges.loc[~edges.true_link,"alpha_mean"].mean(),
            "mean_group_norm_true":edges.loc[edges.true_link,"posterior_mean_group_norm"].mean(),"mean_group_norm_false":edges.loc[~edges.true_link,"posterior_mean_group_norm"].mean(),
            "VB_group_SNR_AUPRC":snr["AUPRC"],"VB_group_SNR_TPR_FPR01":snr["TPR_at_FPR_0p01"],"VB_group_SNR_TPR_FPR05":snr["TPR_at_FPR_0p05"],
            "coverage_nonzero":nz,"coverage_zero":zero,"final_objective":diag["surrogate_objective"],"final_A_change":diag["final_A_change_norm"],
            "final_B_change":diag["final_B_change_norm"],"final_alpha_change":diag["final_alpha_change_norm"]})
         status="success"
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as error:
         runtime=time.perf_counter()-started;status="numerical_error";row=failure_row(meta,runtime,status,error,warning_messages)
        except Exception as error:
         runtime=time.perf_counter()-started
         status="failed_metric" if phase=="metric" else "failed_fit"
         row=failure_row(meta,runtime,status,error,warning_messages)
        tables["parameter"].append(row);tables["runtime"].append({**meta,"runtime_seconds":runtime,"run_status":status,
          "failure_rate":float(status!="success"),"n_iter":row.get("n_iter",np.nan),"converged_all":row.get("converged_all",False),
          "practically_converged":row.get("practically_converged",False),"hit_max_iter":row.get("hit_max_iter",False),
          "spectral_radius_A":row.get("spectral_radius_A",np.nan)});statuses.append(status)
        print({key:row.get(key) for key in ("T","outer_run","method","runtime_seconds","run_status","converged_all","practically_converged","hit_max_iter","n_iter","spectral_radius_A","A_offdiag_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p05","precision_at_FPR_0p05","F1_at_FPR_0p05","B_relative_frobenius_error","B_energy_ratio_hat_to_true","filtered_signal_mse","smoothed_signal_mse")})
       run_rows.append({"T":T,"outer_run":outer,"n_methods":len(statuses),"n_failed":sum(s!="success" for s in statuses),
         "failure_rate":sum(s!="success" for s in statuses)/max(len(statuses),1),"completed":True,"timestamp":time.time()})
       frames=materialize_tables(tables);parameter,b,support,edge,calibration,roc,_,fixed,sample=all_summaries(frames)
       partial={"parameter_results_partial.csv":frames["parameter"],"parameter_summary_partial.csv":parameter,"B_recovery_summary_partial.csv":b,
        "A_support_summary_partial.csv":support,"vb_edge_scores_partial.csv":frames["vb_edges"],"vb_edge_score_summary_partial.csv":edge,
        "vb_uncertainty_calibration_summary_partial.csv":calibration,"fixed_fpr_operating_points_partial.csv":fixed,"roc_summary_partial.csv":roc,
        "convergence_diagnostics_partial.csv":frames["convergence"],"sample_size_summary_partial.csv":sample}
       for name,frame in partial.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
       for key,name in checkpoint.items():
        if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
       atomic_csv(pd.DataFrame(run_rows),ledger)
    frames=materialize_tables(tables);parameter,b,support,edge,calibration,roc,points,fixed,sample=all_summaries(frames)
    decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
    outputs={"run_summary.csv":pd.DataFrame(run_rows),"runtime_summary.csv":grouped_stats(frames["runtime"],GROUPS,["runtime_seconds","n_iter","converged_all","practically_converged","hit_max_iter","spectral_radius_A","failure_rate"]),
      "parameter_results.csv":frames["parameter"],"parameter_summary.csv":parameter,"B_recovery_summary.csv":b,"A_support_summary.csv":support,
      "network_results.csv":frames["network"],"link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","method","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
      "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,
      "lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],"vb_edge_scores.csv":frames["vb_edges"],
      "vb_edge_score_summary.csv":edge,"vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":calibration,
      "objective_history.csv":frames["objective"],"convergence_diagnostics.csv":frames["convergence"],"sample_size_summary.csv":sample}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    save_plots(frames["parameter"],edge,frames["vb_edges"],calibration)
    print("\nInterpretation guide: identify the T where VB first beats group-lasso, where group-SNR retains useful TPR at FPR 0.01/0.05, whether T=50/100 is viable, how B error explains A error, how practical convergence/runtime/coverage scale, and whether a minimum-T recommendation is needed. Moderate-T gains support VB above 250/500; gains only at 1000/2000 imply sample hunger; controlled short-T success is strong evidence; T=50 failures are retained as expected underdetermination. Credible intervals remain diagnostics, not significance tests. Structured q_x and posterior spectral GC remain deferred.")


if __name__=="__main__":main()
