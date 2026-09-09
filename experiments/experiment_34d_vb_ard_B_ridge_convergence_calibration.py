"""Experiment 34D: explicit B-ridge and convergence calibration for VB-ARD."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34D_BLAS_THREADS", "1")

import json
import time

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
from src.ssm.vb_ard_varx_ssm_b_controls import HybridVBARDVARXSSMKnownCWithBControls


scaleup.LGC_LAMBDA_GRID = [0.0]
na, nb, N_SOURCES, T_VALUE = 2, 3, 20, 2000
N_OUTER_RUNS = 5
B_RIDGE_LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CONVERGENCE_CONFIGS = [{"name":"max100","max_iter":100},
                       {"name":"max200","max_iter":200}]
METHODS = ["hybrid_vb_ard_free_B", "hybrid_vb_ard_ridge_B",
           "hybrid_vb_ard_fixed_B", "group_lasso_em_free_B"]
VB_HYPERPARAM = {"a0":1e-3,"b0":1e-3}
TOL_A_CHANGE = TOL_B_CHANGE = TOL_ALPHA_CHANGE = TOL_LOGLIKE_CHANGE = 1e-6
BASE_SEED = 4400000
SMOKE_TEST = os.environ.get("EXPERIMENT_34D_SMOKE", "0") == "1"
if SMOKE_TEST:
    N_OUTER_RUNS=1;B_RIDGE_LAMBDA_GRID=[0.0,1e-2]
    CONVERGENCE_CONFIGS=[{"name":"max100","max_iter":100}]
    METHODS=["hybrid_vb_ard_free_B","hybrid_vb_ard_ridge_B"]
N_OUTER_RUNS=int(os.environ.get("EXPERIMENT_34D_N_OUTER_RUNS",N_OUTER_RUNS))
RESULTS_DIR=os.environ.get("EXPERIMENT_34D_RESULTS_DIR","results/experiment_34d")


def fit_vb(data,mode,ridge_lambda,config,seed):
    model=HybridVBARDVARXSSMKnownCWithBControls(na,nb,data["C"],data["Q"],data["R"],
        B_update_mode=mode,fixed_B_matrices=data["B"] if mode=="fixed" else None,
        B_ridge_lambda=ridge_lambda,base_B_ridge=0.0,ridge_B_multiplier=1.0,
        max_iter=config["max_iter"],tol_objective=TOL_LOGLIKE_CHANGE,
        tol_A_change=TOL_A_CHANGE,tol_B_change=TOL_B_CHANGE,
        tol_alpha_change=TOL_ALPHA_CHANGE,a0=1e-3,b0=1e-3,
        diagonal_prior_precision=1e-4,posterior_jitter=1e-8,
        random_state=seed).fit(data["y"],data["u"])
    return model,model.get_A_posterior_mean(),np.asarray(model.B_matrices),\
           model.filtered_state_mean_,model.smoothed_state_mean_


def convergence_flags(model,mode,max_iter):
    A=model.A_change_history_[-1];B=model.B_change_history_[-1]
    alpha=model.alpha_change_history_[-1];ll=model.loglike_relative_change_history_[-1]
    flags={"converged_A":A<TOL_A_CHANGE,
           "converged_B":True if mode=="fixed" else B<TOL_B_CHANGE,
           "converged_alpha":alpha<TOL_ALPHA_CHANGE,
           "converged_loglike":ll<TOL_LOGLIKE_CHANGE}
    flags["converged_all"]=all(flags.values())
    flags["hit_max_iter"]=model.n_iter_>=max_iter and not flags["converged_all"]
    return flags


def vb_diagnostics(model,mode,max_iter):
    flags=convergence_flags(model,mode,max_iter)
    return {"n_iter":model.n_iter_,"converged":flags["converged_all"],**flags,
      "final_kalman_log_likelihood":model.objective_history_[-1],
      "surrogate_objective":model.objective_history_[-1],
      "final_A_change_norm":model.A_absolute_change_history_[-1],
      "final_B_change_norm":model.B_absolute_change_history_[-1],
      "final_alpha_change_norm":model.alpha_absolute_change_history_[-1],
      "relative_A_change_norm":model.A_change_history_[-1],
      "relative_B_change_norm":model.B_change_history_[-1],
      "relative_alpha_change_norm":model.alpha_change_history_[-1],
      "loglike_relative_change":model.loglike_relative_change_history_[-1],
      "spectral_radius_A":model.spectral_radius_history_[-1],
      "stability_rescaling_flag":bool(any(model.rescaling_history_)),
      "numerical_warning_flag":not model.diagnostics_["all_finite"]}


def iteration_rows(model,data,meta):
    rows=[];truth_B=np.asarray(data["B"]);mask=~np.eye(model.n_states,dtype=bool)
    for i in range(model.n_iter_):
        B=np.asarray(model.B_matrices_history_[i]);A=np.asarray(model.A_mean_history_[i]);alpha=model.alpha_mean_history_[i]
        labels=[];scores=[]
        for target in range(model.n_states):
            for source in range(model.n_states):
                if target!=source:
                    labels.append(data["mask"][target,source]);scores.append(np.linalg.norm(A[:,target,source]))
        support,_=safe_curve(labels,scores)
        rows.append({**meta,"iteration":i+1,
          "kalman_log_likelihood":model.objective_history_[i],"surrogate_objective":model.objective_history_[i],
          "A_change_norm":model.A_absolute_change_history_[i],"relative_A_change_norm":model.A_change_history_[i],
          "B_change_norm":model.B_absolute_change_history_[i],"relative_B_change_norm":model.B_change_history_[i],
          "alpha_change_norm":model.alpha_absolute_change_history_[i],"relative_alpha_change_norm":model.alpha_change_history_[i],
          "loglike_absolute_change":model.loglike_absolute_change_history_[i],"loglike_relative_change":model.loglike_relative_change_history_[i],
          "B_relative_error":np.linalg.norm(B-truth_B)/np.linalg.norm(truth_B),
          "B_energy_ratio_hat_to_true":np.sum(B**2)/np.sum(truth_B**2),
          "spectral_radius_A":model.spectral_radius_history_[i],"rescaled_A_flag":model.rescaling_history_[i],
          "min_alpha_mean":np.nanmin(alpha[mask]),"median_alpha_mean":np.nanmedian(alpha[mask]),"max_alpha_mean":np.nanmax(alpha[mask]),
          "mean_alpha_true_edges":np.nanmean(alpha[data["mask"]]),"mean_alpha_false_edges":np.nanmean(alpha[mask & ~data["mask"]]),
          "A_support_AUPRC":support["AUPRC"],
          "nan_inf_warning_flag":not (np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(alpha[mask])))})
    return pd.DataFrame(rows)


def score_summaries(scores):
    groups=["n_sources","T","method","B_MODEL_VARIANT","B_ridge_lambda",
      "convergence_config_name","signal_type","score_type","a0","b0","lgc_lambda"]
    summaries=[];points=[];fixed=[]
    for keys,g in scores.groupby(groups,dropna=False):
        meta=dict(zip(groups,keys));metrics,curve=safe_curve(g.true_link,g.score);summaries.append({**meta,**metrics})
        for threshold,item in curve:points.append({**meta,"threshold":threshold,**item})
        for level in (.01,.03,.05,.10):
            eligible=[p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"]<=level]
            threshold,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
            fixed.append({**meta,"target_fpr_level":level,"threshold":threshold,"actual_fpr":item["fpr"],
              **{key:item[key] for key in ("tpr","precision","f1","tp","fp","tn","fn")}})
    return pd.DataFrame(summaries),pd.DataFrame(points),pd.DataFrame(fixed)


def edge_summary(frame):
    rows=[];groups=["n_sources","T","B_MODEL_VARIANT","B_ridge_lambda","convergence_config_name","a0","b0"]
    for keys,g in frame.groupby(groups,dropna=False):
        for score in ("posterior_mean_group_norm","posterior_second_moment_group_norm","inverse_alpha_score","group_snr_score"):
            metrics,_=safe_curve(g.true_link,g[score]);rows.append({**dict(zip(groups,keys)),"vb_score_type":score,**metrics})
    return pd.DataFrame(rows)


GROUPS=["n_sources","T","method","B_MODEL_VARIANT","B_ridge_lambda","convergence_config_name","a0","b0"]
PARAM_METRICS=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error",
 "A_group_norm_pearson_correlation","A_group_norm_spearman_correlation","A_support_ROC_AUC","A_support_AUPRC",
 "A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10",
 "precision_at_FPR_0p05","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse","runtime_seconds","n_iter",
 "converged_all","hit_max_iter","spectral_radius_A"]
B_METRICS=["B_relative_frobenius_error","B_lag0_relative_frobenius_error","B_lag1_relative_frobenius_error",
 "B_lag2_relative_frobenius_error","B_mean_absolute_error","B_pearson_correlation","B_spearman_correlation",
 "B_energy_ratio_hat_to_true","B_change_norm","relative_B_change_norm","initial_B_relative_error","final_B_relative_error"]


def ridge_summary(parameter):
    vb=parameter.loc[parameter.method.isin(["hybrid_vb_ard_free_B","hybrid_vb_ard_ridge_B"])]
    columns={"A_support_AUPRC":"mean_A_support_AUPRC","A_support_TPR_at_FPR_0p05":"mean_A_support_TPR_at_FPR_0p05",
      "precision_at_FPR_0p05":"mean_precision_at_FPR_0p05","F1_at_FPR_0p05":"mean_F1_at_FPR_0p05",
      "B_relative_frobenius_error":"mean_B_relative_error","B_energy_ratio_hat_to_true":"mean_B_energy_ratio_hat_to_true",
      "runtime_seconds":"mean_runtime_seconds","n_iter":"mean_n_iter","converged_all":"convergence_rate",
      "hit_max_iter":"hit_max_iter_rate","group_snr_AUPRC":"mean_group_snr_AUPRC",
      "group_snr_TPR_at_FPR_0p05":"mean_group_snr_TPR_at_FPR_0p05",
      "offdiag_nonzero_coverage":"mean_offdiag_nonzero_coverage","offdiag_zero_coverage":"mean_offdiag_zero_coverage"}
    return vb.groupby(["B_ridge_lambda","convergence_config_name"],dropna=False)[list(columns)].mean().reset_index().rename(columns=columns)


def all_summaries(frames):
    parameter=grouped_stats(frames["parameter"],GROUPS,PARAM_METRICS)
    b=grouped_stats(frames["parameter"],GROUPS,B_METRICS)
    support=grouped_stats(frames["parameter"],GROUPS,["A_support_ROC_AUC","A_support_AUPRC","A_support_best_youden_J",
      "A_support_best_F1","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05",
      "A_support_TPR_at_FPR_0p10","precision_at_FPR_0p05","F1_at_FPR_0p05"])
    edge=edge_summary(frames["vb_edges"]);calibration=uncertainty_summary(frames["uncertainty"])
    roc,points,fixed=score_summaries(frames["scores"])
    return parameter,b,support,edge,calibration,roc,points,fixed,ridge_summary(frames["parameter"])


def save_plots(parameter,history,edges,calibration):
    path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
    vb=parameter.loc[parameter.method.isin(["hybrid_vb_ard_free_B","hybrid_vb_ard_ridge_B"])]
    def line(metric,name,label):
        fig,ax=plt.subplots()
        for config,g in vb.groupby("convergence_config_name"):
            s=g.groupby("B_ridge_lambda")[metric].mean();ax.plot(s.index,s.values,marker="o",label=config)
        ax.set(xlabel="B ridge lambda",ylabel=label);ax.set_xscale("symlog",linthresh=1e-4);ax.legend();fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    for metric,name,label in (("A_support_AUPRC","AUPRC_vs_B_ridge.png","A-support AUPRC"),
      ("A_support_TPR_at_FPR_0p05","TPR_vs_B_ridge.png","TPR at FPR <= 0.05"),
      ("B_relative_frobenius_error","B_error_vs_B_ridge.png","B relative error"),
      ("B_energy_ratio_hat_to_true","B_energy_vs_B_ridge.png","B energy ratio"),
      ("runtime_seconds","runtime_vs_B_ridge.png","seconds"),("n_iter","iterations_vs_B_ridge.png","iterations")):line(metric,name,label)
    fig,ax=plt.subplots()
    for name in ("relative_A_change_norm","relative_B_change_norm","relative_alpha_change_norm"):
        s=history.groupby("iteration")[name].median();ax.plot(s.index,s.values,label=name)
    ax.set(xlabel="iteration",ylabel="relative change",yscale="log");ax.legend();fig.tight_layout();fig.savefig(os.path.join(path,"changes_vs_iteration.png"));plt.close(fig)
    fig,ax=plt.subplots()
    for _,g in history.groupby(["B_ridge_lambda","convergence_config_name"]):ax.plot(g.iteration,g.kalman_log_likelihood,alpha=.25)
    ax.set(xlabel="iteration",ylabel="Kalman log likelihood");fig.tight_layout();fig.savefig(os.path.join(path,"loglike_vs_iteration.png"));plt.close(fig)
    selected=edges.loc[edges.B_ridge_lambda.isin([0.,1e-3,1e-2,1e-1])]
    fig,ax=plt.subplots();values=[];labels=[]
    for ridge in sorted(selected.B_ridge_lambda.unique()):
        for truth,label in ((True,"true"),(False,"false")):
            values.append(selected.loc[(selected.B_ridge_lambda==ridge)&(selected.true_link==truth),"group_snr_score"]);labels.append(f"{ridge:g}\n{label}")
    ax.boxplot(values,labels=labels);ax.set_ylabel("VB group-SNR");fig.tight_layout();fig.savefig(os.path.join(path,"group_SNR_selected_ridges.png"));plt.close(fig)
    selected=calibration.loc[(calibration.coefficient_type!="all") & calibration.B_ridge_lambda.isin([0.,1e-3,1e-2,1e-1])]
    pivot=selected.groupby(["B_ridge_lambda","coefficient_type"]).empirical_coverage_95.mean().unstack()
    fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel("empirical 95% coverage");ax.tick_params(axis="x",rotation=0);fig.tight_layout();fig.savefig(os.path.join(path,"coverage_selected_ridges.png"));plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    config={"experiment":"34D","method":"Level-1 Hybrid Kalman + VB-ARD","is_full_structured_vi":False,
      "M":N_SOURCES,"T":T_VALUE,"N_OUTER_RUNS":N_OUTER_RUNS,"B_RIDGE_LAMBDA_GRID":B_RIDGE_LAMBDA_GRID,
      "CONVERGENCE_CONFIGS":CONVERGENCE_CONFIGS,"METHODS":METHODS,"VB_HYPERPARAM":VB_HYPERPARAM,
      "tolerances":{"A":TOL_A_CHANGE,"B":TOL_B_CHANGE,"alpha":TOL_ALPHA_CHANGE,"loglike":TOL_LOGLIKE_CHANGE},
      "LGC_LAMBDA_GRID":[0.0],"B_initialization_mode":"pinv_proxy_joint_ridge for estimated B; fixed_true_B for control",
      "smoke_test":SMOKE_TEST,"TODO":"Experiment 35 structured q_x only if justified; spectral GC uncertainty after practical validation"}
    config=json.loads(json.dumps(config));config_path=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
    if os.path.exists(config_path) and os.path.exists(ledger):
        with open(config_path,encoding="utf-8") as h:old=json.load(h)
        keys=("M","T","N_OUTER_RUNS","B_RIDGE_LAMBDA_GRID","CONVERGENCE_CONFIGS","METHODS","VB_HYPERPARAM","tolerances")
        if any(old.get(k)!=config.get(k) for k in keys):raise ValueError("Existing 34D checkpoint has a different numerical configuration.")
    with open(config_path,"w",encoding="utf-8") as h:json.dump(config,h,indent=2)
    checkpoint={"parameter":"parameter_results_partial.csv","support":"_support_checkpoint.csv","vb_edges":"vb_edge_scores_partial.csv",
      "uncertainty":"_uncertainty_checkpoint.csv","objective":"_objective_checkpoint.csv","convergence":"convergence_diagnostics_partial.csv",
      "network":"_network_checkpoint.csv","lgc":"_lgc_checkpoint.csv","scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
    tables={k:[] for k in checkpoint if k!="run"};run_rows=[]
    for key,name in checkpoint.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            loaded=pd.read_csv(path)
            if key=="run":run_rows=loaded.to_dict("records")
            elif len(loaded):tables[key]=[loaded]
    completed={int(r["outer_run"]) for r in run_rows}
    for outer in range(N_OUTER_RUNS):
      if outer in completed:continue
      seed=BASE_SEED+outer*10000;data=simulate(N_SOURCES,T_VALUE,"sparse_random",seed)
      base={"n_sources":N_SOURCES,"T":T_VALUE,"outer_run":outer,"case_name":"sparse_random","C_MODE":"identity"}
      baseline={**base,"method":"dataset_baseline","B_MODEL_VARIANT":"dataset_baseline","B_ridge_lambda":np.nan,
        "B_ridge_multiplier":np.nan,"convergence_config_name":"not_applicable","a0":np.nan,"b0":np.nan}
      proxy=data["y"]@np.linalg.pinv(data["C"]).T
      for signal_type,signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
        network,lgcs,scores=network_scores(signal,signal_type,data,baseline,compute_lgc=signal_type=="observed_y")
        tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
      specs=[]
      if "hybrid_vb_ard_fixed_B" in METHODS:specs.append(("hybrid_vb_ard_fixed_B","fixed",np.nan,CONVERGENCE_CONFIGS[0]))
      for config_item in CONVERGENCE_CONFIGS:
        if "hybrid_vb_ard_free_B" in METHODS:specs.append(("hybrid_vb_ard_free_B","free",0.0,config_item))
        if "hybrid_vb_ard_ridge_B" in METHODS:
            specs.extend(("hybrid_vb_ard_ridge_B","ridge",ridge,config_item) for ridge in B_RIDGE_LAMBDA_GRID if ridge>0)
      if "group_lasso_em_free_B" in METHODS:specs.append(("group_lasso_em_free_B","em",0.0,{"name":"em_default","max_iter":100}))
      for method,mode,ridge,conv in specs:
        variant={"fixed":"fixed_B_true","free":"free_B","ridge":"ridge_B_calibrated","em":"free_B"}[mode]
        meta={**base,"method":method,"B_MODEL_VARIANT":variant,"B_ridge_lambda":ridge,
          "B_ridge_multiplier":np.nan,"convergence_config_name":conv["name"],
          "a0":1e-3 if mode!="em" else np.nan,"b0":1e-3 if mode!="em" else np.nan,"lambda_A_group_fraction":.003 if mode=="em" else np.nan}
        started=time.perf_counter()
        if mode=="em":
            model,Ah,Bh,filtered,smoothed,initialization=fit_em_variant(data,"free_B",.003,seed+500);diag=em_diagnostics(model)
            diag.update({"converged_A":diag["converged"],"converged_B":diag["converged"],"converged_alpha":np.nan,
              "converged_loglike":diag["converged"],"converged_all":diag["converged"],"hit_max_iter":not diag["converged"],
              "relative_A_change_norm":diag["final_A_change_norm"],
              "relative_B_change_norm":diag.get("relative_B_change_norm",np.nan),
              "relative_alpha_change_norm":np.nan})
        else:
            model,Ah,Bh,filtered,smoothed=fit_vb(data,mode,0.0 if mode=="free" else ridge,conv,seed+500);initialization=model.B_initialization_mode_;diag=vb_diagnostics(model,mode,conv["max_iter"])
        runtime=time.perf_counter()-started
        row,support=recovery(Ah,data,filtered,smoothed,meta,runtime,diag["n_iter"],diag["converged_all"]);row.update(diag)
        bm=B_metrics(Bh,data,initialization,diag["final_B_change_norm"]);initial=np.asarray(model.B_initial_matrices_)
        bm.update({"relative_B_change_norm":diag["relative_B_change_norm"],
          "initial_B_relative_error":np.linalg.norm(initial-np.asarray(data["B"]))/np.linalg.norm(data["B"]),
          "initial_B_energy_ratio_hat_to_true":np.sum(initial**2)/np.sum(np.asarray(data["B"])**2),
          "final_B_relative_error":bm["B_relative_frobenius_error"],"final_B_energy_ratio_hat_to_true":bm["B_energy_ratio_hat_to_true"]})
        row.update(bm);row["precision_at_FPR_0p05"]=row.get("A_support_precision_at_FPR_0p05",np.nan);row["F1_at_FPR_0p05"]=row.get("A_support_F1_at_FPR_0p05",np.nan)
        tables["parameter"].append(row);tables["support"].append(support);tables["runtime"].append({**meta,"runtime_seconds":runtime,**diag})
        for signal_type,signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
          network,lgcs,scores=network_scores(signal,signal_type,data,meta,compute_lgc=True);tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
        tables["scores"].append(support_score_rows(support,meta))
        if mode!="em":
          coeff,edges,history=vb_details(model,data,meta);edges["true_A_group_norm"]=edges.true_group_norm
          history=iteration_rows(model,data,meta);tables["objective"].append(history);tables["convergence"].append(history)
          tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["scores"].extend(vb_score_rows(edges,meta))
          snr,_=safe_curve(edges.true_link,edges.group_snr_score);nz=coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean();zero=coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean()
          row.update({"group_snr_AUPRC":snr["AUPRC"],"group_snr_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"],"offdiag_nonzero_coverage":nz,"offdiag_zero_coverage":zero})
          print({"mean_alpha_true":edges.loc[edges.true_link,"alpha_mean"].mean(),"mean_alpha_false":edges.loc[~edges.true_link,"alpha_mean"].mean(),
            "mean_group_norm_true":edges.loc[edges.true_link,"posterior_mean_group_norm"].mean(),"mean_group_norm_false":edges.loc[~edges.true_link,"posterior_mean_group_norm"].mean(),
            "VB_group_SNR_AUPRC":snr["AUPRC"],"VB_group_SNR_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"],"coverage_nonzero":nz,"coverage_zero":zero,
            "final_surrogate_objective":diag["surrogate_objective"],"final_A_change_norm":diag["final_A_change_norm"],"final_B_change_norm":diag["final_B_change_norm"],"final_alpha_change_norm":diag["final_alpha_change_norm"]})
        print({key:row.get(key) for key in ("n_sources","outer_run","method","B_MODEL_VARIANT","B_ridge_lambda","convergence_config_name","runtime_seconds","converged_all","hit_max_iter","n_iter","spectral_radius_A","A_offdiag_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p05","B_relative_frobenius_error","B_energy_ratio_hat_to_true","B_change_norm","relative_B_change_norm","precision_at_FPR_0p05","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse")})
      run_rows.append({"outer_run":outer,"completed":True,"timestamp":time.time()});frames=materialize_tables(tables)
      parameter,b,support,edge,calibration,roc,_,fixed,ridge_summary_frame=all_summaries(frames)
      partial={"parameter_results_partial.csv":frames["parameter"],"parameter_summary_partial.csv":parameter,"B_recovery_summary_partial.csv":b,
       "A_support_summary_partial.csv":support,"vb_edge_scores_partial.csv":frames["vb_edges"],"vb_edge_score_summary_partial.csv":edge,
       "vb_uncertainty_calibration_summary_partial.csv":calibration,"fixed_fpr_operating_points_partial.csv":fixed,"roc_summary_partial.csv":roc,
       "convergence_diagnostics_partial.csv":frames["convergence"],"B_ridge_summary_partial.csv":ridge_summary_frame}
      for name,frame in partial.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
      for key,name in checkpoint.items():
        if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
      atomic_csv(pd.DataFrame(run_rows),ledger)
    frames=materialize_tables(tables);parameter,b,support,edge,calibration,roc,points,fixed,ridge_summary_frame=all_summaries(frames)
    decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
    outputs={"run_summary.csv":pd.DataFrame(run_rows),"runtime_summary.csv":grouped_stats(frames["runtime"],GROUPS,["runtime_seconds","n_iter","converged_all","hit_max_iter","spectral_radius_A"]),
      "parameter_results.csv":frames["parameter"],"parameter_summary.csv":parameter,"B_recovery_summary.csv":b,"A_support_summary.csv":support,
      "network_results.csv":frames["network"],"link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","method","B_MODEL_VARIANT","B_ridge_lambda","convergence_config_name","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
      "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,
      "lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],"vb_edge_scores.csv":frames["vb_edges"],
      "vb_edge_score_summary.csv":edge,"vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":calibration,
      "objective_history.csv":frames["objective"],"convergence_diagnostics.csv":frames["convergence"],"B_ridge_summary.csv":ridge_summary_frame}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    save_plots(frames["parameter"],frames["convergence"],frames["vb_edges"],calibration)
    print("\nInterpretation guide: compare max100/max200 convergence and performance, identify the ridge maximizing AUPRC and low-FPR TPR, inspect B recovery and strong-ridge underfit, relate B error to false A scores, verify group-SNR robustness, choose free/weak/moderate ridge, and treat intervals as uncalibrated unless coverage improves. Moderate-ridge gains support calibrated VB; free-B dominance supports longer runs or revised stopping; convergence-only gains at max200 imply max100/early stopping is adequate; universal instability requires profiling before more science. TODO Experiment 35: structured q_x only if justified. Posterior spectral GC remains deferred.")


if __name__=="__main__":main()
