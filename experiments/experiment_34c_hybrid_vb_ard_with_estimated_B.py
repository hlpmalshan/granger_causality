"""Experiment 34C: Level-1 Hybrid VB-ARD A with practical B estimation."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34C_BLAS_THREADS", "1")

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import (
    atomic_csv, correlations, grouped_stats, relative, spearman,
)
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import (
    recovery, safe_curve, simulate, uncertainty_summary, vb_details,
)
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import (
    add_meta, materialize_tables, network_scores, support_score_rows,
    vb_score_rows,
)
from src.ssm.em_varx_p_known_c_group_lasso_b_controls import (
    EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge,
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB,
)
from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
)
from src.ssm.kalman_varx_p import extract_current_latent_state, kalman_smooth_varx_p_companion
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.ssm.vb_ard_varx_ssm_b_controls import HybridVBARDVARXSSMKnownCWithBControls


na, nb = 2, 3
SOURCE_COUNT_VALUES, T_VALUES, N_OUTER_RUNS = [20], [2000], 10
B_MODEL_VARIANTS = ["fixed_B_true", "free_B", "ridge_B_medium"]
METHOD_FAMILIES = ["group_lasso_em", "hybrid_vb_ard"]
LAMBDA_A_GROUP_FRACTION_GRID = [0.003, 0.03]
VB_HYPERPARAM = {"a0": 1e-3, "b0": 1e-3}
BASE_B_RIDGE, RIDGE_B_MULTIPLIER = 1e-6, 10.0
BASE_SEED = 4300000
SMOKE_TEST = os.environ.get("EXPERIMENT_34C_SMOKE", "0") == "1"
if SMOKE_TEST:
    N_OUTER_RUNS = 1
N_OUTER_RUNS = int(os.environ.get("EXPERIMENT_34C_N_OUTER_RUNS", N_OUTER_RUNS))
RESULTS_DIR = os.environ.get("EXPERIMENT_34C_RESULTS_DIR", "results/experiment_34c")


class _TrackBMixin:
    def _initialize_from_proxy(self, y, u):
        super()._initialize_from_proxy(y, u)
        self.B_initial_matrices_ = np.asarray(self.B_matrices).copy()

    def _m_step(self, y, u, smooth_result):
        old = np.asarray(self.B_matrices).copy()
        super()._m_step(y, u, smooth_result)
        new = np.asarray(self.B_matrices)
        self.B_absolute_change_norm = float(np.linalg.norm(new-old))
        self.B_change_norm = float(self.B_absolute_change_norm /
                                   max(np.linalg.norm(old),1e-12))


class _TrackedFreeB(_TrackBMixin,
        EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov):
    pass


class _TrackedRidgeB(_TrackBMixin, EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge):
    pass


class _TrackedFixedB(_TrackBMixin, EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB):
    pass


def em_kwargs(data, fraction, seed):
    return dict(na=na,nb=nb,C=data["C"],D=None,max_iter=100,tol=1e-6,
        ridge_m_step=1e-4,covariance_floor=1e-6,R_init=data["R"],Q_init=data["Q"],
        estimate_Q=False,estimate_R=False,R_floor=.30,zero_constraints=[],
        initial_parameters=None,jitter_scale=0.,random_seed=seed,verbose=False,
        alpha_Q=0.,alpha_R=0.,shrinkage_target_Q="spherical",shrinkage_target_R="spherical",
        lambda_A_group_fraction=fraction,ridge_A_offdiag=1e-4,ridge_A_diag=1e-4,
        group_solver_max_iter=5000,group_solver_tol=1e-7,stabilize_A=True,target_radius=.98)


def fit_em_variant(data, variant, fraction, seed):
    kwargs = em_kwargs(data,fraction,seed)
    if variant == "fixed_B_true":
        model = _TrackedFixedB(fixed_B_matrices=data["B"],ridge_B=BASE_B_RIDGE,**kwargs)
        initialization = "fixed_true_B"
    elif variant == "ridge_B_medium":
        model = _TrackedRidgeB(ridge_B_base=BASE_B_RIDGE,
            ridge_B_multiplier=RIDGE_B_MULTIPLIER,**kwargs)
        initialization = "existing_EM_pinv_proxy_joint_initialization"
    else:
        model = _TrackedFreeB(ridge_B=BASE_B_RIDGE,**kwargs)
        initialization = "existing_EM_pinv_proxy_joint_initialization"
    model.B_change_norm = 0.0
    model.B_absolute_change_norm = 0.0
    model.fit(data["y"],data["u"])
    posterior = kalman_smooth_varx_p_companion(data["y"],data["u"],model.F,model.G,
        model.Q_aug,model.R,model.C_aug,nb,model.D)
    n=data["C"].shape[1]
    return model,np.asarray(model.A_matrices),np.asarray(model.B_matrices),\
        extract_current_latent_state(posterior["filter"]["x_filt"],n),\
        extract_current_latent_state(posterior["smoother"]["x_smooth"],n),initialization


def fit_vb_variant(data, variant, seed):
    mode={"fixed_B_true":"fixed","free_B":"free","ridge_B_medium":"ridge"}[variant]
    model=HybridVBARDVARXSSMKnownCWithBControls(na,nb,data["C"],data["Q"],data["R"],
        B_update_mode=mode,fixed_B_matrices=data["B"] if mode=="fixed" else None,
        base_B_ridge=BASE_B_RIDGE,ridge_B_multiplier=RIDGE_B_MULTIPLIER,
        max_iter=100,tol_objective=1e-6,tol_A_change=1e-6,tol_B_change=1e-6,
        tol_alpha_change=1e-6,a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,
        posterior_jitter=1e-8,random_state=seed).fit(data["y"],data["u"])
    return model,model.get_A_posterior_mean(),np.asarray(model.B_matrices),\
        model.filtered_state_mean_,model.smoothed_state_mean_,model.B_initialization_mode_


def em_diagnostics(model):
    history=np.asarray(model.log_likelihoods,float)
    change=(abs(history[-1]-history[-2])/max(abs(history[-2]),1.0)) if len(history)>1 else np.nan
    return {"n_iter":len(history),"converged":len(history)<model.max_iter,
        "final_kalman_log_likelihood":float(model.smooth_result["log_likelihood"]),
        "surrogate_objective":np.nan,"final_A_change_norm":change,
        "final_B_change_norm":float(model.B_absolute_change_norm),
        "relative_B_change_norm":float(model.B_change_norm),
        "final_alpha_change_norm":np.nan,
        "spectral_radius_A":float(model.spectral_radius()),
        "stability_rescaling_flag":bool(any(np.asarray(getattr(model,"A_sparse_scale_history",[1.]))!=1.)),
        "numerical_warning_flag":not np.all(np.isfinite(history))}


def vb_diagnostics(model):
    return {"n_iter":model.n_iter_,"converged":model.converged_,
        "final_kalman_log_likelihood":model.objective_history_[-1],
        "surrogate_objective":model.objective_history_[-1],
        "final_A_change_norm":model.A_change_history_[-1],
        "final_B_change_norm":model.B_change_history_[-1],
        "final_alpha_change_norm":model.alpha_change_history_[-1],
        "spectral_radius_A":var_companion_spectral_radius(model.A_mean_matrices_),
        "stability_rescaling_flag":bool(any(model.rescaling_history_)),
        "numerical_warning_flag":not model.diagnostics_["all_finite"]}


def B_metrics(Bh,data,initialization,change):
    true,hat=np.asarray(data["B"]),np.asarray(Bh)
    result={"B_relative_frobenius_error":relative(hat,true),
        "B_mean_absolute_error":float(np.mean(abs(hat-true))),
        "B_pearson_correlation":correlations(true.ravel(),hat.ravel()),
        "B_spearman_correlation":spearman(true.ravel(),hat.ravel()),
        "B_energy_true":float(np.sum(true**2)),"B_energy_hat":float(np.sum(hat**2)),
        "B_energy_ratio_hat_to_true":float(np.sum(hat**2)/np.sum(true**2)),
        "B_change_norm":change,"B_initialization_mode":initialization}
    for lag in range(nb):result[f"B_lag{lag}_relative_frobenius_error"]=relative(hat[lag],true[lag])
    return result


def summarize_scores(scores):
    groups=["n_sources","T","method","B_MODEL_VARIANT","signal_type","score_type",
            "lambda_A_group_fraction","a0","b0","lgc_lambda"]
    summaries=[];points=[];fixed=[]
    for keys,g in scores.groupby(groups,dropna=False):
        meta=dict(zip(groups,keys));metrics,curve=safe_curve(g.true_link,g.score)
        summaries.append({**meta,"n_edges":len(g),**metrics})
        for threshold,item in curve:points.append({**meta,"threshold":threshold,**item})
        for level in (.01,.03,.05,.10):
            eligible=[p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"]<=level]
            threshold,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
            fixed.append({**meta,"target_fpr_level":level,"threshold":threshold,"actual_fpr":item["fpr"],
                **{key:item[key] for key in ("tpr","precision","f1","tp","fp","tn","fn")}})
    return pd.DataFrame(summaries),pd.DataFrame(points),pd.DataFrame(fixed)


def vb_edge_summary(frame):
    rows=[];groups=["n_sources","T","B_MODEL_VARIANT","a0","b0"]
    for keys,g in frame.groupby(groups,dropna=False):
        for score in ("posterior_mean_group_norm","posterior_second_moment_group_norm","inverse_alpha_score","group_snr_score"):
            metrics,_=safe_curve(g.true_link,g[score]);rows.append({**dict(zip(groups,keys)),"vb_score_type":score,**metrics})
    return pd.DataFrame(rows)


GROUPS=["n_sources","T","method","B_MODEL_VARIANT","lambda_A_group_fraction","a0","b0"]
PARAMETER_METRICS=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error",
 "A_diagonal_relative_frobenius_error","A_group_norm_pearson_correlation","A_group_norm_spearman_correlation",
 "A_support_ROC_AUC","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03",
 "A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10","precision_at_FPR_0p05","F1_at_FPR_0p05",
 "filtered_signal_mse","smoothed_signal_mse","runtime_seconds","n_iter","converged","spectral_radius_A"]
B_METRICS=["B_relative_frobenius_error","B_lag0_relative_frobenius_error",
 "B_lag1_relative_frobenius_error","B_lag2_relative_frobenius_error","B_mean_absolute_error",
 "B_pearson_correlation","B_spearman_correlation","B_energy_ratio_hat_to_true","B_change_norm"]


def summaries(frames):
    parameter=grouped_stats(frames["parameter"],GROUPS,PARAMETER_METRICS)
    brecovery=grouped_stats(frames["parameter"],GROUPS,B_METRICS)
    support=grouped_stats(frames["parameter"],GROUPS,["A_support_ROC_AUC","A_support_AUPRC",
        "A_support_best_youden_J","A_support_best_F1","A_support_TPR_at_FPR_0p01",
        "A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10",
        "precision_at_FPR_0p05","F1_at_FPR_0p05"])
    edge=vb_edge_summary(frames["vb_edges"]);calibration=uncertainty_summary(frames["uncertainty"])
    roc,points,fixed=summarize_scores(frames["scores"])
    return parameter,brecovery,support,edge,calibration,roc,points,fixed


def save_plots(parameter,edges,edge_summary,calibration,runtime):
    path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
    def bars(metric,name,label):
        pivot=parameter.groupby(["B_MODEL_VARIANT","method"])[metric].mean().unstack()
        fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel(label);ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    bars("A_support_AUPRC","A_support_AUPRC_by_method_B.png","A-support AUPRC")
    bars("A_support_TPR_at_FPR_0p05","TPR_FPR_0p05_by_method_B.png","TPR at FPR <= 0.05")
    bars("A_offdiag_relative_frobenius_error","A_offdiag_error_by_method_B.png","off-diagonal relative error")
    bars("B_relative_frobenius_error","B_relative_error_by_method_B.png","B relative error")
    bars("B_energy_ratio_hat_to_true","B_energy_ratio_by_method_B.png","B energy ratio")
    bars("runtime_seconds","runtime_by_method_B.png","seconds")
    if len(edges):
        variants=list(edges.B_MODEL_VARIANT.unique());values=[];labels=[]
        for variant in variants:
            for truth,label in ((True,"true"),(False,"false")):
                values.append(edges.loc[(edges.B_MODEL_VARIANT==variant)&(edges.true_link==truth),"group_snr_score"]);labels.append(variant+"\n"+label)
        fig,ax=plt.subplots();ax.boxplot(values,labels=labels);ax.set_ylabel("VB group-SNR");ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(os.path.join(path,"VB_group_SNR_true_false_by_B.png"));plt.close(fig)
    if len(calibration):
        selected=calibration.loc[calibration.coefficient_type!="all"]
        pivot=selected.groupby(["B_MODEL_VARIANT","coefficient_type"]).empirical_coverage_95.mean().unstack()
        fig,ax=plt.subplots();pivot.plot.bar(ax=ax);ax.set_ylabel("empirical 95% coverage");ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(os.path.join(path,"calibration_by_type_B.png"));plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    config={"experiment":"34C","method":"Level-1 Hybrid Kalman + VB-ARD","is_full_structured_vi":False,
      "SOURCE_COUNT_VALUES":SOURCE_COUNT_VALUES,"T_VALUES":T_VALUES,"N_OUTER_RUNS":N_OUTER_RUNS,
      "B_MODEL_VARIANTS":B_MODEL_VARIANTS,"METHOD_FAMILIES":METHOD_FAMILIES,
      "LAMBDA_A_GROUP_FRACTION_GRID":LAMBDA_A_GROUP_FRACTION_GRID,"VB_HYPERPARAM":VB_HYPERPARAM,
      "base_B_ridge":BASE_B_RIDGE,"ridge_B_multiplier":RIDGE_B_MULTIPLIER,
      "B_initialization_mode":{"fixed_B_true":"fixed_true_B","free_B":"pinv_proxy_joint_ridge",
      "ridge_B_medium":"pinv_proxy_joint_ridge"},"na":na,"nb":nb,"Q_true":"0.50 I","R_true":"0.60 I",
      "C_MODE":"identity","smoke_test":SMOKE_TEST,
      "TODO_34D":"full structured q_x only if justified; posterior spectral GC only after practical A-B validation"}
    config=json.loads(json.dumps(config));config_path=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
    if os.path.exists(config_path) and os.path.exists(ledger):
        with open(config_path,encoding="utf-8") as handle:old=json.load(handle)
        keys=("SOURCE_COUNT_VALUES","T_VALUES","N_OUTER_RUNS","B_MODEL_VARIANTS","METHOD_FAMILIES","LAMBDA_A_GROUP_FRACTION_GRID","VB_HYPERPARAM","base_B_ridge","ridge_B_multiplier")
        if any(old.get(k)!=config.get(k) for k in keys):raise ValueError("Existing 34C checkpoint has a different numerical configuration.")
    with open(config_path,"w",encoding="utf-8") as handle:json.dump(config,handle,indent=2)
    checkpoint={"parameter":"parameter_results_partial.csv","support":"_A_support_checkpoint.csv",
      "vb_edges":"vb_edge_scores_partial.csv","uncertainty":"_uncertainty_checkpoint.csv",
      "objective":"_objective_checkpoint.csv","network":"_network_checkpoint.csv","lgc":"_lgc_checkpoint.csv",
      "scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
    tables={k:[] for k in checkpoint if k!="run"};run_rows=[]
    for key,name in checkpoint.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            loaded=pd.read_csv(path)
            if key=="run":run_rows=loaded.to_dict("records")
            elif len(loaded):tables[key]=[loaded]
    completed={int(row["outer_run"]) for row in run_rows}
    for outer in range(N_OUTER_RUNS):
      if outer in completed:continue
      for n in SOURCE_COUNT_VALUES:
       for T in T_VALUES:
        seed=BASE_SEED+outer*10000+n*100+T;data=simulate(n,T,"sparse_random",seed)
        base={"n_sources":n,"T":T,"outer_run":outer,"case_name":"sparse_random","C_MODE":"identity"}
        baseline={**base,"method":"dataset_baseline","B_MODEL_VARIANT":"dataset_baseline","lambda_A_group_fraction":np.nan,"a0":np.nan,"b0":np.nan}
        proxy=data["y"]@np.linalg.pinv(data["C"]).T
        for signal_type,signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
            network,lgcs,scores=network_scores(signal,signal_type,data,baseline,compute_lgc=signal_type=="observed_y")
            tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
        specs=[]
        method_suffix={"fixed_B_true":"fixed_B","free_B":"free_B",
                       "ridge_B_medium":"ridge_B_medium"}
        if "group_lasso_em" in METHOD_FAMILIES:
            specs.extend((f"group_lasso_em_{method_suffix[variant]}",variant,{"lambda_A_group_fraction":lam}) for variant in B_MODEL_VARIANTS for lam in LAMBDA_A_GROUP_FRACTION_GRID)
        if "hybrid_vb_ard" in METHOD_FAMILIES:
            specs.extend((f"hybrid_vb_ard_{method_suffix[variant]}",variant,VB_HYPERPARAM) for variant in B_MODEL_VARIANTS)
        for method,variant,hyper in specs:
            meta={**base,"method":method,"B_MODEL_VARIANT":variant,
              "lambda_A_group_fraction":hyper.get("lambda_A_group_fraction",np.nan),"a0":hyper.get("a0",np.nan),"b0":hyper.get("b0",np.nan)}
            started=time.perf_counter()
            if method.startswith("hybrid"):
                model,Ah,Bh,filtered,smoothed,initialization=fit_vb_variant(data,variant,seed+500);diag=vb_diagnostics(model)
            else:
                model,Ah,Bh,filtered,smoothed,initialization=fit_em_variant(data,variant,hyper["lambda_A_group_fraction"],seed+500);diag=em_diagnostics(model)
            runtime=time.perf_counter()-started
            row,support=recovery(Ah,data,filtered,smoothed,meta,runtime,diag["n_iter"],diag["converged"])
            row.update(diag);row.update(B_metrics(Bh,data,initialization,diag["final_B_change_norm"]))
            row["precision_at_FPR_0p05"]=row.get("A_support_precision_at_FPR_0p05",np.nan);row["F1_at_FPR_0p05"]=row.get("A_support_F1_at_FPR_0p05",np.nan)
            tables["parameter"].append(row);tables["support"].append(support);tables["runtime"].append({**meta,"runtime_seconds":runtime,**diag})
            for signal_type,signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
                network,lgcs,scores=network_scores(signal,signal_type,data,meta,compute_lgc=True);tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
            tables["scores"].append(support_score_rows(support,meta))
            if method.startswith("hybrid"):
                coeff,edges,history=vb_details(model,data,meta);edges["true_A_group_norm"]=edges.true_group_norm
                history["B_change_norm"]=model.B_change_history_[:len(history)]
                tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["objective"].append(history);tables["scores"].extend(vb_score_rows(edges,meta))
                snr,_=safe_curve(edges.true_link,edges.group_snr_score)
                print({"mean_alpha_true":edges.loc[edges.true_link,"alpha_mean"].mean(),"mean_alpha_false":edges.loc[~edges.true_link,"alpha_mean"].mean(),
                  "mean_group_norm_true":edges.loc[edges.true_link,"posterior_mean_group_norm"].mean(),"mean_group_norm_false":edges.loc[~edges.true_link,"posterior_mean_group_norm"].mean(),
                  "VB_group_SNR_AUPRC":snr["AUPRC"],"VB_group_SNR_TPR_at_FPR_0p05":snr["TPR_at_FPR_0p05"],
                  "coverage_offdiag_nonzero":coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean(),
                  "coverage_offdiag_zero":coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean(),
                  "final_surrogate_objective":diag["surrogate_objective"],"final_B_change_norm":diag["final_B_change_norm"]})
            print({key:row.get(key) for key in ("n_sources","outer_run","method","B_MODEL_VARIANT","lambda_A_group_fraction","a0","b0","runtime_seconds","converged","n_iter","spectral_radius_A","A_offdiag_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p05","B_relative_frobenius_error","B_energy_ratio_hat_to_true","precision_at_FPR_0p05","F1_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse")})
      run_rows.append({"outer_run":outer,"completed":True,"timestamp":time.time()});frames=materialize_tables(tables)
      parameter,brecovery,support,edge,calibration,roc,_,fixed=summaries(frames)
      partial={"parameter_results_partial.csv":frames["parameter"],"parameter_summary_partial.csv":parameter,
       "B_recovery_summary_partial.csv":brecovery,"A_support_summary_partial.csv":support,"vb_edge_scores_partial.csv":frames["vb_edges"],
       "vb_edge_score_summary_partial.csv":edge,"vb_uncertainty_calibration_summary_partial.csv":calibration,
       "fixed_fpr_operating_points_partial.csv":fixed,"roc_summary_partial.csv":roc}
      for name,frame in partial.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
      for key,name in checkpoint.items():
       if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
      atomic_csv(pd.DataFrame(run_rows),ledger)
    frames=materialize_tables(tables);parameter,brecovery,support,edge,calibration,roc,points,fixed=summaries(frames)
    decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
    outputs={"run_summary.csv":pd.DataFrame(run_rows),"runtime_summary.csv":grouped_stats(frames["runtime"],GROUPS,["runtime_seconds","n_iter","converged","spectral_radius_A","final_B_change_norm"]),
     "parameter_results.csv":frames["parameter"],"parameter_summary.csv":parameter,"B_recovery_summary.csv":brecovery,"A_support_summary.csv":support,
     "network_results.csv":frames["network"],"link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","method","B_MODEL_VARIANT","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
     "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,
     "lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],"vb_edge_scores.csv":frames["vb_edges"],
     "vb_edge_score_summary.csv":edge,"vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":calibration,"objective_history.csv":frames["objective"]}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    save_plots(frames["parameter"],frames["vb_edges"],edge,calibration,frames["runtime"])
    print("\nInterpretation guide: compare fixed/free/ridge B for VB reproduction of 34B, retained A-support, group-SNR ranking, low-FPR TPR, B-error/false-edge coupling, matched EM performance, interval under-coverage, and runtime. Robust free B supports VB-ARD as the main A branch; recovery only with ridge B supports VB A + ridge-B; degradation in both estimated-B modes identifies joint A-B estimation as the bottleneck; persistent under-coverage means estimator/ranker use only. TODO 34D: full structured q_x only if justified, and posterior spectral GC only after practical A-B validation.")


if __name__=="__main__":main()
