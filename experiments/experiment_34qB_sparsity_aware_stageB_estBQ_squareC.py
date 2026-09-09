"""Experiment 34Q-B: sparsity-aware Stage-B LRVB with estimated B/Q and known square C.

This remains Level-1 Hybrid Kalman + VB-ARD.  Intervals are centered at A_VB.
Independent Monte Carlo work units run in separate processes; BLAS and Numba are
kept single-threaded inside each worker to avoid nested oversubscription.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34QB_INNER_THREADS", "1")

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34n_lrvb_A_covariance as n
import experiments.experiment_34o_lrvb_stageB_smoother_feedback as o
import experiments.experiment_34p_sparsity_aware_stageB_lrvb as p
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats, relative
from experiments.experiment_34f_vb_ard_Q_estimation_shrinkage import Q_metrics
from experiments.experiment_34g_vb_ard_C_mixing_robustness import C_metrics, make_C
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls
from src.varx.varx_generator import generate_colored_input
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics

CONFIG_LIST = [
    {"label":"small_estBQ_identity_M5", "M":5, "T_VALUES":[1000,2000], "C_MODE":"identity", "N_TRUE_NETWORKS":2, "N_REPLICATES_PER_NETWORK":5, "covariance_mode":"selected"},
    {"label":"small_estBQ_mild_M5", "M":5, "T_VALUES":[1000,2000], "C_MODE":"mild_mixing", "N_TRUE_NETWORKS":2, "N_REPLICATES_PER_NETWORK":5, "covariance_mode":"selected"},
    {"label":"main_estBQ_identity_M20_subset", "M":20, "T_VALUES":[1000,2000], "C_MODE":"identity", "N_TRUE_NETWORKS":1, "N_REPLICATES_PER_NETWORK":3, "covariance_mode":"selected"},
    {"label":"main_estBQ_mild_M20_subset", "M":20, "T_VALUES":[1000,2000], "C_MODE":"mild_mixing", "N_TRUE_NETWORKS":1, "N_REPLICATES_PER_NETWORK":3, "covariance_mode":"selected"},
    {"label":"main_estBQ_strong_M20_subset", "M":20, "T_VALUES":[1000,2000], "C_MODE":"strong_mixing", "N_TRUE_NETWORKS":1, "N_REPLICATES_PER_NETWORK":3, "covariance_mode":"selected"},
]
FINITE_DIFF_EPS_GRID = [3e-5, 1e-4, 3e-4]
FINITE_DIFF_EPS_PRIMARY = 1e-4
PERTURBED_VB_MAX_ITER = int(os.environ.get("EXPERIMENT_34QB_PERTURBED_MAX_ITER", "50"))
PERTURBED_CONVERGENCE_TOL = 1e-4
VB_MAX_ITER = int(os.environ.get("EXPERIMENT_34QB_VB_MAX_ITER", "100"))
BASE_SEED = 5710000
SMOKE_TEST = os.environ.get("EXPERIMENT_34QB_SMOKE", "0") == "1"
REDUCED_RUN = os.environ.get("EXPERIMENT_34QB_REDUCED", "0") == "1"
REDUCED_REPLICATES = int(os.environ.get("EXPERIMENT_34QB_REPLICATES", "3"))
if REDUCED_RUN and not SMOKE_TEST:
    CONFIG_LIST = [
        {"label":f"main_estBQ_{mode.replace('_mixing','')}_M20_subset", "M":20,
         "T_VALUES":[2000], "C_MODE":mode, "N_TRUE_NETWORKS":1,
         "N_REPLICATES_PER_NETWORK":REDUCED_REPLICATES, "covariance_mode":"selected"}
        for mode in ("identity", "mild_mixing", "strong_mixing")
    ]
if SMOKE_TEST:
    CONFIG_LIST = [{"label":"small_estBQ_identity_M5", "M":5, "T_VALUES":[1000], "C_MODE":"identity", "N_TRUE_NETWORKS":1, "N_REPLICATES_PER_NETWORK":1, "covariance_mode":"selected"}]
    FINITE_DIFF_EPS_GRID = [1e-4]
    PERTURBED_VB_MAX_ITER = 30
RESULTS_DIR = os.environ.get("EXPERIMENT_34QB_RESULTS_DIR", "results/experiment_34qB")
MAX_WORKERS = max(1, int(os.environ.get("EXPERIMENT_34QB_WORKERS", str(min(4, os.cpu_count() or 1)))))
METHOD = "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_C_known_R"


class ProgressBar:
    def __init__(self, total, width=32):
        self.total, self.width, self.done = max(int(total), 0), width, 0
        if not self.total:
            print("Experiment 34Q-B: all checkpointed work is complete.")
    def update(self, label):
        self.done += 1
        f = min(self.done/max(self.total, 1), 1.0); k = round(self.width*f)
        print(f"\rExperiment 34Q-B [{'#'*k}{'-'*(self.width-k)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:48]}", end="\n" if self.done >= self.total else "", flush=True)


def simulate(A, B, M, T, C_MODE, seed):
    C = np.eye(M) if C_MODE == "identity" else make_C(M, C_MODE, seed)
    Q, R = .5*np.eye(M), .6*np.eye(M)
    u = generate_colored_input(T+j.BURN_IN, .95, 1., seed+200)
    result = generate_ssm_varx_p_data(A, B, u, Q, R, C=C, D=None, burn_in=j.BURN_IN, random_seed=seed+300, return_augmented=True)
    return {"A":A, "B":B, "Q":Q, "R":R, "C":C, "mask":np.any(np.asarray(A)!=0,axis=0)&~np.eye(M,dtype=bool), "x":result["x"], "y":result["y"], "u":result["u"]}


def fit_estBQ(data, seed):
    return HybridVBARDVARXSSMKnownCWithBQControls(
        j.na, j.nb, data["C"], data["Q"], data["R"], B_update_mode="free",
        B_ridge_lambda=0., base_B_ridge=0., estimate_Q=True,
        Q_update_mode="diag_shrink_scalar", Q_floor_mode="none", Q_floor_value=0.,
        Q_shrinkage_rho=.25, Q_update_damping=.5,
        include_A_posterior_uncertainty_in_Q=True, max_iter=VB_MAX_ITER,
        tol_objective=1e-6, tol_A_change=1e-6, tol_B_change=1e-6,
        tol_Q_change=1e-6, tol_alpha_change=1e-6, a0=1e-3, b0=1e-3,
        diagonal_prior_precision=1e-4, posterior_jitter=1e-8,
        random_state=seed).fit(data["y"], data["u"])


def latent_metrics(model, data):
    proxy = data["y"] @ np.linalg.pinv(data["C"]).T
    def corr(x, y): return float(np.nanmean([correlations(x[:,i], y[:,i]) for i in range(x.shape[1])]))
    return {"smoothed_state_MSE":np.mean((model.smoothed_state_mean_-data["x"])**2),
            "smoothed_state_correlation":corr(model.smoothed_state_mean_,data["x"]),
            "filtered_state_MSE":np.mean((model.filtered_state_mean_-data["x"])**2),
            "filtered_state_correlation":corr(model.filtered_state_mean_,data["x"]),
            "pseudo_inverse_state_MSE":np.mean((proxy-data["x"])**2),
            "pseudo_inverse_state_correlation":corr(proxy,data["x"])}


def q_recovery(model, data, meta):
    row = Q_metrics(model.Q, data, model)
    return {**meta, **row, "Q_shrinkage_rho":.25, "Q_damping":.5,
            "include_A_posterior_uncertainty_in_Q":True}


def a_recovery(model, data, meta):
    row,_=j.recovery(np.asarray(model.A_mean_matrices_),data,model.filtered_state_mean_,model.smoothed_state_mean_,meta,0.,model.n_iter_,model.converged_)
    return {**meta,**row,"A_diag_relative_frobenius_error":row["A_diagonal_relative_frobenius_error"],"active_group_norm_correlation":row["A_group_norm_pearson_correlation"],"support_AUPRC_model_A_group_norm":row["A_support_AUPRC"],"support_ROC_AUC_model_A_group_norm":row["A_support_ROC_AUC"],"TPR_at_FPR_0p05_model_A_group_norm":row["A_support_TPR_at_FPR_0p05"],"precision_at_FPR_0p05_model_A_group_norm":row.get("precision_at_FPR_0p05",np.nan)}


def calibration_summary(frame):
    keys=["config_label","M","T","C_MODE","covariance_estimator","coefficient_type"]; rows=[]
    for values,g in frame.groupby(keys,dropna=False):
        rows.append({**dict(zip(keys,values)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_interval_width_95":g.interval_width_95.mean(),"median_interval_width_95":g.interval_width_95.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"n_coefficients":len(g)})
    return pd.DataFrame(rows)


def group_summary(frame):
    keys=["config_label","M","T","C_MODE","covariance_estimator","edge_group_type"]
    return frame.groupby(keys,dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),n_groups=("D2","size")).reset_index()


def edge_summary(frame):
    rows=[]; keys=["config_label","M","T","C_MODE","covariance_estimator"]
    for values,g in frame.groupby(keys,dropna=False):
        metrics,_=j.safe_curve(g.true_link,g.score); rows.append({**dict(zip(keys,values)),**metrics})
    return pd.DataFrame(rows)


def empirical(frame):
    rows=[]; keys=["config_label","true_network_id","M","T","C_MODE","coefficient_global_index","covariance_estimator","coefficient_type"]
    for values,g in frame.groupby(keys,dropna=False):
        ev=g.signed_error.var(ddof=1) if len(g)>1 else g.squared_error.mean(); pv=g.posterior_variance.mean(); ratio=pv/max(ev,1e-15)
        rows.append({**dict(zip(keys,values)),"posterior_variance":pv,"empirical_error_variance":ev,"variance_ratio_posterior_to_empirical":ratio,"posterior_sd":np.sqrt(max(pv,0)),"empirical_sd":np.sqrt(max(ev,0)),"sd_ratio_posterior_to_empirical":np.sqrt(max(ratio,0)),"n_replicates":len(g)})
    return pd.DataFrame(rows)


def task_specifications():
    tasks=[]
    for cfg in CONFIG_LIST:
        for network in range(cfg["N_TRUE_NETWORKS"]):
            for T in cfg["T_VALUES"]:
                for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
                    tasks.append((cfg,T,network,replicate))
    return tasks


def worker(task):
    cfg,T,network_id,replicate=task; M=cfg["M"]; mode=cfg["C_MODE"]
    network_seed=BASE_SEED+M*100000+network_id*1000+(0 if mode=="identity" else 2000000 if mode=="mild_mixing" else 4000000)
    seed=network_seed+T*10+replicate
    meta={"config_label":cfg["label"],"M":M,"T":T,"C_MODE":mode,"true_network_id":network_id,"replicate_id":replicate,"method":METHOD}
    out={k:[] for k in ("selected","groupsel","coeff","group","scores","network","B","Q","A","latent","covdiag","perturb","activity","gating","runtime","run")}; started=time.perf_counter()
    try:
        A,B,_=j.fixed_network(M,network_seed); data=simulate(A,B,M,T,mode,seed)
        base=time.perf_counter(); model=fit_estBQ(data,seed+500); base_runtime=time.perf_counter()-base
        lstart=time.perf_counter(); louis=n._louis_covariances(model,data,seed+700); louis_runtime=time.perf_counter()-lstart
        selected=qa.select_directions(model,data,louis,seed+900)
        # Reuse 34Q-A's selected finite-difference driver with full B/Q feedback.
        qa.FINITE_DIFF_EPS_GRID=FINITE_DIFF_EPS_GRID; qa.FINITE_DIFF_EPS_PRIMARY=FINITE_DIFF_EPS_PRIMARY
        qa.PERTURBED_VB_MAX_ITER=PERTURBED_VB_MAX_ITER; qa.PERTURBED_CONVERGENCE_TOL=PERTURBED_CONVERGENCE_TOL
        sstart=time.perf_counter(); valid,raw,psd,perturb,projection,times=qa.run_stageB_free(model,data,selected,meta,True,True); stage_runtime=time.perf_counter()-sstart
        selected=selected.set_index("coefficient_global_index").loc[valid].reset_index()
        out["selected"].append(pd.DataFrame([{**meta,**r} for r in selected.to_dict("records")]))
        global_sets={"ordinary_vb":o.global_covariance_block(model.A_row_covariances_),"stabilized_louis_eta_0p70_tau_0p90":o.global_covariance_block(louis)}
        blocks={name:value[np.ix_(valid,valid)] for name,value in global_sets.items()}; blocks["lrvb_stageB_smoother_feedback_raw"]=raw; blocks["lrvb_stageB_smoother_feedback_psd_projected"]=psd
        records=p.group_records(model,data,selected,{"ordinary_vb":blocks["ordinary_vb"],"stabilized_louis_eta_0p70_tau_0p90":blocks["stabilized_louis_eta_0p70_tau_0p90"],"lrvb_stageB_smoother_feedback":psd})
        out["groupsel"].append(pd.DataFrame([{**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"],"complete_group":r["complete_group"]} for r in records]))
        variants=qa.covariance_variants(blocks,records)
        for variant in variants:
            out["coeff"].append(qa.coefficient_rows(model,data,selected,variant,meta)); g,s=qa.group_rows(model,records,variant,meta); out["group"].append(g); out["scores"].append(s); out["gating"].append(qa.gating_row(meta,variant,records))
            proj=variant["projection"]; out["covdiag"].append({**meta,"covariance_estimator":variant["covariance_estimator"],"number_selected_directions":len(selected),"number_negative_variances":proj.get("number_negative_variances",0),"fraction_negative_variances":proj.get("fraction_negative_variances",0.),"number_psd_projections":int(proj.get("psd_projection_used",False)),"fraction_psd_projected":float(proj.get("psd_projection_used",False)),**covariance_diagnostics(variant["covariance"],blocks["ordinary_vb"],blocks["stabilized_louis_eta_0p70_tau_0p90"],selected.coefficient_type)})
        Arow=a_recovery(model,data,meta); Brow=qa.B_recovery(model,data,meta); Qrow=q_recovery(model,data,meta); Lrow={**meta,**latent_metrics(model,data),**C_metrics(data,data["y"]@np.linalg.pinv(data["C"]).T)}
        out["network"].append(qa.full_network(model,data,meta,louis)); out["B"].append(Brow); out["Q"].append(Qrow); out["A"].append(Arow); out["latent"].append(Lrow)
        out["perturb"].append(perturb); out["activity"].append(qa.activity_rows(meta,records,blocks))
        out["runtime"].append({**meta,"baseline_VB_runtime_seconds":base_runtime,"louis_runtime_seconds":louis_runtime,"stageB_total_runtime_seconds":stage_runtime,"average_perturbed_fit_runtime_seconds":np.mean(times),"number_perturbed_fits":len(times),"number_selected_directions":len(selected),"B_estimation_enabled":True,"Q_estimation_enabled":True,"total_runtime_seconds":time.perf_counter()-started,"run_status":"success"})
        out["run"].append({**meta,"fit_status":"success","total_runtime_seconds":time.perf_counter()-started,**{k:v for row in (Arow,Brow,Qrow,Lrow) for k,v in row.items() if k not in meta}})
    except Exception as error:
        failure={**meta,"fit_status":"failed","total_runtime_seconds":time.perf_counter()-started,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()}; out["run"].append(failure); out["runtime"].append({**failure,"run_status":"failed"})
    return {k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in out.items()}


PARTIAL={"run":"run_summary_partial.csv","coeff":"_coefficient_calibration_partial.csv","group":"_group_calibration_partial.csv","selected":"_selected_coefficients_partial.csv","groupsel":"_selected_groups_partial.csv","scores":"_selected_scores_partial.csv","network":"_network_partial.csv","B":"_B_checkpoint.csv","Q":"_Q_checkpoint.csv","A":"_A_checkpoint.csv","latent":"_latent_checkpoint.csv","covdiag":"covariance_diagnostics_partial.csv","perturb":"stageB_perturbation_diagnostics_partial.csv","activity":"_activity_partial.csv","gating":"_gating_partial.csv","runtime":"_runtime_checkpoint.csv"}


def combine(tables):
    return {k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in tables.items()}


def save_checkpoints(frames):
    for key,name in PARTIAL.items(): atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
    if len(frames["coeff"]): atomic_csv(calibration_summary(frames["coeff"]),os.path.join(RESULTS_DIR,"calibration_summary_partial.csv"))
    if len(frames["group"]): atomic_csv(group_summary(frames["group"]),os.path.join(RESULTS_DIR,"group_calibration_summary_partial.csv"))
    if len(frames["A"]):
        _,_,_,_,A,B,Q,latent,runtime=summaries(frames)
        for name,table in (("A_recovery_summary_partial.csv",A),("B_recovery_summary_partial.csv",B),("Q_recovery_summary_partial.csv",Q),("latent_state_recovery_summary_partial.csv",latent),("runtime_summary_partial.csv",runtime),("decision_summary_partial.csv",decision_summary(frames))): atomic_csv(table,os.path.join(RESULTS_DIR,name))


def summaries(frames):
    keys=["config_label","M","T","C_MODE"]
    cal=calibration_summary(frames["coeff"]); group=group_summary(frames["group"]); edge=edge_summary(frames["scores"]); emp=empirical(frames["coeff"])
    A=grouped_stats(frames["A"],keys,["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diag_relative_frobenius_error","active_group_norm_correlation","support_AUPRC_model_A_group_norm","support_ROC_AUC_model_A_group_norm","TPR_at_FPR_0p05_model_A_group_norm","precision_at_FPR_0p05_model_A_group_norm"])
    B=grouped_stats(frames["B"],keys,["B_relative_frobenius_error","B_lag0_relative_error","B_lag1_relative_error","B_lag2_relative_error","B_energy_ratio_hat_to_true","B_correlation_hat_true"])
    Q=grouped_stats(frames["Q"],keys,["Q_relative_frobenius_error","Q_diag_relative_error","Q_trace_ratio_hat_to_true","Q_min_diag","Q_max_diag","Q_mean_diag"])
    latent=grouped_stats(frames["latent"],keys,["smoothed_state_MSE","smoothed_state_correlation","filtered_state_MSE","filtered_state_correlation","pseudo_inverse_state_MSE","pseudo_inverse_state_correlation","C_condition_number"])
    runtime=grouped_stats(frames["runtime"],keys,["baseline_VB_runtime_seconds","louis_runtime_seconds","stageB_total_runtime_seconds","average_perturbed_fit_runtime_seconds","number_perturbed_fits","number_selected_directions","total_runtime_seconds"])
    repkeys=keys+["true_network_id","replicate_id"]; soft_name="soft_sparsity_aware_stageB_logistic_c4_tau2p5"
    active=frames["coeff"].loc[(frames["coeff"].covariance_estimator==soft_name)&(frames["coeff"].coefficient_type=="offdiag_nonzero")].groupby(repkeys).agg(active_coverage=("ci95_contains_true","mean"),active_std_z=("standardized_error",lambda x:x.std(ddof=0))).reset_index(); score=[]
    for values,g in frames["scores"].loc[frames["scores"].covariance_estimator==soft_name].groupby(repkeys):
        metrics,_=j.safe_curve(g.true_link,g.score); score.append({**dict(zip(repkeys,values)),"selected_edge_AUPRC":metrics["AUPRC"]})
    diagnostic=active.merge(pd.DataFrame(score),on=repkeys,how="outer").merge(frames["B"],on=repkeys,how="left").merge(frames["Q"],on=repkeys,how="left").merge(frames["latent"],on=repkeys,how="left")
    for values,g in diagnostic.groupby(keys,dropna=False):
        for table,prefix,column in ((B,"B", "B_relative_frobenius_error"),(Q,"Q","Q_relative_frobenius_error"),(latent,"smoothed_state_MSE","smoothed_state_MSE")):
            mask=(table.config_label==values[0])&(table.M==values[1])&(table["T"]==values[2])&(table.C_MODE==values[3]); table.loc[mask,f"correlation_{prefix}_error_active_coverage" if prefix in ("B","Q") else "correlation_smoothed_state_MSE_active_coverage"]=correlations(g[column],g.active_coverage); table.loc[mask,f"correlation_{prefix}_error_offdiag_nonzero_std_z" if prefix in ("B","Q") else "correlation_smoothed_state_MSE_offdiag_nonzero_std_z"]=correlations(g[column],g.active_std_z); table.loc[mask,f"correlation_{prefix}_error_selected_edge_AUPRC" if prefix in ("B","Q") else "correlation_smoothed_state_MSE_selected_edge_AUPRC"]=correlations(g[column],g.selected_edge_AUPRC)
    return cal,group,edge,emp,A,B,Q,latent,runtime


def decision_summary(frames):
    if not len(frames["coeff"]): return pd.DataFrame()
    cal,groups,edges,emp,As,Bs,Qs,Ls,Rt=summaries(frames); rows=[]
    for keys,g in cal.groupby(["config_label","M","T","C_MODE","covariance_estimator"],dropna=False):
        by={kind:item for kind,item in g.groupby("coefficient_type")}; val=lambda kind,col:by[kind][col].mean() if kind in by else np.nan
        match=lambda f:(f.config_label==keys[0])&(f.M==keys[1])&(f["T"]==keys[2])&(f.C_MODE==keys[3]); gr=groups.loc[match(groups)&(groups.covariance_estimator==keys[4])]; ed=edges.loc[match(edges)&(edges.covariance_estimator==keys[4])]; ev=emp.loc[match(emp)&(emp.covariance_estimator==keys[4])]; net=frames["network"].loc[match(frames["network"])]; cd=frames["covdiag"].loc[match(frames["covdiag"])&(frames["covdiag"].covariance_estimator==keys[4])]; rt=frames["runtime"].loc[match(frames["runtime"])]
        rows.append({"config_label":keys[0],"M":keys[1],"T":keys[2],"C_MODE":keys[3],"covariance_estimator":keys[4],"diag_coverage":val("diagonal","empirical_coverage_95"),"offdiag_nonzero_coverage":val("offdiag_nonzero","empirical_coverage_95"),"offdiag_zero_coverage":val("offdiag_zero","empirical_coverage_95"),"diag_std_z":val("diagonal","std_standardized_error"),"offdiag_nonzero_std_z":val("offdiag_nonzero","std_standardized_error"),"offdiag_zero_std_z":val("offdiag_zero","std_standardized_error"),"diag_mean_interval_width_95":val("diagonal","mean_interval_width_95"),"offdiag_nonzero_mean_interval_width_95":val("offdiag_nonzero","mean_interval_width_95"),"offdiag_zero_mean_interval_width_95":val("offdiag_zero","mean_interval_width_95"),"true_edge_group_chi2_95_coverage":gr.loc[gr.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"false_edge_group_chi2_95_coverage":gr.loc[gr.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"selected_edge_AUC":ed.ROC_AUC.mean(),"selected_edge_AUPRC":ed.AUPRC.mean(),"selected_edge_TPR_at_FPR_0p05":ed.TPR_at_FPR_0p05.mean(),"selected_edge_precision_at_FPR_0p05":ed.precision_at_FPR_0p05.mean(),"full_network_AUPRC_model_A_group_norm":net.loc[net.score_type=="model_A_group_norm","AUPRC"].mean(),"full_network_AUPRC_stabilized_louis_group_snr":net.loc[net.score_type=="stabilized_louis_group_snr","AUPRC"].mean(),"A_relative_frobenius_error":frames["A"].loc[match(frames["A"]),"A_relative_frobenius_error"].mean(),"A_offdiag_relative_frobenius_error":frames["A"].loc[match(frames["A"]),"A_offdiag_relative_frobenius_error"].mean(),"B_relative_frobenius_error":frames["B"].loc[match(frames["B"]),"B_relative_frobenius_error"].mean(),"B_energy_ratio_hat_to_true":frames["B"].loc[match(frames["B"]),"B_energy_ratio_hat_to_true"].mean(),"Q_relative_frobenius_error":frames["Q"].loc[match(frames["Q"]),"Q_relative_frobenius_error"].mean(),"Q_trace_ratio_hat_to_true":frames["Q"].loc[match(frames["Q"]),"Q_trace_ratio_hat_to_true"].mean(),"smoothed_state_MSE":frames["latent"].loc[match(frames["latent"]),"smoothed_state_MSE"].mean(),"smoothed_state_correlation":frames["latent"].loc[match(frames["latent"]),"smoothed_state_correlation"].mean(),"variance_ratio_to_empirical_diag":ev.loc[ev.coefficient_type=="diagonal","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_nonzero":ev.loc[ev.coefficient_type=="offdiag_nonzero","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_zero":ev.loc[ev.coefficient_type=="offdiag_zero","variance_ratio_posterior_to_empirical"].mean(),"fraction_negative_variances":cd.fraction_negative_variances.mean(),"fraction_psd_projected":cd.fraction_psd_projected.mean(),"total_runtime_seconds":rt.total_runtime_seconds.mean(),"recommended_for_uncertainty":bool(.88<=np.nanmean([val("diagonal","empirical_coverage_95"),val("offdiag_nonzero","empirical_coverage_95")])<=.97),"recommended_for_ranking":bool(ed.AUPRC.mean()>=.5),"notes":"A_center=A_VB; B and Q re-estimated in primary perturbed fits."})
    return pd.DataFrame(rows)


def plots(cal,activity,A,B,Q,latent,edges,covdiag,runtime):
    path=os.path.join(RESULTS_DIR,"plots"); os.makedirs(path,exist_ok=True)
    def bar(frame,groups,y,name):
        s=frame.groupby(groups,dropna=False)[y].mean(); fig,ax=plt.subplots(); s.plot.bar(ax=ax); ax.set_ylabel(y); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(os.path.join(path,name)); plt.close(fig)
    bar(cal,["covariance_estimator","coefficient_type","C_MODE"],"empirical_coverage_95","coverage.png"); bar(cal,["covariance_estimator","coefficient_type","C_MODE"],"std_standardized_error","standardized_error_sd.png"); bar(cal,["covariance_estimator","coefficient_type","C_MODE"],"mean_interval_width_95","interval_width.png"); bar(cal,["covariance_estimator","C_MODE"],"mean_posterior_sd","posterior_sd.png"); bar(cal.loc[cal.coefficient_type.isin(["offdiag_nonzero","offdiag_zero"])],["covariance_estimator","coefficient_type"],"empirical_coverage_95","active_vs_zero_coverage.png"); bar(activity,["true_edge_group","C_MODE"],"soft_logistic_weight_c4_tau2p5","soft_weights.png"); bar(activity,["true_edge_group","C_MODE"],"group_snr_louis","louis_snr.png"); bar(edges,["covariance_estimator","C_MODE"],"AUPRC","selected_edge_AUPRC.png"); bar(covdiag,["covariance_estimator","C_MODE"],"trace_over_louis_selected","stageB_louis_ratio.png"); bar(runtime,["C_MODE","T"],"total_runtime_seconds_mean","runtime.png")
    soft=cal.loc[(cal.covariance_estimator=="soft_sparsity_aware_stageB_logistic_c4_tau2p5")&(cal.coefficient_type=="offdiag_nonzero")]
    for frame,x,name in ((B,"B_relative_frobenius_error","B_error_coverage.png"),(Q,"Q_relative_frobenius_error","Q_error_coverage.png"),(latent,"smoothed_state_MSE","state_MSE_coverage.png")):
        column=x if x in frame else x+"_mean"; merged=frame.merge(soft,on=["config_label","M","T","C_MODE"]); fig,ax=plt.subplots(); ax.scatter(merged[column],merged.empirical_coverage_95); ax.set(xlabel=x,ylabel="active coverage"); fig.tight_layout(); fig.savefig(os.path.join(path,name)); plt.close(fig)
    sources=[(A,"A_relative_frobenius_error"),(B,"B_relative_frobenius_error"),(Q,"Q_relative_frobenius_error"),(latent,"smoothed_state_MSE")]; fig,axes=plt.subplots(2,2,figsize=(8,6));
    for ax,(frame,column) in zip(axes.ravel(),sources):
        value=column if column in frame else column+"_mean"; pivot=frame.groupby(["C_MODE","T"])[value].mean().unstack(); pivot.plot.bar(ax=ax,legend=False); ax.set_ylabel(column); ax.tick_params(axis="x",rotation=20)
    fig.tight_layout(); fig.savefig(os.path.join(path,"C_MODE_recovery_comparison.png")); plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    config={"experiment":"34Q-B","CONFIG_LIST":CONFIG_LIST,"FINITE_DIFF_EPS_GRID":FINITE_DIFF_EPS_GRID,"FINITE_DIFF_EPS_PRIMARY":FINITE_DIFF_EPS_PRIMARY,"PERTURBED_VB_MAX_ITER":PERTURBED_VB_MAX_ITER,"PERTURBED_CONVERGENCE_TOL":PERTURBED_CONVERGENCE_TOL,"VB_MAX_ITER":VB_MAX_ITER,"B_REESTIMATED_IN_PERTURBED_FITS":True,"Q_REESTIMATED_IN_PERTURBED_FITS":True,"Q_shrinkage_rho":.25,"Q_damping":.5,"C_known":True,"R_fixed_true":True,"A_center":"A_VB","max_workers":MAX_WORKERS,"inner_threads":int(os.environ.get("EXPERIMENT_34QB_INNER_THREADS","1")),"acceleration":{"monte_carlo":"ProcessPoolExecutor","louis_kernels":"Numba when installed","nested_threading":"disabled"},"smoke_test":SMOKE_TEST,"reduced_run":REDUCED_RUN,"reduced_replicates":REDUCED_REPLICATES}
    cp=os.path.join(RESULTS_DIR,"experiment_config.json"); ledger=os.path.join(RESULTS_DIR,PARTIAL["run"])
    if os.path.exists(cp) and os.path.exists(ledger):
        with open(cp,encoding="utf8") as h: old=json.load(h)
        for key in ("CONFIG_LIST","FINITE_DIFF_EPS_GRID","FINITE_DIFF_EPS_PRIMARY","PERTURBED_VB_MAX_ITER","VB_MAX_ITER","Q_shrinkage_rho"):
            if old.get(key)!=config.get(key): raise ValueError("Existing 34Q-B checkpoint numerical configuration differs.")
    with open(cp,"w",encoding="utf8") as h: json.dump(config,h,indent=2)
    tables={k:[] for k in PARTIAL}
    for key,name in PARTIAL.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            try: f=pd.read_csv(path)
            except pd.errors.EmptyDataError: f=pd.DataFrame()
            if len(f): tables[key].append(f)
    prior=combine(tables)["run"]; completed=set(zip(prior.config_label,prior["T"].astype(int),prior.true_network_id.astype(int),prior.replicate_id.astype(int))) if len(prior) else set()
    tasks=[task for task in task_specifications() if (task[0]["label"],task[1],task[2],task[3]) not in completed]; progress=ProgressBar(len(tasks))
    if tasks:
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS,len(tasks))) as pool:
            futures={pool.submit(worker,task):task for task in tasks}
            for future in as_completed(futures):
                task=futures[future]; cfg,T,network,replicate=task
                try: result=future.result()
                except Exception as error:
                    meta={"config_label":cfg["label"],"M":cfg["M"],"T":T,"C_MODE":cfg["C_MODE"],"true_network_id":network,"replicate_id":replicate,"fit_status":"worker_failed","error_type":type(error).__name__,"error_message":str(error)}; result={k:pd.DataFrame() for k in PARTIAL}; result["run"]=pd.DataFrame([meta]); result["runtime"]=pd.DataFrame([{**meta,"run_status":"worker_failed"}])
                for key,frame in result.items():
                    if len(frame): tables[key].append(frame)
                frames=combine(tables); save_checkpoints(frames); status=result["run"].iloc[-1].get("fit_status","failed")
                progress.update(f"{cfg['label']} T={T} net={network+1} rep={replicate+1} {status}")
    frames=combine(tables)
    if not len(frames["coeff"]): return
    cal,group,edge,emp,A,B,Q,latent,runtime=summaries(frames); decision=decision_summary(frames)
    outputs={"run_summary.csv":frames["run"],"selected_coefficients.csv":frames["selected"],"selected_groups.csv":frames["groupsel"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":group,"network_recovery_summary.csv":frames["network"],"selected_edge_recovery_summary.csv":edge,"A_recovery_summary.csv":A,"B_recovery_summary.csv":B,"Q_recovery_summary.csv":Q,"latent_state_recovery_summary.csv":latent,"covariance_diagnostics.csv":frames["covdiag"],"stageB_perturbation_diagnostics.csv":frames["perturb"],"activity_score_diagnostics.csv":frames["activity"],"gating_diagnostics.csv":frames["gating"],"empirical_variance_comparison.csv":emp,"deviance_diagnostics.csv":pd.DataFrame([{"status":"not_computed_secondary_runtime_priority","reason":"No new deviance implementation; 34Q-B prioritizes covariance calibration."}]),"runtime_summary.csv":runtime,"decision_summary.csv":decision}
    for name,frame in outputs.items(): atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    if not SMOKE_TEST: plots(cal,frames["activity"],A,B,Q,latent,edge,frames["covdiag"],runtime)


if __name__ == "__main__":
    main()
