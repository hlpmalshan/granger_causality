"""Experiment 34R Stage 2C: validate tau=1 Stage-B gates for rectangular C.

This remains Level-1 Hybrid Kalman + VB-ARD.  Stage-B is evaluated only in
selected coefficient directions, A_VB remains the posterior center, and B/Q
are re-estimated in every central finite-difference fit.
"""
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[_key] = os.environ.get("EXPERIMENT_34R_STAGE2C_INNER_THREADS", "1")

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34i_static_spectral_varx_gc as spi
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as qb
import experiments.experiment_34qC_practical_model_choice_squareC as qc
import experiments.experiment_34r_stage1_rectangular_C_breakpoint as r1
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls
from src.stats.lrvb_stageB_smoother_feedback import global_index, project_selected_covariance, sensitivity_column
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics, gated_covariance, globally_damped_covariance, logistic_weight

RESULTS_DIR = os.environ.get("EXPERIMENT_34R_STAGE2C_RESULTS_DIR", "results/experiment_34r_stage2c_validate_tau1")
WORKERS = max(1, int(os.environ.get("EXPERIMENT_34R_STAGE2C_WORKERS", "2")))
SMOKE = os.environ.get("EXPERIMENT_34R_STAGE2C_SMOKE", "0") == "1"
CONFIGS = [
    {"label":"gaussian_positive_control_Mx20_My40_T1000", "M_x":20, "M_y":40, "T":1000},
    {"label":"gaussian_transition_Mx20_My15_T1000", "M_x":20, "M_y":15, "T":1000},
]
N_TRUE_NETWORKS, N_REPLICATES_PER_NETWORK = 3, 2
if SMOKE:
    CONFIGS = [{"label":"smoke_gaussian_Mx5_My4_T200", "M_x":5, "M_y":4, "T":200}]
    N_TRUE_NETWORKS = N_REPLICATES_PER_NETWORK = 1

METHOD = "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_gaussian_rectangular_C_known_R"
EPS = 1e-4
PERTURB_MAX = 5 if SMOKE else 30
RESCUE_MAX = 5 if SMOKE else 50
VB_MAX_ITER = 8 if SMOKE else 75
TOL = 1e-4
CAP_SECONDS = float(os.environ.get("EXPERIMENT_34R_STAGE2C_STAGEB_CAP_SECONDS", str(10*3600)))
SELECT_LIMIT = 14 if SMOKE else 95
TAUS = (1.0, 1.5, 2.0, 2.5)

TABLES = ("run","C","selected","groupsel","coeff","group","covdiag","perturb","activity",
          "latent","A","B","Q","network","spectral","spectra","deviance","runtime")
CHECKPOINTS = {key:f"_{key}_checkpoint.csv" for key in TABLES}
CHECKPOINTS["run"] = "run_summary_partial.csv"
PARTIAL_DIRECT = {
    "covdiag":"covariance_diagnostics_partial.csv", "perturb":"stageB_perturbation_diagnostics_partial.csv",
    "latent":"latent_state_recovery_summary_partial.csv", "A":"A_recovery_summary_partial.csv",
    "B":"B_recovery_summary_partial.csv", "Q":"Q_recovery_summary_partial.csv",
    "network":"network_recovery_summary_partial.csv", "spectra":"spectral_gc_summary_partial.csv",
    "deviance":"deviance_diagnostics_partial.csv", "runtime":"runtime_summary_partial.csv",
}


class Progress:
    def __init__(self, total):
        self.total, self.done, self.start = max(int(total), 0), 0, time.perf_counter()
        if not self.total:
            print("Experiment 34R Stage 2C: all checkpointed runs are complete.")
    def update(self, label):
        self.done += 1; frac = self.done/max(self.total,1); width = 30; filled = round(width*frac)
        elapsed = time.perf_counter()-self.start; eta = elapsed/max(self.done,1)*(self.total-self.done)
        print(f"\r34R-2C [{'#'*filled}{'-'*(width-filled)}] {self.done}/{self.total} "
              f"{100*frac:5.1f}% ETA {eta/3600:5.2f}h {label[:38]}",
              end="\n" if self.done == self.total else "", flush=True)


def meta_for(cfg, network, replicate):
    return {"config_label":cfg["label"], "M_x":cfg["M_x"], "M_y":cfg["M_y"],
            "observation_ratio":cfg["M_y"]/cfg["M_x"], "C_family":"gaussian_isotropic",
            "C_MODE":"gaussian_isotropic", "T":cfg["T"], "true_network_id":network,
            "replicate_id":replicate, "method":METHOD,
            "run_id":f"{cfg['label']}_network{network}_replicate{replicate}"}


def fit_model(data, seed, max_iter=VB_MAX_ITER):
    return HybridVBARDVARXSSMKnownCWithBQControls(
        2, 3, data["C"], data["Q"], data["R"], B_update_mode="free", B_ridge_lambda=0.,
        base_B_ridge=0., estimate_Q=True, Q_update_mode="diag_shrink_scalar",
        Q_floor_mode="none", Q_floor_value=0., Q_shrinkage_rho=.25, Q_update_damping=.5,
        include_A_posterior_uncertainty_in_Q=True, max_iter=max_iter, tol_objective=TOL,
        tol_A_change=TOL, tol_B_change=TOL, tol_Q_change=TOL, tol_alpha_change=TOL,
        a0=1e-3, b0=1e-3, diagonal_prior_precision=1e-4, posterior_jitter=1e-8,
        random_state=seed).fit(data["y"], data["u"])


def select_directions(model, data, louis, meta, seed, limit=SELECT_LIMIT):
    M = model.n_states; rng = np.random.default_rng(seed); column_obs = np.linalg.norm(data["C"], axis=0)
    groups, seen = [], set()
    def add(target, source, reason):
        if (target,source) not in seen:
            seen.add((target,source)); groups.append((int(target),int(source),reason))
    for target,source in zip(*np.where(data["mask"])):
        add(target,source,"true_offdiag_all")
    order = np.argsort(column_obs)
    diagonal = np.unique(np.r_[order[:3],order[max(0,M//2-1):M//2+2],order[-3:]])[:min(8,M)]
    for source in diagonal: add(source,source,"diagonal_observability_spread")
    false = [(t,s) for t in range(M) for s in range(M) if t != s and not data["mask"][t,s]]
    shuffled = false.copy(); rng.shuffle(shuffled)
    true_edge_obs=sorted(column_obs[t]*column_obs[s] for t,s in zip(*np.where(data["mask"])))
    desired=np.quantile(true_edge_obs,np.linspace(.05,.95,min(8,len(true_edge_obs)))) if true_edge_obs else []
    available=shuffled.copy()
    for value in desired:
        target,source=min(available,key=lambda pair:abs(column_obs[pair[0]]*column_obs[pair[1]]-value))
        add(target,source,"matched_false_observability"); available.remove((target,source))
    scored = []
    for target,source in false:
        cols=[source,M+source]; mean=np.asarray(model.A_mean_matrices_)[:,target,source]
        scored.append((float(np.sqrt(max(mean@np.linalg.pinv(louis[target][np.ix_(cols,cols)])@mean,0))),target,source))
    for _,target,source in sorted(scored,reverse=True)[:5]: add(target,source,"high_snr_false")
    if 2*(len(groups)+5) <= limit:
        for _,target,source in sorted(scored)[:5]: add(target,source,"low_snr_false")
    true_obs = [column_obs[t]*column_obs[s] for t,s,_ in groups if data["mask"][t,s]]
    cuts = np.quantile(true_obs,[.25,.5,.75]) if true_obs else [np.nan]*3
    def quartile(value):
        if not np.isfinite(cuts).all(): return np.nan
        return ("Q1_low","Q2","Q3","Q4_high")[int(np.searchsorted(cuts,value,side="right"))]
    coefficient_rows, group_rows = [], []
    for target,source,reason in groups:
        cols=[source,M+source]; mean=np.asarray(model.A_mean_matrices_)[:,target,source]
        snr=float(np.sqrt(max(mean@np.linalg.pinv(louis[target][np.ix_(cols,cols)])@mean,0)))
        edge_obs=float(column_obs[target]*column_obs[source]); true=bool(data["mask"][target,source])
        kind="diagonal" if target==source else ("offdiag_nonzero" if true else "offdiag_zero")
        common={**meta,"target":target,"source":source,"true_edge":true,"coefficient_type":kind,
                "selected_direction_type":reason,"selected_reason":reason,
                "source_observability_score":column_obs[source],"target_observability_score":column_obs[target],
                "edge_observability_score":edge_obs,"group_snr_louis":snr,
                "estimated_group_norm":np.linalg.norm(mean),
                "observability_quartile":quartile(edge_obs) if true else np.nan}
        group_rows.append(common)
        for lag in range(2):
            coefficient_rows.append({**common,"coefficient_global_index":global_index(target,lag,source,M,2),"lag":lag+1})
    return pd.DataFrame(coefficient_rows).head(limit), pd.DataFrame(group_rows)


def perturb_one(args):
    model,y,u,index = args; start=time.perf_counter()
    column,plus,minus=sensitivity_column(model,y,u,index,EPS,PERTURB_MAX,TOL,reestimate_B=True,reestimate_Q=True)
    rescue=not (plus.converged and minus.converged)
    if rescue and RESCUE_MAX>PERTURB_MAX:
        column,plus,minus=sensitivity_column(model,y,u,index,EPS,RESCUE_MAX,TOL,reestimate_B=True,reestimate_Q=True)
    row={"perturbation_direction_index":index,"eps":EPS,"use_central_difference":True,
         "plus_converged":plus.converged,"minus_converged":minus.converged,
         "plus_n_iter":plus.n_iter,"minus_n_iter":minus.n_iter,
         "plus_rescue_used":rescue,"minus_rescue_used":rescue,
         "plus_final_relative_A_change":plus.final_relative_A_change,"minus_final_relative_A_change":minus.final_relative_A_change,
         "plus_final_relative_B_change":plus.final_relative_B_change,"minus_final_relative_B_change":minus.final_relative_B_change,
         "plus_final_relative_Q_change":plus.final_relative_Q_change,"minus_final_relative_Q_change":minus.final_relative_Q_change,
         "plus_loglikelihood":plus.final_loglikelihood,"minus_loglikelihood":minus.final_loglikelihood,
         "variance_estimate_raw":column[index],"finite_difference_valid":bool(np.all(np.isfinite(column))),
         "B_reestimated_in_perturbed_fits":True,"Q_reestimated_in_perturbed_fits":True,
         "warning_flag":";".join(filter(None,[plus.warning_flag,minus.warning_flag])),
         "runtime_seconds":time.perf_counter()-start}
    return index,column,row


def covariance_variants(blocks, records):
    louis=blocks["stabilized_louis_eta_0p70_tau_0p90"]; stage=blocks["lrvb_stageB_smoother_feedback_psd_projected"]
    variants=[]
    def add(name,cov,weights=None,projection=None):
        variants.append({"covariance_estimator":name,"covariance":cov,"weights":weights,
                         "projection":projection or {"psd_projection_used":False,"number_negative_variances":0,"fraction_negative_variances":0.}})
    for name in ("ordinary_vb","stabilized_louis_eta_0p70_tau_0p90","lrvb_stageB_smoother_feedback_raw","lrvb_stageB_smoother_feedback_psd_projected"):
        add(name,blocks[name])
    def weights(tau,hard=False):
        values={}
        for record in records:
            weight=1. if record["diagonal"] else (float(record["group_snr_louis"]>=tau) if hard else logistic_weight(record["group_snr_louis"],tau,4.))
            for position in record["positions"]: values[position]=weight
        return np.asarray([values.get(position,0.) for position in range(len(louis))])
    for tau in TAUS:
        w=weights(tau); gated=gated_covariance(louis,stage,w)
        add(f"soft_stageB_logistic_c4_tau{str(tau).replace('.','p')}",gated.covariance,w,gated.diagnostics)
    w=weights(1.,True); gated=gated_covariance(louis,stage,w)
    add("hard_stageB_group_snr_tau1p0",gated.covariance,w,gated.diagnostics)
    damped=globally_damped_covariance(louis,stage,.75)
    add("global_lambda_stageB_0p75",damped.covariance,np.full(len(louis),np.sqrt(.75)),damped.diagnostics)
    return variants


def frames(tables):
    return {key:(pd.concat(value,ignore_index=True) if value else pd.DataFrame()) for key,value in tables.items()}


def load_tables():
    tables={key:[] for key in TABLES}
    for key,name in CHECKPOINTS.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            try: frame=pd.read_csv(path)
            except pd.errors.EmptyDataError: frame=pd.DataFrame()
            if len(frame): tables[key]=[frame]
    return tables


def save_raw(tables):
    current=frames(tables)
    for key,name in CHECKPOINTS.items(): atomic_csv(current[key],os.path.join(RESULTS_DIR,name))
    for key,name in PARTIAL_DIRECT.items(): atomic_csv(current[key],os.path.join(RESULTS_DIR,name))
    return current


def column_path(meta,index):
    directory=os.path.join(RESULTS_DIR,"stageB_column_checkpoints",meta["config_label"],
                           f"network_{meta['true_network_id']}",f"replicate_{meta['replicate_id']}")
    os.makedirs(directory,exist_ok=True)
    return os.path.join(directory,f"direction_{int(index)}.npy")


def atomic_column(path,column):
    temporary=path+".tmp"
    with open(temporary,"wb") as handle: np.save(handle,np.asarray(column,float))
    os.replace(temporary,path)


def latent_metrics(model,data):
    proxy=r1.ridge_proxy(data)
    def corr(estimate):
        return float(np.nanmean([correlations(estimate[:,i],data["x"][:,i]) for i in range(data["x"].shape[1])]))
    return {"smoothed_state_MSE":float(np.mean((model.smoothed_state_mean_-data["x"])**2)),
            "smoothed_state_correlation":corr(model.smoothed_state_mean_),
            "filtered_state_MSE":float(np.mean((model.filtered_state_mean_-data["x"])**2)),
            "filtered_state_correlation":corr(model.filtered_state_mean_),
            "pseudo_inverse_state_MSE":float(np.mean((proxy-data["x"])**2)),
            "pseudo_inverse_state_correlation":corr(proxy)}


def activity_rows(meta,records,blocks):
    rows=[]
    for record in records:
        if record["diagonal"]: continue
        pos=record["positions"]; stage=blocks["lrvb_stageB_smoother_feedback_psd_projected"][np.ix_(pos,pos)]
        snr=float(record["group_snr_louis"])
        rows.append({**meta,"group_target":record["target"],"group_source":record["source"],
                     "true_edge_group":record["true_edge"],"group_norm":record["group_norm"],
                     "group_snr_current":record["group_snr_current"],"group_snr_louis":snr,
                     "group_snr_stageB":np.sqrt(max(record["mean"]@np.linalg.pinv(stage)@record["mean"],0)),
                     "E_alpha":record["E_alpha"],"inverse_alpha_score":record["inverse_alpha_score"],
                     "soft_tau1_weight":logistic_weight(snr,1.,4.),"hard_tau1_selected":snr>=1.,
                     "soft_tau1p5_weight":logistic_weight(snr,1.5,4.),
                     "soft_tau2_weight":logistic_weight(snr,2.,4.),
                     "soft_tau2p5_weight":logistic_weight(snr,2.5,4.),
                     "suppressed_by_soft_tau1":logistic_weight(snr,1.,4.)<.5,
                     "suppressed_by_original_tau2p5":logistic_weight(snr,2.5,4.)<.5,
                     "stageB_over_louis_trace_ratio":record["stageB_over_louis_trace_ratio"],
                     "needed_ratio_if_truth_available":record["needed_variance_ratio_if_truth_available"]})
    return pd.DataFrame(rows)


def run_replicate(cfg,network,replicate,tables):
    meta=meta_for(cfg,network,replicate); data,seed=r1.simulate(cfg["M_x"],cfg["M_y"],cfg["T"],"gaussian_isotropic",network,replicate)
    total_start=time.perf_counter(); baseline_start=time.perf_counter(); model=fit_model(data,seed+500)
    baseline_seconds=time.perf_counter()-baseline_start; louis_start=time.perf_counter(); louis=r1.louis(model,data,seed+700)
    louis_seconds=time.perf_counter()-louis_start; selected,selected_groups=select_directions(model,data,louis,meta,seed+900)
    indices=selected.coefficient_global_index.astype(int).tolist()
    columns={index:np.load(column_path(meta,index)) for index in indices if os.path.exists(column_path(meta,index))}
    pending=[index for index in indices if index not in columns]; diagnostics=[]; stage_start=time.perf_counter()
    if pending:
        print(f"\nStage-B {cfg['label']} net={network} rep={replicate}: {len(pending)} directions")
    completed_count=0; truncated=False
    with ProcessPoolExecutor(max_workers=min(WORKERS,max(len(pending),1))) as pool:
        for offset in range(0,len(pending),WORKERS):
            batch=pending[offset:offset+WORKERS]
            futures={pool.submit(perturb_one,(model,data["y"],data["u"],index)):index for index in batch}
            for future in as_completed(futures):
                index,column,row=future.result(); columns[index]=column; atomic_column(column_path(meta,index),column)
                diagnostics.append({**meta,**row}); completed_count+=1
                prior=frames(tables)["perturb"]; checkpoint=pd.concat([prior,pd.DataFrame(diagnostics)],ignore_index=True)
                atomic_csv(checkpoint,os.path.join(RESULTS_DIR,CHECKPOINTS["perturb"]))
                atomic_csv(checkpoint,os.path.join(RESULTS_DIR,PARTIAL_DIRECT["perturb"]))
                frac=(len(columns))/max(len(indices),1); width=24; fill=round(width*frac)
                print(f"\r  perturbations [{'#'*fill}{'-'*(width-fill)}] {len(columns)}/{len(indices)}",end="",flush=True)
            if time.perf_counter()-stage_start > CAP_SECONDS and offset+len(batch)<len(pending):
                truncated=True; break
    if pending: print()
    valid=[index for index in indices if index in columns and np.all(np.isfinite(columns[index]))]
    if not valid: raise FloatingPointError("No valid selected Stage-B directions were completed.")
    matrix=np.column_stack([columns[index] for index in valid]); raw,psd,stage_projection=project_selected_covariance(matrix,valid)
    selected=selected.set_index("coefficient_global_index").loc[valid].reset_index()
    ordinary=qa.o.global_covariance_block(model.A_row_covariances_)[np.ix_(valid,valid)]
    louis_cov=qa.o.global_covariance_block(louis)[np.ix_(valid,valid)]
    blocks={"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":louis_cov,
            "lrvb_stageB_smoother_feedback_raw":raw,"lrvb_stageB_smoother_feedback_psd_projected":psd}
    records=qa.p.group_records(model,data,selected,{"ordinary_vb":ordinary,
             "stabilized_louis_eta_0p70_tau_0p90":louis_cov,"lrvb_stageB_smoother_feedback":psd})
    lookup=selected_groups.set_index(["target","source"]); column_obs=np.linalg.norm(data["C"],axis=0)
    for variant in covariance_variants(blocks,records):
        coeff=qa.coefficient_rows(model,data,selected,variant,meta)
        for name in ("source_observability_score","target_observability_score","edge_observability_score","observability_quartile"):
            coeff[name]=[lookup.loc[(int(t),int(s)),name] if (int(t),int(s)) in lookup.index else np.nan for t,s in zip(coeff.target,coeff.source)]
        group,_=qa.group_rows(model,records,variant,meta)
        if len(group):
            group["source_observability_score"]=[column_obs[int(s)] for s in group.group_source]
            group["target_observability_score"]=[column_obs[int(t)] for t in group.group_target]
            group["edge_observability_score"]=group.source_observability_score*group.target_observability_score
            group["observability_quartile"]=[lookup.loc[(int(t),int(s)),"observability_quartile"] if (int(t),int(s)) in lookup.index else np.nan for t,s in zip(group.group_target,group.group_source)]
            group["fraction_D2_below_chi2_95_df2"]=group.D2_below_chi2_95_df2.astype(float)
            group["fraction_D2_below_chi2_99_df2"]=group.D2_below_chi2_99_df2.astype(float)
        tables["coeff"].append(coeff); tables["group"].append(group)
        projection=variant["projection"]
        raw_negative=int(np.sum(np.diag(raw)<0)); raw_negative_fraction=raw_negative/max(len(raw),1)
        is_raw=variant["covariance_estimator"]=="lrvb_stageB_smoother_feedback_raw"
        is_stage_psd=variant["covariance_estimator"]=="lrvb_stageB_smoother_feedback_psd_projected"
        tables["covdiag"].append(pd.DataFrame([{**meta,"covariance_estimator":variant["covariance_estimator"],
            "number_selected_directions":len(valid),"raw_and_psd_stageB_differ":bool(np.max(np.abs(raw-psd))>1e-12),
            "number_negative_variances":raw_negative if is_raw else projection.get("number_negative_variances",0),
            "fraction_negative_variances":raw_negative_fraction if is_raw else projection.get("fraction_negative_variances",0.),
            "number_psd_projections":int(stage_projection.get("psd_projection_used",False)) if is_stage_psd else int(projection.get("psd_projection_used",False)),
            "fraction_psd_projected":float(stage_projection.get("psd_projection_used",False)) if is_stage_psd else float(projection.get("psd_projection_used",False)),
            **covariance_diagnostics(variant["covariance"],ordinary,louis_cov,selected.coefficient_type)}]))
    tables["activity"].append(activity_rows(meta,records,blocks)); tables["selected"].append(selected)
    selected_keys=set(zip(selected.target.astype(int),selected.source.astype(int)))
    tables["groupsel"].append(selected_groups[[tuple(x) in selected_keys for x in selected_groups[["target","source"]].to_numpy()]])
    tables["perturb"].append(pd.DataFrame(diagnostics))
    A=qb.a_recovery(model,data,meta); B=qa.B_recovery(model,data,meta); Q=qb.q_recovery(model,data,meta); Q["Q_update_damping"]=Q.get("Q_damping",.5)
    latent={**meta,**latent_metrics(model,data)}; network_frame=qa.full_network(model,data,meta,louis)
    old_M=spi.M; spi.M=cfg["M_x"]
    spectra,bands,_,_,_=spi.compute_spectral_tables(model.A_mean_matrices_,model.B_matrices,model.Q,np.linspace(0,np.pi,128),data,meta,"estimated_AQ")
    spectral_metrics,_=spi.metric_rows(bands); spi.M=old_M
    for key,value in meta.items(): spectral_metrics[key]=value
    deviance=qc.deviance(model,data,meta); stage_seconds=time.perf_counter()-stage_start
    runtime={**meta,"baseline_VB_runtime_seconds":baseline_seconds,"louis_runtime_seconds":louis_seconds,
             "stageB_total_runtime_seconds":stage_seconds,
             "average_perturbed_fit_runtime_seconds":np.mean([row["runtime_seconds"] for row in diagnostics]) if diagnostics else np.nan,
             "number_perturbed_fits":2*len(valid),"number_selected_directions":len(valid),"number_workers":WORKERS,
             "stageB_truncated":bool(truncated or len(valid)<len(indices)),"total_runtime_seconds":time.perf_counter()-total_start,
             "run_status":"success"}
    run={**meta,**r1.c_metrics(data["C"]),**latent,**A,**B,**Q,**runtime}
    additions={"run":pd.DataFrame([run]),"C":pd.DataFrame([{**meta,**r1.c_metrics(data["C"])}]),
               "latent":pd.DataFrame([latent]),"A":pd.DataFrame([A]),"B":pd.DataFrame([B]),"Q":pd.DataFrame([Q]),
               "network":network_frame,"spectral":spectral_metrics,"spectra":bands,"deviance":deviance,
               "runtime":pd.DataFrame([runtime])}
    for key,frame in additions.items(): tables[key].append(frame)
    save_raw(tables)


def calibration_summary(frame):
    keys=["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","coefficient_type"]
    return frame.groupby(keys,dropna=False).agg(
        empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),
        median_signed_error=("signed_error","median"),mean_abs_error=("abs_error","mean"),
        median_abs_error=("abs_error","median"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),
        mean_posterior_sd=("posterior_sd","mean"),median_posterior_sd=("posterior_sd","median"),
        mean_interval_width_95=("interval_width_95","mean"),median_interval_width_95=("interval_width_95","median"),
        mean_standardized_error=("standardized_error","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),
        median_abs_standardized_error=("standardized_error",lambda x:x.abs().median()),n_coefficients=("posterior_sd","size")).reset_index()


def calibration_replicate_summary(frame):
    runkeys=["config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","covariance_estimator","coefficient_type"]
    per=frame.groupby(runkeys,dropna=False).agg(coverage=("ci95_contains_true","mean"),std_z=("standardized_error",lambda x:x.std(ddof=0))).reset_index()
    keys=["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","coefficient_type"]
    out=per.groupby(keys,dropna=False).agg(mean_coverage_across_replicates=("coverage","mean"),std_coverage_across_replicates=("coverage","std"),mean_std_z_across_replicates=("std_z","mean"),std_std_z_across_replicates=("std_z","std"),n_replicates=("replicate_id","size"),n_networks=("true_network_id","nunique")).reset_index()
    out["sem_coverage_across_replicates"]=out.std_coverage_across_replicates/np.sqrt(out.n_replicates)
    out["sem_std_z_across_replicates"]=out.std_std_z_across_replicates/np.sqrt(out.n_replicates)
    return out


def _expanded_group_types(frame):
    if not len(frame): return frame.assign(summary_group_type=pd.Series(dtype=str))
    pieces=[]; base=frame.copy(); base["summary_group_type"]=np.where(base.true_edge_group,"true_edge_group","false_edge_group"); pieces.append(base)
    true=frame[frame.true_edge_group.astype(bool)].copy()
    if len(true):
        med=true.groupby(["config_label","M_y"])["edge_observability_score"].transform("median")
        true["summary_group_type"]=np.where(true.edge_observability_score<=med,"low_observability_true_edge_group","high_observability_true_edge_group"); pieces.append(true)
        quart=true.copy(); quart["summary_group_type"]=quart.observability_quartile; pieces.append(quart)
    return pd.concat(pieces,ignore_index=True)


def group_calibration_summary(frame):
    expanded=_expanded_group_types(frame); keys=["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","summary_group_type"]
    return expanded.groupby(keys,dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),
        fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),
        fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),
        mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("D2","size")).reset_index()


def group_replicate_summary(frame):
    expanded=_expanded_group_types(frame); runkeys=["config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","covariance_estimator","summary_group_type"]
    per=expanded.groupby(runkeys,dropna=False).agg(coverage=("D2_below_chi2_95_df2","mean"),mean_D2=("D2","mean")).reset_index()
    keys=["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","summary_group_type"]
    out=per.groupby(keys,dropna=False).agg(mean_coverage_across_replicates=("coverage","mean"),std_coverage_across_replicates=("coverage","std"),mean_D2_across_replicates=("mean_D2","mean"),std_D2_across_replicates=("mean_D2","std"),n_replicates=("replicate_id","size"),n_networks=("true_network_id","nunique")).reset_index()
    out["sem_coverage_across_replicates"]=out.std_coverage_across_replicates/np.sqrt(out.n_replicates)
    out["sem_D2_across_replicates"]=out.std_D2_across_replicates/np.sqrt(out.n_replicates)
    return out


def observability_summaries(f):
    activity=f["activity"].copy(); group=f["group"].copy(); selected=f["groupsel"].copy()
    if not len(activity): return pd.DataFrame(),pd.DataFrame()
    keys=["config_label","M_y","observation_ratio","true_network_id","replicate_id","group_target","group_source"]
    true=activity[activity.true_edge_group.astype(bool)].copy()
    sel=selected.rename(columns={"target":"group_target","source":"group_source"})
    true=true.merge(sel[keys+["edge_observability_score","observability_quartile"]],on=keys,how="left")
    wanted={"lrvb_stageB_smoother_feedback_psd_projected":"full_StageB_ci_contains_true_group",
            "soft_stageB_logistic_c4_tau1p0":"soft_tau1_ci_contains_true_group",
            "hard_stageB_group_snr_tau1p0":"hard_tau1_ci_contains_true_group",
            "global_lambda_stageB_0p75":"global_lambda_ci_contains_true_group"}
    pivot=group[group.true_edge_group.astype(bool)&group.covariance_estimator.isin(wanted)].pivot_table(index=keys,columns="covariance_estimator",values="D2_below_chi2_95_df2",aggfunc="first").rename(columns=wanted).reset_index()
    true=true.merge(pivot,on=keys,how="left").rename(columns={"stageB_over_louis_trace_ratio":"StageB_over_Louis_variance_ratio"})
    rows=[]
    for values,g in true.groupby(["config_label","M_x","M_y","observation_ratio","T"],dropna=False):
        rows.append({**dict(zip(["config_label","M_x","M_y","observation_ratio","T"],values)),
            "correlation_edge_observability_group_snr_louis":_corr(g.edge_observability_score,g.group_snr_louis),
            "correlation_edge_observability_soft_tau1_weight":_corr(g.edge_observability_score,g.soft_tau1_weight),
            "mean_observability_selected_true_edges_tau1":g.loc[~g.suppressed_by_soft_tau1,"edge_observability_score"].mean(),
            "mean_observability_suppressed_true_edges_tau1":g.loc[g.suppressed_by_soft_tau1,"edge_observability_score"].mean(),
            "fraction_true_edges_suppressed_tau1":g.suppressed_by_soft_tau1.mean(),
            "fraction_true_edges_suppressed_tau2p5":g.suppressed_by_original_tau2p5.mean()})
    quart=true.groupby(["config_label","M_x","M_y","observation_ratio","T","observability_quartile"],dropna=False).agg(
        full_StageB_group_coverage=("full_StageB_ci_contains_true_group","mean"),soft_tau1_group_coverage=("soft_tau1_ci_contains_true_group","mean"),
        hard_tau1_group_coverage=("hard_tau1_ci_contains_true_group","mean"),global_lambda_group_coverage=("global_lambda_ci_contains_true_group","mean"),
        mean_StageB_Louis_variance_ratio=("StageB_over_Louis_variance_ratio","mean"),mean_soft_tau1_weight=("soft_tau1_weight","mean"),
        fraction_true_edges_suppressed_tau1=("suppressed_by_soft_tau1","mean"),fraction_true_edges_suppressed_tau2p5=("suppressed_by_original_tau2p5","mean"),n_true_edges=("true_edge_group","size")).reset_index()
    return true.merge(pd.DataFrame(rows),on=["config_label","M_x","M_y","observation_ratio","T"],how="left"),quart


def _corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    return float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>1 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan


def balance_scores(coeff,group):
    calrep=calibration_replicate_summary(coeff); grep=group_replicate_summary(group); rows=[]
    for (est,my,ratio),g in calrep.groupby(["covariance_estimator","M_y","observation_ratio"],dropna=False):
        by=g.set_index("coefficient_type"); val=lambda typ,col:by.loc[typ,col] if typ in by.index else np.nan
        ac=val("offdiag_nonzero","mean_coverage_across_replicates"); zc=val("offdiag_zero","mean_coverage_across_replicates")
        az=val("offdiag_nonzero","mean_std_z_across_replicates"); zz=val("offdiag_zero","mean_std_z_across_replicates")
        components=[np.clip(1-abs(ac-.95)/.95,0,1),np.clip(1-abs(zc-.95)/.95,0,1),np.clip(1-abs(az-1),0,1),np.clip(1-abs(zz-1),0,1)]
        gg=grep[(grep.covariance_estimator==est)&(grep.M_y==my)].set_index("summary_group_type")
        tc=gg.loc["true_edge_group","mean_coverage_across_replicates"] if "true_edge_group" in gg.index else np.nan
        fc=gg.loc["false_edge_group","mean_coverage_across_replicates"] if "false_edge_group" in gg.index else np.nan
        td=gg.loc["true_edge_group","mean_D2_across_replicates"] if "true_edge_group" in gg.index else np.nan
        variability=gg.loc["true_edge_group","std_coverage_across_replicates"] if "true_edge_group" in gg.index else np.nan
        group_score=np.nanmean([np.clip(1-abs(tc-.95)/.95,0,1),np.clip(1-abs(fc-.95)/.95,0,1),np.clip(1-abs(td-2)/2,0,1)])-.1*np.nan_to_num(variability)
        rows.append({"covariance_estimator":est,"M_y":my,"observation_ratio":ratio,"active_score":components[0],"zero_score":components[1],"active_z_score":components[2],"zero_z_score":components[3],"balance_score":.35*components[0]+.25*components[1]+.25*components[2]+.15*components[3],"true_edge_group_chi2_95_coverage":tc,"false_edge_group_chi2_95_coverage":fc,"true_edge_group_mean_D2":td,"group_replicate_coverage_std":variability,"group_balance_score":group_score})
    out=pd.DataFrame(rows)
    common=out.groupby("covariance_estimator").agg(mean_balance=("balance_score","mean"),range_balance=("balance_score",lambda x:x.max()-x.min()),mean_group_balance=("group_balance_score","mean"),range_group_balance=("group_balance_score",lambda x:x.max()-x.min())).reset_index()
    common["common_score"]=common.mean_balance-.25*common.range_balance;common["common_group_score"]=common.mean_group_balance-.25*common.range_group_balance
    return out.merge(common,on="covariance_estimator",how="left")


def decision_summary(f,calrep,grep,obs,quart,balance):
    rows=[]; estimators=sorted(calrep.covariance_estimator.unique()) if len(calrep) else []
    for estimator in estimators:
        for my in sorted(calrep.M_y.unique()):
            c=calrep[(calrep.covariance_estimator==estimator)&(calrep.M_y==my)].set_index("coefficient_type")
            if not len(c): continue
            ratio=float(c.observation_ratio.iloc[0]); get=lambda typ,col:c.loc[typ,col] if typ in c.index else np.nan
            g=grep[(grep.covariance_estimator==estimator)&(grep.M_y==my)].set_index("summary_group_type")
            gv=lambda typ,col:g.loc[typ,col] if typ in g.index else np.nan
            runs=f["run"][(f["run"].M_y==my)&(f["run"].run_status.eq("success"))]
            network=f["network"][f["network"].M_y==my]; spectral=f["spectral"][(f["spectral"].M_y==my)&(f["spectral"].score_type.eq("transfer_spectral_gc_diagQ"))]
            spectral_means=spectral.groupby("band_name").AUPRC.mean() if len(spectral) else pd.Series(dtype=float)
            best_band=spectral_means.idxmax() if len(spectral_means) else np.nan
            dev=f["deviance"][f["deviance"].M_y==my]; deviance_best=dev.groupby(["signal_type","score_type"]).AUPRC.mean().max() if len(dev) else np.nan
            selected_obs=obs[obs.M_y==my]; b=balance[(balance.covariance_estimator==estimator)&(balance.M_y==my)]
            low=gv("low_observability_true_edge_group","mean_coverage_across_replicates"); high=gv("high_observability_true_edge_group","mean_coverage_across_replicates")
            rows.append({"covariance_estimator":estimator,"M_y":my,"observation_ratio":ratio,
                "n_networks":int(runs.true_network_id.nunique()),"n_replicates":int(len(runs)),
                "smoothed_state_correlation_mean":runs.smoothed_state_correlation.mean(),"smoothed_state_correlation_std":runs.smoothed_state_correlation.std(),
                "A_support_AUPRC_model_A_group_norm_mean":network.loc[network.score_type.eq("model_A_group_norm"),"AUPRC"].mean(),
                "A_support_AUPRC_stabilized_louis_group_snr_mean":network.loc[network.score_type.eq("stabilized_louis_group_snr"),"AUPRC"].mean(),
                "spectral_best_band":best_band,"spectral_best_band_AUPRC_mean":spectral_means.max() if len(spectral_means) else np.nan,
                "deviance_best_AUPRC_mean":deviance_best,"B_relative_frobenius_error_mean":runs.B_relative_frobenius_error.mean(),
                "B_correlation_hat_true_mean":runs.B_correlation_hat_true.mean(),"Q_relative_frobenius_error_mean":runs.Q_relative_frobenius_error.mean(),
                "Q_trace_ratio_hat_to_true_mean":runs.Q_trace_ratio_hat_to_true.mean(),
                "diag_coverage_mean":get("diagonal","mean_coverage_across_replicates"),"diag_coverage_sem":get("diagonal","sem_coverage_across_replicates"),
                "offdiag_nonzero_coverage_mean":get("offdiag_nonzero","mean_coverage_across_replicates"),"offdiag_nonzero_coverage_sem":get("offdiag_nonzero","sem_coverage_across_replicates"),
                "offdiag_zero_coverage_mean":get("offdiag_zero","mean_coverage_across_replicates"),"offdiag_zero_coverage_sem":get("offdiag_zero","sem_coverage_across_replicates"),
                "diag_std_z_mean":get("diagonal","mean_std_z_across_replicates"),"offdiag_nonzero_std_z_mean":get("offdiag_nonzero","mean_std_z_across_replicates"),
                "offdiag_zero_std_z_mean":get("offdiag_zero","mean_std_z_across_replicates"),
                "true_edge_group_chi2_95_coverage_mean":gv("true_edge_group","mean_coverage_across_replicates"),
                "true_edge_group_chi2_95_coverage_sem":gv("true_edge_group","sem_coverage_across_replicates"),
                "false_edge_group_chi2_95_coverage_mean":gv("false_edge_group","mean_coverage_across_replicates"),
                "false_edge_group_chi2_95_coverage_sem":gv("false_edge_group","sem_coverage_across_replicates"),
                "low_observability_true_edge_coverage_mean":low,"high_observability_true_edge_coverage_mean":high,
                "fraction_true_edges_suppressed_tau1":selected_obs.suppressed_by_soft_tau1.mean() if len(selected_obs) else np.nan,
                "fraction_true_edges_suppressed_tau2p5":selected_obs.suppressed_by_original_tau2p5.mean() if len(selected_obs) else np.nan,
                "balance_score":b.balance_score.mean(),"group_balance_score":b.group_balance_score.mean(),"common_score":b.common_score.mean(),
                "total_runtime_seconds":runs.total_runtime_seconds.sum(),"stageB_truncated_fraction":runs.stageB_truncated.mean(),
                "recommended_for_rectangular_C_uncertainty":False,"recommended_for_full_pipeline":False,"notes":"A_center=A_VB; selected-direction Stage-B; B/Q re-estimated."})
    out=pd.DataFrame(rows)
    if not len(out): return out
    primary="soft_stageB_logistic_c4_tau1p0"; louis="stabilized_louis_eta_0p70_tau_0p90"; old="soft_stageB_logistic_c4_tau2p5"; full="lrvb_stageB_smoother_feedback_psd_projected"
    for my in out.M_y.unique():
        q=out[out.M_y==my].set_index("covariance_estimator")
        if not {primary,louis,old,full}.issubset(q.index): continue
        p,L,O,F=q.loc[primary],q.loc[louis],q.loc[old],q.loc[full]
        success=(p.offdiag_nonzero_coverage_mean>L.offdiag_nonzero_coverage_mean and
                 abs(p.offdiag_nonzero_coverage_mean-.925)<abs(O.offdiag_nonzero_coverage_mean-.925) and
                 (p.offdiag_zero_coverage_mean<=.98 or p.offdiag_zero_coverage_mean<=F.offdiag_zero_coverage_mean) and
                 abs(p.offdiag_nonzero_std_z_mean-1)<abs(O.offdiag_nonzero_std_z_mean-1) and
                 p.offdiag_nonzero_coverage_sem<=.15 and
                 p.low_observability_true_edge_coverage_mean>=O.low_observability_true_edge_coverage_mean)
        mask=(out.M_y==my)&out.covariance_estimator.eq(primary);out.loc[mask,"recommended_for_rectangular_C_uncertainty"]=success
        out.loc[mask,"recommended_for_full_pipeline"]=bool(success and p.smoothed_state_correlation_mean>=.7 and p.spectral_best_band_AUPRC_mean>=.5)
        out.loc[mask,"notes"] += f" primary_success={success}."
    return out


def build_outputs(tables):
    f=frames(tables); coeff=f["coeff"]; group=f["group"]
    cal=calibration_summary(coeff) if len(coeff) else pd.DataFrame(); calrep=calibration_replicate_summary(coeff) if len(coeff) else pd.DataFrame()
    groups=group_calibration_summary(group) if len(group) else pd.DataFrame(); greprep=group_replicate_summary(group) if len(group) else pd.DataFrame()
    obs,quart=observability_summaries(f); balance=balance_scores(coeff,group) if len(coeff) and len(group) else pd.DataFrame()
    decision=decision_summary(f,calrep,greprep,obs,quart,balance) if len(calrep) else pd.DataFrame()
    return f,{"run_summary.csv":f["run"],"C_diagnostics.csv":f["C"],"selected_groups.csv":f["groupsel"],
        "selected_coefficients.csv":f["selected"],"coefficient_calibration_results.csv":coeff,"calibration_summary.csv":cal,
        "calibration_replicate_summary.csv":calrep,"group_calibration_results.csv":group,"group_calibration_summary.csv":groups,
        "group_calibration_replicate_summary.csv":greprep,"covariance_diagnostics.csv":f["covdiag"],
        "stageB_perturbation_diagnostics.csv":f["perturb"],"activity_score_diagnostics.csv":f["activity"],
        "observability_stageB_summary.csv":obs,"observability_quartile_summary.csv":quart,
        "latent_state_recovery_summary.csv":f["latent"],"A_recovery_summary.csv":f["A"],"B_recovery_summary.csv":f["B"],
        "Q_recovery_summary.csv":f["Q"],"network_recovery_summary.csv":f["network"],"spectral_gc_summary.csv":f["spectra"],
        "spectral_band_recovery_summary.csv":f["spectral"],"deviance_diagnostics.csv":f["deviance"],
        "balance_score_summary.csv":balance,"runtime_summary.csv":f["runtime"],"decision_summary.csv":decision}


PARTIAL_SUMMARIES={
    "run_summary.csv":"run_summary_partial.csv","calibration_summary.csv":"calibration_summary_partial.csv",
    "calibration_replicate_summary.csv":"calibration_replicate_summary_partial.csv",
    "group_calibration_summary.csv":"group_calibration_summary_partial.csv",
    "group_calibration_replicate_summary.csv":"group_calibration_replicate_summary_partial.csv",
    "covariance_diagnostics.csv":"covariance_diagnostics_partial.csv",
    "stageB_perturbation_diagnostics.csv":"stageB_perturbation_diagnostics_partial.csv",
    "observability_stageB_summary.csv":"observability_stageB_summary_partial.csv",
    "observability_quartile_summary.csv":"observability_quartile_summary_partial.csv",
    "latent_state_recovery_summary.csv":"latent_state_recovery_summary_partial.csv","A_recovery_summary.csv":"A_recovery_summary_partial.csv",
    "B_recovery_summary.csv":"B_recovery_summary_partial.csv","Q_recovery_summary.csv":"Q_recovery_summary_partial.csv",
    "network_recovery_summary.csv":"network_recovery_summary_partial.csv","spectral_gc_summary.csv":"spectral_gc_summary_partial.csv",
    "deviance_diagnostics.csv":"deviance_diagnostics_partial.csv","balance_score_summary.csv":"balance_score_summary_partial.csv",
    "runtime_summary.csv":"runtime_summary_partial.csv","decision_summary.csv":"decision_summary_partial.csv"}


def save_summaries(tables,final=False):
    f,outputs=build_outputs(tables)
    if final:
        for name,frame in outputs.items(): atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    for source,target in PARTIAL_SUMMARIES.items(): atomic_csv(outputs[source],os.path.join(RESULTS_DIR,target))
    return f,outputs


def _save_figure(path,name):
    plt.tight_layout(); plt.savefig(os.path.join(path,name),dpi=160); plt.close()


def _bar(frame,index,value,path,name,ylabel=None):
    if not len(frame) or value not in frame:
        plt.text(.5,.5,"No completed data",ha="center",va="center"); plt.axis("off"); _save_figure(path,name); return
    series=frame.groupby(index,dropna=False)[value].mean(); series.plot.bar()
    plt.ylabel(ylabel or value); plt.xticks(rotation=25,ha="right"); _save_figure(path,name)


def make_plots(f,outputs):
    path=os.path.join(RESULTS_DIR,"plots"); os.makedirs(path,exist_ok=True)
    crep=outputs["calibration_replicate_summary.csv"]; grep=outputs["group_calibration_replicate_summary.csv"]
    balance=outputs["balance_score_summary.csv"]; obs=outputs["observability_stageB_summary.csv"]; quart=outputs["observability_quartile_summary.csv"]
    _bar(crep[crep.coefficient_type.eq("offdiag_nonzero")],["M_y","covariance_estimator"],"mean_coverage_across_replicates",path,"01_active_coverage.png")
    _bar(crep[crep.coefficient_type.eq("offdiag_zero")],["M_y","covariance_estimator"],"mean_coverage_across_replicates",path,"02_zero_coverage.png")
    _bar(crep[crep.coefficient_type.eq("offdiag_nonzero")],["M_y","covariance_estimator"],"mean_std_z_across_replicates",path,"03_active_std_z.png")
    _bar(crep[crep.coefficient_type.eq("offdiag_zero")],["M_y","covariance_estimator"],"mean_std_z_across_replicates",path,"04_zero_std_z.png")
    _bar(grep[grep.summary_group_type.eq("true_edge_group")],["M_y","covariance_estimator"],"mean_coverage_across_replicates",path,"05_true_group_coverage.png")
    _bar(grep[grep.summary_group_type.eq("false_edge_group")],["M_y","covariance_estimator"],"mean_coverage_across_replicates",path,"06_false_group_coverage.png")
    _bar(balance,["M_y","covariance_estimator"],"balance_score",path,"07_balance_score.png")
    _bar(balance,"covariance_estimator","common_score",path,"08_common_score.png")
    lowhigh=grep[grep.summary_group_type.isin(["low_observability_true_edge_group","high_observability_true_edge_group"])]
    _bar(lowhigh,["M_y","covariance_estimator","summary_group_type"],"mean_coverage_across_replicates",path,"09_low_high_observability_coverage.png")
    if len(obs):
        suppression=obs.groupby("M_y")[["suppressed_by_soft_tau1","suppressed_by_original_tau2p5"]].mean(); suppression.plot.bar()
        plt.ylabel("fraction true edges suppressed"); plt.xticks(rotation=0); _save_figure(path,"10_suppression_tau1_vs_tau2p5.png")
    _bar(quart,["M_y","observability_quartile"],"mean_StageB_Louis_variance_ratio",path,"11_stageB_louis_ratio_by_observability.png")
    _bar(quart,["M_y","observability_quartile"],"mean_soft_tau1_weight",path,"12_soft_tau1_weight_by_observability.png")
    _bar(f["latent"],"M_y","smoothed_state_correlation",path,"13_smoothed_state_correlation.png")
    spectral=f["spectral"]; spectral=spectral[spectral.score_type.eq("transfer_spectral_gc_diagQ")] if len(spectral) else spectral
    _bar(spectral,["M_y","band_name"],"AUPRC",path,"14_spectral_AUPRC.png")
    _bar(f["network"],["M_y","score_type"],"AUPRC",path,"15_A_support_AUPRC.png")
    runtime=f["runtime"]
    if len(runtime):
        runtime.groupby("M_y")[["baseline_VB_runtime_seconds","louis_runtime_seconds","stageB_total_runtime_seconds"]].mean().plot.bar()
        plt.ylabel("seconds"); plt.xticks(rotation=0); _save_figure(path,"16_runtime_breakdown.png")
        q=runtime.copy(); q["stageB_seconds_per_selected_direction"]=q.stageB_total_runtime_seconds/np.maximum(q.number_selected_directions,1)
        _bar(q,"M_y","stageB_seconds_per_selected_direction",path,"17_stageB_runtime_per_direction.png")


def validation_conclusion(decision):
    if not len(decision):
        print("34R Stage 2C ended without enough completed runs for a decision."); return
    names={"soft":"soft_stageB_logistic_c4_tau1p0","hard":"hard_stageB_group_snr_tau1p0",
           "global":"global_lambda_stageB_0p75","full":"lrvb_stageB_smoother_feedback_psd_projected"}
    mean=decision.groupby("covariance_estimator").agg(balance=("balance_score","mean"),group=("group_balance_score","mean"),active=("offdiag_nonzero_coverage_mean","mean")).fillna(-np.inf)
    soft_valid=bool(decision[decision.covariance_estimator.eq(names["soft"])].recommended_for_rectangular_C_uncertainty.all()) if names["soft"] in mean.index else False
    per=decision.set_index(["covariance_estimator","M_y"])
    regimes=sorted(decision.M_y.unique())
    hard_better=all((names["hard"],my) in per.index and (names["soft"],my) in per.index and per.loc[(names["hard"],my),"balance_score"]>per.loc[(names["soft"],my),"balance_score"] for my in regimes)
    global_better=all((names["global"],my) in per.index and per.loc[(names["global"],my),"balance_score"]>max(per.loc[(names["soft"],my),"balance_score"],per.loc[(names["hard"],my),"balance_score"]) for my in regimes)
    full_needed=names["full"] in mean.index and mean.loc[names["full"],"group"]>max(mean.loc[names["soft"],"group"],mean.loc[names["hard"],"group"])+.05
    primary=decision[decision.covariance_estimator.eq(names["soft"])].set_index("M_y")
    my40=40 in primary.index and bool(primary.loc[40,"recommended_for_rectangular_C_uncertainty"])
    my15=15 in primary.index and bool(primary.loc[15,"recommended_for_rectangular_C_uncertainty"])
    low_failure=bool((primary.low_observability_true_edge_coverage_mean<primary.high_observability_true_edge_coverage_mean-.10).any()) if len(primary) else True
    action="adopt soft tau=1.0" if soft_valid else ("adopt hard tau=1.0" if hard_better and not global_better else "use global lambda" if global_better else "move to observability-aware gating" if low_failure else "use full selected-direction Stage-B" if full_needed else "improve latent recovery")
    print("\n34R Stage 2C conclusion")
    print(f"1. Soft tau=1.0 validates across regimes: {soft_valid}")
    print(f"2. Hard tau=1.0 consistently beats soft tau=1.0: {hard_better}")
    print(f"3. Global lambda=0.75 is better than SNR gating: {global_better}")
    print(f"4. Full selected-direction Stage-B remains necessary: {full_needed}")
    print(f"5. M_y=40 positive control validates: {my40}")
    print(f"6. M_y=15 is usable: {my15}")
    print(f"7. Low-observability true edges remain a main failure mode: {low_failure}")
    print(f"8. Recommended next step: {action}")


def experiment_config():
    return {"experiment":"34R_stage2C_validate_rectangular_C_tau1","CONFIG_LIST":CONFIGS,
        "M_x":20 if not SMOKE else 5,"na":2,"nb":3,"n_inputs":1,"T":1000 if not SMOKE else 200,
        "Q_true":"0.50 I_Mx","R_true":"0.60 I_My","C_known":True,"R_fixed_true":True,
        "N_TRUE_NETWORKS":N_TRUE_NETWORKS,"N_REPLICATES_PER_NETWORK":N_REPLICATES_PER_NETWORK,
        "execution_order":"alternate M_y within replicate within true network","N_WORKERS":WORKERS,
        "inner_threads":int(os.environ.get("EXPERIMENT_34R_STAGE2C_INNER_THREADS","1")),"N_FFBS_SAMPLES":50,
        "FINITE_DIFF_EPS_PRIMARY":EPS,"USE_CENTRAL_DIFFERENCE":True,"PERTURBED_VB_MAX_ITER":PERTURB_MAX,
        "PERTURBED_RESCUE_MAX_ITER":RESCUE_MAX,"VB_MAX_ITER":VB_MAX_ITER,"VB_CONVERGENCE_TOL":TOL,
        "PRACTICAL_CONVERGENCE_TOL":TOL,"MAX_STAGEB_SECONDS_PER_CONFIG_REPLICATE":CAP_SECONDS,
        "selected_direction_limit":SELECT_LIMIT,"A_center":"A_VB","B_REESTIMATED_IN_PERTURBED_FITS":True,
        "Q_REESTIMATED_IN_PERTURBED_FITS":True,"soft_tau_grid":list(TAUS),"primary_soft_tau":1.0,
        "logistic_slope":4.0,"Louis_eta":.70,"Louis_tau":.90,"global_lambda":.75,
        "a0":1e-3,"b0":1e-3,"Q_SHRINKAGE_RHO":.25,"Q_UPDATE_DAMPING":.5,
        "C_generation":"iid N(0,1/M_y), then Stage-1 median-column-norm normalization","smoke_test":SMOKE}


def write_or_validate_config(config):
    path=os.path.join(RESULTS_DIR,"experiment_config.json")
    if os.path.exists(path):
        with open(path,encoding="utf8") as handle: previous=json.load(handle)
        operational={"N_WORKERS","inner_threads"}
        previous_numerical={key:value for key,value in previous.items() if key not in operational}
        current_numerical={key:value for key,value in config.items() if key not in operational}
        if previous_numerical != current_numerical:
            raise RuntimeError("Existing Stage 2C checkpoint configuration does not match this run. Use a new results directory or restore the original environment settings.")
        return
    temporary=path+".tmp"
    with open(temporary,"w",encoding="utf8") as handle: json.dump(config,handle,indent=2)
    os.replace(temporary,path)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True); write_or_validate_config(experiment_config()); tables=load_tables(); current=frames(tables)
    successful=current["run"][current["run"].run_status.eq("success")] if len(current["run"]) else pd.DataFrame()
    done=set(zip(successful.config_label,successful.true_network_id.astype(int),successful.replicate_id.astype(int))) if len(successful) else set()
    tasks=[(cfg,network,replicate) for network in range(N_TRUE_NETWORKS) for replicate in range(N_REPLICATES_PER_NETWORK) for cfg in CONFIGS]
    pending=[task for task in tasks if (task[0]["label"],task[1],task[2]) not in done]; progress=Progress(len(pending)); pair_count=0
    for cfg,network,replicate in pending:
        try:
            run_replicate(cfg,network,replicate,tables)
        except Exception as error:
            meta=meta_for(cfg,network,replicate); failure={**meta,"run_status":"failed","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()}
            tables["run"].append(pd.DataFrame([failure])); tables["runtime"].append(pd.DataFrame([failure])); save_raw(tables)
        _,partial=save_summaries(tables,final=False); pair_count+=1
        if pair_count%len(CONFIGS)==0: atomic_csv(partial["decision_summary.csv"],os.path.join(RESULTS_DIR,"interim_validation_summary.csv"))
        progress.update(f"My={cfg['M_y']} net={network} rep={replicate}")
    f,outputs=save_summaries(tables,final=True)
    if not SMOKE and len(outputs["decision_summary.csv"]): make_plots(f,outputs)
    validation_conclusion(outputs["decision_summary.csv"])


if __name__ == "__main__":
    main()
