"""Experiment 34O: LRVB Stage-B Kalman-smoother feedback pilot."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34O_BLAS_THREADS", "1")

import json
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34n_lrvb_A_covariance as n
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats
from src.stats.lrvb_stageB_smoother_feedback import (
    decode_global_index, flatten_A, global_index, project_selected_covariance,
    sensitivity_column,
)

FULL_M5 = os.environ.get("EXPERIMENT_34O_FULL_M5", "0") == "1"
CONFIG_LIST = [
    {"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 2, "N_REPLICATES_PER_NETWORK": 5, "covariance_mode": "full" if FULL_M5 else "selected"},
    {"label": "diagnostic_M20_subset", "M": 20, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 1, "N_REPLICATES_PER_NETWORK": 3, "covariance_mode": "selected"},
]
FINITE_DIFF_EPS_GRID = [3e-5, 1e-4, 3e-4]
FINITE_DIFF_EPS_PRIMARY = 1e-4
PERTURBED_VB_MAX_ITER = 50
PERTURBED_CONVERGENCE_TOL = 1e-4
BASE_SEED = 5550000
SMOKE_TEST = os.environ.get("EXPERIMENT_34O_SMOKE", "0") == "1"
if SMOKE_TEST:
    CONFIG_LIST = [{"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000], "N_TRUE_NETWORKS": 1, "N_REPLICATES_PER_NETWORK": 1, "covariance_mode": "selected"}]
    FINITE_DIFF_EPS_GRID = [1e-4]
RESULTS_DIR = os.environ.get("EXPERIMENT_34O_RESULTS_DIR", "results/experiment_34o")
STAGE_B = "stageB_smoother_feedback"


class ProgressBar:
    def __init__(self, total, width=30): self.total, self.width, self.done = max(int(total), 0), width, 0
    def update(self, label):
        self.done += 1; fraction = min(self.done / max(self.total, 1), 1.); filled = round(self.width * fraction)
        print(f"\rExperiment 34O [{'#'*filled}{'-'*(self.width-filled)}] {self.done}/{self.total} "
              f"({100*fraction:5.1f}%) {label[:58]}", end="\n" if self.done >= self.total else "", flush=True)


def coefficient_kind(data, target, lag, source):
    if target == source: return "diagonal"
    return "offdiag_nonzero" if np.asarray(data["A"])[lag, target, source] != 0 else "offdiag_zero"


def select_directions(data, mode, seed, smoke_test=None):
    M = np.asarray(data["A"]).shape[1]; rng = np.random.default_rng(seed); rows = []
    smoke_test = SMOKE_TEST if smoke_test is None else bool(smoke_test)
    if mode == "full":
        indices = range(j.na * M * M)
        for index in indices:
            target, lag, source = decode_global_index(index, M, j.na)
            rows.append({"coefficient_global_index": index, "target": target, "lag": lag + 1, "source": source,
                "coefficient_type": coefficient_kind(data, target, lag, source), "selected_direction_type": "M5_full"})
        return pd.DataFrame(rows)
    if smoke_test:
        target = int(np.argmax(np.asarray(data["mask"]).sum(axis=1))); chosen = set()
        for lag in range(j.na): chosen.add(global_index(target, lag, target, M, j.na))
        for source in np.flatnonzero(data["mask"][target]):
            for lag in range(j.na): chosen.add(global_index(target, lag, int(source), M, j.na))
        zero = [global_index(target, lag, source, M, j.na) for lag in range(j.na) for source in range(M) if source != target and not data["mask"][target, source]]
        chosen.update(rng.choice(zero, min(5, len(zero)), replace=False).tolist())
        type_lookup = {index: "smoke_target_row" for index in chosen}
    else:
        chosen = set(); type_lookup = {}
        for target in range(M):
            for lag in range(j.na):
                index = global_index(target, lag, target, M, j.na); chosen.add(index); type_lookup[index] = "diagonal"
        true_groups = [(target, source) for target in range(M) for source in range(M) if target != source and data["mask"][target, source]]
        false_groups = [(target, source) for target in range(M) for source in range(M) if target != source and not data["mask"][target, source]]
        matched = [false_groups[index] for index in rng.choice(len(false_groups), min(len(true_groups), len(false_groups)), replace=False)]
        for group_type, groups in (("true_edge_group", true_groups), ("matched_false_edge_group", matched)):
            for target, source in groups:
                for lag in range(j.na):
                    index = global_index(target, lag, source, M, j.na); chosen.add(index); type_lookup[index] = group_type
    for index in sorted(chosen):
        target, lag, source = decode_global_index(index, M, j.na)
        rows.append({"coefficient_global_index": index, "target": target, "lag": lag + 1, "source": source,
            "coefficient_type": coefficient_kind(data, target, lag, source), "selected_direction_type": type_lookup[index]})
    return pd.DataFrame(rows)


def epsilon_directions(selected, M):
    if len(FINITE_DIFF_EPS_GRID) == 1: return set(selected.coefficient_global_index)
    result = []
    for kind in ("diagonal", "offdiag_nonzero", "offdiag_zero"):
        candidates = selected.loc[selected.coefficient_type == kind, "coefficient_global_index"].tolist()
        result.extend(candidates[:(5 if M == 5 else 4)])
    return set(result[:15 if M == 5 else 10])


def global_covariance_block(row_covariances):
    M = len(row_covariances); d = row_covariances[0].shape[0]; result = np.zeros((M*d, M*d))
    for target, covariance in enumerate(row_covariances): result[target*d:(target+1)*d, target*d:(target+1)*d] = covariance
    return result


def run_stage_b(model, data, selected, meta):
    total_dimension = j.na * model.n_states * model.n_states; indices = selected.coefficient_global_index.astype(int).tolist()
    columns = np.full((total_dimension, len(indices)), np.nan); perturb_rows = []; epsilon_rows = []; primary_cache = {}; fit_times = []
    eps_directions = epsilon_directions(selected, model.n_states)
    for column_number, index in enumerate(indices):
        eps_values = FINITE_DIFF_EPS_GRID if index in eps_directions else [FINITE_DIFF_EPS_PRIMARY]
        estimates = []
        for epsilon in eps_values:
            try:
                column, plus, minus = sensitivity_column(model, data["y"], data["u"], index, epsilon,
                    PERTURBED_VB_MAX_ITER, PERTURBED_CONVERGENCE_TOL)
                valid = bool(np.all(np.isfinite(column))); warning = ";".join(filter(None, [plus.warning_flag, minus.warning_flag]))
                if epsilon == FINITE_DIFF_EPS_PRIMARY: columns[:, column_number] = column; primary_cache[index] = column
                fit_times.extend([plus.runtime_seconds, minus.runtime_seconds]); target, lag, source = decode_global_index(index, model.n_states, j.na)
                variance = column[index]; estimates.append(variance)
                perturb_rows.append({**meta, "coefficient_global_index": index, "lag": lag + 1, "target": target, "source": source,
                    "coefficient_type": coefficient_kind(data, target, lag, source), "eps": epsilon,
                    "theta_plus_j": plus.theta[index], "theta_minus_j": minus.theta[index], "baseline_theta_j": flatten_A(model.A_mean_matrices_)[index],
                    "variance_estimate_raw": variance, "covariance_column_norm": np.linalg.norm(column),
                    "relative_column_norm": np.linalg.norm(column) / max(np.linalg.norm(flatten_A(model.A_mean_matrices_)), 1e-15),
                    "max_abs_response": np.max(np.abs(column)), "response_self_fraction": abs(variance) / max(np.sum(np.abs(column)), 1e-15),
                    "perturbed_plus_converged": plus.converged, "perturbed_minus_converged": minus.converged,
                    "plus_n_iter": plus.n_iter, "minus_n_iter": minus.n_iter, "plus_final_relative_A_change": plus.final_relative_A_change,
                    "minus_final_relative_A_change": minus.final_relative_A_change, "plus_loglikelihood": plus.final_loglikelihood,
                    "minus_loglikelihood": minus.final_loglikelihood, "finite_difference_valid": valid, "warning_flag": warning})
            except Exception as error:
                estimates.append(np.nan); perturb_rows.append({**meta, "coefficient_global_index": index, "eps": epsilon,
                    "finite_difference_valid": False, "warning_flag": f"{type(error).__name__}: {error}"})
        finite = np.asarray(estimates)[np.isfinite(estimates)]; cv = np.std(finite) / max(abs(np.mean(finite)), 1e-15) if len(finite) > 1 else 0.
        for epsilon, estimate in zip(eps_values, estimates): epsilon_rows.append({**meta, "coefficient_global_index": index, "eps": epsilon,
            "variance_by_eps": estimate, "coefficient_of_variation_across_eps": cv, "selected_eps": FINITE_DIFF_EPS_PRIMARY,
            "eps_stability_flag": bool(cv < .20)})
    valid_columns = np.flatnonzero(np.all(np.isfinite(columns), axis=0)); valid_indices = [indices[position] for position in valid_columns]
    if not valid_indices: raise FloatingPointError("No valid Stage-B sensitivity columns.")
    raw, psd, projection = project_selected_covariance(columns[:, valid_columns], valid_indices)
    perturb_frame = pd.DataFrame(perturb_rows); psd_variance = dict(zip(valid_indices, np.diag(psd)))
    if len(perturb_frame):
        perturb_frame["variance_estimate_psd"] = [psd_variance.get(int(index), np.nan) if eps == FINITE_DIFF_EPS_PRIMARY else np.nan for index, eps in zip(perturb_frame.coefficient_global_index, perturb_frame.eps)]
    return valid_indices, raw, psd, perturb_frame, pd.DataFrame(epsilon_rows), projection, fit_times


def coefficient_rows(model, data, selected, covariance_blocks, meta):
    rows = []; theta = flatten_A(model.A_mean_matrices_); truth = flatten_A(data["A"]); indices = selected.coefficient_global_index.astype(int).tolist()
    for estimator, block in covariance_blocks.items():
        for position, index in enumerate(indices):
            item = selected.iloc[position]; variance = block[position, position]
            sd = np.sqrt(variance) if variance > 0 else np.nan; error = theta[index] - truth[index]
            rows.append({**meta, "coefficient_global_index": index, "target": item.target, "lag": item.lag, "source": item.source,
                "coefficient_type": item.coefficient_type, "selected_direction_type": item.selected_direction_type,
                "covariance_estimator": estimator, "lrvb_stage": STAGE_B if "stageB" in estimator else (n.LRVB_STAGE if "stageA" in estimator else "not_applicable"),
                "posterior_center": theta[index], "posterior_variance": variance, "posterior_sd": sd, "true_value": truth[index],
                "signed_error": error, "abs_error": abs(error), "squared_error": error**2, "standardized_error": error/sd if sd > 0 else np.nan,
                "ci95_lower": theta[index]-1.96*sd if sd > 0 else np.nan, "ci95_upper": theta[index]+1.96*sd if sd > 0 else np.nan,
                "ci95_contains_true": abs(error) <= 1.96*sd if sd > 0 else False})
    return pd.DataFrame(rows)


def group_and_score_rows(model, data, selected, covariance_blocks, meta):
    positions = {int(index): position for position, index in enumerate(selected.coefficient_global_index)}; groups = []; scores = []; M = model.n_states
    mean = np.asarray(model.A_mean_matrices_); truth = np.asarray(data["A"])
    for target in range(M):
        for source in range(M):
            if target == source: continue
            indices = [global_index(target, lag, source, M, j.na) for lag in range(j.na)]
            if not all(index in positions for index in indices): continue
            pos = [positions[index] for index in indices]; estimate = mean[:, target, source]; actual = truth[:, target, source]
            edge_type = "true_edge_group" if np.linalg.norm(actual) > 0 else "false_edge_group"
            for estimator, covariance in covariance_blocks.items():
                block = covariance[np.ix_(pos, pos)]; difference = estimate-actual; D2 = difference@np.linalg.pinv(block)@difference; snr=np.sqrt(max(estimate@np.linalg.pinv(block)@estimate,0))
                common={**meta,"target":target,"source":source,"edge_group_type":edge_type,"true_link":edge_type=="true_edge_group","covariance_estimator":estimator,"lrvb_stage":STAGE_B if "stageB" in estimator else (n.LRVB_STAGE if "stageA" in estimator else "not_applicable")}
                groups.append({**common,"mahalanobis_D2":D2,"D2_below_chi2_95_df2":D2<=5.991464547,"D2_below_chi2_99_df2":D2<=9.210340372,"group_posterior_sd_trace":np.trace(block),"group_error_norm":np.linalg.norm(difference)})
                scores.append({**common,"score_type":"group_snr","score":snr})
            scores.append({**meta,"target":target,"source":source,"edge_group_type":edge_type,"true_link":edge_type=="true_edge_group","covariance_estimator":"not_applicable","lrvb_stage":"not_applicable","score_type":"model_A_group_norm","score":np.linalg.norm(estimate)})
    return pd.DataFrame(groups),pd.DataFrame(scores)


def calibration_summary(frame):
    keys=["config_label","M","T","covariance_estimator","lrvb_stage","coefficient_type","selected_direction_type"];rows=[]
    for values,g in frame.groupby(keys,dropna=False):rows.append({**dict(zip(keys,values)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"n_coefficients":len(g)})
    return pd.DataFrame(rows)


def group_summary(frame):
    keys=["config_label","M","T","covariance_estimator","lrvb_stage","edge_group_type"]
    if frame.empty:return pd.DataFrame(columns=keys)
    return frame.groupby(keys,dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2","mean"),median_mahalanobis_D2=("mahalanobis_D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_posterior_sd_trace=("group_posterior_sd_trace","mean"),mean_group_error_norm=("group_error_norm","mean"),n_groups=("mahalanobis_D2","size")).reset_index()


def network_summary(frame):
    keys=["config_label","M","T","covariance_estimator","lrvb_stage","score_type"];rows=[]
    for values,g in frame.groupby(keys,dropna=False):
        metrics,_=j.safe_curve(g.true_link,g.score);rows.append({**dict(zip(keys,values)),**metrics})
    return pd.DataFrame(rows)


def empirical_variance(frame):
    keys=["config_label","true_network_id","M","T","coefficient_global_index","covariance_estimator","coefficient_type"];rows=[]
    for values,g in frame.groupby(keys,dropna=False):
        empirical=g.signed_error.var(ddof=1) if len(g)>1 else g.squared_error.mean();posterior=g.posterior_variance.mean()
        rows.append({**dict(zip(keys,values)),"posterior_variance":posterior,"empirical_error_variance":empirical,"variance_ratio_posterior_to_empirical":posterior/max(empirical,1e-15),"posterior_sd":np.sqrt(max(posterior,0)),"empirical_sd":np.sqrt(max(empirical,0)),"sd_ratio_posterior_to_empirical":np.sqrt(max(posterior,0))/max(np.sqrt(max(empirical,0)),1e-15),"n_replicates":len(g)})
    return pd.DataFrame(rows)


def covariance_diagnostic(meta, selected, raw, psd, base_blocks, projection):
    diagonal=np.diag(raw);kinds=selected.coefficient_type.to_numpy();safe=np.sqrt(np.maximum(np.diag(psd),0));louis=np.sqrt(np.maximum(np.diag(base_blocks["stabilized_louis_eta_0p70_tau_0p90"]),0));row={**meta,"covariance_mode":"full" if len(selected)==j.na*meta["M"]**2 else "selected","number_selected_directions":len(selected),"number_negative_variances":int(np.sum(diagonal<0)),"fraction_negative_variances":np.mean(diagonal<0),"mean_variance":np.mean(diagonal),"median_variance":np.median(diagonal),"min_variance":np.min(diagonal),"max_variance":np.max(diagonal),"trace_stageB_selected":np.trace(psd),"trace_stageB_over_vb_selected":np.trace(psd)/max(np.trace(base_blocks["ordinary_vb"]),1e-15),"trace_stageB_over_louis_selected":np.trace(psd)/max(np.trace(base_blocks["stabilized_louis_eta_0p70_tau_0p90"]),1e-15),"symmetry_error_M5_full":projection["symmetry_error"] if meta["M"]==5 and len(selected)==j.na*meta["M"]**2 else np.nan,"min_eigenvalue_M5_full_raw":projection["min_eigenvalue_raw"] if meta["M"]==5 and len(selected)==j.na*meta["M"]**2 else np.nan,"psd_projection_used_M5_full":projection["psd_projection_used"] if meta["M"]==5 and len(selected)==j.na*meta["M"]**2 else np.nan}
    for kind in ("diagonal","offdiag_nonzero","offdiag_zero"):row[f"mean_stageB_sd_over_louis_sd_{kind}"]=np.mean(safe[kinds==kind]/np.maximum(louis[kinds==kind],1e-15)) if np.any(kinds==kind) else np.nan
    return row


def empirical_covariance_m5(covstore):
    rows=[]
    for values,g in covstore.loc[(covstore.M==5)&(covstore.covariance_mode=="full")].groupby(["config_label","true_network_id","T","covariance_estimator"],dropna=False):
        errors=np.asarray([json.loads(value) for value in g.error_vector_json]);posterior=np.mean(np.asarray([json.loads(value) for value in g.covariance_json]),axis=0);emp=np.cov(errors,rowvar=False,ddof=1) if len(errors)>1 else np.diag(errors[0]**2);rows.append({"config_label":values[0],"true_network_id":values[1],"M":5,"T":values[2],"covariance_estimator":values[3],"trace_posterior_cov":np.trace(posterior),"trace_empirical_cov":np.trace(emp),"trace_ratio_posterior_to_empirical":np.trace(posterior)/max(np.trace(emp),1e-15),"frobenius_difference_to_empirical":np.linalg.norm(posterior-emp),"n_replicates":len(errors)})
    return pd.DataFrame(rows)


def decision_summary(cal,groups,selected_edges,empirical,covdiag,epsilon,runtime):
    rows=[]
    for keys,g in cal.groupby(["config_label","M","T","covariance_estimator","lrvb_stage"],dropna=False):
        values={kind:part for kind,part in g.groupby("coefficient_type")};v=lambda kind,col:getattr(values[kind],col).mean() if kind in values else np.nan;edge=selected_edges.loc[(selected_edges.M==keys[1])&(selected_edges["T"]==keys[2])&(selected_edges.covariance_estimator==keys[3])];group=groups.loc[(groups.M==keys[1])&(groups["T"]==keys[2])&(groups.covariance_estimator==keys[3])];ev=empirical.loc[(empirical.M==keys[1])&(empirical["T"]==keys[2])&(empirical.covariance_estimator==keys[3])];cd=covdiag.loc[(covdiag.M==keys[1])&(covdiag["T"]==keys[2])];active=np.nanmean([v("diagonal","empirical_coverage_95"),v("offdiag_nonzero","empirical_coverage_95")]);rows.append({"config_label":keys[0],"M":keys[1],"T":keys[2],"covariance_estimator":keys[3],"lrvb_stage":keys[4],"diag_coverage":v("diagonal","empirical_coverage_95"),"offdiag_nonzero_coverage":v("offdiag_nonzero","empirical_coverage_95"),"offdiag_zero_coverage":v("offdiag_zero","empirical_coverage_95"),"diag_std_z":v("diagonal","std_standardized_error"),"offdiag_nonzero_std_z":v("offdiag_nonzero","std_standardized_error"),"offdiag_zero_std_z":v("offdiag_zero","std_standardized_error"),"true_edge_group_chi2_95_coverage":group.loc[group.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean() if len(group) else np.nan,"false_edge_group_chi2_95_coverage":group.loc[group.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean() if len(group) else np.nan,"selected_edge_AUC":edge.ROC_AUC.mean(),"selected_edge_AUPRC":edge.AUPRC.mean(),"variance_ratio_to_empirical_diag":ev.loc[ev.coefficient_type=="diagonal","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_nonzero":ev.loc[ev.coefficient_type=="offdiag_nonzero","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_zero":ev.loc[ev.coefficient_type=="offdiag_zero","variance_ratio_posterior_to_empirical"].mean(),"fraction_negative_variances":cd.fraction_negative_variances.mean() if "stageB" in keys[3] else 0.,"fraction_psd_projected":cd.psd_projection_used_M5_full.mean() if "stageB" in keys[3] else 0.,"finite_difference_stability_flag_rate":epsilon.eps_stability_flag.mean(),"total_runtime_seconds":runtime.loc[(runtime.M==keys[1])&(runtime["T"]==keys[2])].filter(like="total_runtime_seconds_mean").mean(axis=1).mean(),"recommended_for_uncertainty":bool(.88<=active<=.97 and .90<=v("offdiag_zero","empirical_coverage_95")<=.99),"recommended_for_ranking":bool(edge.AUPRC.mean()>=selected_edges.loc[(selected_edges.M==keys[1])&(selected_edges["T"]==keys[2]),"AUPRC"].max()-.05),"notes":"Stage-B finite differences include repeated smoother, A, and alpha feedback; center remains A_VB."})
    return pd.DataFrame(rows)


def summaries(frames):
    cal=calibration_summary(frames["coeff"]);groups=group_summary(frames["group"]);edges=network_summary(frames["scores"]);emp=empirical_variance(frames["coeff"]);runtime=grouped_stats(frames["runtime"],["config_label","M","T"],["baseline_VB_runtime_seconds","louis_runtime_seconds","stageA_lrvb_runtime_seconds","stageB_total_runtime_seconds","average_perturbed_fit_runtime_seconds","number_perturbed_fits","number_selected_directions","total_runtime_seconds"]);decision=decision_summary(cal,groups,edges,emp,frames["covdiag"],frames["epsilon"],runtime);return cal,groups,edges,emp,runtime,decision


def save_plots(cal,coeff,emp,covdiag,epsilon,edges,runtime):
    path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
    def bar(frame,x,y,name):
        s=frame.groupby(x,dropna=False)[y].mean();fig,ax=plt.subplots();s.plot.bar(ax=ax);ax.set_ylabel(y);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    bar(cal,"covariance_estimator","empirical_coverage_95","coverage.png");bar(cal,"covariance_estimator","std_standardized_error","standardized_sd.png");bar(cal,"covariance_estimator","mean_posterior_sd","posterior_sd.png");ratio=cal.copy();louis=ratio.loc[ratio.covariance_estimator=="stabilized_louis_eta_0p70_tau_0p90"].groupby("coefficient_type").mean_posterior_sd.mean();stage=ratio.loc[ratio.covariance_estimator=="lrvb_stageB_smoother_feedback_psd_projected"].groupby("coefficient_type").mean_posterior_sd.mean();pd.DataFrame({"ratio":stage/louis}).plot.bar();plt.tight_layout();plt.savefig(os.path.join(path,"stageB_louis_ratio.png"));plt.close();stagecoeff=coeff.loc[coeff.covariance_estimator=="lrvb_stageB_smoother_feedback_psd_projected"];fig,ax=plt.subplots();ax.scatter(stagecoeff.posterior_variance,stagecoeff.squared_error,s=3);ax.set(xlabel="Stage-B variance",ylabel="squared error");fig.tight_layout();fig.savefig(os.path.join(path,"variance_squared_error.png"));plt.close(fig);bar(emp,"covariance_estimator","variance_ratio_posterior_to_empirical","variance_ratio.png");bar(covdiag,"M","trace_stageB_over_louis_selected","trace_ratio.png");bar(epsilon,"eps","variance_by_eps","epsilon.png");bar(covdiag,"M","fraction_negative_variances","negative_rate.png");bar(covdiag,"M","psd_projection_used_M5_full","psd_rate.png");bar(edges,"covariance_estimator","AUPRC","selected_edge_AUPRC.png");fig,ax=plt.subplots();ax.scatter(runtime.number_selected_directions_mean,runtime.stageB_total_runtime_seconds_mean);ax.set(xlabel="directions",ylabel="seconds");fig.tight_layout();fig.savefig(os.path.join(path,"runtime_directions.png"));plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34O","CONFIG_LIST":CONFIG_LIST,"FINITE_DIFF_EPS_GRID":FINITE_DIFF_EPS_GRID,"FINITE_DIFF_EPS_PRIMARY":FINITE_DIFF_EPS_PRIMARY,"PERTURBED_VB_MAX_ITER":PERTURBED_VB_MAX_ITER,"PERTURBED_CONVERGENCE_TOL":PERTURBED_CONVERGENCE_TOL,"center_estimator":"A_VB","stageB_mode":"full_M5_selected_M20" if FULL_M5 else "selected_M5_and_M20","full_M5_opt_in_environment_variable":"EXPERIMENT_34O_FULL_M5=1","smoke_test":SMOKE_TEST};cp=os.path.join(RESULTS_DIR,"experiment_config.json")
    names={"selected":"_selected_partial.csv","perturb":"stageB_perturbation_diagnostics_partial.csv","covdiag":"stageB_covariance_diagnostics_partial.csv","epsilon":"_epsilon_partial.csv","coeff":"_coeff_partial.csv","group":"_group_partial.csv","scores":"_scores_partial.csv","covstore":"_covstore_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={key:[] for key in names}
    if os.path.exists(cp) and os.path.exists(os.path.join(RESULTS_DIR,"run_summary_partial.csv")):
        with open(cp,encoding="utf8") as handle:old=json.load(handle)
        for key in ("CONFIG_LIST","FINITE_DIFF_EPS_GRID","FINITE_DIFF_EPS_PRIMARY","PERTURBED_VB_MAX_ITER","PERTURBED_CONVERGENCE_TOL"):
            if old.get(key)!=config.get(key):raise ValueError("Existing 34O checkpoint configuration differs.")
    with open(cp,"w",encoding="utf8") as handle:json.dump(config,handle,indent=2)
    for key,name in names.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            try:frame=pd.read_csv(path)
            except pd.errors.EmptyDataError:frame=pd.DataFrame()
            if len(frame):tables[key]=[frame]
    previous=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame();completed=set(zip(previous.config_label,previous["T"].astype(int),previous.true_network_id.astype(int),previous.replicate_id.astype(int))) if len(previous) else set();total=sum((cfg["label"],T,net,rep) not in completed for cfg in CONFIG_LIST for T in cfg["T_VALUES"] for net in range(cfg["N_TRUE_NETWORKS"]) for rep in range(cfg["N_REPLICATES_PER_NETWORK"]));progress=ProgressBar(total)
    for cfg in CONFIG_LIST:
        M=cfg["M"]
        for network_id in range(cfg["N_TRUE_NETWORKS"]):
            network_seed=BASE_SEED+M*100000+network_id*1000;A,B,mask=j.fixed_network(M,network_seed)
            for T in cfg["T_VALUES"]:
                for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
                    key=(cfg["label"],T,network_id,replicate)
                    if key in completed:continue
                    meta={"config_label":cfg["label"],"M":M,"T":T,"true_network_id":network_id,"replicate_id":replicate};seed=network_seed+T*10+replicate;started=time.perf_counter();status="success"
                    try:
                        data=j.simulate(A,B,M,T,seed);fit_start=time.perf_counter();model=j.fit_model(data,seed+500);fit_runtime=time.perf_counter()-fit_start;louis_start=time.perf_counter();louis=n._louis_covariances(model,data,seed+700);louis_runtime=time.perf_counter()-louis_start;stageA_start=time.perf_counter();stageA,_,_=n._lrvb_covariances(model,data,meta);stageA_runtime=time.perf_counter()-stageA_start;selected=select_directions(data,cfg["covariance_mode"],seed+900);stageB_start=time.perf_counter();valid_indices,raw,psd,perturb,epsilon,projection,fit_times=run_stage_b(model,data,selected,meta);stageB_runtime=time.perf_counter()-stageB_start;selected=selected.set_index("coefficient_global_index").loc[valid_indices].reset_index();tables["selected"].append(pd.DataFrame([{**meta,**row} for row in selected.to_dict("records")]))
                        global_sets={"ordinary_vb":global_covariance_block(model.A_row_covariances_),"stabilized_louis_eta_0p70_tau_0p90":global_covariance_block(louis),"lrvb_stageA_alpha_feedback":global_covariance_block(stageA)};blocks={name:value[np.ix_(valid_indices,valid_indices)] for name,value in global_sets.items()};blocks["lrvb_stageB_smoother_feedback_raw"]=raw;blocks["lrvb_stageB_smoother_feedback_psd_projected"]=psd;hybrid=np.zeros_like(psd);np.fill_diagonal(hybrid,np.maximum(np.diag(blocks["stabilized_louis_eta_0p70_tau_0p90"]),np.diag(psd)));blocks["hybrid_max_louis_stageB_diag"]=hybrid
                        coeff=coefficient_rows(model,data,selected,blocks,meta);group,scores=group_and_score_rows(model,data,selected,blocks,meta);covdiag=covariance_diagnostic(meta,selected,raw,psd,blocks,projection);truth=flatten_A(data["A"])[valid_indices];estimate=flatten_A(model.A_mean_matrices_)[valid_indices];covstore=pd.DataFrame([{**meta,"covariance_mode":covdiag["covariance_mode"],"covariance_estimator":name,"error_vector_json":json.dumps((estimate-truth).tolist()),"covariance_json":json.dumps(value.tolist())} for name,value in blocks.items()]);tables["perturb"].append(perturb);tables["epsilon"].append(epsilon);tables["covdiag"].append(covdiag);tables["coeff"].append(coeff);tables["group"].append(group);tables["scores"].append(scores);tables["covstore"].append(covstore);tables["runtime"].append({**meta,"baseline_VB_runtime_seconds":fit_runtime,"louis_runtime_seconds":louis_runtime,"stageA_lrvb_runtime_seconds":stageA_runtime,"stageB_total_runtime_seconds":stageB_runtime,"average_perturbed_fit_runtime_seconds":np.mean(fit_times),"number_perturbed_fits":len(fit_times),"number_selected_directions":len(selected),"total_runtime_seconds":time.perf_counter()-started,"run_status":"success"})
                    except Exception as error:status="failed";tables["runtime"].append({**meta,"total_runtime_seconds":time.perf_counter()-started,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()})
                    tables["run"].append({**meta,"fit_status":status,"total_runtime_seconds":time.perf_counter()-started});frames={name:(pd.concat(items,ignore_index=True) if items and isinstance(items[0],pd.DataFrame) else pd.DataFrame(items)) for name,items in tables.items()}
                    for name,file in names.items():atomic_csv(frames[name],os.path.join(RESULTS_DIR,file))
                    if len(frames["coeff"]):
                        cal,groups,edges,emp,runtime,decision=summaries(frames)
                        for file,frame in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",groups),("empirical_variance_comparison_partial.csv",emp),("runtime_summary_partial.csv",runtime),("decision_summary_partial.csv",decision)):atomic_csv(frame,os.path.join(RESULTS_DIR,file))
                    progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
    frames={name:(pd.concat(items,ignore_index=True) if items and isinstance(items[0],pd.DataFrame) else pd.DataFrame(items)) for name,items in tables.items()};cal,groups,edges,emp,runtime,decision=summaries(frames);emp_cov=empirical_covariance_m5(frames["covstore"]);outputs={"run_summary.csv":frames["run"],"selected_coefficients.csv":frames["selected"],"stageB_perturbation_diagnostics.csv":frames["perturb"],"stageB_covariance_diagnostics.csv":frames["covdiag"],"epsilon_stability_diagnostics.csv":frames["epsilon"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":groups,"network_recovery_summary.csv":edges,"selected_edge_recovery_summary.csv":edges,"empirical_variance_comparison.csv":emp,"empirical_covariance_comparison_M5.csv":emp_cov,"runtime_summary.csv":runtime,"decision_summary.csv":decision}
    for file,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,file))
    save_plots(cal,frames["coeff"],emp,frames["covdiag"],frames["epsilon"],edges,runtime)


if __name__=="__main__":main()
