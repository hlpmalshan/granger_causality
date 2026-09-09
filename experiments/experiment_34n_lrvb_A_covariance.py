"""Experiment 34N: Stage-A linear-response VB covariance for latent VARX A."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34N_BLAS_THREADS", "1")

import json
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats
from src.stats.louis_missing_information import (
    SAMPLER_LABEL, estimate_missing_information_numba, prior_precision_for_row,
    sample_companion_trajectories_ffbs, stabilized_louis_covariance,
)
from src.stats.lrvb_A_covariance import epsilon_stability, stage_a_lrvb_covariance

CONFIG_LIST = [
    {"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 3, "N_REPLICATES_PER_NETWORK": 10},
    {"label": "main_controlled_M20", "M": 20, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 2, "N_REPLICATES_PER_NETWORK": 5},
]
FINITE_DIFF_EPS_GRID = [1e-5, 3e-5, 1e-4, 3e-4]
FINITE_DIFF_EPS_PRIMARY = 1e-4
N_MC_SMOOTHER_SAMPLES = 100
BASE_SEED = 5450000
SMOKE_TEST = os.environ.get("EXPERIMENT_34N_SMOKE", "0") == "1"
if SMOKE_TEST:
    CONFIG_LIST = [{"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000], "N_TRUE_NETWORKS": 1, "N_REPLICATES_PER_NETWORK": 2}]
    FINITE_DIFF_EPS_GRID = [1e-4, 3e-4]; N_MC_SMOOTHER_SAMPLES = 20
if os.environ.get("EXPERIMENT_34N_MC_SAMPLES"):
    N_MC_SMOOTHER_SAMPLES = int(os.environ["EXPERIMENT_34N_MC_SAMPLES"])
RESULTS_DIR = os.environ.get("EXPERIMENT_34N_RESULTS_DIR", "results/experiment_34n")
LRVB_STAGE = "stageA_alpha_feedback"


class ProgressBar:
    def __init__(self, total, width=30): self.total, self.width, self.done = max(int(total), 0), width, 0
    def update(self, label):
        self.done += 1; fraction = min(self.done / max(self.total, 1), 1.); filled = round(self.width * fraction)
        print(f"\rExperiment 34N [{'#'*filled}{'-'*(self.width-filled)}] {self.done}/{self.total} "
              f"({100*fraction:5.1f}%) {label[:58]}", end="\n" if self.done >= self.total else "", flush=True)


def _selected_epsilon_rows(data):
    active = np.flatnonzero(np.asarray(data["mask"]).sum(axis=1) > 0)
    inactive = np.flatnonzero(np.asarray(data["mask"]).sum(axis=1) == 0)
    selected = {0}
    if len(active): selected.add(int(active[0]))
    if len(inactive): selected.add(int(inactive[0]))
    return selected


def _louis_covariances(model, data, seed):
    samples = sample_companion_trajectories_ffbs(model.smooth_result_, model.F, N_MC_SMOOTHER_SAMPLES, seed)
    beta = model._pack_A(); M = model.n_states
    priors = np.asarray([prior_precision_for_row(M, j.na, row, model.alpha_mean_, model.diagonal_prior_precision) for row in range(M)])
    missing = [item["missing_information"] for item in estimate_missing_information_numba(samples, data["u"], beta, model.B_matrices, data["Q"], j.na, j.nb, priors)]
    covariances = []
    for current, miss in zip(model.A_row_covariances_, missing):
        covariance, _, _ = stabilized_louis_covariance(np.linalg.pinv(current), miss, .70, .90)
        covariances.append(covariance)
    return covariances


def _lrvb_covariances(model, data, meta):
    stats = model._compute_posterior_sufficient_statistics(model.smooth_result_, data["u"])
    beta = model._pack_A(); M = model.n_states; selected = _selected_epsilon_rows(data)
    covariances = []; diagnostics = []; epsilon_rows = []
    for target in range(M):
        sources = [source for source in range(M) if source != target]
        alpha = model.alpha_mean_[target, sources]
        started = time.perf_counter()
        primary = stage_a_lrvb_covariance(beta[target], alpha, stats["S_zz"], stats["S_zx"][:, target],
            data["Q"][target, target], target, M, j.na, epsilon=FINITE_DIFF_EPS_PRIMARY,
            a0=model.a0, b0=model.b0, diagonal_prior_precision=model.diagonal_prior_precision,
            posterior_jitter=model.posterior_jitter)
        elapsed = time.perf_counter() - started; covariances.append(primary.covariance_psd)
        row = {**meta, "target_row": target, "lrvb_stage": LRVB_STAGE, **primary.diagnostics,
            "trace_ordinary_vb": np.trace(model.A_row_covariances_[target]),
            "trace_lrvb_over_vb": np.trace(primary.covariance_psd) / max(np.trace(model.A_row_covariances_[target]), 1e-15),
            "runtime_lrvb_row_seconds": elapsed}
        diagnostics.append(row)
        if target in selected:
            results = {FINITE_DIFF_EPS_PRIMARY: primary}
            for epsilon in FINITE_DIFF_EPS_GRID:
                if epsilon not in results:
                    results[epsilon] = stage_a_lrvb_covariance(beta[target], alpha, stats["S_zz"], stats["S_zx"][:, target],
                        data["Q"][target, target], target, M, j.na, epsilon=epsilon,
                        a0=model.a0, b0=model.b0, diagonal_prior_precision=model.diagonal_prior_precision,
                        posterior_jitter=model.posterior_jitter)
            stability = epsilon_stability(results)
            diagnostics[-1].update({"finite_difference_stability_score": stability["trace_coefficient_of_variation_across_eps"]})
            for epsilon, result in results.items():
                epsilon_rows.append({**meta, "target_row": target, "lrvb_stage": LRVB_STAGE, "eps_used": epsilon,
                    "lrvb_trace": np.trace(result.covariance_psd),
                    "diagonal_sd_json": json.dumps(np.sqrt(np.maximum(np.diag(result.covariance_psd), 0)).tolist()), **stability})
    return covariances, pd.DataFrame(diagnostics), pd.DataFrame(epsilon_rows)


def _frames_for_covariances(model, data, meta, covariance_sets):
    corrected = {row: {name: covariances[row] for name, covariances in covariance_sets.items()} for row in range(model.n_states)}
    coefficients = j.coefficient_rows(model, data, corrected, meta).rename(columns={"posterior_mean": "posterior_center"})
    coefficients["lrvb_stage"] = np.where(coefficients.covariance_estimator.str.startswith("lrvb"), LRVB_STAGE, "not_applicable")
    groups, wide_scores = j.group_rows(model, data, corrected, meta)
    groups["lrvb_stage"] = np.where(groups.covariance_estimator.str.startswith("lrvb"), LRVB_STAGE, "not_applicable")
    scores = []
    for estimator in covariance_sets:
        column = "group_snr_" + estimator
        frame = wide_scores.copy(); frame["covariance_estimator"] = estimator; frame["lrvb_stage"] = LRVB_STAGE if estimator.startswith("lrvb") else "not_applicable"
        frame["score_type"] = "group_snr"; frame["score"] = frame[column]; frame["true_link"] = frame.true_group_norm > 0; scores.append(frame)
    model_score = wide_scores.copy(); model_score["covariance_estimator"] = "not_applicable"; model_score["lrvb_stage"] = "not_applicable"
    model_score["score_type"] = "model_A_group_norm"; model_score["score"] = model_score.posterior_mean_group_norm; model_score["true_link"] = model_score.true_group_norm > 0; scores.append(model_score)
    rowcov = []
    true = np.asarray(data["A"]); beta = model._pack_A()
    for target in range(model.n_states):
        truth = np.concatenate([true[lag, target] for lag in range(j.na)])
        for estimator, covariances in covariance_sets.items():
            rowcov.append({**meta, "target_row": target, "covariance_estimator": estimator,
                "lrvb_stage": LRVB_STAGE if estimator.startswith("lrvb") else "not_applicable",
                "error_vector_json": json.dumps((beta[target] - truth).tolist()),
                "covariance_json": json.dumps(np.asarray(covariances[target]).tolist())})
    return coefficients, groups, pd.concat(scores, ignore_index=True), pd.DataFrame(rowcov)


def calibration_summary(frame):
    keys = ["config_label", "M", "T", "covariance_estimator", "lrvb_stage", "coefficient_type"]; rows = []
    for values, group in frame.groupby(keys, dropna=False):
        rows.append({**dict(zip(keys, values)), "empirical_coverage_95": group.ci95_contains_true.mean(),
            "mean_signed_error": group.signed_error.mean(), "median_signed_error": group.signed_error.median(),
            "mean_abs_error": group.abs_error.mean(), "median_abs_error": group.abs_error.median(),
            "rmse": np.sqrt(group.squared_error.mean()), "mean_posterior_sd": group.posterior_sd.mean(),
            "median_posterior_sd": group.posterior_sd.median(), "mean_standardized_error": group.standardized_error.mean(),
            "std_standardized_error": group.standardized_error.std(ddof=0), "median_abs_standardized_error": group.standardized_error.abs().median(),
            "posterior_sd_abs_error_correlation": correlations(group.posterior_sd, group.abs_error), "n_coefficients": len(group)})
    return pd.DataFrame(rows)


def group_summary(frame):
    keys = ["config_label", "M", "T", "covariance_estimator", "lrvb_stage", "edge_group_type"]
    return frame.groupby(keys, dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2", "mean"), median_mahalanobis_D2=("mahalanobis_D2", "median"),
        fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df_p", "mean"), fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df_p", "mean"),
        mean_group_posterior_sd_trace=("group_posterior_sd_trace", "mean"), mean_group_error_norm=("group_error_norm", "mean"), n_groups=("mahalanobis_D2", "size")).reset_index()


def network_summary(frame):
    keys = ["config_label", "M", "T", "score_type", "covariance_estimator", "lrvb_stage"]; rows = []
    for values, group in frame.groupby(keys, dropna=False):
        metrics, _ = j.safe_curve(group.true_link, group.score); rows.append({**dict(zip(keys, values)), **metrics})
    return pd.DataFrame(rows)


def empirical_summary(frame):
    keys = ["config_label", "true_network_id", "M", "T", "target_row", "covariance_estimator", "lrvb_stage"]; rows = []
    for values, group in frame.groupby(keys, dropna=False):
        errors = np.asarray([json.loads(value) for value in group.error_vector_json]); posterior = np.mean(np.asarray([json.loads(value) for value in group.covariance_json]), axis=0)
        empirical = np.cov(errors, rowvar=False, ddof=1) if len(errors) > 1 else np.diag(errors[0] ** 2); difference = posterior - empirical
        rows.append({**dict(zip(keys, values)), "trace_posterior_cov": np.trace(posterior), "trace_empirical_cov": np.trace(empirical),
            "trace_ratio_posterior_to_empirical": np.trace(posterior) / max(np.trace(empirical), 1e-15),
            "diagonal_mean_ratio_posterior_to_empirical": np.mean(np.diag(posterior)) / max(np.mean(np.diag(empirical)), 1e-15),
            "frobenius_difference_to_empirical": np.linalg.norm(difference), "diagonal_frobenius_difference_to_empirical": np.linalg.norm(np.diag(difference)), "n_replicates": len(errors)})
    return pd.DataFrame(rows)


def decision_summary(cal, groups, network, empirical, diagnostics, runtime):
    rows = []
    for keys, subset in cal.groupby(["M", "T", "covariance_estimator", "lrvb_stage"], dropna=False):
        values = {row.coefficient_type: row for _, row in subset.iterrows()}
        def val(kind, field): return getattr(values[kind], field) if kind in values else np.nan
        group = groups.loc[(groups.M == keys[0]) & (groups["T"] == keys[1]) & (groups.covariance_estimator == keys[2])]
        net = network.loc[(network.M == keys[0]) & (network["T"] == keys[1]) & (network.covariance_estimator == keys[2]) & (network.score_type == "group_snr")]
        emp = empirical.loc[(empirical.M == keys[0]) & (empirical["T"] == keys[1]) & (empirical.covariance_estimator == keys[2])]
        diag = diagnostics.loc[(diagnostics.M == keys[0]) & (diagnostics["T"] == keys[1])] if keys[2].startswith("lrvb") else pd.DataFrame()
        active_coverage = np.nanmean([val("diagonal", "empirical_coverage_95"), val("offdiag_nonzero", "empirical_coverage_95")])
        rows.append({"M": keys[0], "T": keys[1], "covariance_estimator": keys[2], "lrvb_stage": keys[3],
            "diag_coverage": val("diagonal", "empirical_coverage_95"), "offdiag_nonzero_coverage": val("offdiag_nonzero", "empirical_coverage_95"),
            "offdiag_zero_coverage": val("offdiag_zero", "empirical_coverage_95"), "diag_std_z": val("diagonal", "std_standardized_error"),
            "offdiag_nonzero_std_z": val("offdiag_nonzero", "std_standardized_error"), "offdiag_zero_std_z": val("offdiag_zero", "std_standardized_error"),
            "true_edge_group_chi2_95_coverage": group.loc[group.edge_group_type == "true_edge_group", "fraction_D2_below_chi2_95_df2"].mean(),
            "false_edge_group_chi2_95_coverage": group.loc[group.edge_group_type == "false_edge_group", "fraction_D2_below_chi2_95_df2"].mean(),
            "group_snr_AUPRC": net.AUPRC.mean(), "group_snr_TPR_at_FPR_0p05": net.TPR_at_FPR_0p05.mean(),
            "group_snr_precision_at_FPR_0p05": net.precision_at_FPR_0p05.mean(), "trace_ratio_posterior_to_empirical": emp.trace_ratio_posterior_to_empirical.mean(),
            "fraction_rows_psd_projected": diag.psd_projection_used.mean() if len(diag) else 0., "mean_spectral_radius_J": diag.spectral_radius_J.mean() if len(diag) else np.nan,
            "max_spectral_radius_J": diag.spectral_radius_J.max() if len(diag) else np.nan, "mean_condition_number_I_minus_J": diag.condition_number_I_minus_J.mean() if len(diag) else np.nan,
            "runtime_seconds": runtime.loc[(runtime.M == keys[0]) & (runtime["T"] == keys[1])].filter(like="total_runtime_seconds_mean").mean(axis=1).mean(),
            "recommended_for_uncertainty": bool(.88 <= active_coverage <= .97 and .90 <= val("offdiag_zero", "empirical_coverage_95") <= .99),
            "recommended_for_ranking": bool(net.AUPRC.mean() >= network.loc[(network.M == keys[0]) & (network["T"] == keys[1]), "AUPRC"].max() - .05),
            "notes": "Stage-A LRVB freezes Kalman smoother statistics and retains A_VB as center."})
    return pd.DataFrame(rows)


def save_plots(cal, coeff, empirical, diagnostics, network, runtime):
    path = os.path.join(RESULTS_DIR, "plots"); os.makedirs(path, exist_ok=True)
    def bar(frame, x, y, name):
        series = frame.groupby(x, dropna=False)[y].mean(); fig, ax = plt.subplots(); series.plot.bar(ax=ax); ax.set_ylabel(y); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)
    bar(cal, "covariance_estimator", "empirical_coverage_95", "coverage.png"); bar(cal, "covariance_estimator", "std_standardized_error", "standardized_sd.png"); bar(cal, "covariance_estimator", "mean_posterior_sd", "posterior_sd.png")
    keys = ["config_label", "true_network_id", "replicate_id", "M", "T", "target_row", "lag", "source"]
    lrvb = coeff.loc[coeff.covariance_estimator == "lrvb_stageA_alpha_feedback", keys + ["posterior_sd"]].rename(columns={"posterior_sd": "lrvb_sd"})
    for estimator, name in (("ordinary_vb", "vb_lrvb_sd.png"), ("stabilized_louis_eta_0p70_tau_0p90", "louis_lrvb_sd.png")):
        other = coeff.loc[coeff.covariance_estimator == estimator, keys + ["posterior_sd"]].rename(columns={"posterior_sd": "other_sd"}); pair = lrvb.merge(other, on=keys)
        fig, ax = plt.subplots(); ax.scatter(pair.other_sd, pair.lrvb_sd, s=2); ax.set(xlabel=estimator, ylabel="LRVB SD"); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)
    bar(empirical, "covariance_estimator", "trace_ratio_posterior_to_empirical", "empirical_trace_ratio.png"); bar(diagnostics, "target_row", "trace_lrvb_over_vb", "trace_lrvb_vb.png"); bar(diagnostics, "target_row", "trace_lrvb_over_louis", "trace_lrvb_louis.png")
    bar(diagnostics, "target_row", "spectral_radius_J", "jacobian_radius.png"); bar(diagnostics, "target_row", "condition_number_I_minus_J", "jacobian_condition.png"); bar(diagnostics, "target_row", "min_eigenvalue_lrvb_raw", "min_eigenvalue.png"); bar(diagnostics, "M", "psd_projection_used", "psd_fraction.png")
    bar(network, "covariance_estimator", "AUPRC", "group_snr_AUPRC.png"); bar(network, "covariance_estimator", "TPR_at_FPR_0p05", "group_snr_TPR05.png"); bar(runtime, "M", "total_runtime_seconds_mean", "runtime.png")


def _summaries(frames):
    cal = calibration_summary(frames["coeff"]); groups = group_summary(frames["group"]); network = network_summary(frames["scores"]); empirical = empirical_summary(frames["rowcov"])
    runtime = grouped_stats(frames["runtime"], ["config_label", "M", "T"], ["VB_fit_runtime_seconds", "louis_correction_runtime_seconds", "lrvb_stageA_runtime_seconds", "total_runtime_seconds", "average_lrvb_row_runtime_seconds", "number_rows_completed"])
    decision = decision_summary(cal, groups, network, empirical, frames["diagnostics"], runtime)
    return cal, groups, network, empirical, runtime, decision


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {"experiment": "34N", "CONFIG_LIST": CONFIG_LIST, "LRVB_STAGE_LIST": [LRVB_STAGE], "FINITE_DIFF_EPS_GRID": FINITE_DIFF_EPS_GRID,
        "FINITE_DIFF_EPS_PRIMARY": FINITE_DIFF_EPS_PRIMARY, "LRVB_USE_CENTRAL_DIFFERENCE": True, "N_MC_SMOOTHER_SAMPLES": N_MC_SMOOTHER_SAMPLES,
        "C_MODE": "identity", "B_MODE": "fixed_B_true", "Q_MODE": "fixed_Q_true", "R_MODE": "fixed_R_true", "smoother_sampler": SAMPLER_LABEL,
        "center_estimator": "A_VB", "stageB_implemented": False, "smoke_test": SMOKE_TEST}
    config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    names = {"diagnostics": "lrvb_row_diagnostics_partial.csv", "epsilon": "_epsilon_partial.csv", "coeff": "_coeff_partial.csv", "group": "_group_partial.csv",
        "scores": "_scores_partial.csv", "rowcov": "_rowcov_partial.csv", "runtime": "_runtime_partial.csv", "run": "run_summary_partial.csv"}
    if os.path.exists(config_path) and os.path.exists(os.path.join(RESULTS_DIR, "run_summary_partial.csv")):
        with open(config_path, encoding="utf8") as handle: old = json.load(handle)
        for key in ("CONFIG_LIST", "FINITE_DIFF_EPS_GRID", "FINITE_DIFF_EPS_PRIMARY", "N_MC_SMOOTHER_SAMPLES"):
            if old.get(key) != config.get(key): raise ValueError("Existing 34N checkpoint configuration differs.")
    with open(config_path, "w", encoding="utf8") as handle: json.dump(config, handle, indent=2)
    tables = {key: [] for key in names}
    for key, filename in names.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            try: frame = pd.read_csv(path)
            except pd.errors.EmptyDataError: frame = pd.DataFrame()
            if len(frame): tables[key] = [frame]
    previous = pd.concat(tables["run"], ignore_index=True) if tables["run"] else pd.DataFrame()
    completed = set(zip(previous.config_label, previous["T"].astype(int), previous.true_network_id.astype(int), previous.replicate_id.astype(int))) if len(previous) else set()
    total = sum((cfg["label"], T, network, replicate) not in completed for cfg in CONFIG_LIST for T in cfg["T_VALUES"] for network in range(cfg["N_TRUE_NETWORKS"]) for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]))
    progress = ProgressBar(total)
    for cfg in CONFIG_LIST:
        M = cfg["M"]
        for network_id in range(cfg["N_TRUE_NETWORKS"]):
            network_seed = BASE_SEED + M * 100000 + network_id * 1000; A, B, mask = j.fixed_network(M, network_seed)
            for T in cfg["T_VALUES"]:
                for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
                    key = (cfg["label"], T, network_id, replicate)
                    if key in completed: continue
                    meta = {"config_label": cfg["label"], "M": M, "T": T, "true_network_id": network_id, "replicate_id": replicate}; seed = network_seed + T * 10 + replicate; started = time.perf_counter(); status = "success"
                    try:
                        data = j.simulate(A, B, M, T, seed); fit_start = time.perf_counter(); model = j.fit_model(data, seed + 500); fit_runtime = time.perf_counter() - fit_start
                        louis_start = time.perf_counter(); louis = _louis_covariances(model, data, seed + 700); louis_runtime = time.perf_counter() - louis_start
                        lrvb_start = time.perf_counter(); lrvb, diagnostic, epsilon = _lrvb_covariances(model, data, meta); lrvb_runtime = time.perf_counter() - lrvb_start
                        diagnostic["trace_stabilized_louis"] = [np.trace(value) for value in louis]; diagnostic["trace_lrvb_over_louis"] = diagnostic.trace_lrvb_psd / diagnostic.trace_stabilized_louis
                        covariance_sets = {"ordinary_vb": model.A_row_covariances_, "stabilized_louis_eta_0p70_tau_0p90": louis, "lrvb_stageA_alpha_feedback": lrvb}
                        coeff, group, scores, rowcov = _frames_for_covariances(model, data, meta, covariance_sets)
                        tables["diagnostics"].append(diagnostic); tables["epsilon"].append(epsilon); tables["coeff"].append(coeff); tables["group"].append(group); tables["scores"].append(scores); tables["rowcov"].append(rowcov)
                        tables["runtime"].append({**meta, "VB_fit_runtime_seconds": fit_runtime, "louis_correction_runtime_seconds": louis_runtime,
                            "lrvb_stageA_runtime_seconds": lrvb_runtime, "lrvb_stageB_runtime_seconds": 0., "total_runtime_seconds": time.perf_counter() - started,
                            "average_lrvb_row_runtime_seconds": lrvb_runtime / M, "number_rows_completed": M, "run_status": "success"})
                    except Exception as error:
                        status = "failed"; tables["runtime"].append({**meta, "total_runtime_seconds": time.perf_counter() - started, "run_status": status,
                            "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()})
                    tables["run"].append({**meta, "fit_status": status, "total_runtime_seconds": time.perf_counter() - started})
                    frames = {name: (pd.concat(items, ignore_index=True) if items and isinstance(items[0], pd.DataFrame) else pd.DataFrame(items)) for name, items in tables.items()}
                    for name, filename in names.items(): atomic_csv(frames[name], os.path.join(RESULTS_DIR, filename))
                    if len(frames["coeff"]):
                        cal, group_sum, network, empirical, runtime, decision = _summaries(frames)
                        for filename, frame in (("calibration_summary_partial.csv", cal), ("group_calibration_summary_partial.csv", group_sum),
                            ("network_recovery_summary_partial.csv", network), ("empirical_covariance_comparison_partial.csv", empirical),
                            ("runtime_summary_partial.csv", runtime), ("decision_summary_partial.csv", decision)):
                            atomic_csv(frame, os.path.join(RESULTS_DIR, filename))
                    progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
    frames = {name: (pd.concat(items, ignore_index=True) if items and isinstance(items[0], pd.DataFrame) else pd.DataFrame(items)) for name, items in tables.items()}
    cal, group_sum, network, empirical, runtime, decision = _summaries(frames)
    outputs = {"run_summary.csv": frames["run"], "lrvb_row_diagnostics.csv": frames["diagnostics"], "lrvb_epsilon_diagnostics.csv": frames["epsilon"],
        "coefficient_calibration_results.csv": frames["coeff"], "calibration_summary.csv": cal, "group_calibration_results.csv": frames["group"],
        "group_calibration_summary.csv": group_sum, "network_recovery_summary.csv": network, "empirical_covariance_comparison.csv": empirical,
        "runtime_summary.csv": runtime, "decision_summary.csv": decision}
    for filename, frame in outputs.items(): atomic_csv(frame, os.path.join(RESULTS_DIR, filename))
    save_plots(cal, frames["coeff"], empirical, frames["diagnostics"], network, runtime)


if __name__ == "__main__":
    main()
