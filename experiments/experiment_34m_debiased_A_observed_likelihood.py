"""Experiment 34M: one-step observed-likelihood de-biasing of latent VARX A."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34M_BLAS_THREADS", "1")

import json
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats, relative
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.stats.debiased_A_estimator import (
    apply_debias_step, build_information_variants, observed_score_rows,
    observed_score_rows_numba, predicted_quadratic_gain, solve_row_steps,
)
from src.stats.louis_missing_information import (
    SAMPLER_LABEL, estimate_missing_information_numba, prior_precision_for_row,
    sample_companion_trajectories_ffbs,
)

CONFIG_LIST = [
    {"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 3, "N_REPLICATES_PER_NETWORK": 10},
    {"label": "main_controlled_M20", "M": 20, "T_VALUES": [1000, 2000], "N_TRUE_NETWORKS": 2, "N_REPLICATES_PER_NETWORK": 5},
]
DEBIAS_STEP_GRID = [.25, .50, .75, 1.00]
INFORMATION_VARIANTS = ["currentInfo", "louis0p50", "stabilizedLouis"]
N_MC_SMOOTHER_SAMPLES = 100
BASE_SEED = 5350000
SMOKE_TEST = os.environ.get("EXPERIMENT_34M_SMOKE", "0") == "1"
if SMOKE_TEST:
    CONFIG_LIST = [{"label": "small_controlled_M5", "M": 5, "T_VALUES": [1000], "N_TRUE_NETWORKS": 1, "N_REPLICATES_PER_NETWORK": 2}]
    N_MC_SMOOTHER_SAMPLES = 20
if os.environ.get("EXPERIMENT_34M_MC_SAMPLES"):
    N_MC_SMOOTHER_SAMPLES = int(os.environ["EXPERIMENT_34M_MC_SAMPLES"])
RESULTS_DIR = os.environ.get("EXPERIMENT_34M_RESULTS_DIR", "results/experiment_34m")


class ProgressBar:
    def __init__(self, total, width=30):
        self.total, self.width, self.done = max(int(total), 0), width, 0

    def update(self, label):
        self.done += 1; fraction = min(self.done / max(self.total, 1), 1.0)
        filled = round(self.width * fraction)
        print(f"\rExperiment 34M [{'#' * filled}{'-' * (self.width-filled)}] "
              f"{self.done}/{self.total} ({100*fraction:5.1f}%) {label[:58]}",
              end="\n" if self.done >= self.total else "", flush=True)


def _meta(config, T, network_id, replicate):
    return {"config_label": config["label"], "M": config["M"], "T": T,
            "true_network_id": network_id, "replicate_id": replicate}


def _center_label(info, step, projected):
    suffix = f"step_{step:.2f}".replace(".", "p")
    return f"A_db_{info}_{suffix}_{'stable' if projected else 'raw'}"


def _coefficient_rows(center_A, covariance_rows, data, meta, center_name,
                      covariance_name, info, step, projected):
    rows = []; truth = np.asarray(data["A"]); M = truth.shape[1]
    for target in range(M):
        covariance = covariance_rows[target]
        for lag in range(j.na):
            for source in range(M):
                column = lag * M + source; sd = np.sqrt(max(covariance[column, column], 0.0))
                error = center_A[lag, target, source] - truth[lag, target, source]
                kind = "diagonal" if source == target else ("offdiag_nonzero" if truth[lag, target, source] != 0 else "offdiag_zero")
                rows.append({**meta, "center_estimator": center_name, "covariance_estimator": covariance_name,
                    "information_used_for_debias": info, "debias_step_size": step,
                    "stable_projection_used": projected, "target_row": target, "lag": lag + 1,
                    "source": source, "coefficient_type": kind, "posterior_center": center_A[lag, target, source],
                    "posterior_sd": sd, "true_value": truth[lag, target, source], "signed_error": error,
                    "abs_error": abs(error), "squared_error": error ** 2,
                    "standardized_error": error / max(sd, 1e-15),
                    "ci95_lower": center_A[lag, target, source] - 1.96 * sd,
                    "ci95_upper": center_A[lag, target, source] + 1.96 * sd,
                    "ci95_contains_true": abs(error) <= 1.96 * sd})
    return pd.DataFrame(rows)


def _group_and_score_rows(center_A, covariance_rows, data, meta, center_name,
                          covariance_name, info, step, projected):
    groups = []; scores = []; truth = np.asarray(data["A"]); M = truth.shape[1]
    for target in range(M):
        for source in range(M):
            if target == source: continue
            cols = np.asarray([lag * M + source for lag in range(j.na)])
            block = covariance_rows[target][np.ix_(cols, cols)]
            actual, estimate = truth[:, target, source], center_A[:, target, source]
            difference = estimate - actual; true_link = bool(np.linalg.norm(actual) > 0)
            common = {**meta, "center_estimator": center_name, "covariance_estimator": covariance_name,
                "information_used_for_debias": info, "debias_step_size": step,
                "stable_projection_used": projected, "target": target, "source": source,
                "true_link": true_link, "edge_group_type": "true_edge_group" if true_link else "false_edge_group"}
            D2 = float(difference @ np.linalg.pinv(block) @ difference)
            snr = np.sqrt(max(float(estimate @ np.linalg.pinv(block) @ estimate), 0.0))
            groups.append({**common, "mahalanobis_D2": D2, "D2_below_chi2_95_df2": D2 <= 5.991464547,
                "D2_below_chi2_99_df2": D2 <= 9.210340372, "group_posterior_sd_trace": np.trace(block),
                "group_error_norm": np.linalg.norm(difference)})
            for score_type, score in (("model_A_group_norm", np.linalg.norm(estimate)), ("group_snr", snr)):
                scores.append({**common, "score_type": score_type, "score": score})
    return pd.DataFrame(groups), pd.DataFrame(scores)


def _recovery_row(center_A, data, meta, center_name, info, step, projected, beta_vb):
    true = np.asarray(data["A"]); result, _ = j.recovery(center_A, data, data["x"], data["x"], meta, 0., 0, True)
    beta_center = np.stack([np.concatenate([center_A[k, i] for k in range(j.na)]) for i in range(center_A.shape[1])])
    step_matrix = beta_center - beta_vb
    step_A = np.asarray([step_matrix[:, lag * center_A.shape[1]:(lag + 1) * center_A.shape[1]] for lag in range(j.na)])
    mask_diag = np.eye(center_A.shape[1], dtype=bool)[None].repeat(j.na, axis=0)
    active = (true != 0) & ~mask_diag; zero = (true == 0) & ~mask_diag
    active_step = np.mean(np.abs(step_A[active])) if active.any() else np.nan
    zero_step = np.mean(np.abs(step_A[zero])) if zero.any() else np.nan
    return {**meta, "center_estimator": center_name, "information_used_for_debias": info,
        "debias_step_size": step, "stable_projection_used": projected, **result,
        "A_db_step_norm": np.linalg.norm(step_matrix),
        "A_db_step_norm_relative_to_A_VB": np.linalg.norm(step_matrix) / max(np.linalg.norm(beta_vb), 1e-12),
        "mean_abs_debias_step_diag": np.mean(np.abs(step_A[mask_diag])),
        "mean_abs_debias_step_offdiag_nonzero": active_step,
        "mean_abs_debias_step_offdiag_zero": zero_step,
        "ratio_step_active_to_zero": active_step / max(zero_step, 1e-12)}


def calibration_summary(frame):
    keys = ["config_label", "M", "T", "center_estimator", "covariance_estimator",
            "information_used_for_debias", "debias_step_size", "stable_projection_used", "coefficient_type"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        rows.append({**dict(zip(keys, values)), "empirical_coverage_95": group.ci95_contains_true.mean(),
            "mean_signed_error": group.signed_error.mean(), "median_signed_error": group.signed_error.median(),
            "mean_abs_error": group.abs_error.mean(), "median_abs_error": group.abs_error.median(),
            "rmse": np.sqrt(group.squared_error.mean()), "mean_posterior_sd": group.posterior_sd.mean(),
            "median_posterior_sd": group.posterior_sd.median(), "mean_standardized_error": group.standardized_error.mean(),
            "std_standardized_error": group.standardized_error.std(ddof=0),
            "median_abs_standardized_error": group.standardized_error.abs().median(),
            "posterior_sd_abs_error_correlation": correlations(group.posterior_sd, group.abs_error),
            "n_coefficients": len(group)})
    return pd.DataFrame(rows)


def bias_summary(coeff):
    keys = ["config_label", "M", "T", "center_estimator", "debias_step_size", "stable_projection_used", "coefficient_type"]
    baseline_keys = ["config_label", "true_network_id", "replicate_id", "M", "T", "target_row", "lag", "source"]
    baseline = coeff.loc[(coeff.center_estimator == "A_VB_current") & (coeff.covariance_estimator == "current"), baseline_keys + ["abs_error", "squared_error"]].rename(columns={"abs_error": "vb_abs_error", "squared_error": "vb_squared_error"})
    merged = coeff.merge(baseline, on=baseline_keys, how="left"); rows = []
    for values, group in merged.groupby(keys, dropna=False):
        rows.append({**dict(zip(keys, values)), "mean_signed_error": group.signed_error.mean(),
            "median_signed_error": group.signed_error.median(), "mean_abs_error": group.abs_error.mean(),
            "rmse": np.sqrt(group.squared_error.mean()),
            "fraction_abs_error_reduced_vs_VB": np.mean(group.abs_error < group.vb_abs_error),
            "mean_abs_error_reduction_vs_VB": np.mean(group.vb_abs_error - group.abs_error),
            "rmse_reduction_vs_VB": np.sqrt(group.vb_squared_error.mean()) - np.sqrt(group.squared_error.mean()),
            "n_coefficients": len(group)})
    return pd.DataFrame(rows)


def group_summary(frame):
    keys = ["config_label", "M", "T", "center_estimator", "covariance_estimator",
            "information_used_for_debias", "debias_step_size", "stable_projection_used", "edge_group_type"]
    return frame.groupby(keys, dropna=False).agg(mean_D2=("mahalanobis_D2", "mean"), median_D2=("mahalanobis_D2", "median"),
        fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2", "mean"),
        fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2", "mean"),
        mean_group_posterior_sd_trace=("group_posterior_sd_trace", "mean"),
        mean_group_error_norm=("group_error_norm", "mean"), n_groups=("mahalanobis_D2", "size")).reset_index()


def network_summary(frame):
    keys = ["config_label", "M", "T", "score_type", "center_estimator", "debias_step_size",
            "stable_projection_used", "covariance_estimator"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        metrics, _ = j.safe_curve(group.true_link, group.score)
        rows.append({**dict(zip(keys, values)), **metrics})
    return pd.DataFrame(rows)


def empirical_summary(row_errors):
    keys = ["config_label", "true_network_id", "M", "T", "target_row", "center_estimator"]
    rows = []
    for values, group in row_errors.groupby(keys, dropna=False):
        errors = np.asarray([json.loads(x) for x in group.error_vector_json])
        empirical = np.cov(errors, rowvar=False, ddof=1) if len(errors) > 1 else np.diag(errors[0] ** 2)
        stable_cov = np.mean(np.asarray([json.loads(x) for x in group.stabilized_covariance_json]), axis=0)
        current_cov = np.mean(np.asarray([json.loads(x) for x in group.current_covariance_json]), axis=0)
        rows.append({**dict(zip(keys, values)), "trace_empirical": np.trace(empirical),
            "trace_current": np.trace(current_cov), "trace_stabilizedLouis": np.trace(stable_cov),
            "trace_ratio_stabilizedLouis_to_empirical": np.trace(stable_cov) / max(np.trace(empirical), 1e-12),
            "diagonal_mean_ratio": np.mean(np.diag(stable_cov)) / max(np.mean(np.diag(empirical)), 1e-12),
            "frobenius_difference_to_empirical": np.linalg.norm(stable_cov - empirical), "n_replicates": len(errors)})
    return pd.DataFrame(rows)


def _summaries(frames):
    calibration = calibration_summary(frames["coeff"]); bias = bias_summary(frames["coeff"])
    groups = group_summary(frames["group"]); network = network_summary(frames["scores"])
    recovery_keys = ["config_label", "M", "T", "center_estimator", "debias_step_size", "stable_projection_used"]
    recovery_cols = ["A_relative_frobenius_error", "A_offdiag_relative_frobenius_error", "A_diagonal_relative_frobenius_error", "A_mean_absolute_error", "A_offdiag_mean_absolute_error", "A_diagonal_mean_absolute_error", "A_group_norm_pearson_correlation", "A_group_norm_spearman_correlation", "A_db_step_norm_relative_to_A_VB"]
    recovery = grouped_stats(frames["recovery"], recovery_keys, recovery_cols)
    runtime = grouped_stats(frames["runtime"], ["config_label", "M", "T"], ["VB_fit_runtime_seconds", "score_computation_runtime_seconds", "information_computation_runtime_seconds", "debiasing_runtime_seconds", "total_runtime_seconds"])
    empirical = empirical_summary(frames["rowerror"])
    return calibration, bias, groups, network, recovery, runtime, empirical


def decision_summary(calibration, network, recovery, stability, score_diag, runtime):
    keys = ["M", "T", "center_estimator", "covariance_estimator", "information_used_for_debias", "debias_step_size", "stable_projection_used"]
    rows = []
    for values, group in calibration.groupby(keys, dropna=False):
        by_type = {row.coefficient_type: row for _, row in group.iterrows()}
        def value(kind, column):
            return getattr(by_type[kind], column) if kind in by_type else np.nan
        net = network.loc[(network.M == values[0]) & (network["T"] == values[1]) & (network.center_estimator == values[2]) & (network.score_type == "group_snr")]
        rec = recovery.loc[(recovery.M == values[0]) & (recovery["T"] == values[1]) & (recovery.center_estimator == values[2])]
        stab = stability.loc[(stability.M == values[0]) & (stability["T"] == values[1]) & (stability.center_estimator == values[2])]
        log = score_diag.loc[(score_diag.M == values[0]) & (score_diag["T"] == values[1]) & (score_diag.center_estimator == values[2])]
        baseline = network.loc[(network.M == values[0]) & (network["T"] == values[1]) & (network.center_estimator == "A_VB_current") & (network.score_type == "group_snr")]
        active = np.nanmean([value("diagonal", "empirical_coverage_95"), value("offdiag_nonzero", "empirical_coverage_95")])
        rows.append({**dict(zip(keys, values)), "diag_coverage": value("diagonal", "empirical_coverage_95"),
            "offdiag_nonzero_coverage": value("offdiag_nonzero", "empirical_coverage_95"), "offdiag_zero_coverage": value("offdiag_zero", "empirical_coverage_95"),
            "diag_mean_signed_error": value("diagonal", "mean_signed_error"), "offdiag_nonzero_mean_signed_error": value("offdiag_nonzero", "mean_signed_error"),
            "offdiag_zero_mean_signed_error": value("offdiag_zero", "mean_signed_error"), "diag_rmse": value("diagonal", "rmse"),
            "offdiag_nonzero_rmse": value("offdiag_nonzero", "rmse"), "offdiag_zero_rmse": value("offdiag_zero", "rmse"),
            "diag_std_z": value("diagonal", "std_standardized_error"), "offdiag_nonzero_std_z": value("offdiag_nonzero", "std_standardized_error"),
            "offdiag_zero_std_z": value("offdiag_zero", "std_standardized_error"), "AUPRC": net.AUPRC.mean(),
            "TPR_at_FPR_0p05": net.TPR_at_FPR_0p05.mean(), "precision_at_FPR_0p05": net.precision_at_FPR_0p05.mean(),
            "A_relative_frobenius_error": rec.filter(like="A_relative_frobenius_error_mean").mean(axis=1).mean(),
            "A_offdiag_relative_frobenius_error": rec.filter(like="A_offdiag_relative_frobenius_error_mean").mean(axis=1).mean(),
            "stability_failure_rate": np.mean(~stab.stable_before_projection) if len(stab) else 0.,
            "mean_loglikelihood_gain": log.loglikelihood_gain.mean() if len(log) else np.nan,
            "runtime_seconds": runtime.loc[(runtime.M == values[0]) & (runtime["T"] == values[1])].filter(like="total_runtime_seconds_mean").mean(axis=1).mean(),
            "recommended_for_uncertainty": bool(.88 <= active <= .97 and abs(value("offdiag_nonzero", "mean_signed_error")) < .05),
            "recommended_for_ranking": bool(net.AUPRC.mean() >= baseline.AUPRC.mean() - .05),
            "notes": "One-step unpenalized Fisher-score correction; stabilized Louis intervals."})
    return pd.DataFrame(rows)


def save_plots(cal, bias, network, stability, score_diag, coeff, diagnostics):
    path = os.path.join(RESULTS_DIR, "plots"); os.makedirs(path, exist_ok=True)
    def bar(frame, x, y, name, subset=None):
        data = frame if subset is None else frame.loc[subset(frame)]
        series = data.groupby(x, dropna=False)[y].mean(); fig, ax = plt.subplots(); series.plot.bar(ax=ax)
        ax.set_ylabel(y); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)
    bar(cal, "center_estimator", "empirical_coverage_95", "active_coverage.png", lambda x: x.coefficient_type.isin(["diagonal", "offdiag_nonzero"]))
    bar(cal, "center_estimator", "empirical_coverage_95", "zero_coverage.png", lambda x: x.coefficient_type == "offdiag_zero")
    bar(bias, "center_estimator", "mean_signed_error", "signed_error.png"); bar(bias, "center_estimator", "rmse", "rmse.png")
    bar(cal, "center_estimator", "std_standardized_error", "standardized_error_sd.png")
    bar(diagnostics, "coefficient_type", "abs_debias_step", "debias_step_distribution.png")
    bar(network, "center_estimator", "AUPRC", "AUPRC.png"); bar(network, "center_estimator", "TPR_at_FPR_0p05", "TPR05.png"); bar(network, "center_estimator", "precision_at_FPR_0p05", "precision05.png")
    bar(stability, "debias_step_size", "stable_before_projection", "stability_rate.png"); bar(score_diag, "debias_step_size", "loglikelihood_gain", "loglikelihood_gain.png")
    base = coeff.loc[(coeff.center_estimator == "A_VB_current") & (coeff.covariance_estimator == "current")]; deb = coeff.loc[(coeff.information_used_for_debias == "stabilizedLouis") & (coeff.debias_step_size == 1.) & (coeff.stable_projection_used == True)]
    keys = ["config_label", "true_network_id", "replicate_id", "M", "T", "target_row", "lag", "source"]
    pair = base[keys + ["posterior_center", "true_value", "coefficient_type"]].rename(columns={"posterior_center": "vb"}).merge(deb[keys + ["posterior_center"]].rename(columns={"posterior_center": "db"}), on=keys)
    for x, y, name, mask in (("vb", "db", "vb_vs_debiased.png", np.ones(len(pair), bool)), ("true_value", "db", "true_vs_active.png", pair.coefficient_type.isin(["diagonal", "offdiag_nonzero"]))):
        fig, ax = plt.subplots(); ax.scatter(pair.loc[mask, x], pair.loc[mask, y], s=3); ax.set(xlabel=x, ylabel=y); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {"experiment": "34M", "CONFIG_LIST": CONFIG_LIST, "DEBIAS_STEP_GRID": DEBIAS_STEP_GRID,
        "INFORMATION_VARIANTS": INFORMATION_VARIANTS, "N_MC_SMOOTHER_SAMPLES": N_MC_SMOOTHER_SAMPLES,
        "eta_stabilized": .70, "tau_stabilized": .90, "C_MODE": "identity", "B_MODE": "fixed_B_true",
        "Q_MODE": "fixed_Q_true", "R_MODE": "fixed_R_true", "smoother_sampler": SAMPLER_LABEL,
        "numba_kernels": ["observed_score_rows_numba", "complete_data_score_numba"], "smoke_test": SMOKE_TEST}
    config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    partial_names = {"coeff": "_coefficient_partial.csv", "group": "_group_partial.csv", "scores": "_scores_partial.csv",
        "diagnostics": "debiasing_diagnostics_partial.csv", "stability": "_stability_partial.csv", "scorelog": "_scorelog_partial.csv",
        "rowerror": "_rowerror_partial.csv", "recovery": "_recovery_partial.csv", "runtime": "_runtime_partial.csv", "run": "run_summary_partial.csv"}
    if os.path.exists(config_path) and os.path.exists(os.path.join(RESULTS_DIR, "run_summary_partial.csv")):
        with open(config_path, encoding="utf8") as handle: old = json.load(handle)
        for key in ("CONFIG_LIST", "DEBIAS_STEP_GRID", "INFORMATION_VARIANTS", "N_MC_SMOOTHER_SAMPLES"):
            if old.get(key) != config.get(key): raise ValueError("Existing 34M checkpoint configuration differs.")
    with open(config_path, "w", encoding="utf8") as handle: json.dump(config, handle, indent=2)
    tables = {key: [] for key in partial_names}
    for key, filename in partial_names.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            try: frame = pd.read_csv(path)
            except pd.errors.EmptyDataError: frame = pd.DataFrame()
            if len(frame): tables[key] = [frame]
    previous = pd.concat(tables["run"], ignore_index=True) if tables["run"] else pd.DataFrame()
    completed = set(zip(previous.config_label, previous["T"].astype(int), previous.true_network_id.astype(int), previous.replicate_id.astype(int))) if len(previous) else set()
    total = sum((c["label"], T, n, r) not in completed for c in CONFIG_LIST for T in c["T_VALUES"] for n in range(c["N_TRUE_NETWORKS"]) for r in range(c["N_REPLICATES_PER_NETWORK"]))
    progress = ProgressBar(total); observed_score_rows_numba(np.eye(2), np.zeros((2, 1)), np.zeros((1, 2)), np.ones(1))
    for cfg in CONFIG_LIST:
        M = cfg["M"]
        for network_id in range(cfg["N_TRUE_NETWORKS"]):
            network_seed = BASE_SEED + M * 100000 + network_id * 1000; A, B, mask = j.fixed_network(M, network_seed)
            for T in cfg["T_VALUES"]:
                for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
                    key = (cfg["label"], T, network_id, replicate)
                    if key in completed: continue
                    meta = _meta(cfg, T, network_id, replicate); seed = network_seed + T * 10 + replicate; started = time.perf_counter(); status = "success"
                    try:
                        data = j.simulate(A, B, M, T, seed); fit_start = time.perf_counter(); model = j.fit_model(data, seed + 500); fit_runtime = time.perf_counter() - fit_start
                        beta_vb = model._pack_A(); stats = model._compute_posterior_sufficient_statistics(model.smooth_result_, data["u"])
                        score_start = time.perf_counter(); scores = observed_score_rows(stats, beta_vb, data["Q"]); score_runtime = time.perf_counter() - score_start
                        info_start = time.perf_counter(); samples = sample_companion_trajectories_ffbs(model.smooth_result_, model.F, N_MC_SMOOTHER_SAMPLES, seed + 700)
                        priors = np.asarray([prior_precision_for_row(M, j.na, i, model.alpha_mean_, model.diagonal_prior_precision) for i in range(M)])
                        missing = [x["missing_information"] for x in estimate_missing_information_numba(samples, data["u"], beta_vb, model.B_matrices, data["Q"], j.na, j.nb, priors)]
                        information, covariance, info_diag = build_information_variants(model.A_row_covariances_, missing); deltas = solve_row_steps(scores, information); info_runtime = time.perf_counter() - info_start
                        baseline_A = np.asarray(model.A_mean_matrices_); baseline_ll = float(model.smooth_result_["log_likelihood"])
                        centers = [("A_VB_current", baseline_A, "current", "none", 0., False, model.A_row_covariances_),
                                   ("A_VB_stabilizedLouis", baseline_A, "stabilizedLouis", "none", 0., False, covariance["stabilizedLouis"])]
                        debias_start = time.perf_counter(); stability_rows = []; scorelog_rows = []; diagnostic_rows = []
                        for info in INFORMATION_VARIANTS:
                            for step in DEBIAS_STEP_GRID:
                                estimate = apply_debias_step(beta_vb, deltas[info], step, M, j.na)
                                predicted = predicted_quadratic_gain(scores, deltas[info], information[info], step)
                                for projected, center_A in ((False, estimate.raw_A), (True, estimate.stable_A)):
                                    label = _center_label(info, step, projected); centers.append((label, center_A, "stabilizedLouis", info, step, projected, covariance["stabilizedLouis"]))
                                    stability_rows.append({**meta, "center_estimator": label, "information_used_for_debias": info, "debias_step_size": step,
                                        "stable_projection_used": projected, "spectral_radius_before_projection": estimate.spectral_radius_before,
                                        "spectral_radius_after_projection": estimate.spectral_radius_after if projected else estimate.spectral_radius_before,
                                        "stable_before_projection": estimate.stable_before, "rescaling_factor": estimate.rescaling_factor if projected else 1.})
                                    if step in (.50, 1.00):
                                        actual_ll = float(model.smooth(data["y"], data["u"], center_A)["log_likelihood"])
                                        scorelog_rows.append({**meta, "center_estimator": label, "information_used_for_debias": info,
                                            "debias_step_size": step, "stable_projection_used": projected, "actual_loglikelihood_before": baseline_ll,
                                            "actual_loglikelihood_after": actual_ll, "loglikelihood_gain": actual_ll - baseline_ll,
                                            "predicted_loglikelihood_gain": predicted.sum()})
                                for row in range(M):
                                    row_common = {**meta, "center_estimator": _center_label(info, step, False), "information_used_for_debias": info,
                                        "debias_step_size": step, "target_row": row, "norm_s_obs_i": np.linalg.norm(scores[row]),
                                        "norm_delta_i": np.linalg.norm(deltas[info][row]), "relative_delta_i": np.linalg.norm(deltas[info][row]) / max(np.linalg.norm(beta_vb[row]), 1e-12),
                                        "s_obs_i_T_delta_i": scores[row] @ deltas[info][row], "predicted_loglikelihood_gain": predicted[row]}
                                    for lag in range(j.na):
                                        for source in range(M):
                                            column = lag * M + source; actual = np.asarray(data["A"])[lag, row, source]
                                            kind = "diagonal" if source == row else ("offdiag_nonzero" if actual != 0 else "offdiag_zero")
                                            diagnostic_rows.append({**row_common, "lag": lag + 1, "source": source,
                                                "coefficient_type": kind, "abs_debias_step": abs(step * deltas[info][row, column])})
                        for center_name, center_A, cov_name, info, step, projected, cov_rows in centers:
                            coeff = _coefficient_rows(center_A, cov_rows, data, meta, center_name, cov_name, info, step, projected); tables["coeff"].append(coeff)
                            group, score_frame = _group_and_score_rows(center_A, cov_rows, data, meta, center_name, cov_name, info, step, projected); tables["group"].append(group); tables["scores"].append(score_frame)
                            tables["recovery"].append(_recovery_row(center_A, data, meta, center_name, info, step, projected, beta_vb))
                            for target in range(M):
                                truth_row = np.concatenate([np.asarray(data["A"])[lag, target] for lag in range(j.na)])
                                center_row = np.concatenate([center_A[lag, target] for lag in range(j.na)])
                                tables["rowerror"].append({**meta, "target_row": target, "center_estimator": center_name,
                                    "error_vector_json": json.dumps((center_row - truth_row).tolist()),
                                    "current_covariance_json": json.dumps(np.asarray(model.A_row_covariances_[target]).tolist()),
                                    "stabilized_covariance_json": json.dumps(np.asarray(covariance["stabilizedLouis"][target]).tolist())})
                        debias_runtime = time.perf_counter() - debias_start
                        tables["diagnostics"].append(pd.DataFrame(diagnostic_rows)); tables["stability"].append(pd.DataFrame(stability_rows)); tables["scorelog"].append(pd.DataFrame(scorelog_rows))
                        tables["runtime"].append({**meta, "VB_fit_runtime_seconds": fit_runtime, "score_computation_runtime_seconds": score_runtime,
                            "information_computation_runtime_seconds": info_runtime, "debiasing_runtime_seconds": debias_runtime,
                            "total_runtime_seconds": time.perf_counter() - started})
                    except Exception as error:
                        status = "failed"; tables["runtime"].append({**meta, "total_runtime_seconds": time.perf_counter() - started,
                            "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()})
                    tables["run"].append({**meta, "fit_status": status, "total_runtime_seconds": time.perf_counter() - started})
                    frames = {name: (pd.concat(items, ignore_index=True) if items and isinstance(items[0], pd.DataFrame) else pd.DataFrame(items)) for name, items in tables.items()}
                    for name, filename in partial_names.items(): atomic_csv(frames[name], os.path.join(RESULTS_DIR, filename))
                    if len(frames["coeff"]):
                        cal, bias, group_sum, network, recovery, runtime, empirical = _summaries(frames)
                        decision = decision_summary(cal, network, recovery, frames["stability"], frames["scorelog"], runtime)
                        for filename, frame in (("calibration_summary_partial.csv", cal), ("bias_summary_partial.csv", bias),
                            ("network_recovery_summary_partial.csv", network), ("A_recovery_summary_partial.csv", recovery),
                            ("runtime_summary_partial.csv", runtime), ("decision_summary_partial.csv", decision)):
                            atomic_csv(frame, os.path.join(RESULTS_DIR, filename))
                    progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
    frames = {name: (pd.concat(items, ignore_index=True) if items and isinstance(items[0], pd.DataFrame) else pd.DataFrame(items)) for name, items in tables.items()}
    cal, bias, group_sum, network, recovery, runtime, empirical = _summaries(frames)
    decision = decision_summary(cal, network, recovery, frames["stability"], frames["scorelog"], runtime)
    outputs = {"run_summary.csv": frames["run"], "debiasing_diagnostics.csv": frames["diagnostics"],
        "coefficient_calibration_results.csv": frames["coeff"], "calibration_summary.csv": cal, "bias_summary.csv": bias,
        "group_calibration_results.csv": frames["group"], "group_calibration_summary.csv": group_sum,
        "network_recovery_summary.csv": network, "A_recovery_summary.csv": recovery,
        "stability_diagnostics.csv": frames["stability"], "score_loglikelihood_diagnostics.csv": frames["scorelog"],
        "empirical_covariance_comparison.csv": empirical, "runtime_summary.csv": runtime, "decision_summary.csv": decision}
    for filename, frame in outputs.items(): atomic_csv(frame, os.path.join(RESULTS_DIR, filename))
    save_plots(cal, bias, network, frames["stability"], frames["scorelog"], frames["coeff"], frames["diagnostics"])


if __name__ == "__main__":
    main()
