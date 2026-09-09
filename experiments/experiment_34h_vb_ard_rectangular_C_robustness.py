"""Experiment 34H: full-row-rank rectangular known-C robustness.

This remains Level-1 Hybrid Kalman + VB-ARD.  It intentionally does not treat
the 15 observed mixtures as a 20-source network representation.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34H_BLAS_THREADS", "1")

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
from experiments.experiment_33c_source_count_scaling_lgc_metric import (
    atomic_csv, correlations, grouped_stats, make_network, relative,
)
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import (
    recovery, safe_curve, vb_details,
)
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import (
    materialize_tables, network_scores, support_score_rows, vb_score_rows,
)
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import B_metrics
from experiments.experiment_34f_vb_ard_Q_estimation_shrinkage import (
    Q_METRICS, Q_metrics, diagnostics, fit_vb, iteration_rows,
)
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.varx.varx_generator import generate_colored_input

scaleup.LGC_LAMBDA_GRID = [0.0]
M_LATENT, M_OBSERVED, na, nb, BURN_IN = 20, 15, 2, 3, 300
OBSERVATION_FRACTION = M_OBSERVED / M_LATENT
C_MODE_LIST = ["rectangular_75_well_conditioned"]
T_VALUES = [1000, 2000]
N_OUTER_RUNS_BY_T = {1000: 10, 2000: 5}
METHOD = "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25"
Q_VARIANT = "estimate_Q_diag_shrink_scalar_rho_0p25"
BASE_SEED = 4800000
SMOKE_TEST = os.environ.get("EXPERIMENT_34H_SMOKE", "0") == "1"
if SMOKE_TEST:
    T_VALUES, N_OUTER_RUNS_BY_T = [1000], {1000: 1}
override = os.environ.get("EXPERIMENT_34H_N_OUTER_RUNS")
if override is not None:
    N_OUTER_RUNS_BY_T = {T: int(override) for T in T_VALUES}
RESULTS_DIR = os.environ.get("EXPERIMENT_34H_RESULTS_DIR", "results/experiment_34h")


class ProgressBar:
    def __init__(self, total, width=30):
        self.total, self.width, self.completed = max(int(total), 0), width, 0
        if not self.total:
            print("Experiment 34H: all checkpointed work is already complete.")

    def update(self, label):
        self.completed += 1
        fraction = min(self.completed / max(self.total, 1), 1.0)
        filled = int(round(self.width * fraction))
        print(
            f"\rExperiment 34H [{'#' * filled}{'-' * (self.width-filled)}] "
            f"{self.completed}/{self.total} ({100*fraction:5.1f}%) {label[:64]}",
            end="\n" if self.completed >= self.total else "", flush=True,
        )


def make_rectangular_C(n_observed, n_latent, seed):
    """Construct a reproducible, controlled, full-row-rank observation map."""
    rng = np.random.default_rng(seed + 150)
    left, _ = np.linalg.qr(rng.normal(size=(n_observed, n_observed)))
    right, _ = np.linalg.qr(rng.normal(size=(n_latent, n_observed)), mode="reduced")
    singular = np.linspace(1.0, 0.35, n_observed)
    C = left @ np.diag(singular) @ right.T
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    if C.shape != (n_observed, n_latent) or np.linalg.matrix_rank(C) != n_observed:
        raise np.linalg.LinAlgError("Rectangular C is not full row rank.")
    return C


def simulate(T, seed):
    A, mask = make_network(M_LATENT, seed)
    rng = np.random.default_rng(seed + 100)
    B = [rng.normal(0, scale, (M_LATENT, 1)) for scale in (.45, .25, .15)]
    C = make_rectangular_C(M_OBSERVED, M_LATENT, seed)
    u = generate_colored_input(T + BURN_IN, .95, 1.0, seed + 200)
    Q, R = .5 * np.eye(M_LATENT), .6 * np.eye(M_OBSERVED)
    result = generate_ssm_varx_p_data(
        A, B, u, Q, R, C=C, D=None, burn_in=BURN_IN,
        random_seed=seed + 300, return_augmented=True,
    )
    if result["x"].shape[1] != M_LATENT or result["y"].shape[1] != M_OBSERVED:
        raise ValueError("Simulator returned incompatible latent/observation dimensions.")
    return {"A": A, "B": B, "C": C, "Q": Q, "R": R, "mask": mask,
            "x": result["x"], "y": result["y"], "u": result["u"]}


def mean_component_correlation(estimate, truth):
    return float(np.nanmean([correlations(estimate[:, i], truth[:, i])
                             for i in range(truth.shape[1])]))


def rectangular_C_metrics(data, proxy):
    C = data["C"]
    singular = np.linalg.svd(C, compute_uv=False)
    row_norms, column_norms = np.linalg.norm(C, axis=1), np.linalg.norm(C, axis=0)
    row_corr = np.corrcoef(C)
    column_corr = np.corrcoef(C.T)
    row_mask = ~np.eye(row_corr.shape[0], dtype=bool)
    col_mask = ~np.eye(column_corr.shape[0], dtype=bool)
    return {
        "C_shape_rows": C.shape[0], "C_shape_cols": C.shape[1],
        "C_rank": np.linalg.matrix_rank(C), "C_condition_number": singular[0] / singular[-1],
        "C_min_singular_value": singular[-1], "C_max_singular_value": singular[0],
        "C_spectral_norm": singular[0], "C_frobenius_norm": np.linalg.norm(C),
        "C_row_norm_mean": row_norms.mean(), "C_row_norm_std": row_norms.std(),
        "C_column_norm_mean": column_norms.mean(), "C_column_norm_std": column_norms.std(),
        "C_row_correlation_mean_abs_offdiag": np.nanmean(np.abs(row_corr[row_mask])),
        "C_column_correlation_mean_abs_offdiag": np.nanmean(np.abs(column_corr[col_mask])),
        "C_projection_condition_number": np.linalg.cond(C @ C.T),
        "pseudoinverse_norm": np.linalg.norm(np.linalg.pinv(C), 2),
        "pinv_proxy_mse": np.mean((proxy - data["x"]) ** 2),
        "pinv_proxy_correlation": mean_component_correlation(proxy, data["x"]),
    }


DIM_GROUPS = ["n_latent", "n_observed", "observation_fraction", "T", "C_MODE"]
GROUPS = DIM_GROUPS + ["method", "Q_VARIANT", "Q_shrinkage_rho",
                       "Q_update_damping", "a0", "b0"]
PARAM_METRICS = [
    "A_relative_frobenius_error", "A_offdiag_relative_frobenius_error",
    "A_diagonal_relative_frobenius_error", "A_group_norm_pearson_correlation",
    "A_group_norm_spearman_correlation", "A_support_ROC_AUC", "A_support_AUPRC",
    "A_support_TPR_at_FPR_0p01", "A_support_TPR_at_FPR_0p03",
    "A_support_TPR_at_FPR_0p05", "A_support_TPR_at_FPR_0p10",
    "precision_at_FPR_0p01", "precision_at_FPR_0p05", "F1_at_FPR_0p01",
    "F1_at_FPR_0p05", "filtered_signal_mse", "smoothed_signal_mse",
    "runtime_seconds", "n_iter", "converged_all", "practically_converged",
    "hit_max_iter", "spectral_radius_A", "failure_rate",
]
B_METRICS = [
    "B_relative_frobenius_error", "B_lag0_relative_frobenius_error",
    "B_lag1_relative_frobenius_error", "B_lag2_relative_frobenius_error",
    "B_mean_absolute_error", "B_pearson_correlation", "B_spearman_correlation",
    "B_energy_ratio_hat_to_true", "B_change_norm", "relative_B_change_norm",
    "initial_B_relative_error", "final_B_relative_error",
]
C_METRICS = [
    "C_rank", "C_condition_number", "C_min_singular_value", "C_max_singular_value",
    "C_spectral_norm", "C_frobenius_norm", "C_row_norm_mean", "C_row_norm_std",
    "C_column_norm_mean", "C_column_norm_std", "C_projection_condition_number",
    "pseudoinverse_norm", "pinv_proxy_mse", "pinv_proxy_correlation",
]


def score_summaries(scores):
    groups = DIM_GROUPS + ["method", "Q_VARIANT", "signal_type", "score_type",
                           "a0", "b0", "lgc_lambda"]
    summaries, points, fixed = [], [], []
    for keys, group in scores.groupby(groups, dropna=False):
        meta = dict(zip(groups, keys)); metrics, curve = safe_curve(group.true_link, group.score)
        summaries.append({**meta, **metrics})
        for threshold, item in curve:
            points.append({**meta, "threshold": threshold, **item})
        for level in (.01, .03, .05, .10):
            eligible = [p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"] <= level]
            threshold, item = max(eligible, key=lambda p: (np.nan_to_num(p[1]["tpr"], nan=-1), -p[0])) if eligible else curve[0]
            fixed.append({**meta, "target_fpr_level": level, "threshold": threshold,
                          "actual_fpr": item["fpr"], **{k: item[k] for k in
                          ("tpr", "precision", "f1", "tp", "fp", "tn", "fn")}})
    return pd.DataFrame(summaries), pd.DataFrame(points), pd.DataFrame(fixed)


def edge_summary(frame):
    rows = []; groups = DIM_GROUPS + ["Q_VARIANT", "a0", "b0"]
    for keys, group in frame.groupby(groups, dropna=False):
        for score in ("posterior_mean_group_norm", "posterior_second_moment_group_norm",
                      "inverse_alpha_score", "group_snr_score"):
            metrics, _ = safe_curve(group.true_link, group[score])
            rows.append({**dict(zip(groups, keys)), "vb_score_type": score, **metrics})
    return pd.DataFrame(rows)


def uncertainty_calibration(frame):
    frame = frame.copy()
    frame["signed_error"] = frame.posterior_mean - frame.true_value
    frame["squared_error"] = frame.signed_error ** 2
    frame["standardized_error"] = frame.signed_error / frame.posterior_std.replace(0, np.nan)
    rows = []
    groups = DIM_GROUPS + ["Q_VARIANT", "a0", "b0"]
    expanded = pd.concat([frame.assign(_type=frame.coefficient_type), frame.assign(_type="all")])
    for keys, group in expanded.groupby(groups + ["_type"], dropna=False):
        corr = correlations(group.posterior_std, group.abs_error) if len(group) > 1 else np.nan
        rows.append({**dict(zip(groups + ["coefficient_type"], keys)),
            "n_coefficients": len(group), "empirical_coverage_95": group.ci95_contains_true.mean(),
            "mean_posterior_std": group.posterior_std.mean(), "median_posterior_std": group.posterior_std.median(),
            "mean_signed_error": group.signed_error.mean(), "median_signed_error": group.signed_error.median(),
            "mean_abs_error": group.abs_error.mean(), "median_abs_error": group.abs_error.median(),
            "rmse": np.sqrt(group.squared_error.mean()), "mean_standardized_error": group.standardized_error.mean(),
            "std_standardized_error": group.standardized_error.std(ddof=0),
            "median_standardized_abs_error": group.standardized_error.abs().median(),
            "posterior_std_abs_error_correlation": corr})
    return pd.DataFrame(rows)


def latent_summary(parameter):
    rows = []
    for _, row in parameter.loc[parameter.run_status == "success"].iterrows():
        for signal, mse, corr in (
            ("pinv_proxy", row.pinv_proxy_mse, row.pinv_proxy_correlation),
            ("method_filtered", row.filtered_signal_mse, row.filtered_signal_correlation),
            ("method_smoothed", row.smoothed_signal_mse, row.smoothed_signal_correlation),
        ):
            rows.append({**{k: row[k] for k in DIM_GROUPS}, "signal_type": signal,
                "signal_mse": mse, "signal_correlation": corr,
                "mse_improvement_over_pinv": row.pinv_proxy_mse - mse,
                "corr_improvement_over_pinv": corr - row.pinv_proxy_correlation})
    return pd.DataFrame(rows).groupby(DIM_GROUPS + ["signal_type"], dropna=False).mean(numeric_only=True).reset_index()


def robustness_summary(parameter):
    rows = []
    for _, row in parameter.iterrows():
        for score_type in ("model_A", "vb_group_snr"):
            rows.append({**{k: row[k] for k in DIM_GROUPS}, "score_type": score_type,
                "A_support_AUPRC": row.get("A_support_AUPRC"),
                "A_support_TPR_at_FPR_0p01": row.get("A_support_TPR_at_FPR_0p01"),
                "A_support_TPR_at_FPR_0p05": row.get("A_support_TPR_at_FPR_0p05"),
                "precision_at_FPR_0p01": row.get("precision_at_FPR_0p01"),
                "precision_at_FPR_0p05": row.get("precision_at_FPR_0p05"),
                "F1_at_FPR_0p05": row.get("F1_at_FPR_0p05"),
                "VB_group_SNR_AUPRC": row.get("group_snr_AUPRC") if score_type == "vb_group_snr" else np.nan,
                "VB_group_SNR_TPR_at_FPR_0p01": row.get("group_snr_TPR_at_FPR_0p01") if score_type == "vb_group_snr" else np.nan,
                "VB_group_SNR_TPR_at_FPR_0p05": row.get("group_snr_TPR_at_FPR_0p05") if score_type == "vb_group_snr" else np.nan,
                "B_relative_error": row.get("B_relative_frobenius_error"),
                "Q_relative_error": row.get("Q_relative_frobenius_error"),
                "smoothed_signal_mse": row.get("smoothed_signal_mse"),
                "smoothed_signal_correlation": row.get("smoothed_signal_correlation"),
                "pinv_proxy_mse": row.get("pinv_proxy_mse"), "pinv_proxy_correlation": row.get("pinv_proxy_correlation"),
                "runtime_seconds": row.get("runtime_seconds"), "practically_converged": row.get("practically_converged"),
                "failure_rate": row.get("failure_rate")})
    frame = pd.DataFrame(rows)
    columns = {name: "mean_" + name for name in (
        "A_support_AUPRC", "A_support_TPR_at_FPR_0p01", "A_support_TPR_at_FPR_0p05",
        "precision_at_FPR_0p01", "precision_at_FPR_0p05", "F1_at_FPR_0p05",
        "VB_group_SNR_AUPRC", "VB_group_SNR_TPR_at_FPR_0p01", "VB_group_SNR_TPR_at_FPR_0p05",
        "B_relative_error", "Q_relative_error", "smoothed_signal_mse", "smoothed_signal_correlation",
        "pinv_proxy_mse", "pinv_proxy_correlation", "runtime_seconds")}
    columns["practically_converged"] = "practical_convergence_rate"
    return frame.groupby(DIM_GROUPS + ["score_type"], dropna=False)[list(columns) + ["failure_rate"]].mean().reset_index().rename(columns=columns)


def summaries(frames):
    parameter = grouped_stats(frames["parameter"], GROUPS, PARAM_METRICS)
    b = grouped_stats(frames["parameter"], DIM_GROUPS + ["method"], B_METRICS)
    q = grouped_stats(frames["parameter"], DIM_GROUPS + ["method", "Q_VARIANT"], Q_METRICS)
    support = grouped_stats(frames["parameter"], GROUPS, [m for m in PARAM_METRICS if m.startswith("A_support_") or m.startswith("precision_") or m.startswith("F1_")])
    c = grouped_stats(frames["parameter"], DIM_GROUPS, C_METRICS + [
        "smoothed_signal_mse", "smoothed_signal_correlation", "filtered_signal_mse",
        "filtered_signal_correlation", "smoothed_mse_improvement_over_pinv",
        "smoothed_corr_improvement_over_pinv"])
    edge = edge_summary(frames["vb_edges"])
    calibration = uncertainty_calibration(frames["uncertainty"])
    roc, points, fixed = score_summaries(frames["scores"])
    return parameter, b, q, support, c, latent_summary(frames["parameter"]), edge, calibration, roc, points, fixed, robustness_summary(frames["parameter"])


def failure_row(meta, runtime, status, error, messages):
    row = {**meta, "runtime_seconds": runtime, "run_status": status,
           "error_type": type(error).__name__, "error_message": str(error),
           "traceback": traceback.format_exc(), "warning_count": len(messages),
           "warning_messages": " | ".join(messages), "failure_rate": 1.0,
           "converged_all": False, "practically_converged": False, "hit_max_iter": False}
    for key in PARAM_METRICS + B_METRICS + Q_METRICS + C_METRICS:
        row.setdefault(key, np.nan)
    return row


def save_plots(parameter, edge_summary_frame, edges, calibration):
    path = os.path.join(RESULTS_DIR, "plots"); os.makedirs(path, exist_ok=True)
    success = parameter.loc[parameter.run_status == "success"]

    def by_T(metric, name, label):
        values = success.groupby("T")[metric].mean()
        fig, ax = plt.subplots(); ax.plot(values.index, values.values, marker="o")
        ax.set(xlabel="T", ylabel=label); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)

    for args in (("A_support_AUPRC", "AUPRC_by_T.png", "A-support AUPRC"),
                 ("A_support_TPR_at_FPR_0p01", "TPR01_by_T.png", "TPR at FPR <= .01"),
                 ("A_support_TPR_at_FPR_0p05", "TPR05_by_T.png", "TPR at FPR <= .05"),
                 ("precision_at_FPR_0p05", "precision05_by_T.png", "precision at FPR <= .05"),
                 ("A_offdiag_relative_frobenius_error", "A_error_by_T.png", "A offdiag relative error"),
                 ("B_relative_frobenius_error", "B_error_by_T.png", "B relative error"),
                 ("Q_relative_frobenius_error", "Q_error_by_T.png", "Q relative error"),
                 ("runtime_seconds", "runtime_by_T.png", "seconds"),
                 ("practically_converged", "practical_convergence_by_T.png", "practical convergence rate")):
        by_T(*args)
    snr = edge_summary_frame.loc[edge_summary_frame.vb_score_type == "group_snr_score"].groupby("T").AUPRC.mean()
    fig, ax = plt.subplots(); ax.plot(snr.index, snr.values, marker="o"); ax.set(xlabel="T", ylabel="VB group-SNR AUPRC"); fig.tight_layout(); fig.savefig(os.path.join(path, "group_SNR_AUPRC_by_T.png")); plt.close(fig)

    def comparison(left, right, name, label):
        group = success.groupby("T")[[left, right]].mean()
        fig, ax = plt.subplots(); ax.plot(group.index, group[left], marker="o", label="pinv proxy"); ax.plot(group.index, group[right], marker="o", label="smoothed"); ax.set(xlabel="T", ylabel=label); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)
    comparison("pinv_proxy_mse", "smoothed_signal_mse", "pinv_vs_smoothed_MSE.png", "MSE")
    comparison("pinv_proxy_correlation", "smoothed_signal_correlation", "pinv_vs_smoothed_correlation.png", "correlation")
    values, labels = [], []
    for T in T_VALUES:
        for truth, label in ((True, "true"), (False, "false")):
            values.append(edges.loc[(edges["T"] == T) & (edges.true_link == truth), "group_snr_score"]); labels.append(f"T={T}\n{label}")
    fig, ax = plt.subplots(); ax.boxplot(values, labels=labels); ax.set_ylabel("VB group-SNR"); fig.tight_layout(); fig.savefig(os.path.join(path, "group_SNR_true_false_by_T.png")); plt.close(fig)
    selected = calibration.loc[calibration.coefficient_type != "all"]
    for metric, name, label in (("empirical_coverage_95", "coverage_by_type_T.png", "95% coverage"),
                                ("std_standardized_error", "standardized_error_SD_by_type_T.png", "standardized-error SD")):
        pivot = selected.pivot_table(index="coefficient_type", columns="T", values=metric)
        fig, ax = plt.subplots(); pivot.plot.bar(ax=ax); ax.set_ylabel(label); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(os.path.join(path, name)); plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {"experiment": "34H", "methodology": "Level-1 Hybrid Kalman + VB-ARD",
        "M_LATENT": M_LATENT, "M_OBSERVED": M_OBSERVED, "OBSERVATION_FRACTION": OBSERVATION_FRACTION,
        "C_MODE_LIST": C_MODE_LIST, "T_VALUES": T_VALUES, "N_OUTER_RUNS_BY_T": N_OUTER_RUNS_BY_T,
        "METHODS": [METHOD], "C_generation": {"construction": "SVD-style full-row-rank, unit row norm", "singular_values": "linspace(1.0, 0.35, 15)"},
        "Q_VARIANT": Q_VARIANT, "Q_shrinkage_rho": .25, "Q_update_damping": .5,
        "include_A_posterior_uncertainty_in_Q": True, "B_ridge_lambda": 0.0,
        "VB_MAX_ITER": 100, "a0": 1e-3, "b0": 1e-3, "LGC_LAMBDA_GRID": [0.0],
        "smoke_test": SMOKE_TEST, "square_C_reference": "Compare against Experiments 34F and 34G; square C is intentionally not rerun",
        "TODO": "Static frequency-domain VARX-GC in 34I/34J; Louis correction in a separate covariance-calibration experiment"}
    config = json.loads(json.dumps(config)); config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    ledger = os.path.join(RESULTS_DIR, "run_summary_partial.csv")
    if os.path.exists(config_path) and os.path.exists(ledger):
        with open(config_path, encoding="utf-8") as handle: old = json.load(handle)
        for key in ("M_LATENT", "M_OBSERVED", "C_MODE_LIST", "T_VALUES", "N_OUTER_RUNS_BY_T", "METHODS", "Q_VARIANT"):
            if old.get(key) != config.get(key):
                raise ValueError("Existing 34H checkpoint has a different numerical configuration.")
    with open(config_path, "w", encoding="utf-8") as handle: json.dump(config, handle, indent=2)
    checkpoint = {"parameter": "parameter_results_partial.csv", "support": "_support_checkpoint.csv",
        "vb_edges": "vb_edge_scores_partial.csv", "uncertainty": "_uncertainty_checkpoint.csv",
        "objective": "_objective_checkpoint.csv", "convergence": "convergence_diagnostics_partial.csv",
        "network": "_network_checkpoint.csv", "lgc": "_lgc_checkpoint.csv", "scores": "_scores_checkpoint.csv",
        "runtime": "_runtime_checkpoint.csv", "run": "run_summary_partial.csv"}
    tables = {key: [] for key in checkpoint if key != "run"}; run_rows = []
    for key, name in checkpoint.items():
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            try: loaded = pd.read_csv(path)
            except pd.errors.EmptyDataError: loaded = pd.DataFrame()
            if key == "run": run_rows = loaded.to_dict("records")
            elif len(loaded): tables[key] = [loaded]
    completed = {(int(row["T"]), int(row["outer_run"])) for row in run_rows}
    remaining = sum((T, outer) not in completed for T in T_VALUES for outer in range(N_OUTER_RUNS_BY_T[T]))
    progress = ProgressBar(remaining)
    for T in T_VALUES:
        for outer in range(N_OUTER_RUNS_BY_T[T]):
            if (T, outer) in completed: continue
            seed = BASE_SEED + T * 100 + outer; messages = []; started = time.perf_counter(); phase = "simulate"
            base = {"n_sources": M_LATENT, "n_latent": M_LATENT, "n_observed": M_OBSERVED,
                "observation_fraction": OBSERVATION_FRACTION, "T": T, "outer_run": outer,
                "case_name": C_MODE_LIST[0], "C_MODE": C_MODE_LIST[0], "method": METHOD,
                "Q_VARIANT": Q_VARIANT, "Q_update_mode": "diag_shrink_scalar", "Q_shrinkage_rho": .25,
                "Q_update_damping": .5, "a0": 1e-3, "b0": 1e-3, "lambda_A_group_fraction": np.nan,
                "B_MODEL_VARIANT": "free_B", "B_ridge_lambda": 0.0, "convergence_config_name": "max100"}
            try:
                data = simulate(T, seed)
                # For full row rank C, this is C.T @ inv(C @ C.T), evaluated stably.
                C_pinv = np.linalg.solve(data["C"] @ data["C"].T, data["C"]).T
                proxy = data["y"] @ C_pinv.T
                cmetrics = rectangular_C_metrics(data, proxy)
                baseline = {**base, "method": "dataset_baseline"}
                for signal_type, signal in (("oracle_latent", data["x"]), ("pinv_proxy", proxy)):
                    network, lgcs, scores = network_scores(signal, signal_type, data, baseline, compute_lgc=True)
                    tables["network"].append(network); tables["lgc"].extend(lgcs); tables["scores"].extend(scores)
                phase = "fit"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model, Ah, Bh, filtered, smoothed = fit_vb(data, Q_VARIANT, seed + 500)
                    diag = diagnostics(model); messages = [str(item.message) for item in caught]
                phase = "metric"; runtime = time.perf_counter() - started
                row, support = recovery(Ah, data, filtered, smoothed, base, runtime, diag["n_iter"], diag["converged_all"])
                row.update(diag); row.update(B_metrics(Bh, data, model.B_initialization_mode_, diag["final_B_change_norm"]))
                row.update(Q_metrics(model.Q, data, model)); row.update(cmetrics)
                initial_B = np.asarray(model.B_initial_matrices_)
                row.update({"initial_B_relative_error": relative(initial_B, np.asarray(data["B"])),
                    "initial_B_energy_ratio_hat_to_true": np.sum(initial_B ** 2) / np.sum(np.asarray(data["B"]) ** 2),
                    "final_B_relative_error": row["B_relative_frobenius_error"],
                    "final_B_energy_ratio_hat_to_true": row["B_energy_ratio_hat_to_true"],
                    "relative_B_change_norm": diag["final_relative_B_change_norm"],
                    "precision_at_FPR_0p01": row.get("A_support_precision_at_FPR_0p01", np.nan),
                    "precision_at_FPR_0p05": row.get("A_support_precision_at_FPR_0p05", np.nan),
                    "F1_at_FPR_0p01": row.get("A_support_F1_at_FPR_0p01", np.nan),
                    "F1_at_FPR_0p05": row.get("A_support_F1_at_FPR_0p05", np.nan),
                    "pinv_proxy_mse": cmetrics["pinv_proxy_mse"], "pinv_proxy_correlation": cmetrics["pinv_proxy_correlation"],
                    "filtered_mse_improvement_over_pinv": cmetrics["pinv_proxy_mse"] - row["filtered_signal_mse"],
                    "smoothed_mse_improvement_over_pinv": cmetrics["pinv_proxy_mse"] - row["smoothed_signal_mse"],
                    "filtered_corr_improvement_over_pinv": row["filtered_signal_correlation"] - cmetrics["pinv_proxy_correlation"],
                    "smoothed_corr_improvement_over_pinv": row["smoothed_signal_correlation"] - cmetrics["pinv_proxy_correlation"],
                    "run_status": "success", "failure_rate": 0.0, "warning_count": len(messages),
                    "warning_messages": " | ".join(messages),
                    "numerical_warning_flag": bool(messages) or diag["numerical_warning_flag"]})
                tables["support"].append(support)
                for signal_type, signal in (("method_filtered", filtered), ("method_smoothed", smoothed)):
                    network, lgcs, scores = network_scores(signal, signal_type, data, base, compute_lgc=True)
                    tables["network"].append(network); tables["lgc"].extend(lgcs); tables["scores"].extend(scores)
                tables["scores"].append(support_score_rows(support, base))
                coeff, edges, _ = vb_details(model, data, base); edges["true_A_group_norm"] = edges.true_group_norm
                coeff["signed_error"] = coeff.posterior_mean - coeff.true_value
                coeff["squared_error"] = coeff.signed_error ** 2
                coeff["standardized_error"] = coeff.signed_error / coeff.posterior_std.replace(0, np.nan)
                history = iteration_rows(model, data, base)
                tables["uncertainty"].append(coeff); tables["vb_edges"].append(edges)
                tables["objective"].append(history); tables["convergence"].append(history)
                tables["scores"].extend(vb_score_rows(edges, base))
                snr, _ = safe_curve(edges.true_link, edges.group_snr_score)
                row.update({"group_snr_AUPRC": snr["AUPRC"],
                            "group_snr_TPR_at_FPR_0p01": snr["TPR_at_FPR_0p01"],
                            "group_snr_TPR_at_FPR_0p05": snr["TPR_at_FPR_0p05"]})
                status = "success"
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
                runtime = time.perf_counter() - started; status = "numerical_error"
                row = failure_row(base, runtime, status, error, messages)
            except Exception as error:
                runtime = time.perf_counter() - started; status = "failed_metric" if phase == "metric" else "failed_fit"
                row = failure_row(base, runtime, status, error, messages)
            tables["parameter"].append(row)
            tables["runtime"].append({**base, "runtime_seconds": runtime, "run_status": status,
                "failure_rate": float(status != "success"), "n_iter": row.get("n_iter", np.nan),
                "converged_all": row.get("converged_all", False), "practically_converged": row.get("practically_converged", False),
                "hit_max_iter": row.get("hit_max_iter", False), "spectral_radius_A": row.get("spectral_radius_A", np.nan)})
            run_rows.append({**base, "run_status": status, "runtime_seconds": runtime,
                             "n_iter": row.get("n_iter", np.nan), "error_message": row.get("error_message", "")})
            frames = materialize_tables(tables)
            for key, name in checkpoint.items():
                atomic_csv(pd.DataFrame(run_rows) if key == "run" else frames[key], os.path.join(RESULTS_DIR, name))
            if len(frames["parameter"]):
                partial = summaries(frames)
                for name, frame in zip(("parameter_summary_partial.csv", "B_recovery_summary_partial.csv", "Q_recovery_summary_partial.csv",
                    "A_support_summary_partial.csv", "rectangular_C_summary_partial.csv", "latent_recovery_summary_partial.csv",
                    "vb_edge_score_summary_partial.csv", "vb_uncertainty_calibration_summary_partial.csv", "roc_summary_partial.csv",
                    "_roc_points_partial.csv", "fixed_fpr_operating_points_partial.csv", "rectangular_C_robustness_summary_partial.csv"), partial):
                    atomic_csv(frame, os.path.join(RESULTS_DIR, name))
            progress.update(f"T={T} run={outer+1}/{N_OUTER_RUNS_BY_T[T]} {status}")
    frames = materialize_tables(tables)
    parameter, b, q, support, c, latent, edge, calibration, roc, points, fixed, robust = summaries(frames)
    decision = fixed.copy(); decision["FPR"] = decision.actual_fpr; decision["TPR"] = decision.tpr; decision["specificity"] = 1 - decision.actual_fpr
    outputs = {"run_summary.csv": pd.DataFrame(run_rows),
        "runtime_summary.csv": grouped_stats(frames["runtime"], GROUPS, ["runtime_seconds", "n_iter", "converged_all", "practically_converged", "hit_max_iter", "spectral_radius_A", "failure_rate"]),
        "parameter_results.csv": frames["parameter"], "parameter_summary.csv": parameter,
        "B_recovery_summary.csv": b, "Q_recovery_summary.csv": q, "A_support_summary.csv": support,
        "rectangular_C_summary.csv": c, "latent_recovery_summary.csv": latent,
        "network_results.csv": frames["network"],
        "link_strength_summary.csv": grouped_stats(frames["network"], DIM_GROUPS + ["method", "signal_type", "true_link"], ["raw_deviance", "debiased_deviance"]),
        "roc_summary.csv": roc, "roc_curve_points.csv": points, "fixed_fpr_operating_points.csv": fixed,
        "decision_summary.csv": decision,
        "lgc_decision_summary.csv": decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],
        "vb_edge_scores.csv": frames["vb_edges"], "vb_edge_score_summary.csv": edge,
        "vb_coefficient_uncertainty.csv": frames["uncertainty"],
        "vb_uncertainty_calibration_summary.csv": calibration,
        "objective_history.csv": frames["objective"], "convergence_diagnostics.csv": frames["convergence"],
        "rectangular_C_robustness_summary.csv": robust}
    for name, frame in outputs.items(): atomic_csv(frame, os.path.join(RESULTS_DIR, name))
    save_plots(frames["parameter"], edge, frames["vb_edges"], calibration)
    print(f"Experiment 34H complete. Outputs saved under {RESULTS_DIR}/")
    print("Interpretation guide: compare rectangular-C recovery with 34G; assess T=2000 rescue, group-SNR, VB smoothing versus the right-pseudoinverse proxy, B/Q recovery, convergence, and whether rectangular C belongs in static spectral-GC validation. Credible intervals remain diagnostics only; static GC and Louis correction remain deferred.")


if __name__ == "__main__":
    main()
