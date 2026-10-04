"""Experiment 34UE: historical multi-domain greedy Stage-B validation.

This experiment is a read-only replay of existing Stage-B results.  It never
simulates data, fits a state-space model, or starts LRVB finite differences.
The independent validation unit is always the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from itertools import combinations
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, spearmanr

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.diagnostic_subset_design import add_diagnostic_strata, stratified_maximin_order
from src.stats.greedy_stageb_selection import (
    exact_greedy_ratios,
    facility_objective,
    facility_order,
    logdet_objective,
    logdet_order,
    randomized_submodularity_check,
    rbf_similarity,
)


EPS = 1e-12
ALLOW_NEW_STAGEB = False
EXPERIMENT = "34UE_historical_multidomain_greedy_validation"
OUTPUT_ROOT = Path("results/experiment_34ue_historical_multidomain_greedy_validation")
SOURCES = {
    "34UC": Path("results/experiment_34uc_greedy_selector_validation/prospective_validation_5refs"),
    "34UA": Path("results/experiment_34u_a_fullrun_observability_calibration"),
    "34R": Path("results/experiment_34r_stage2"),
    "34Q-C": Path("results/experiment_34qC"),
    "34O": Path("results/experiment_34o"),
    "34Q-A": Path("results/experiment_34qA"),
    "34UD": Path("results/experiment_34ud_diagnostic_subset_design"),
}
RHO_GRID = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]
RAW_FEATURES = [
    "A_group_norm", "VB_cov_trace", "Louis_cov_trace", "logdet_Louis_cov",
    "Louis_group_SNR", "observability_product", "transfer_spectral_score",
]
FEATURES = [f"x__{name}" for name in RAW_FEATURES]
DETERMINISTIC = [
    "D0_current_ordering", "P0_purpose_topk", "D2_stratified_maximin",
    "D3_purpose_facility", "D4_purpose_logdet",
]
METRICS = [
    "retained_active_gain", "zero_coverage_delta_vs_full", "relative_Frobenius_error",
    "trace_relative_error", "mean_abs_logdet_error", "inflation_correlation",
    "correction_capture", "signed_correction_capture", "Q1_retained_active_gain",
]


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UE_SMOKE", "0") == "1")
    output = Path(args.results_dir) if args.results_dir else (OUTPUT_ROOT / "smoke_test" if smoke else OUTPUT_ROOT)
    return {
        "experiment": EXPERIMENT,
        "title": "Historical Multi-Domain Greedy Stage-B Selector Validation",
        "results_directory": str(output), "smoke_test": smoke,
        "primary_domain": "34UC", "supplemental_domain": "34UA",
        "stress_domains": ["34R", "34Q-C", "34O", "34Q-A"],
        "rho_grid": [0.50, 1.00] if smoke else RHO_GRID,
        "purpose_ridge": 1e-3, "facility_lambda_grid": [0.50, 0.75, 0.90],
        "facility_lambda_prior_anchor": 0.75,
        "facility_bandwidth": "training_median_nonzero_pairwise_distance",
        "logdet_delta": 1e-3,
        "random_repeats": env_int("EXPERIMENT_34UE_RANDOM_REPEATS", 10 if smoke else 500),
        "network_bootstrap_repeats": env_int("EXPERIMENT_34UE_BOOTSTRAP_REPEATS", 100 if smoke else 5000),
        "master_seed": 3405001,
        "truth_used_by_selector": False, "new_simulations_allowed": False,
        "new_stageB_fits_allowed": False, "ALLOW_NEW_STAGEB": ALLOW_NEW_STAGEB,
        "matrixfree_lrvb_enabled": False, "A_center": "A_VB",
        "stageB_need_target": "log1p(max((trace(Sigma_SB)-trace(Sigma_L))/max(trace(Sigma_L),1e-12),0))",
        "independent_unit": "network; never edge, direction, dataset replicate, C mode, or T",
        "core_feature_set": RAW_FEATURES,
        "selector_labels": {
            "D0": "current ordering", "U0": "uniform random", "D1": "stratified random",
            "P0": "purpose-only modular top-k", "D2": "34UD stratified maximin",
            "D3": "purpose-weighted facility location", "D4": "purpose-weighted log-det",
        },
        "primary_gain_threshold": 0.95, "catastrophic_gain_threshold": 0.80,
        "zero_coverage_tolerance": 0.02, "low_observability_gain_floor": 0.70,
        "material_rho_reduction": 0.10,
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def cv_checkpoint_signature(frame, config):
    """Fingerprint the scientific CV configuration and normalized input rows."""
    settings = {
        key: config[key]
        for key in (
            "rho_grid", "purpose_ridge", "facility_lambda_grid",
            "facility_lambda_prior_anchor", "logdet_delta", "random_repeats",
            "master_seed", "stageB_need_target", "core_feature_set",
        )
    }
    columns = [
        "run_id", "edge_id", "target_i", "source_j", "stageB_need_target",
        "true_active", *FEATURES,
    ]
    ordered = frame.sort_values(["run_id", "edge_id"])[columns].reset_index(drop=True)
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf8"))
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def prepare_cv_checkpoint(frame, config, output):
    checkpoint = Path(output) / "_checkpoints" / "34uc_cv"
    checkpoint.mkdir(parents=True, exist_ok=True)
    signature = cv_checkpoint_signature(frame, config)
    manifest_path = checkpoint / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf8") as handle:
            previous = json.load(handle)
        if previous.get("signature") != signature:
            raise RuntimeError(
                "Existing 34UE CV checkpoint does not match the current numerical "
                "configuration or historical input rows. Use a different results directory."
            )
    else:
        atomic_json({"signature": signature, "completed_folds": []}, manifest_path)
    return checkpoint, signature


def update_cv_checkpoint_manifest(checkpoint, signature, completed_folds):
    atomic_json(
        {"signature": signature, "completed_folds": sorted(map(int, completed_folds))},
        checkpoint / "manifest.json",
    )


class Progress:
    def __init__(self, total):
        self.total = max(int(total), 1); self.done = 0; self.started = time.perf_counter()
    def update(self, label=""):
        self.done += 1; fraction = min(self.done / self.total, 1.0); filled = round(30 * fraction)
        eta = (time.perf_counter() - self.started) / max(self.done, 1) * max(self.total - self.done, 0)
        print(f"\r34UE [{'#' * filled}{'-' * (30 - filled)}] {self.done}/{self.total} "
              f"{100 * fraction:5.1f}% ETA {eta / 60:5.1f}m {label[:28]}",
              end="\n" if self.done >= self.total else "", flush=True)


def initialize(config):
    if ALLOW_NEW_STAGEB or os.environ.get("EXPERIMENT_34UE_ALLOW_NEW_STAGEB", "0") == "1":
        raise RuntimeError("34UE is permanently read-only: new Stage-B execution is prohibited.")
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    atomic_json(config, output / "experiment_config.json"); (output / "plots").mkdir(exist_ok=True)
    return output


def safe_read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def config_value(path, *names, default=np.nan):
    if not path.exists(): return default
    data = json.loads(path.read_text(encoding="utf8"))
    for name in names:
        if name in data: return data[name]
    return default


def normalize_wide(frame, experiment, domain, reference_type):
    frame = frame.copy()
    rename = {"target": "target_i", "source": "source_j", "true_edge": "true_active",
              "estimated_group_norm": "A_group_norm", "vb_cov_trace": "VB_cov_trace",
              "louis_cov_trace": "Louis_cov_trace", "louis_group_snr": "Louis_group_SNR",
              "edge_o_inst_product": "observability_product",
              "spectral_transfer_full_band_score": "transfer_spectral_score",
              "edge_o_dyn_estA_mean_K10": "dynamic_observability",
              "stageb_cov_trace": "stageB_cov_trace", "relative_stageb_inflation": "stageB_relative_inflation"}
    frame = frame.rename(columns=rename)
    frame["experiment"] = experiment; frame["domain"] = domain
    frame["config_label"] = frame.get("candidate_support", reference_type)
    frame["Mx"] = frame.get("M_x", frame.get("M", np.nan)); frame["My"] = frame.get("M_y", np.nan)
    frame["C_mode"] = frame.get("C_MODE", frame.get("C_family", np.nan))
    frame["edge_id"] = frame.target_i.astype(int).astype(str) + "<-" + frame.source_j.astype(int).astype(str)
    frame["stageB_covariance_available"] = np.isfinite(frame.stageB_cov_trace)
    frame["stageB_minus_louis_trace"] = frame.stageB_cov_trace - frame.Louis_cov_trace
    frame["stageB_reference_type"] = reference_type
    frame["logdet_Louis_cov"] = frame.get("log_louis_cov_determinant",
                                           np.log(np.maximum(frame.get("louis_cov_determinant", np.nan), EPS)))
    frame["observability_quartile"] = frame.get("observability_quartile", np.nan)
    frame["run_status"] = "success"
    frame["selector_compatible"] = True
    frame["A_center"] = "A_VB"; frame["B_mode"] = "estimated_free"
    frame["Q_mode"] = "estimated_diag_shrink_scalar_rho_0p25"; frame["R_mode"] = "fixed_true"
    return frame


def normalize_long(path, experiment, domain, trace_column, target_column, source_column,
                   true_column, stageb_estimator, reference_type):
    raw = safe_read(path / "group_calibration_results.csv")
    if not len(raw): return pd.DataFrame()
    keys = [c for c in ["config_label", "M_x", "M_y", "M", "T", "C_MODE",
                         "true_network_id", "replicate_id", "run_id", target_column, source_column,
                         true_column, "edge_group_type", "group_error_norm"] if c in raw]
    value_columns = [trace_column]
    for name in ["D2", "mahalanobis_D2", "D2_below_chi2_95_df2", "group_snr",
                 "source_observability_score", "target_observability_score", "edge_observability_score"]:
        if name in raw: value_columns.append(name)
    pivots = {}
    for value in value_columns:
        pivots[value] = raw.pivot_table(index=keys, columns="covariance_estimator", values=value,
                                        aggfunc="first").reset_index()
    base = pivots[trace_column]
    louis = "stabilized_louis_eta_0p70_tau_0p90"; vb = "ordinary_vb"
    if louis not in base or stageb_estimator not in base: return pd.DataFrame()
    output = base[keys].copy()
    output["VB_cov_trace"] = base.get(vb, np.nan)
    output["Louis_cov_trace"] = base[louis]
    output["stageB_cov_trace"] = base[stageb_estimator]
    d2 = pivots.get("D2", pivots.get("mahalanobis_D2", pd.DataFrame()))
    covered = pivots.get("D2_below_chi2_95_df2", pd.DataFrame())
    if len(d2):
        output["Louis_group_SNR"] = np.sqrt(np.maximum(d2.get(louis, np.nan), 0.0))
    elif "group_snr" in pivots:
        output["Louis_group_SNR"] = pivots["group_snr"].get(louis, np.nan)
    else: output["Louis_group_SNR"] = np.nan
    if len(covered):
        output["louis_group_covered_95"] = covered.get(louis, np.nan)
        output["stageb_group_covered_95"] = covered.get(stageb_estimator, np.nan)
    output = output.rename(columns={target_column: "target_i", source_column: "source_j", true_column: "true_active"})
    output["experiment"] = experiment; output["domain"] = domain
    output["Mx"] = output.get("M_x", output.get("M", np.nan)); output["My"] = output.get("M_y", np.nan)
    output["C_mode"] = output.get("C_MODE", np.nan)
    if "run_id" not in output:
        output["run_id"] = (output.config_label.astype(str) + "_M" + output["Mx"].astype(int).astype(str)
                            + "_T" + output["T"].astype(int).astype(str)
                            + "_net" + output.true_network_id.astype(str)
                            + "_rep" + output.replicate_id.astype(str))
    output["edge_id"] = output.target_i.astype(int).astype(str) + "<-" + output.source_j.astype(int).astype(str)
    output["A_group_norm"] = np.nan; output["logdet_Louis_cov"] = np.nan
    output["observability_product"] = output.get("edge_observability_score", np.nan)
    output["transfer_spectral_score"] = np.nan; output["dynamic_observability"] = np.nan
    output["stageB_relative_inflation"] = ((output.stageB_cov_trace - output.Louis_cov_trace)
                                            / np.maximum(output.Louis_cov_trace, EPS))
    output["stageB_minus_louis_trace"] = output.stageB_cov_trace - output.Louis_cov_trace
    output["stageB_covariance_available"] = np.isfinite(output.stageB_cov_trace)
    output["stageB_reference_type"] = reference_type; output["observability_quartile"] = np.nan
    output["run_status"] = "success"; output["selector_compatible"] = False
    output["A_center"] = "A_VB"
    output["B_mode"] = (
        "reestimated_in_perturbed_fits" if experiment == "34Q-A"
        else "historical_variant"
    )
    output["Q_mode"] = "historical_variant"; output["R_mode"] = "fixed_true"
    return output


def discover_and_normalize(config):
    manifest = []
    uc_path, ua_path = SOURCES["34UC"], SOURCES["34UA"]
    uc_raw = safe_read(uc_path / "new_reference_edge_rows_partial.csv")
    uc_runtime = safe_read(uc_path / "new_reference_runtime_partial.csv")
    success = set(uc_runtime.loc[uc_runtime.run_status.eq("success"), "run_id"])
    uc_raw = uc_raw[uc_raw.run_id.isin(success)].copy()
    if config["smoke_test"]: uc_raw = uc_raw[uc_raw.run_id.isin(sorted(uc_raw.run_id.unique())[:2])]
    uc = normalize_wide(uc_raw, "34UC", "primary", "deployable_reference_90")
    ua_raw = safe_read(ua_path / "calibration_edge_rows_partial.csv")
    ua_runtime = safe_read(ua_path / "runtime_summary_partial.csv")
    ua_success = set(ua_runtime.loc[ua_runtime.run_status.eq("success"), "run_id"])
    ua_raw = ua_raw[(ua_raw.run_id.isin(ua_success)) & (ua_raw.M_y.eq(40)) &
                    (ua_raw.candidate_support.eq("full_candidate"))].copy()
    if config["smoke_test"]: ua_raw = ua_raw[ua_raw.run_id.eq(sorted(ua_raw.run_id.unique())[0])]
    ua = normalize_wide(ua_raw, "34UA", "supplemental_transfer", "truth_stratified_benchmark_64")

    r = normalize_long(SOURCES["34R"], "34R", "rectangular_C_stress", "group_cov_trace",
                       "group_target", "group_source", "true_edge_group",
                       "lrvb_stageB_smoother_feedback_psd_projected", "selected_subset")
    qc = normalize_long(SOURCES["34Q-C"], "34Q-C", "C_geometry_stress", "group_cov_trace",
                        "group_target", "group_source", "true_edge_group",
                        "lrvb_stageB_smoother_feedback_psd_projected", "selected_subset")
    o = normalize_long(SOURCES["34O"], "34O", "repeatability_fixedB", "group_posterior_sd_trace",
                       "target", "source", "true_link",
                       "lrvb_stageB_smoother_feedback_psd_projected", "selected_subset")
    qa = normalize_long(SOURCES["34Q-A"], "34Q-A", "repeatability_reestimatedB", "group_cov_trace",
                        "group_target", "group_source", "true_edge_group",
                        "lrvb_stageB_smoother_feedback_psd_projected", "selected_subset")
    if config["smoke_test"]:
        r = r.iloc[:0]; qc = qc.iloc[:0]; o = o.iloc[:0]; qa = qa.iloc[:0]
    panels = [item for item in [uc, ua, r, qc, o, qa] if len(item)]
    panel = pd.concat(panels, ignore_index=True, sort=False)
    for experiment, path in SOURCES.items():
        if experiment == "34UD":
            data_file = path / "decision_summary.csv"
        elif experiment == "34UC": data_file = path / "new_reference_edge_rows_partial.csv"
        elif experiment == "34UA": data_file = path / "calibration_edge_rows_partial.csv"
        else: data_file = path / "group_calibration_results.csv"
        subset = panel[panel.experiment.eq(experiment)] if experiment in set(panel.experiment) else pd.DataFrame()
        config_path = path / "experiment_config.json"
        manifest.append({
            "experiment": experiment, "source_directory": str(path), "source_file": str(data_file),
            "directory_exists": path.exists(), "config_exists": config_path.exists(),
            "completion_marker_exists": (path / "_COMPLETED.json").exists(),
            "file_exists": data_file.exists(), "usable_runs": subset.run_id.nunique() if len(subset) and "run_id" in subset else 0,
            "usable_edge_rows": len(subset), "selector_compatible_runs": subset.loc[subset.selector_compatible].run_id.nunique() if len(subset) else 0,
            "audit_status": "usable_primary" if experiment == "34UC" and len(subset) else
                            "usable_supplemental" if experiment == "34UA" and len(subset) else
                            "usable_limited_schema" if len(subset) else
                            "metadata_only" if experiment == "34UD" and data_file.exists() else "missing_or_incompatible",
            "limitations": "none" if experiment == "34UC" else
                           "truth-stratified benchmark pool; analyze separately" if experiment == "34UA" else
                           "missing full core selector features; repeatability/stress metrics only" if experiment in {"34R", "34Q-C", "34O", "34Q-A"} else
                           "selector definitions and labels reused; not a Stage-B source",
        })
    return panel, pd.DataFrame(manifest)


def transform_features(frame):
    frame = frame.copy()
    for raw, feature in zip(RAW_FEATURES, FEATURES):
        values = pd.to_numeric(frame[raw], errors="coerce")
        if raw == "logdet_Louis_cov": frame[feature] = values
        else: frame[feature] = np.log1p(np.maximum(values, 0.0))
    frame["stageB_need_target"] = np.log1p(np.maximum(frame.stageB_relative_inflation, 0.0))
    return frame


def manifests(panel):
    rows = []
    for run_id, group in panel.groupby("run_id", sort=False):
        first = group.iloc[0]
        rows.append({
            "experiment": first.experiment, "domain": first.domain, "source_directory": str(SOURCES[first.experiment]),
            "config_label": first.config_label, "Mx": first.Mx, "My": first.My, "T": first["T"],
            "C_mode": first.C_mode, "true_network_id": first.true_network_id,
            "replicate_id": first.replicate_id, "run_id": run_id,
            "candidate_support": first.get("candidate_support", first.stageB_reference_type),
            "number_candidate_groups": len(group),
            "number_available_stageB_groups": int(group.stageB_covariance_available.sum()),
            "A_center": first.A_center, "B_mode": first.B_mode, "Q_mode": first.Q_mode, "R_mode": first.R_mode,
            "finite_difference_epsilon": config_value(SOURCES[first.experiment] / "experiment_config.json", "PERTURB_EPS", "FINITE_DIFFERENCE_EPSILON"),
            "perturbed_VB_max_iter": config_value(SOURCES[first.experiment] / "experiment_config.json", "PERTURBED_VB_MAX_ITER"),
            "StageB_reference_type": first.stageB_reference_type, "run_status": first.run_status,
            "completeness": bool(group.stageB_covariance_available.all()),
            "selector_compatible": bool(group.selector_compatible.all()),
            "independent_network_unit": f"{first.experiment}:network_{int(first.true_network_id)}",
            "statistical_role": "independent_primary_network" if first.experiment == "34UC" else
                                "independent_supplemental_network" if first.experiment == "34UA" else
                                "paired/repeated stress realization; not an independent network",
        })
    return pd.DataFrame(rows)


def assert_no_leakage(train, heldout_id=None):
    forbidden = {"true_active", "true_A", "oracle_support", "oracle_lambda", "true_group_norm"}
    assert not forbidden.intersection(FEATURES)
    if heldout_id is not None: assert heldout_id not in set(train.run_id)
    assert np.isfinite(train[FEATURES].to_numpy(float)).all()


def fit_selector_state(train, config, fixed_lambda=None):
    assert_no_leakage(train)
    X = train[FEATURES].to_numpy(float); y = train.stageB_need_target.to_numpy(float)
    mean = X.mean(axis=0); scale = X.std(axis=0); scale[scale <= EPS] = 1.0
    Z = (X - mean) / scale
    design = np.column_stack([np.ones(len(Z)), Z])
    penalty = config["purpose_ridge"] * np.eye(design.shape[1]); penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    distance = pdist(Z); positive = distance[np.isfinite(distance) & (distance > 0)]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    state = {"mean": mean, "scale": scale, "coefficients": coefficients, "bandwidth": bandwidth,
             "facility_lambda": fixed_lambda, "delta": config["logdet_delta"]}
    return state


def predict_state(frame, state):
    Z = (frame[FEATURES].to_numpy(float) - state["mean"]) / state["scale"]
    predicted = np.column_stack([np.ones(len(Z)), Z]) @ state["coefficients"]
    q = np.maximum(predicted, 0.0)
    if np.max(q, initial=0.0) <= EPS: q = np.full(len(q), EPS)
    return Z, predicted, q


def build_orders(frame, state, config, include_random=False, seed=0):
    frame = frame.reset_index(drop=True); Z, predicted, q = predict_state(frame, state); n = len(frame)
    current_key = "source_row_order" if "source_row_order" in frame and frame.source_row_order.notna().all() else None
    current = (np.argsort(frame[current_key].to_numpy(float), kind="stable") if current_key
               else np.argsort(-frame.Louis_group_SNR.to_numpy(float), kind="stable"))
    strata_frame = frame.rename(columns={"target_i": "target", "source_j": "source"})
    strata_frame["log_edge_o_inst_product"] = np.log(np.maximum(strata_frame.observability_product, EPS))
    strata_frame["louis_group_snr"] = strata_frame.Louis_group_SNR
    strata_frame = add_diagnostic_strata(strata_frame)
    maximin = stratified_maximin_order(Z, strata_frame.diagnostic_stratum, n, "uncertainty")
    similarity = rbf_similarity(Z, state["bandwidth"])
    lam = float(state["facility_lambda"] if state["facility_lambda"] is not None else .75)
    facility = [row["index"] for row in facility_order(similarity, q, lam, "purpose")]
    phi = np.column_stack([np.ones(n), Z])
    logdet = [row["index"] for row in logdet_order(phi, q, state["delta"])]
    orders = {
        "D0_current_ordering": list(map(int, current)),
        "P0_purpose_topk": list(map(int, np.argsort(-q, kind="stable"))),
        "D2_stratified_maximin": list(map(int, maximin)),
        "D3_purpose_facility": list(map(int, facility)),
        "D4_purpose_logdet": list(map(int, logdet)),
    }
    return orders, Z, predicted, q, strata_frame


def covariance_blocks(frame, prefix):
    values = np.empty((len(frame), 2, 2), float)
    values[:, 0, 0] = frame[f"Sigma_{prefix}_00"]
    values[:, 0, 1] = frame[f"Sigma_{prefix}_01"]
    values[:, 1, 0] = frame[f"Sigma_{prefix}_10"]
    values[:, 1, 1] = frame[f"Sigma_{prefix}_11"]
    return values


def replay_metrics(frame, order, rho):
    frame = frame.reset_index(drop=True); n = len(frame); k = int(np.clip(np.ceil(rho * n), 1, n))
    selected = np.zeros(n, bool); selected[np.asarray(order[:k], int)] = True
    active = frame.true_active.astype(bool).to_numpy(); zero = ~active
    louis_cov = frame.louis_group_covered_95.astype(bool).to_numpy()
    stage_cov = frame.stageb_group_covered_95.astype(bool).to_numpy()
    hybrid_covered = np.where(selected, stage_cov, louis_cov)
    c_l = louis_cov[active].mean() if active.any() else np.nan
    c_sb = stage_cov[active].mean() if active.any() else np.nan
    c_k = hybrid_covered[active].mean() if active.any() else np.nan
    retained = (c_k - c_l) / max(c_sb - c_l, EPS) if np.isfinite(c_l + c_sb + c_k) else np.nan
    zero_l = louis_cov[zero].mean() if zero.any() else np.nan
    zero_full = stage_cov[zero].mean() if zero.any() else np.nan
    zero_k = hybrid_covered[zero].mean() if zero.any() else np.nan
    louis = covariance_blocks(frame, "louis"); stage = covariance_blocks(frame, "stageb")
    hybrid = louis.copy(); hybrid[selected] = stage[selected]
    relative_fro = np.linalg.norm(hybrid - stage) / max(np.linalg.norm(stage), EPS)
    trace_stage = np.trace(stage, axis1=1, axis2=2); trace_louis = np.trace(louis, axis1=1, axis2=2)
    trace_hybrid = np.trace(hybrid, axis1=1, axis2=2)
    trace_error = np.linalg.norm(trace_hybrid - trace_stage) / max(np.linalg.norm(trace_stage), EPS)
    logdet_stage = np.log(np.maximum(np.linalg.det(stage), EPS)); logdet_hybrid = np.log(np.maximum(np.linalg.det(hybrid), EPS))
    correction = trace_stage - trace_louis; hybrid_correction = trace_hybrid - trace_louis
    positive = np.maximum(correction, 0.0)
    capture = float(positive[selected].sum() / max(positive.sum(), EPS))
    signed = float(correction[selected].sum() / max(abs(correction).sum(), EPS))
    corr = np.corrcoef(correction, hybrid_correction)[0, 1] if np.std(correction) > EPS and np.std(hybrid_correction) > EPS else np.nan
    qbin = pd.qcut(frame.observability_product.rank(method="first"), 4, labels=False).to_numpy(int)
    q1_active = active & (qbin == 0)
    q1_l = louis_cov[q1_active].mean() if q1_active.any() else np.nan
    q1_sb = stage_cov[q1_active].mean() if q1_active.any() else np.nan
    q1_k = hybrid_covered[q1_active].mean() if q1_active.any() else np.nan
    q1_denominator = q1_sb - q1_l if np.isfinite(q1_l + q1_sb) else np.nan
    q1_gain = ((q1_k - q1_l) / q1_denominator
               if np.isfinite(q1_l + q1_sb + q1_k) and q1_denominator > EPS else np.nan)
    return {
        "N": n, "rho": float(rho), "k": k, "selected_fraction": k / n,
        "active_coverage_Louis": c_l, "active_coverage_full_stageB": c_sb,
        "active_coverage_k": c_k, "retained_active_gain": retained,
        "zero_coverage_Louis": zero_l, "zero_coverage_full_stageB": zero_full,
        "zero_coverage_k": zero_k, "zero_coverage_delta_vs_full": zero_k - zero_full,
        "relative_Frobenius_error": relative_fro, "trace_relative_error": trace_error,
        "mean_abs_logdet_error": float(np.mean(np.abs(logdet_hybrid - logdet_stage))),
        "inflation_correlation": corr, "correction_capture": capture,
        "signed_correction_capture": signed, "Q1_retained_active_gain": q1_gain,
        "unique_source_nodes": int(frame.loc[selected, "source_j"].nunique()),
        "unique_target_nodes": int(frame.loc[selected, "target_i"].nunique()),
    }


def method_required_rho(curves, config):
    rows = []
    for (method, rho), group in curves.groupby(["selector", "rho"], sort=False):
        rows.append({"selector": method, "rho": rho,
                     "mean_gain": group.retained_active_gain.mean(), "min_gain": group.retained_active_gain.min(),
                     "mean_abs_zero_delta": group.zero_coverage_delta_vs_full.abs().mean(),
                     "min_Q1_gain": group.Q1_retained_active_gain.min()})
    summary = pd.DataFrame(rows)
    required = {}
    for method, group in summary.groupby("selector"):
        valid = group[(group.mean_gain >= config["primary_gain_threshold"] - EPS) &
                      (group.min_gain >= config["catastrophic_gain_threshold"] - EPS) &
                      (group.mean_abs_zero_delta <= config["zero_coverage_tolerance"] + EPS) &
                      ((group.min_Q1_gain >= config["low_observability_gain_floor"] - EPS) | group.min_Q1_gain.isna())]
        required[method] = float(valid.rho.min()) if len(valid) else np.nan
    return required, summary


def choose_lambda(train, base_state, config):
    best = None
    for lam in config["facility_lambda_grid"]:
        state = dict(base_state); state["facility_lambda"] = lam; rows = []
        for run_id, group in train.groupby("run_id"):
            orders, *_ = build_orders(group, state, config)
            for rho in config["rho_grid"]:
                rows.append(replay_metrics(group, orders["D3_purpose_facility"], rho))
        curve = pd.DataFrame(rows)
        candidate = curve.groupby("rho").retained_active_gain.mean()
        passing = candidate[candidate >= config["primary_gain_threshold"]]
        required = float(passing.index.min()) if len(passing) else 9.0
        score = (required, -candidate.max(), abs(lam - config["facility_lambda_prior_anchor"]))
        if best is None or score < best[0]: best = (score, lam)
    return float(best[1])


def proportional_random_order(labels, rng):
    labels = np.asarray(labels, int); bins = {label: list(np.where(labels == label)[0]) for label in sorted(set(labels))}
    for values in bins.values(): rng.shuffle(values)
    counts = {label: 0 for label in bins}; population = {label: len(values) for label, values in bins.items()}
    positions = {label: 0 for label in bins}; total = len(labels); order = []
    while len(order) < total:
        rank = len(order) + 1
        available = [label for label in bins if positions[label] < len(bins[label])]
        label = max(available, key=lambda item: (rank * population[item] / total - counts[item], -item))
        order.append(int(bins[label][positions[label]])); positions[label] += 1; counts[label] += 1
    return order


def random_replays(frame, q, fold, config):
    rows = []; n = len(frame); labels = pd.qcut(pd.Series(q).rank(method="first"), 4, labels=False).to_numpy(int)
    for repeat in range(config["random_repeats"]):
        rng = np.random.default_rng(config["master_seed"] + 100003 * fold + repeat)
        orders = {"U0_uniform_random": list(map(int, rng.permutation(n))),
                  "D1_stratified_random": proportional_random_order(labels, rng)}
        for method, order in orders.items():
            for rho in config["rho_grid"]:
                rows.append({"fold": fold, "selector": method, "random_repeat": repeat,
                             **replay_metrics(frame, order, rho)})
    raw = pd.DataFrame(rows); summary = []
    for (method, rho), group in raw.groupby(["selector", "rho"]):
        row = {"fold": fold, "run_id": frame.run_id.iloc[0], "selector": method, "rho": rho,
               "random_repeats": config["random_repeats"]}
        for metric in METRICS:
            values = group[metric]
            row.update({f"{metric}_mean": values.mean(), f"{metric}_median": values.median(),
                        f"{metric}_q05": values.quantile(.05), f"{metric}_q95": values.quantile(.95)})
        summary.append(row)
    return pd.DataFrame(summary)


def primary_cv(uc, config, output, progress=None):
    runs = sorted(uc.run_id.unique()); folds = []; predictions = []; order_rows = []; curves = []; random_rows = []
    checkpoint, signature = prepare_cv_checkpoint(uc, config, output)
    completed_folds = []
    for fold, heldout in enumerate(runs):
        paths = {
            "fold": checkpoint / f"fold_{fold:02d}_config.csv",
            "predictions": checkpoint / f"fold_{fold:02d}_predictions.csv",
            "orderings": checkpoint / f"fold_{fold:02d}_orderings.csv",
            "curves": checkpoint / f"fold_{fold:02d}_curves.csv",
            "random": checkpoint / f"fold_{fold:02d}_random.csv",
        }
        if all(path.exists() for path in paths.values()):
            fold_frame = pd.read_csv(paths["fold"])
            if len(fold_frame) != 1 or fold_frame.heldout_run_id.iloc[0] != heldout:
                raise RuntimeError(f"Invalid 34UE checkpoint identity for fold {fold}.")
            folds.extend(fold_frame.to_dict("records"))
            predictions.extend(pd.read_csv(paths["predictions"]).to_dict("records"))
            order_rows.extend(pd.read_csv(paths["orderings"]).to_dict("records"))
            curves.extend(pd.read_csv(paths["curves"]).to_dict("records"))
            random_rows.append(pd.read_csv(paths["random"]))
            completed_folds.append(fold)
            if progress: progress.update("restored 34UC network CV checkpoint")
            continue
        train = uc[~uc.run_id.eq(heldout)].copy(); test = uc[uc.run_id.eq(heldout)].copy().reset_index(drop=True)
        assert_no_leakage(train, heldout)
        state = fit_selector_state(train, config); state["facility_lambda"] = choose_lambda(train, state, config)
        orders, Z, predicted, q, strata = build_orders(test, state, config)
        fold_rows = [{"fold": fold, "heldout_run_id": heldout, "training_run_ids": "|".join(sorted(train.run_id.unique())),
                      "n_training_networks": train.run_id.nunique(), "n_training_edges": len(train),
                      "facility_lambda": state["facility_lambda"], "facility_bandwidth": state["bandwidth"],
                      "purpose_ridge": config["purpose_ridge"], "scaler_mean": json.dumps(state["mean"].tolist()),
                      "scaler_scale": json.dumps(state["scale"].tolist()),
                      "purpose_coefficients": json.dumps(state["coefficients"].tolist()),
                      "heldout_used_in_training": False}]
        prediction_rows = []
        for index, row in test.iterrows():
            prediction_rows.append({"fold": fold, "run_id": heldout, "target_i": row.target_i, "source_j": row.source_j,
                                    "predicted_stageB_need": predicted[index], "nonnegative_purpose_weight": q[index],
                                    "observed_stageB_need": row.stageB_need_target})
        ordering_rows = []; curve_rows = []
        for method, order in orders.items():
            for rank, index in enumerate(order, 1):
                row = test.iloc[index]
                ordering_rows.append({"fold": fold, "run_id": heldout, "selector": method, "rank": rank,
                                      "target_i": row.target_i, "source_j": row.source_j, "edge_id": row.edge_id,
                                      "predicted_stageB_need": predicted[index], "true_active_evaluation_only": row.true_active})
            for rho in config["rho_grid"]:
                curve_rows.append({"fold": fold, "run_id": heldout, "selector": method,
                                   **replay_metrics(test, order, rho)})
        random_frame = random_replays(test, q, fold, config)
        fold_frame = pd.DataFrame(fold_rows); prediction_frame = pd.DataFrame(prediction_rows)
        ordering_frame = pd.DataFrame(ordering_rows); curve_frame = pd.DataFrame(curve_rows)
        atomic_csv(fold_frame, str(paths["fold"])); atomic_csv(prediction_frame, str(paths["predictions"]))
        atomic_csv(ordering_frame, str(paths["orderings"])); atomic_csv(curve_frame, str(paths["curves"]))
        atomic_csv(random_frame, str(paths["random"]))
        folds.extend(fold_rows); predictions.extend(prediction_rows); order_rows.extend(ordering_rows); curves.extend(curve_rows)
        random_rows.append(random_frame); completed_folds.append(fold)
        update_cv_checkpoint_manifest(checkpoint, signature, completed_folds)
        if progress: progress.update("34UC network CV")
    return (pd.DataFrame(folds), pd.DataFrame(predictions), pd.DataFrame(order_rows),
            pd.DataFrame(curves), pd.concat(random_rows, ignore_index=True))


def aggregate_curves(curves, config):
    rows = []; rng = np.random.default_rng(config["master_seed"] + 99)
    for (method, rho), group in curves.groupby(["selector", "rho"], sort=False):
        row = {"selector": method, "rho": rho, "n_independent_networks": group.run_id.nunique()}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            row.update({f"{metric}_mean": np.mean(values) if len(values) else np.nan,
                        f"{metric}_median": np.median(values) if len(values) else np.nan,
                        f"{metric}_min": np.min(values) if len(values) else np.nan,
                        f"{metric}_max": np.max(values) if len(values) else np.nan})
            if len(values):
                samples = rng.choice(values, (config["network_bootstrap_repeats"], len(values)), replace=True).mean(axis=1)
                row[f"{metric}_network_bootstrap_q025"] = np.quantile(samples, .025)
                row[f"{metric}_network_bootstrap_q975"] = np.quantile(samples, .975)
        rows.append(row)
    return pd.DataFrame(rows)


def final_state_and_transfer(uc, ua, config):
    state = fit_selector_state(uc, config); state["facility_lambda"] = choose_lambda(uc, state, config)
    per_network = []; order_rows = []
    for run_id, group in ua.groupby("run_id", sort=False):
        group = group.reset_index(drop=True); orders, _, predicted, q, _ = build_orders(group, state, config)
        for method, order in orders.items():
            for rank, index in enumerate(order, 1):
                edge = group.iloc[index]
                order_rows.append({"domain": "34UA", "run_id": run_id, "selector": method, "rank": rank,
                                   "edge_id": edge.edge_id, "target_i": edge.target_i, "source_j": edge.source_j})
            for rho in config["rho_grid"]:
                per_network.append({"run_id": run_id, "selector": method, **replay_metrics(group, order, rho)})
    per_network = pd.DataFrame(per_network)
    summary = aggregate_curves(per_network.assign(fold=np.nan), config) if len(per_network) else pd.DataFrame()
    return state, per_network, summary, pd.DataFrame(order_rows)


def pair_stability(first, second, rho):
    merged = first[["edge_id", "stageB_relative_inflation"]].merge(
        second[["edge_id", "stageB_relative_inflation"]], on="edge_id", suffixes=("_a", "_b"))
    if len(merged) < 3: return None
    n_a, n_b = len(first), len(second); k_a = int(np.ceil(rho * n_a)); k_b = int(np.ceil(rho * n_b))
    top_a = set(first.nlargest(k_a, "stageB_relative_inflation").edge_id)
    top_b = set(second.nlargest(k_b, "stageB_relative_inflation").edge_id)
    intersection = len(top_a & top_b); union = len(top_a | top_b)
    return {"n_common_edges": len(merged),
            "stageB_need_spearman": spearmanr(merged.stageB_relative_inflation_a, merged.stageB_relative_inflation_b).statistic,
            "stageB_need_kendall": kendalltau(merged.stageB_relative_inflation_a, merged.stageB_relative_inflation_b).statistic,
            "topk_jaccard": intersection / max(union, 1),
            "topk_overlap_coefficient": intersection / max(min(len(top_a), len(top_b)), 1)}


def stress_repeatability(panel, recommended_rho):
    all_rows = []
    for experiment in ["34R", "34Q-C", "34O", "34Q-A"]:
        domain = panel[panel.experiment.eq(experiment)]
        runs = {run_id: group for run_id, group in domain.groupby("run_id")}
        for run_a, run_b in combinations(sorted(runs), 2):
            a, b = runs[run_a], runs[run_b]
            same_network = int(a.true_network_id.iloc[0]) == int(b.true_network_id.iloc[0])
            if not same_network: continue
            if int(a.Mx.iloc[0]) != int(b.Mx.iloc[0]): continue
            if experiment in {"34O", "34Q-A"} and str(a.config_label.iloc[0]) != str(b.config_label.iloc[0]):
                continue
            result = pair_stability(a, b, recommended_rho)
            if result is None: continue
            my_a, my_b = a.My.iloc[0], b.My.iloc[0]
            t_a, t_b = a["T"].iloc[0], b["T"].iloc[0]
            c_a, c_b = a.C_mode.iloc[0], b.C_mode.iloc[0]
            if experiment == "34R":
                comparison_type = "within_My40_repeat" if my_a == my_b == 40 else (
                    "within_My15_repeat" if my_a == my_b == 15 else "My40_vs_My15")
            elif experiment == "34Q-C":
                comparison_type = "same_C_repeat" if c_a == c_b and t_a == t_b else (
                    "cross_C_mode" if c_a != c_b and t_a == t_b else "cross_T")
            else:
                comparison_type = "same_T_repeat" if t_a == t_b else "T1000_vs_T2000"
            all_rows.append({"experiment": experiment, "run_id_a": run_a, "run_id_b": run_b,
                             "config_a": a.config_label.iloc[0], "config_b": b.config_label.iloc[0],
                             "My_a": my_a, "My_b": my_b, "T_a": t_a, "T_b": t_b,
                             "C_mode_a": c_a, "C_mode_b": c_b, "comparison_type": comparison_type,
                             "rho": recommended_rho, "frozen_selector_applied": False,
                             "limitation": "full seven-feature frozen selector unavailable; historical Stage-B-need repeatability only",
                             **result})
    pairs = pd.DataFrame(all_rows)
    def summarize(experiment):
        subset = pairs[pairs.experiment.eq(experiment)] if "experiment" in pairs else pd.DataFrame()
        if not len(subset): return pd.DataFrame([{"experiment": experiment, "status": "insufficient_compatible_pairs"}])
        rows = []
        for comparison_type, group in subset.groupby("comparison_type", dropna=False):
            rows.append({"experiment": experiment, "status": "historical_need_repeatability_only",
                         "comparison_type": comparison_type, "n_paired_comparisons": len(group),
                         "mean_spearman": group.stageB_need_spearman.mean(),
                         "mean_kendall": group.stageB_need_kendall.mean(),
                         "mean_topk_jaccard": group.topk_jaccard.mean(),
                         "mean_topk_overlap_coefficient": group.topk_overlap_coefficient.mean(),
                         "frozen_selector_applied": False})
        return pd.DataFrame(rows)
    return pairs, {name: summarize(name) for name in ["34R", "34Q-C", "34O", "34Q-A"]}


def selector_stability(orderings, rho):
    rows = []
    for method, block in orderings.groupby("selector"):
        groups = {run_id: group.sort_values("rank") for run_id, group in block.groupby("run_id")}
        for a, b in combinations(sorted(groups), 2):
            ga, gb = groups[a], groups[b]
            merged = ga[["edge_id", "rank"]].merge(gb[["edge_id", "rank"]], on="edge_id", suffixes=("_a", "_b"))
            ka, kb = int(np.ceil(rho * len(ga))), int(np.ceil(rho * len(gb)))
            sa, sb = set(ga.head(ka).edge_id), set(gb.head(kb).edge_id)
            rows.append({"experiment": "34UC", "selector": method, "run_id_a": a, "run_id_b": b,
                         "rho": rho, "n_common_edges": len(merged),
                         "spearman_rank": spearmanr(merged.rank_a, merged.rank_b).statistic if len(merged) >= 3 else np.nan,
                         "kendall_tau": kendalltau(merged.rank_a, merged.rank_b).statistic if len(merged) >= 3 else np.nan,
                         "topk_jaccard": len(sa & sb) / max(len(sa | sb), 1),
                         "topk_overlap_coefficient": len(sa & sb) / max(min(len(sa), len(sb)), 1)})
    return pd.DataFrame(rows)


def theoretical_checks(config):
    rng = np.random.default_rng(config["master_seed"]); Z = rng.normal(size=(12, 7)); q = rng.uniform(.1, 1, 12)
    similarity = rbf_similarity(Z, np.median(pdist(Z))); phi = np.column_stack([np.ones(12), Z])
    rows = []; exact_rows = []
    for method in ("facility", "logdet"):
        if method == "facility":
            objective = lambda selected: facility_objective(selected, similarity, q, .75, "purpose")
            order = facility_order(similarity, q, .75, "purpose")
        else:
            objective = lambda selected: logdet_objective(selected, phi, q, config["logdet_delta"])
            order = logdet_order(phi, q, config["logdet_delta"])
        check = randomized_submodularity_check(objective, 12, trials=1000, seed=config["master_seed"])
        rows.append({"selector": method, **check})
        for item in exact_greedy_ratios(objective, order, n_small=12, budgets=(3, 4)):
            exact_rows.append({"selector": method, **item})
    return pd.DataFrame(rows), pd.DataFrame(exact_rows)


def decide(primary_summary, primary_curves, transfer_summary, config):
    required, diagnostics = method_required_rho(primary_curves, config)
    candidates = ["P0_purpose_topk", "D2_stratified_maximin", "D3_purpose_facility", "D4_purpose_logdet"]
    complexity = {"P0_purpose_topk": 0, "D2_stratified_maximin": 1,
                  "D3_purpose_facility": 2, "D4_purpose_logdet": 3}
    ranked = sorted(candidates, key=lambda name: (required.get(name, np.inf) if np.isfinite(required.get(name, np.nan)) else np.inf,
                                                   complexity[name]))
    winner, runner = ranked[:2]; rho = required.get(winner, np.nan)
    if not np.isfinite(rho): rho = 1.0
    chosen = primary_summary[(primary_summary.selector.eq(winner)) & np.isclose(primary_summary.rho, rho)].iloc[0]
    transfer_required, _ = method_required_rho(
        transfer_summary.rename(columns={c: c.replace("_mean", "") for c in []}), config
    ) if False else ({}, pd.DataFrame())
    if len(transfer_summary):
        t = transfer_summary[transfer_summary.selector.eq(winner)]
        valid = t[(t.retained_active_gain_mean >= .95 - EPS) &
                  (t.retained_active_gain_min >= .80 - EPS)]
        ua_rho = float(valid.rho.min()) if len(valid) else np.nan
    else: ua_rho = np.nan
    purpose_rho = required.get("P0_purpose_topk", np.nan)
    facility_better = np.isfinite(purpose_rho) and np.isfinite(required.get("D3_purpose_facility", np.nan)) and purpose_rho - required["D3_purpose_facility"] >= .10
    logdet_better = np.isfinite(purpose_rho) and np.isfinite(required.get("D4_purpose_logdet", np.nan)) and purpose_rho - required["D4_purpose_logdet"] >= .10
    ready = bool(np.isfinite(required.get(winner, np.nan)) and rho < 1.0 and
                 chosen.retained_active_gain_min >= .80 - EPS)
    row = {
        "primary_number_34uc_networks": primary_curves.run_id.nunique(),
        "supplemental_number_34ua_networks": int(transfer_summary.n_independent_networks.max()) if len(transfer_summary) else 0,
        "best_selector": winner, "runner_up_selector": runner,
        "recommended_budget_fraction": rho, "recommended_k_if_N90": int(np.ceil(rho * 90)),
        "recommended_k_if_N64": int(np.ceil(rho * 64)), "smallest_rho_mean_gain_0p95": required.get(winner),
        "minimum_network_gain_at_recommended_rho": chosen.retained_active_gain_min,
        "zero_coverage_delta_at_recommended_rho": chosen.zero_coverage_delta_vs_full_mean,
        "low_observability_gain_at_recommended_rho": chosen.Q1_retained_active_gain_mean,
        "covariance_error_at_recommended_rho": chosen.relative_Frobenius_error_mean,
        "correction_capture_at_recommended_rho": chosen.correction_capture_mean,
        "beats_uniform_random": np.nan, "beats_stratified_random": np.nan,
        "transfers_to_34ua": bool(np.isfinite(ua_rho)), "34ua_required_rho": ua_rho,
        "stable_in_34r_My40": np.nan, "degradation_in_34r_My15": np.nan,
        "stable_across_34qc_C_modes": np.nan, "repeatable_in_34o": np.nan, "repeatable_in_34qa": np.nan,
        "facility_materially_beats_purpose_topk": facility_better,
        "logdet_materially_beats_purpose_topk": logdet_better,
        "selected_algorithm_complexity_justified": winner == "P0_purpose_topk" or facility_better or logdet_better,
        "ready_to_freeze_selector_for_future_experiments": ready,
        "ready_to_move_to_matrixfree_lrvb": ready,
        "evidence_limitation": "34UC and 34UA support full replay; stress panels lack the full frozen selector feature schema.",
    }
    return pd.DataFrame([row]), diagnostics


def paired_comparisons(curves, random_summary):
    rows = []
    for _, row in curves.iterrows():
        for control in ["U0_uniform_random", "D1_stratified_random"]:
            random = random_summary[(random_summary.fold.eq(row.fold)) & random_summary.selector.eq(control) &
                                    np.isclose(random_summary.rho, row.rho)]
            if not len(random): continue
            random = random.iloc[0]
            rows.append({"fold": row.fold, "run_id": row.run_id, "selector": row.selector,
                         "random_control": control, "rho": row.rho,
                         "deterministic_retained_active_gain": row.retained_active_gain,
                         "random_median": random.retained_active_gain_median,
                         "random_q05": random.retained_active_gain_q05, "random_q95": random.retained_active_gain_q95,
                         "deterministic_minus_random_median": row.retained_active_gain - random.retained_active_gain_median,
                         "deterministic_percentile_band_position":
                             "above_q95" if row.retained_active_gain > random.retained_active_gain_q95 else
                             "below_q05" if row.retained_active_gain < random.retained_active_gain_q05 else "inside_5_95"})
    return pd.DataFrame(rows)


def make_plots(output, primary, primary_network_curves, transfer, random_summary, stability, stress, decision):
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    specs = [
        ("retained_active_gain_mean", "01_retained_active_gain_vs_rho.png", "retained active gain"),
        ("relative_Frobenius_error_mean", "02_covariance_error_vs_rho.png", "relative Frobenius error"),
        ("correction_capture_mean", "03_correction_capture_vs_rho.png", "correction capture"),
        ("zero_coverage_delta_vs_full_mean", "04_zero_coverage_delta_vs_rho.png", "zero coverage delta"),
        ("Q1_retained_active_gain_mean", "05_Q1_retained_gain_vs_rho.png", "Q1 retained active gain"),
    ]
    for metric, filename, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, group in primary.groupby("selector"):
            ax.plot(group.rho, group[metric], marker="o", label=method)
        ax.set(xlabel="budget fraction rho", ylabel=ylabel); ax.legend(fontsize=6); fig.tight_layout()
        fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for control, group in random_summary.groupby("selector"):
        summary = group.groupby("rho").retained_active_gain_median.mean()
        ax.plot(summary.index, summary.values, marker="o", label=control)
    for method, group in primary.groupby("selector"):
        ax.plot(group.rho, group.retained_active_gain_mean, lw=1, label=method)
    ax.set(xlabel="rho", ylabel="retained active gain"); ax.legend(fontsize=5); fig.tight_layout()
    fig.savefig(plots / "06_deterministic_vs_random.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    winner = decision.best_selector.iloc[0]
    a = primary[primary.selector.eq(winner)]; b = transfer[transfer.selector.eq(winner)] if len(transfer) else pd.DataFrame()
    ax.plot(a.rho, a.retained_active_gain_mean, marker="o", label="34UC")
    if len(b): ax.plot(b.rho, b.retained_active_gain_mean, marker="o", label="34UA")
    ax.set(xlabel="rho", ylabel="retained active gain"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "07_34UC_vs_34UA.png", dpi=150); plt.close(fig)
    chosen_rho = float(decision.recommended_budget_fraction.iloc[0])
    pivot = primary_network_curves[np.isclose(primary_network_curves.rho, chosen_rho)].pivot(
        index="selector", columns="run_id", values="retained_active_gain")
    fig, ax = plt.subplots(figsize=(8, 4)); image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=6); ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plots / "08_selector_network_heatmap.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(stability):
        group = stability.groupby("selector").topk_jaccard.mean(); ax.bar(range(len(group)), group.values); ax.set_xticks(range(len(group)), group.index, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("mean top-k Jaccard"); fig.tight_layout(); fig.savefig(plots / "09_selector_stability.png", dpi=150); plt.close(fig)
    for experiment, filename, title in [("34R", "10_34R_My40_vs_My15.png", "34R stress"),
                                         ("34Q-C", "11_34QC_geometry.png", "34Q-C geometry")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        subset = stress[stress.experiment.eq(experiment)] if len(stress) else pd.DataFrame()
        if len(subset): ax.bar(range(len(subset)), subset.stageB_need_spearman.fillna(0)); ax.set_ylabel("Stage-B-need Spearman")
        else: ax.text(.5, .5, "Insufficient compatible historical pairs", ha="center", va="center")
        ax.set_title(title); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in primary.groupby("selector"):
        ax.plot(group.rho, group.relative_Frobenius_error_mean, marker="o", label=method)
    ax.set(xlabel="cost fraction rho", ylabel="covariance error"); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(plots / "12_cost_fidelity_pareto.png", dpi=150); plt.close(fig)


def main():
    args = parse_args(); config = configuration(args); output = initialize(config); started = time.perf_counter()
    progress = Progress(13 if config["smoke_test"] else 16)
    panel, source_manifest = discover_and_normalize(config); network_manifest = manifests(panel)
    atomic_csv(source_manifest, str(output / "source_manifest.csv")); atomic_csv(network_manifest, str(output / "network_manifest.csv")); progress.update("audited historical sources")
    panel = transform_features(panel)
    feature_schema = pd.DataFrame([{"feature": feature, "raw_source": raw,
                                    "transform": "identity" if raw == "logdet_Louis_cov" else "log1p(max(x,0))",
                                    "selector_feature": True, "truth_derived": False,
                                    "available_34UC": panel.loc[panel.experiment.eq("34UC"), raw].notna().all(),
                                    "available_34UA": panel.loc[panel.experiment.eq("34UA"), raw].notna().all()}
                                   for raw, feature in zip(RAW_FEATURES, FEATURES)])
    atomic_csv(feature_schema, str(output / "selector_feature_schema.csv")); atomic_csv(panel, str(output / "normalized_edge_panel.csv")); progress.update("normalized edge panel")
    theory, exact = theoretical_checks(config); atomic_csv(theory, str(output / "theory_sanity_checks.csv")); atomic_csv(exact, str(output / "exact_smallN_selector_checks.csv")); progress.update("checked selector theory")
    uc = panel[panel.experiment.eq("34UC") & panel.selector_compatible].copy()
    folds, predictions, orderings, curves, random_summary = primary_cv(uc, config, output, progress)
    atomic_csv(folds, str(output / "cv_fold_config.csv")); atomic_csv(predictions, str(output / "cv_predictions.csv")); atomic_csv(orderings, str(output / "cv_selector_orderings.csv")); atomic_csv(curves, str(output / "cv_budget_curves.csv"));
    fold_summary = curves.groupby(["fold", "run_id", "selector"], as_index=False).agg(best_retained_gain=("retained_active_gain", "max"), minimum_covariance_error=("relative_Frobenius_error", "min"))
    atomic_csv(fold_summary, str(output / "cv_fold_summary.csv")); progress.update("saved 34UC CV")
    primary = aggregate_curves(curves, config); atomic_csv(primary, str(output / "primary_34uc_summary.csv"))
    ua = panel[panel.experiment.eq("34UA") & panel.selector_compatible].copy()
    state, transfer_network, transfer, transfer_orders = final_state_and_transfer(uc, ua, config)
    atomic_csv(transfer_network, str(output / "transfer_34ua_per_network.csv")); atomic_csv(transfer, str(output / "transfer_34ua_summary.csv")); progress.update("replayed 34UA transfer")
    preliminary_decision, budget_diagnostics = decide(primary, curves, transfer, config)
    recommended_rho = float(preliminary_decision.recommended_budget_fraction.iloc[0])
    stress_pairs, stress_summaries = stress_repeatability(panel, recommended_rho)
    atomic_csv(stress_summaries["34R"], str(output / "stress_34r_summary.csv")); atomic_csv(stress_summaries["34Q-C"], str(output / "geometry_34qc_summary.csv"));
    atomic_csv(stress_summaries["34O"], str(output / "repeatability_34o_summary.csv")); atomic_csv(stress_summaries["34Q-A"], str(output / "repeatability_34qa_summary.csv")); progress.update("audited stress panels")
    stability = selector_stability(orderings, recommended_rho)
    if len(stress_pairs):
        stress_for_stability = stress_pairs.rename(columns={"stageB_need_spearman": "spearman_rank", "stageB_need_kendall": "kendall_tau"})
        stability = pd.concat([stability, stress_for_stability], ignore_index=True, sort=False)
    atomic_csv(stability, str(output / "selector_stability.csv")); atomic_csv(random_summary, str(output / "random_baseline_summary.csv")); progress.update("summarized stability/random")
    covariance = curves[["fold", "run_id", "selector", "rho", "relative_Frobenius_error", "trace_relative_error", "mean_abs_logdet_error", "inflation_correlation", "correction_capture", "signed_correction_capture"]]
    observability = curves[["fold", "run_id", "selector", "rho", "Q1_retained_active_gain"]]
    paired = paired_comparisons(curves, random_summary)
    winner = preliminary_decision.best_selector.iloc[0]
    winner_rho = float(preliminary_decision.recommended_budget_fraction.iloc[0])
    winner_pairs = paired[(paired.selector.eq(winner)) & np.isclose(paired.rho, winner_rho)]
    for control, column in [("U0_uniform_random", "beats_uniform_random"),
                            ("D1_stratified_random", "beats_stratified_random")]:
        comparison = winner_pairs[winner_pairs.random_control.eq(control)]
        preliminary_decision[column] = bool(
            len(comparison)
            and comparison.deterministic_minus_random_median.mean() > 0
            and comparison.deterministic_minus_random_median.min() >= -1e-12
        )
    r_pairs = stress_pairs[stress_pairs.experiment.eq("34R")] if len(stress_pairs) else pd.DataFrame()
    r_my40 = r_pairs[(r_pairs.My_a.eq(40)) & (r_pairs.My_b.eq(40))] if len(r_pairs) else pd.DataFrame()
    r_cross = r_pairs[r_pairs.My_a.ne(r_pairs.My_b)] if len(r_pairs) else pd.DataFrame()
    preliminary_decision["stable_in_34r_My40"] = bool(len(r_my40) and r_my40.stageB_need_spearman.mean() >= .70)
    preliminary_decision["degradation_in_34r_My15"] = (
        r_cross.topk_jaccard.mean() - r_my40.topk_jaccard.mean() if len(r_cross) and len(r_my40) else np.nan
    )
    qc_pairs = stress_pairs[stress_pairs.experiment.eq("34Q-C")] if len(stress_pairs) else pd.DataFrame()
    qc_cross = qc_pairs[qc_pairs.C_mode_a.ne(qc_pairs.C_mode_b)] if len(qc_pairs) else pd.DataFrame()
    qc_modes = set(panel.loc[panel.experiment.eq("34Q-C"), "C_mode"].dropna().astype(str))
    preliminary_decision["stable_across_34qc_C_modes"] = bool(
        {"identity", "mild_mixing", "strong_mixing"}.issubset(qc_modes)
        and len(qc_cross) and qc_cross.stageB_need_spearman.mean() >= .70
    )
    for experiment, column in [("34O", "repeatable_in_34o"), ("34Q-A", "repeatable_in_34qa")]:
        item = stress_pairs[stress_pairs.experiment.eq(experiment)] if len(stress_pairs) else pd.DataFrame()
        preliminary_decision[column] = bool(len(item) and item.stageB_need_spearman.mean() >= .70)
    preliminary_decision["stress_selector_evidence_complete"] = False
    preliminary_decision["ready_to_freeze_selector_for_future_experiments"] = False
    preliminary_decision["ready_to_move_to_matrixfree_lrvb"] = False
    atomic_csv(covariance, str(output / "covariance_fidelity_summary.csv")); atomic_csv(observability, str(output / "observability_quartile_summary.csv")); atomic_csv(paired, str(output / "paired_method_comparisons.csv")); progress.update("saved fidelity summaries")
    mapping = pd.DataFrame([{"N": N, "rho": rho, "k": int(np.ceil(rho * N))} for N in [64, 90] for rho in config["rho_grid"]])
    atomic_csv(pd.concat([orderings.assign(domain="34UC"), transfer_orders], ignore_index=True, sort=False), str(output / "selector_orderings.csv")); atomic_csv(mapping, str(output / "k_fraction_mapping.csv"))
    atomic_csv(budget_diagnostics, str(output / "selector_budget_diagnostics.csv")); atomic_csv(preliminary_decision, str(output / ("decision_summary_smoke.csv" if config["smoke_test"] else "decision_summary.csv"))); progress.update("synthesized decision")
    runtime = pd.DataFrame([{"component": "34UE_read_only_meta_validation", "runtime_seconds": time.perf_counter() - started,
                             "new_simulations_run": 0, "new_stageB_fits_run": 0, "matrixfree_lrvb_run": False,
                             "primary_independent_networks": uc.run_id.nunique(), "supplemental_independent_networks": ua.run_id.nunique(),
                             "random_repeats": config["random_repeats"], "network_bootstrap_repeats": config["network_bootstrap_repeats"]}])
    atomic_csv(runtime, str(output / "runtime_summary.csv")); make_plots(output, primary, curves, transfer, random_summary, stability, stress_pairs, preliminary_decision); progress.update("saved outputs/figures")
    checkpoint_root = (output / "_checkpoints").resolve()
    checkpoint_cleanup_pending = False
    if checkpoint_root.exists() and checkpoint_root.parent == output.resolve():
        try:
            shutil.rmtree(checkpoint_root)
        except OSError:
            # Some synchronized Windows filesystems temporarily retain directory
            # handles. Scientific outputs are complete; a later run can safely
            # reuse or remove this configuration-validated checkpoint.
            checkpoint_cleanup_pending = bool(
                checkpoint_root.exists()
                and any(path.is_file() for path in checkpoint_root.rglob("*"))
            )
    atomic_json({"completed": True, "completed_at_unix": time.time(), "read_only": True,
                 "new_stageB_fits_run": 0, "primary_networks": int(uc.run_id.nunique()),
                 "supplemental_networks": int(ua.run_id.nunique()),
                 "best_selector": preliminary_decision.best_selector.iloc[0],
                 "recommended_budget_fraction": float(preliminary_decision.recommended_budget_fraction.iloc[0]),
                 "checkpoint_cleanup_pending": checkpoint_cleanup_pending}, output / "_COMPLETED.json")
    progress.update("complete")


if __name__ == "__main__":
    main()
