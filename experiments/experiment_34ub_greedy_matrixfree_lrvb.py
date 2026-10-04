"""Experiment 34U-B: greedy Stage-B selection and matrix-free LRVB.

Phase B replays completed 34U-A full-reference covariance blocks.  Phase C/D
validates a one-update-map implicit derivative against the saved legacy
finite-difference reference.  The active 34U-A directory is read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UB_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import experiments.experiment_34u_a_fullrun_observability_calibration as ua
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations
from src.stats.lrvb_stageB_smoother_feedback import global_index
from src.stats.matrixfree_lrvb_hybrid import (
    FixedPointLinearResponseOperator,
    edge_response_block,
    solve_response,
)

EPS = 1e-12
FEATURES = [
    "estimated_group_norm",
    "vb_cov_trace",
    "louis_cov_trace",
    "log_louis_cov_determinant",
    "louis_group_snr",
    "log_edge_o_inst_product",
    "spectral_transfer_full_band_score",
    "edge_o_dyn_estA_mean_K10",
]
METHOD_LABELS = {
    "current_ordering": "Baseline 0 current 34U-A order",
    "stratified_random": "Baseline 1 stratified random",
    "G1_purpose": "G1 purpose-only",
    "G2_facility": "G2 goal-aware facility location",
    "G3_pivoted_cholesky": "G3 feature-kernel block pivoted Cholesky",
    "G4_d_optimal": "G4 D-optimal",
}


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-matrixfree", action="store_true")
    parser.add_argument("--results-dir", default=None, help="Optional exact output directory for development validation.")
    return parser.parse_args()


def config_object(smoke, skip_matrixfree=False, results_dir=None):
    smoke = bool(smoke or os.environ.get("EXPERIMENT_34UB_SMOKE", "0") == "1")
    cal_reps = env_int("EXPERIMENT_34UB_CAL_REPS", 2)
    eval_reps = env_int("EXPERIMENT_34UB_EVAL_REPS", 2)
    root = Path(os.environ.get("EXPERIMENT_34UB_RESULTS_DIR", "results/experiment_34ub_greedy_matrixfree_lrvb"))
    output = Path(results_dir) if results_dir else (root / "smoke_test" if smoke and "EXPERIMENT_34UB_RESULTS_DIR" not in os.environ else root)
    return {
        "experiment": "34UB_greedy_matrixfree_lrvb",
        "smoke_test": smoke,
        "source_34UA_directory": os.environ.get("EXPERIMENT_34UB_34UA_DIR", "results/experiment_34u_a_fullrun_observability_calibration"),
        "results_directory": str(output),
        "M_x": 20,
        "M_y_primary": 40,
        "candidate_support_primary": "full_candidate",
        "calibration_reference_replicates": cal_reps,
        "evaluation_reference_replicates": eval_reps,
        "workers": max(1, env_int("EXPERIMENT_34UB_WORKERS", 1)),
        "inner_threads": max(1, env_int("EXPERIMENT_34UB_INNER_THREADS", 1)),
        "max_reference_edges": env_int("EXPERIMENT_34UB_MAX_REFERENCE_EDGES", 1 if smoke else 6),
        "greedy_k_max": env_int("EXPERIMENT_34UB_GREEDY_K_MAX", 20 if smoke else 90),
        "random_repeats": env_int("EXPERIMENT_34UB_RANDOM_REPEATS", 3 if smoke else 20),
        "facility_lambda_grid": [0.25, 0.50, 0.75],
        "requested_k_grid": [5, 10, 15, 20, 30, 45, 60, 90],
        "ridge_penalty": env_float("EXPERIMENT_34UB_RIDGE_PENALTY", 1e-3),
        "cg_rtol": env_float("EXPERIMENT_34UB_CG_RTOL", 1e-3 if smoke else 1e-4),
        "cg_maxiter": env_int("EXPERIMENT_34UB_CG_MAXITER", 20 if smoke else 30),
        "hvp_epsilon": env_float("EXPERIMENT_34UB_HVP_EPS", 1e-4),
        "matrixfree_validation_enabled": not skip_matrixfree,
        "matrixfree_derivative": "central directional finite difference of one fixed-point update map",
        "krylov_solver": "GMRES because the sequential hybrid update Jacobian is nonsymmetric",
        "posterior_center": "A_VB unchanged",
        "truth_used_by_selector": False,
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def atomic_text(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def initialize_output(config):
    output = Path(config["results_directory"])
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "_COMPLETED.json"
    if marker.exists():
        raise FileExistsError(f"Completed 34U-B directory exists: {output}")
    path = output / "experiment_config.json"
    if path.exists():
        with path.open(encoding="utf8") as handle:
            old = json.load(handle)
        protected = ["smoke_test", "source_34UA_directory", "calibration_reference_replicates",
                     "evaluation_reference_replicates", "max_reference_edges", "greedy_k_max",
                     "cg_rtol", "cg_maxiter", "hvp_epsilon"]
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Existing checkpoint configuration differs: {changed}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Non-empty output directory has no compatible config: {output}")
    atomic_json(config, path)
    return output


def load_reference(config):
    source = Path(config["source_34UA_directory"])
    edge_path = source / "calibration_edge_rows_partial.csv"
    runtime_path = source / "runtime_summary_partial.csv"
    edge = read_csv(edge_path)
    runtime = read_csv(runtime_path)
    if not len(edge):
        raise FileNotFoundError(f"No completed 34U-A reference edges at {edge_path}")
    eligible = edge[(edge.M_y == config["M_y_primary"]) &
                    (edge.candidate_support == config["candidate_support_primary"])].copy()
    successful = set(runtime.loc[runtime.run_status.eq("success"), "run_id"]) if len(runtime) else set(eligible.run_id)
    eligible = eligible[eligible.run_id.isin(successful)]
    runs = sorted(eligible.run_id.unique())
    needed = config["calibration_reference_replicates"] + config["evaluation_reference_replicates"]
    if len(runs) < needed:
        raise RuntimeError(f"Need {needed} completed 34U-A references but found {len(runs)}.")
    calibration_runs = runs[:config["calibration_reference_replicates"]]
    evaluation_runs = runs[config["calibration_reference_replicates"]:needed]
    eligible["reference_split"] = np.where(eligible.run_id.isin(calibration_runs), "calibration", "evaluation")
    eligible = eligible[eligible.run_id.isin(calibration_runs + evaluation_runs)].copy()
    eligible["source_row_order"] = eligible.groupby("run_id").cumcount() + 1
    return eligible, runtime, calibration_runs, evaluation_runs


def existing_profile(config, reference, runtime):
    source = Path(config["source_34UA_directory"])
    rows = []
    for run_id, group in reference.groupby("run_id", sort=False):
        run = runtime[runtime.run_id == run_id]
        diagnostics = []
        directory = source / "stageB_column_checkpoints" / run_id
        if directory.exists():
            for path in directory.glob("*.json"):
                with path.open(encoding="utf8") as handle:
                    diagnostics.append(json.load(handle))
        frame = pd.DataFrame(diagnostics)
        rows.append({
            "run_id": run_id,
            "reference_split": group.reference_split.iloc[0],
            "M_y": group.M_y.iloc[0],
            "candidate_support": group.candidate_support.iloc[0],
            "selected_edge_groups": len(group),
            "individual_A_coefficient_directions": 2 * len(group),
            "diagonal_groups": int(np.sum(group.target == group.source)),
            "active_offdiagonal_groups": int(group.true_edge.sum()),
            "zero_offdiagonal_groups": int((~group.true_edge.astype(bool)).sum()),
            "selection_semantics": "truth-stratified simulation benchmark across observability quartiles and SNR spread",
            "one_direction_semantics": "one lag coefficient; each p=2 edge group has two directions",
            "repeated_operations": "plus/minus converged hybrid updates; each iteration reruns Kalman/RTS, sufficient statistics, A, alpha, B and Q updates",
            "baseline_VB_runtime_seconds": run.baseline_VB_runtime_seconds.mean() if len(run) else np.nan,
            "louis_runtime_seconds": run.louis_runtime_seconds.mean() if len(run) else np.nan,
            "stageb_runtime_seconds": run.stageb_runtime_seconds.mean() if len(run) else np.nan,
            "total_runtime_seconds": run.total_runtime_seconds.mean() if len(run) else np.nan,
            "mean_seconds_per_direction": frame.runtime_seconds.mean() if len(frame) else np.nan,
            "mean_seconds_per_edge_group": 2 * frame.runtime_seconds.mean() if len(frame) else np.nan,
            "mean_plus_iterations": frame.plus_n_iter.mean() if len(frame) else np.nan,
            "mean_minus_iterations": frame.minus_n_iter.mean() if len(frame) else np.nan,
            "plus_convergence_rate": frame.plus_converged.mean() if len(frame) else np.nan,
            "minus_convergence_rate": frame.minus_converged.mean() if len(frame) else np.nan,
            "finite_direction_rate": frame.finite_difference_valid.mean() if len(frame) else np.nan,
        })
    return pd.DataFrame(rows)


def robust_feature_transform(reference):
    frame = reference.copy()
    frame["log_louis_cov_determinant"] = np.log(np.maximum(frame.louis_cov_determinant, EPS))
    frame["log_ard_group_precision"] = np.nan
    frame["spectral_uncertainty_impact"] = np.nan
    calibration = frame[frame.reference_split == "calibration"]
    parameters = []
    for feature in FEATURES:
        values = calibration[feature].to_numpy(float)
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= EPS:
            scale = float(np.nanstd(values))
        scale = max(scale, EPS)
        frame[f"z__{feature}"] = (frame[feature] - median) / scale
        parameters.append({"feature": feature, "calibration_median": median,
                           "calibration_robust_scale": scale, "available": True})
    parameters.extend([
        {"feature": "log_ard_group_precision", "calibration_median": np.nan,
         "calibration_robust_scale": np.nan, "available": False,
         "reason": "34U-A reference rows did not save alpha_mean"},
        {"feature": "spectral_uncertainty_impact", "calibration_median": np.nan,
         "calibration_robust_scale": np.nan, "available": False,
         "reason": "34U-A did not save complete fitted A/Q needed for cheap spectral Jacobians"},
    ])
    if not np.all(np.isfinite(frame[[f"z__{name}" for name in FEATURES]])):
        raise FloatingPointError("Non-finite standardized cheap features.")
    return frame, pd.DataFrame(parameters)


def ridge_utility(frame, config):
    calibration = frame[frame.reference_split == "calibration"]
    X = calibration[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    X = np.c_[np.ones(len(X)), X]
    y = calibration.relative_stageb_inflation.to_numpy(float)
    penalty = config["ridge_penalty"] * np.eye(X.shape[1]); penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    all_X = np.c_[np.ones(len(frame)), frame[[f"z__{name}" for name in FEATURES]].to_numpy(float)]
    frame = frame.copy()
    frame["purpose_score_raw"] = all_X @ coefficients
    frame["u_e"] = np.maximum(frame.purpose_score_raw, 0.0)
    coefficient_rows = pd.DataFrame({
        "term": ["intercept", *FEATURES], "coefficient": coefficients,
        "ridge_penalty": config["ridge_penalty"], "training_target": "stageb_relative_inflation",
        "training_rows": len(calibration), "truth_feature_used": False,
    })
    return frame, coefficient_rows


def rbf_bandwidth(frame):
    calibration = frame[frame.reference_split == "calibration"]
    values = calibration[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    distances = pdist(values)
    positive = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def similarity_matrix(group, bandwidth):
    Z = group[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    distance2 = squareform(pdist(Z, metric="sqeuclidean")) if len(Z) > 1 else np.zeros((1, 1))
    return np.exp(-distance2 / (2 * max(bandwidth, EPS) ** 2))


def facility_order(group, bandwidth, lam):
    similarity = similarity_matrix(group, bandwidth)
    utility = np.maximum(group.u_e.to_numpy(float), 0.0)
    utility_denominator = max(utility.sum(), EPS)
    coverage = np.zeros(len(group)); remaining = set(range(len(group))); rows = []
    for rank in range(1, len(group) + 1):
        best, best_gain = None, -np.inf
        for candidate in remaining:
            purpose_gain = utility[candidate] / utility_denominator
            coverage_gain = np.mean(np.maximum(coverage, similarity[:, candidate]) - coverage)
            gain = lam * purpose_gain + (1 - lam) * coverage_gain
            if gain > best_gain:
                best, best_gain = candidate, gain
        coverage = np.maximum(coverage, similarity[:, best]); remaining.remove(best)
        objective = lam * utility[[item["index"] for item in rows] + [best]].sum() / utility_denominator + (1-lam) * coverage.mean()
        rows.append({"index": best, "marginal_gain": best_gain, "objective": objective,
                     "selected_reason": "maximum goal-aware facility-location marginal gain"})
    return rows


def purpose_order(group):
    order = np.argsort(-group.u_e.to_numpy(float))
    return [{"index": int(index), "marginal_gain": float(group.u_e.iloc[index]),
             "objective": float(group.u_e.iloc[order[:rank]].sum()),
             "selected_reason": "largest frozen calibration-derived purpose score"}
            for rank, index in enumerate(order, 1)]


def current_order(group):
    order = np.argsort(group.source_row_order.to_numpy())
    return [{"index": int(index), "marginal_gain": np.nan, "objective": np.nan,
             "selected_reason": "prefix of existing 34U-A truth-stratified benchmark order"}
            for index in order]


def stratified_random_order(group, seed):
    rng = np.random.default_rng(seed)
    local = group.reset_index(drop=True).copy()
    local["snr_stratum"] = pd.qcut(local.louis_group_snr.rank(method="first"), 4, labels=False)
    local["obs_stratum"] = pd.qcut(local.edge_o_inst_product.rank(method="first"), 4, labels=False)
    bins = {}
    for key, part in local.groupby(["snr_stratum", "obs_stratum"]):
        values = part.index.tolist(); rng.shuffle(values); bins[key] = values
    order = []
    while any(bins.values()):
        for key in sorted(bins):
            if bins[key]: order.append(bins[key].pop())
    return [{"index": int(index), "marginal_gain": np.nan, "objective": np.nan,
             "selected_reason": "round-robin random from SNR x observability strata"}
            for index in order]


def pivoted_cholesky_order(group, bandwidth):
    kernel = similarity_matrix(group, bandwidth) + 1e-10 * np.eye(len(group))
    residual = kernel.copy(); original_trace = float(np.trace(kernel)); remaining = set(range(len(group))); rows = []
    for rank in range(1, len(group) + 1):
        diagonal = np.diag(residual)
        pivot = max(remaining, key=lambda index: diagonal[index])
        pivot_value = max(float(residual[pivot, pivot]), EPS)
        column = residual[:, pivot].copy()
        residual -= np.outer(column, column) / pivot_value
        residual = 0.5 * (residual + residual.T)
        remaining.remove(pivot)
        remaining_diag = np.maximum(np.diag(residual)[list(remaining)], 0.0) if remaining else np.asarray([0.0])
        rows.append({"index": pivot, "marginal_gain": 2 * pivot_value,
                     "objective": 1 - max(float(np.trace(residual)), 0.0) / max(original_trace, EPS),
                     "residual_trace": 2 * max(float(np.trace(residual)), 0.0),
                     "residual_trace_ratio": max(float(np.trace(residual)), 0.0) / max(original_trace, EPS),
                     "max_residual_block_trace": 2 * float(remaining_diag.max()),
                     "selected_reason": "largest residual 2x2 block trace in RBF feature kernel"})
    return rows


def d_optimal_order(group):
    Z = group[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    information = 1e-6 * np.eye(Z.shape[1]); remaining = set(range(len(group))); rows = []
    for rank in range(1, len(group) + 1):
        gains = {index: float(np.log1p(Z[index] @ np.linalg.solve(information, Z[index]))) for index in remaining}
        selected = max(gains, key=gains.get)
        information += np.outer(Z[selected], Z[selected]); remaining.remove(selected)
        rows.append({"index": selected, "marginal_gain": gains[selected],
                     "objective": float(np.linalg.slogdet(information)[1]),
                     "selected_reason": "maximum D-optimal log-determinant gain"})
    return rows


def order_rows(group, method, ordering, facility_lambda=np.nan, random_repeat=0):
    base = group.reset_index(drop=True)
    rows = []
    for rank, item in enumerate(ordering, 1):
        source = base.iloc[item["index"]]
        row = {"run_id": source.run_id, "reference_split": source.reference_split,
               "method": method, "method_label": METHOD_LABELS[method],
               "facility_lambda": facility_lambda, "random_repeat": random_repeat,
               "greedy_rank": rank, "target_i": int(source.target), "source_j": int(source.source),
               "u_e": source.u_e, "marginal_gain": item.get("marginal_gain", np.nan),
               "objective_value": item.get("objective", np.nan),
               "selected_reason": item["selected_reason"], "true_edge_diagnostic": bool(source.true_edge),
               "edge_o_inst_product": source.edge_o_inst_product}
        for feature in FEATURES:
            row[feature] = source[feature]
            row[f"z__{feature}"] = source[f"z__{feature}"]
        for key in ("residual_trace", "residual_trace_ratio", "max_residual_block_trace"):
            row[key] = item.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_orderings(frame, config, bandwidth):
    outputs = []
    for run_number, (run_id, group) in enumerate(frame.groupby("run_id", sort=False)):
        group = group.reset_index(drop=True)
        outputs.append(order_rows(group, "current_ordering", current_order(group)))
        outputs.append(order_rows(group, "G1_purpose", purpose_order(group)))
        for lam in config["facility_lambda_grid"]:
            outputs.append(order_rows(group, "G2_facility", facility_order(group, bandwidth, lam), lam))
        outputs.append(order_rows(group, "G3_pivoted_cholesky", pivoted_cholesky_order(group, bandwidth)))
        outputs.append(order_rows(group, "G4_d_optimal", d_optimal_order(group)))
        for repeat in range(config["random_repeats"]):
            ordering = stratified_random_order(group, 7400000 + 1000 * run_number + repeat)
            outputs.append(order_rows(group, "stratified_random", ordering, random_repeat=repeat))
    return pd.concat(outputs, ignore_index=True)


def covariance(row, kind):
    prefix = kind.lower()
    value = lambda name: row[name] if isinstance(row, pd.Series) else getattr(row, name)
    return np.asarray([[value(f"Sigma_{prefix}_00"), value(f"Sigma_{prefix}_01")],
                       [value(f"Sigma_{prefix}_10"), value(f"Sigma_{prefix}_11")]], float)


def block_metrics(row, block):
    difference = np.asarray([row.beta_hat_lag1 - row.beta_true_lag1,
                             row.beta_hat_lag2 - row.beta_true_lag2])
    block = 0.5 * (block + block.T)
    values = np.linalg.eigvalsh(block)
    if values.min() <= 0:
        values, vectors = np.linalg.eigh(block)
        values = np.maximum(values, max(1e-12, 1e-8 * max(values.max(), 1e-12)))
        block = vectors @ np.diag(values) @ vectors.T
    D2 = float(difference @ np.linalg.solve(block, difference))
    sd = np.sqrt(np.maximum(np.diag(block), 0.0))
    sign, logdet = np.linalg.slogdet(block)
    area = float(np.pi * 5.991464547 * np.exp(0.5 * logdet)) if sign > 0 else np.nan
    return D2 <= 5.991464547, np.abs(difference) <= 1.96 * sd, float(np.trace(block)), area


def k_grid(config, pool_size):
    values = [value for value in config["requested_k_grid"] if value <= min(pool_size, config["greedy_k_max"])]
    maximum = min(pool_size, config["greedy_k_max"])
    if maximum not in values: values.append(maximum)
    if pool_size not in values: values.append(pool_size)
    return sorted(set(values))


def replay_one(group, selected_pairs, method, k, facility_lambda, random_repeat):
    selected_pairs = set(selected_pairs)
    rows = []
    positive_total = np.maximum(group.delta_trace.to_numpy(float), 0.0).sum()
    for row in group.itertuples():
        chosen = (int(row.target), int(row.source)) in selected_pairs
        used = covariance(row, "stageb") if chosen else covariance(row, "louis")
        full = covariance(row, "stageb")
        cover, coefficient_cover, trace, area = block_metrics(row, used)
        full_trace = max(float(np.trace(full)), EPS)
        full_area = max(float(row.stageb_ellipse_area_95), EPS)
        rows.append({"true_edge": bool(row.true_edge), "selected": chosen,
                     "group_cover": cover, "coefficient_cover": coefficient_cover.mean(),
                     "trace": trace, "area": area,
                     "covariance_frobenius_error": np.linalg.norm(used-full)/max(np.linalg.norm(full), EPS),
                     "trace_relative_error": abs(trace-np.trace(full))/full_trace,
                     "ellipse_area_relative_error": abs(area-row.stageb_ellipse_area_95)/full_area,
                     "positive_inflation_captured": max(row.delta_trace, 0.0) if chosen else 0.0,
                     "observability": row.edge_o_inst_product, "snr": row.louis_group_snr,
                     "u_e": row.u_e})
    values = pd.DataFrame(rows); active = values[values.true_edge]; zero = values[~values.true_edge]
    louis_active = group[group.true_edge].louis_group_covered_95.mean()
    full_active = group[group.true_edge].stageb_group_covered_95.mean()
    louis_zero = group[~group.true_edge].louis_group_covered_95.mean()
    full_zero = group[~group.true_edge].stageb_group_covered_95.mean()
    active_coverage, zero_coverage = active.group_cover.mean(), zero.group_cover.mean()
    selected = values[values.selected]
    selected_indices = np.flatnonzero(values.selected)
    Z = group[[f"z__{feature}" for feature in FEATURES]].to_numpy(float)
    diversity = float(pdist(Z[selected_indices]).mean()) if len(selected_indices) > 1 else 0.0
    active_gain = full_active - louis_active
    zero_gain = full_zero - louis_zero
    return {
        "run_id": group.run_id.iloc[0], "reference_split": group.reference_split.iloc[0],
        "method": method, "facility_lambda": facility_lambda, "random_repeat": random_repeat,
        "k": k, "pool_size": len(group), "direction_fraction": k / len(group),
        "active_coefficient_coverage": active.coefficient_cover.mean(),
        "zero_coefficient_coverage": zero.coefficient_cover.mean(),
        "active_group_coverage": active_coverage, "zero_group_coverage": zero_coverage,
        "full90_active_group_coverage": full_active, "louis_active_group_coverage": louis_active,
        "full90_zero_group_coverage": full_zero, "louis_zero_group_coverage": louis_zero,
        "delta_active_group_coverage_vs_full": active_coverage-full_active,
        "delta_zero_group_coverage_vs_full": zero_coverage-full_zero,
        "retained_active_coverage_gain": (active_coverage-louis_active)/max(active_gain, EPS),
        "retained_zero_coverage_gain": (zero_coverage-louis_zero)/max(zero_gain, EPS),
        "mean_covariance_frobenius_error": values.covariance_frobenius_error.mean(),
        "mean_trace_relative_error": values.trace_relative_error.mean(),
        "mean_ellipse_area_relative_error": values.ellipse_area_relative_error.mean(),
        "captured_inflation_fraction": values.positive_inflation_captured.sum()/max(positive_total, EPS),
        "active_edge_selection_rate": active.selected.mean(), "zero_edge_selection_rate": zero.selected.mean(),
        "mean_selected_observability": selected.observability.mean(), "mean_selected_snr": selected.snr.mean(),
        "mean_selected_purpose": selected.u_e.mean(), "pairwise_feature_diversity": diversity,
    }


def replay_curves(frame, selected, config):
    rows = []
    for run_id, group in frame.groupby("run_id", sort=False):
        selections = selected[selected.run_id == run_id]
        descriptors = selections[["method", "facility_lambda", "random_repeat"]].drop_duplicates()
        for descriptor in descriptors.itertuples(index=False):
            ordering = selections[(selections.method == descriptor.method) &
                                  (selections.random_repeat == descriptor.random_repeat)]
            if np.isfinite(descriptor.facility_lambda):
                ordering = ordering[np.isclose(ordering.facility_lambda, descriptor.facility_lambda)]
            ordering = ordering.sort_values("greedy_rank")
            for k in k_grid(config, len(group)):
                chosen = list(zip(ordering.head(k).target_i.astype(int), ordering.head(k).source_j.astype(int)))
                rows.append(replay_one(group, chosen, descriptor.method, k,
                                       descriptor.facility_lambda, descriptor.random_repeat))
    return pd.DataFrame(rows)


def choose_facility_lambda(curve):
    calibration = curve[(curve.reference_split == "calibration") & (curve.method == "G2_facility")]
    if not len(calibration): return np.nan, pd.DataFrame()
    target_k = min(20, calibration.pool_size.min())
    rows = []
    for lam, group in calibration.groupby("facility_lambda"):
        exact = group[group.k == target_k]
        rows.append({"facility_lambda": lam, "selection_k": target_k,
                     "mean_retained_active_coverage_gain": exact.retained_active_coverage_gain.mean(),
                     "mean_captured_inflation_fraction": exact.captured_inflation_fraction.mean(),
                     "mean_covariance_frobenius_error": exact.mean_covariance_frobenius_error.mean(),
                     "calibration_score": np.nanmean([exact.retained_active_coverage_gain.mean(), exact.captured_inflation_fraction.mean()])})
    summary = pd.DataFrame(rows)
    best = summary.loc[summary.calibration_score.idxmax(), "facility_lambda"]
    summary["selected_on_calibration"] = np.isclose(summary.facility_lambda, best)
    return float(best), summary


def aggregate_curves(curve, best_lambda):
    use = curve[(curve.reference_split == "evaluation") &
                ((curve.method != "G2_facility") | np.isclose(curve.facility_lambda, best_lambda))]
    keys = ["method", "facility_lambda", "k", "pool_size"]
    metrics = [column for column in use.select_dtypes(include=[np.number]).columns
               if column not in ["facility_lambda", "k", "pool_size", "random_repeat"]]
    mean = use.groupby(keys, dropna=False)[metrics].mean().add_suffix("_mean")
    std = use.groupby(keys, dropna=False)[metrics].std().add_suffix("_std")
    return mean.join(std).reset_index(), use


def matrixfree_derivation_text():
    return r"""# Matrix-free LRVB derivation for Experiment 34U-B

The legacy Stage-B routine perturbs one A natural-parameter coordinate and
re-runs a warm-started hybrid iteration to convergence for both signs.  Every
iteration performs an exact Kalman/RTS smoother, rebuilds posterior sufficient
statistics, updates q(A), q(alpha), B, and diagonal Q, and then repeats.

Define the minimal implemented fixed-point state

    theta = (vec(A_mean), alpha_offdiag, vec(B), diag(Q)).

Latent states and q(A) covariance are deterministic intermediates of one update
and therefore are not independent state coordinates.  Let M(theta, h) be one
sequential update:

    theta -> smoother moments -> Szz,Szx -> q(A;h) -> q(alpha) -> B -> Q.

At a fixed point theta*=M(theta*,0), differentiation gives

    (I - J_M) dtheta/dh = partial M / partial h.

The implementation supplies a scipy LinearOperator for I-J_M with

    J_M v ~= [M(theta*+eps*v,0)-M(theta*-eps*v,0)]/(2 eps ||v|| scaling).

The right-hand side is another central difference of one update map with
respect to the selected natural-parameter perturbation.  Neither operation
re-optimizes the variational model to convergence.

The map is sequential and its Jacobian is generally nonsymmetric.  Therefore
GMRES—not CG—is used.  Symmetry is diagnosed by randomized bilinear tests
|u' M v-v' M u|.  A Hutchinson-estimated diagonal of I-J_M is tested as a
simple matrix-free preconditioner.  The same operator and preconditioner are
reused for both lag RHS vectors of every selected p=2 edge.
"""


def recreate_baseline(config, reference_group):
    source_config_path = Path(config["source_34UA_directory"]) / "experiment_config.json"
    with source_config_path.open(encoding="utf8") as handle:
        source_config = json.load(handle)
    first = reference_group.iloc[0]
    meta = ua.metadata(source_config, "calibration", int(first.M_y), str(first.candidate_support), int(first.replicate_id))
    data, seed = ua.simulate(source_config, meta)
    masks = ua.paired_candidate_masks(source_config, int(first.true_network_id), int(first.replicate_id))
    model = ua.fit_model(source_config, data, masks[str(first.candidate_support)], seed + 500)
    return source_config, meta, data, model


def representative_edges(group, count):
    local = group.copy()
    composite = (local.louis_group_snr.rank(pct=True) +
                 local.edge_o_inst_product.rank(pct=True) +
                 local.relative_stageb_inflation.rank(pct=True))
    local = local.assign(_composite=composite).sort_values("_composite")
    indices = np.unique(np.round(np.linspace(0, len(local)-1, min(count, len(local)))).astype(int))
    return local.iloc[indices]


def matrixfree_validation(config, frame, output):
    columns_validation = ["run_id", "target", "source", "true_edge", "legacy_trace", "matrixfree_trace",
                          "relative_frobenius_error", "relative_trace_error", "relative_determinant_error",
                          "elementwise_correlation", "symmetry_error", "minimum_eigenvalue", "coverage_decision_agreement",
                          "legacy_seconds_per_edge", "matrixfree_seconds_per_edge", "speedup_factor", "all_rhs_converged"]
    if not config["matrixfree_validation_enabled"]:
        return (pd.DataFrame(columns=columns_validation), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    evaluation = frame[frame.reference_split == "evaluation"]
    run_id = sorted(evaluation.run_id.unique())[0]
    group = evaluation[evaluation.run_id == run_id].copy()
    source_config, meta, data, model = recreate_baseline(config, group)
    operator = FixedPointLinearResponseOperator(model, data["y"], data["u"], config["hvp_epsilon"])
    started = time.perf_counter(); mapped = operator.map(operator.theta0); one_map_seconds = time.perf_counter()-started
    fixed_residual = np.linalg.norm(mapped-operator.theta0)/max(np.linalg.norm(operator.theta0), EPS)
    asym_median, asym_max = operator.bilinear_asymmetry(1 if config["smoke_test"] else 3, 34)
    preconditioner, preconditioner_diagonal = operator.hutchinson_diagonal_preconditioner(1 if config["smoke_test"] else 3, 35)
    operator_rows = pd.DataFrame([{"run_id": run_id, "state_dimension": operator.layout.size,
        "A_dimension": operator.layout.A_slice.stop-operator.layout.A_slice.start,
        "alpha_dimension": operator.layout.alpha_slice.stop-operator.layout.alpha_slice.start,
        "B_dimension": operator.layout.B_slice.stop-operator.layout.B_slice.start,
        "Q_dimension": operator.layout.Q_slice.stop-operator.layout.Q_slice.start,
        "fixed_point_relative_residual": fixed_residual, "one_update_map_seconds": one_map_seconds,
        "bilinear_asymmetry_median": asym_median, "bilinear_asymmetry_max": asym_max,
        "operator_is_symmetric": bool(asym_max <= 1e-3), "cg_valid_for_operator": bool(asym_max <= 1e-3),
        "solver_selected": "GMRES", "JVP_type": "directional finite difference of one update map",
        "full_refit_per_matvec": False}])
    profile = existing_profile(config, group, read_csv(Path(config["source_34UA_directory"])/"runtime_summary_partial.csv"))
    legacy_seconds = profile.mean_seconds_per_edge_group.mean()
    validation_rows, convergence_rows, preconditioner_rows = [], [], []
    selected = representative_edges(group, config["max_reference_edges"])
    for edge_number, row in enumerate(selected.itertuples()):
        chosen_preconditioner = preconditioner
        chosen_name = "hutchinson_diagonal"
        if edge_number == 0:
            diagnostic_index = global_index(int(row.target), 0, int(row.source), model.n_states, 2)
            diagnostic_rhs = operator.perturbation_rhs(diagnostic_index)
            _, no_pre = solve_response(operator, diagnostic_rhs, config["cg_rtol"], config["cg_maxiter"], None)
            _, with_pre = solve_response(operator, diagnostic_rhs, config["cg_rtol"], config["cg_maxiter"], preconditioner)
            for name, diagnostic in (("none", no_pre), ("hutchinson_diagonal", with_pre)):
                preconditioner_rows.append({"run_id": run_id, "target": int(row.target), "source": int(row.source),
                                            "lag": 1, "preconditioner": name,
                                            **{key:value for key,value in diagnostic.items() if key != "residual_history"}})
            if no_pre["relative_residual"] < with_pre["relative_residual"]:
                chosen_preconditioner, chosen_name = None, "none"
        before_calls = operator.matvec_calls
        raw, block, diagnostics = edge_response_block(operator, int(row.target), int(row.source),
                                                       config["cg_rtol"], config["cg_maxiter"], chosen_preconditioner)
        matrixfree_seconds = sum(item["runtime_seconds"] for item in diagnostics)
        legacy = covariance(row, "stageb")
        difference = np.asarray([row.beta_hat_lag1-row.beta_true_lag1, row.beta_hat_lag2-row.beta_true_lag2])
        legacy_cover = difference @ np.linalg.solve(legacy, difference) <= 5.991464547
        matrixfree_cover = difference @ np.linalg.solve(block, difference) <= 5.991464547
        validation_rows.append({"run_id": run_id, "target": int(row.target), "source": int(row.source),
            "true_edge": bool(row.true_edge), "legacy_trace": np.trace(legacy), "matrixfree_trace": np.trace(block),
            "relative_frobenius_error": np.linalg.norm(block-legacy)/max(np.linalg.norm(legacy), EPS),
            "relative_trace_error": abs(np.trace(block)-np.trace(legacy))/max(abs(np.trace(legacy)), EPS),
            "relative_determinant_error": abs(np.linalg.det(block)-np.linalg.det(legacy))/max(abs(np.linalg.det(legacy)), EPS),
            "elementwise_correlation": correlations(block.ravel(), legacy.ravel()),
            "symmetry_error": np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw), EPS),
            "minimum_eigenvalue": np.linalg.eigvalsh(block).min(),
            "coverage_decision_agreement": bool(legacy_cover == matrixfree_cover),
            "legacy_seconds_per_edge": legacy_seconds, "matrixfree_seconds_per_edge": matrixfree_seconds,
            "speedup_factor": legacy_seconds/max(matrixfree_seconds, EPS),
            "all_rhs_converged": all(item["converged"] for item in diagnostics),
            "matrixfree_matvec_calls": operator.matvec_calls-before_calls,
            "median_solver_iterations": float(np.median([item["iterations"] for item in diagnostics])),
            "selected_preconditioner": chosen_name})
        for item in diagnostics:
            for iteration, residual in enumerate(item.pop("residual_history"), 1):
                convergence_rows.append({"run_id": run_id, "target": int(row.target), "source": int(row.source),
                                         "lag": item["lag"], "preconditioner": chosen_name,
                                         "iteration": iteration, "residual_norm": residual})
            preconditioner_rows.append({"run_id": run_id, "target": int(row.target), "source": int(row.source),
                                        "lag": item["lag"], "preconditioner": chosen_name, **item})
        atomic_csv(pd.DataFrame(validation_rows), str(output/"matrixfree_solver_validation_partial.csv"))
        atomic_csv(pd.DataFrame(convergence_rows), str(output/"cg_convergence_partial.csv"))
        atomic_csv(pd.DataFrame(preconditioner_rows), str(output/"preconditioner_summary_partial.csv"))
    preconditioner_summary = pd.DataFrame(preconditioner_rows)
    projected = projected_runtimes(pd.DataFrame(validation_rows), legacy_seconds)
    return pd.DataFrame(validation_rows), operator_rows, pd.DataFrame(convergence_rows), preconditioner_summary, projected


def projected_runtimes(validation, legacy_seconds):
    matrixfree_seconds = validation.matrixfree_seconds_per_edge.median() if len(validation) else np.nan
    rows = []
    for edges in (90, 30, 20, 15, 10):
        rows.extend([
            {"scenario": f"legacy_{edges}", "n_edges": edges, "solver": "legacy converged central finite difference",
             "projected_runtime_seconds": edges*legacy_seconds, "projected_speedup_vs_legacy90": 90/max(edges, 1)},
            {"scenario": f"matrixfree_{edges}", "n_edges": edges, "solver": "matrix-free GMRES",
             "projected_runtime_seconds": edges*matrixfree_seconds,
             "projected_speedup_vs_legacy90": 90*legacy_seconds/max(edges*matrixfree_seconds, EPS)},
        ])
    return pd.DataFrame(rows)


def decide(summary, lambda_summary, validation):
    deployable = summary[summary.retained_active_coverage_gain_mean.notna() &
                         ~summary.method.isin(["current_ordering", "stratified_random"])].copy()
    screening = deployable[deployable.k <= 30].copy()
    target_k = min(20, screening.k.max())
    at_target = screening[screening.k == target_k].sort_values(
        ["retained_active_coverage_gain_mean", "captured_inflation_fraction_mean", "mean_covariance_frobenius_error_mean"],
        ascending=[False, False, True])
    best_method = at_target.iloc[0].method
    qualifying = deployable[deployable.retained_active_coverage_gain_mean >= 0.90]
    if len(qualifying):
        recommended = qualifying.sort_values(["k", "mean_covariance_frobenius_error_mean"]).iloc[0]
    else:
        deployable["distance_to_target"] = np.abs(deployable.retained_active_coverage_gain_mean - 0.90)
        recommended = deployable.sort_values(["distance_to_target", "k"]).iloc[0]
    comparison = summary[summary.k == target_k].set_index("method")
    value = lambda method, column: by_method.loc[method, column] if method in by_method.index else np.nan
    def smallest(threshold):
        q = deployable[deployable.retained_active_coverage_gain_mean >= threshold]
        return q.k.min() if len(q) else np.nan
    matrixfree_matches = bool(len(validation) and validation.all_rhs_converged.all() and validation.relative_frobenius_error.median() <= 0.10)
    by_method = comparison
    row = {"best_greedy_method": best_method,
        "best_facility_lambda": lambda_summary.loc[lambda_summary.selected_on_calibration, "facility_lambda"].iloc[0] if len(lambda_summary) else np.nan,
        "recommended_k": int(recommended.k), "smallest_k_80pct_gain": smallest(.80),
        "smallest_k_90pct_gain": smallest(.90), "smallest_k_95pct_gain": smallest(.95),
        "G2_beats_purpose_only": value("G2_facility", "retained_active_coverage_gain_mean") > value("G1_purpose", "retained_active_coverage_gain_mean"),
        "G2_beats_random": value("G2_facility", "retained_active_coverage_gain_mean") > value("stratified_random", "retained_active_coverage_gain_mean"),
        "G2_beats_pivoted_cholesky": value("G2_facility", "retained_active_coverage_gain_mean") > value("G3_pivoted_cholesky", "retained_active_coverage_gain_mean"),
        "pivoted_cholesky_beats_random": value("G3_pivoted_cholesky", "retained_active_coverage_gain_mean") > value("stratified_random", "retained_active_coverage_gain_mean"),
        "adaptive_stopping_recommended": bool(len(qualifying) and recommended.k < recommended.pool_size),
        "matrixfree_matches_legacy": matrixfree_matches,
        "median_relative_covariance_error": validation.relative_frobenius_error.median() if len(validation) else np.nan,
        "max_relative_covariance_error": validation.relative_frobenius_error.max() if len(validation) else np.nan,
        "median_solver_iterations": validation.median_solver_iterations.median() if len(validation) else np.nan,
        "median_HVP_or_JVP_calls": validation.matrixfree_matvec_calls.median() if len(validation) else np.nan,
        "median_speedup_per_edge": validation.speedup_factor.median() if len(validation) else np.nan,
        "projected_speedup_at_recommended_k": (90*validation.legacy_seconds_per_edge.median()/max(recommended.k*validation.matrixfree_seconds_per_edge.median(), EPS)) if len(validation) else np.nan,
        "cg_valid_for_operator": False, "alternative_solver_used_if_not": "GMRES",
        "combined_method_validated": matrixfree_matches,
        "recommended_future_stageb_selector": best_method,
        "recommended_future_stageb_solver": "matrix-free GMRES" if matrixfree_matches else "legacy finite-difference reference pending matrix-free debugging",
        "recommended_future_edge_budget": int(recommended.k)}
    return pd.DataFrame([row])


def plots(output, curve, summary, selected, lambda_summary, pivot, validation, convergence, projected):
    path = output / "plots"; path.mkdir(exist_ok=True)
    def save(name): plt.tight_layout(); plt.savefig(path/name, dpi=160); plt.close()
    evaluation = curve[curve.reference_split == "evaluation"]
    for column, name, ylabel in (
        ("active_group_coverage", "01_active_coverage_vs_k.png", "active group coverage"),
        ("retained_active_coverage_gain", "02_retained_gain_vs_k.png", "retained FULL-reference gain"),
        ("mean_covariance_frobenius_error", "03_covariance_error_vs_k.png", "mean covariance error"),
        ("direction_fraction", "04_runtime_proxy_vs_k.png", "naive direction fraction")):
        for method, group in evaluation.groupby("method"):
            means = group.groupby("k")[column].mean(); plt.plot(means.index, means, marker="o", label=method)
        plt.xlabel("selected edge groups k"); plt.ylabel(ylabel); plt.legend(fontsize=6); save(name)
    facility = selected[selected.method == "G2_facility"]
    for lam, group in facility.groupby("facility_lambda"):
        means = group.groupby("greedy_rank").objective_value.mean(); plt.plot(means.index, means, label=f"lambda={lam}")
    plt.xlabel("greedy rank"); plt.ylabel("facility objective"); plt.legend(); save("05_facility_objective_vs_k.png")
    if len(pivot):
        means = pivot.groupby("greedy_rank").residual_trace_ratio.mean(); plt.plot(means.index, means)
    plt.xlabel("greedy rank"); plt.ylabel("residual trace ratio"); save("06_pivoted_cholesky_residual.png")
    top = selected[selected.greedy_rank <= 20]
    top.boxplot(column="edge_o_inst_product", by="method", rot=25); plt.suptitle(""); plt.ylabel("edge observability"); save("07_selected_observability.png")
    top.boxplot(column="louis_group_snr", by="method", rot=25); plt.suptitle(""); plt.ylabel("Louis group-SNR"); save("08_selected_snr.png")
    if len(validation):
        plt.scatter(validation.legacy_trace, validation.matrixfree_trace); plt.xlabel("legacy covariance trace"); plt.ylabel("matrix-free trace")
    else: plt.text(.5,.5,"matrix-free validation unavailable",ha="center")
    save("09_matrixfree_vs_legacy_covariance.png")
    if len(validation): plt.bar(np.arange(len(validation)),validation.relative_frobenius_error)
    plt.ylabel("relative covariance error"); save("10_matrixfree_relative_error.png")
    if len(convergence):
        for values, group in convergence.groupby(["target","source","lag"]): plt.semilogy(group.iteration,group.residual_norm,label=str(values))
        plt.legend(fontsize=6)
    plt.xlabel("GMRES iteration"); plt.ylabel("residual norm"); save("11_gmres_residual.png")
    if len(projected): projected.set_index("scenario").projected_runtime_seconds.plot.bar()
    plt.ylabel("projected runtime seconds"); plt.xticks(rotation=30,ha="right"); save("12_projected_runtime.png")


def empty_outputs():
    return {
        "matrixfree_operator_diagnostics.csv": pd.DataFrame(), "matrixfree_solver_validation.csv": pd.DataFrame(),
        "cg_convergence.csv": pd.DataFrame(), "preconditioner_summary.csv": pd.DataFrame(),
        "combined_greedy_matrixfree_summary.csv": pd.DataFrame(), "projected_runtime_summary.csv": pd.DataFrame(),
    }


def main():
    args = parse_args(); started = time.perf_counter()
    config = config_object(args.smoke, args.skip_matrixfree, args.results_dir); output = initialize_output(config)
    atomic_text(matrixfree_derivation_text(), output/"matrixfree_lrvb_derivation.md")
    reference, runtime_34ua, cal_runs, eval_runs = load_reference(config)
    profile = existing_profile(config, reference, runtime_34ua)
    atomic_csv(profile, str(output/"existing_stageb_profile.csv"))
    features, feature_parameters = robust_feature_transform(reference)
    features, calibration_coefficients = ridge_utility(features, config)
    bandwidth = rbf_bandwidth(features)
    calibration_coefficients["facility_RBF_bandwidth"] = bandwidth
    atomic_csv(features, str(output/"candidate_edge_features.csv"))
    atomic_csv(features[features.reference_split=="calibration"], str(output/"greedy_calibration_rows.csv"))
    atomic_csv(feature_parameters, str(output/"feature_standardization.csv"))
    atomic_csv(calibration_coefficients, str(output/"purpose_model_coefficients.csv"))
    selected = build_orderings(features, config, bandwidth)
    atomic_csv(selected, str(output/"greedy_selected_edges_partial.csv"))
    curve = replay_curves(features, selected, config)
    atomic_csv(curve, str(output/"greedy_subset_curve_partial.csv"))
    best_lambda, lambda_summary = choose_facility_lambda(curve)
    summary, evaluation_curve = aggregate_curves(curve, best_lambda)
    random_summary = evaluation_curve[evaluation_curve.method=="stratified_random"].groupby(["k","pool_size"]).agg(
        active_group_coverage_mean=("active_group_coverage","mean"),active_group_coverage_sd=("active_group_coverage","std"),
        retained_gain_mean=("retained_active_coverage_gain","mean"),retained_gain_sd=("retained_active_coverage_gain","std"),
        covariance_error_mean=("mean_covariance_frobenius_error","mean"),n_random_replays=("run_id","size")).reset_index()
    pivot = selected[selected.method=="G3_pivoted_cholesky"][["run_id","reference_split","greedy_rank","target_i","source_j","residual_trace","residual_trace_ratio","max_residual_block_trace"]]
    atomic_csv(selected, str(output/"greedy_selected_edges.csv")); atomic_csv(curve, str(output/"greedy_subset_curve.csv"))
    atomic_csv(summary, str(output/"greedy_subset_summary.csv")); atomic_csv(lambda_summary, str(output/"facility_location_lambda_summary.csv"))
    atomic_csv(pivot, str(output/"pivoted_cholesky_diagnostics.csv")); atomic_csv(random_summary, str(output/"random_baseline_summary.csv"))
    validation = operator_diag = convergence = preconditioners = projected = pd.DataFrame()
    try:
        validation, operator_diag, convergence, preconditioners, projected = matrixfree_validation(config, features, output)
    except Exception as error:
        operator_diag = pd.DataFrame([{"validation_status":"failed", "error_type":type(error).__name__,
                                      "error_message":str(error), "traceback":traceback.format_exc()}])
    atomic_csv(operator_diag, str(output/"matrixfree_operator_diagnostics.csv")); atomic_csv(validation, str(output/"matrixfree_solver_validation.csv"))
    atomic_csv(convergence, str(output/"cg_convergence.csv")); atomic_csv(preconditioners, str(output/"preconditioner_summary.csv"))
    atomic_csv(projected, str(output/"projected_runtime_summary.csv"))
    decision = decide(summary, lambda_summary, validation)
    combined = pd.DataFrame([{"combined_method_validated":bool(decision.combined_method_validated.iloc[0]),
        "selector":decision.recommended_future_stageb_selector.iloc[0], "solver":decision.recommended_future_stageb_solver.iloc[0],
        "edge_budget":decision.recommended_future_edge_budget.iloc[0],
        "note":"No prospective combined run is performed unless matrix-free validation passes the <=0.10 median error and convergence criteria."}])
    runtime_breakdown = pd.concat([profile.assign(component="legacy_34UA_reference"),
        pd.DataFrame([{"component":"34UB_total", "total_runtime_seconds":time.perf_counter()-started,
                       "reference_edges_reused":len(reference), "new_legacy_stageB_refits":0}])],ignore_index=True,sort=False)
    atomic_csv(combined, str(output/"combined_greedy_matrixfree_summary.csv")); atomic_csv(runtime_breakdown, str(output/"runtime_breakdown.csv"))
    atomic_csv(decision, str(output/"decision_summary.csv")); atomic_csv(runtime_breakdown, str(output/"runtime_summary.csv"))
    plots(output, curve, summary, selected, lambda_summary, pivot, validation, convergence, projected)
    atomic_json({"completed":True,"completed_at_unix":time.time(),"calibration_run_ids":cal_runs,"evaluation_run_ids":eval_runs,
                 "reused_34UA_edges":len(reference),"new_legacy_stageB_refits":0},output/"_COMPLETED.json")
    print(f"34U-B complete: selector={decision.best_greedy_method.iloc[0]}, k={decision.recommended_k.iloc[0]}, matrixfree_matches_legacy={decision.matrixfree_matches_legacy.iloc[0]}")


if __name__ == "__main__":
    main()
