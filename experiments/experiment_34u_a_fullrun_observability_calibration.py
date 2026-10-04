"""Experiment 34U-A: prospective observability-calibrated Stage-B allocation.

The posterior center is always A_VB.  Edge truth is used only to construct a
balanced simulation benchmark and to define calibration/evaluation outcomes;
deployable allocation rules use Louis group-SNR and C/R observability only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_thread_var] = os.environ.get("EXPERIMENT_34U_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import experiments.experiment_34i_static_spectral_varx_gc as spectral_experiment
import experiments.experiment_34j_louis_missing_information_A_uncertainty as louis_experiment
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as stageb_experiment
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as recovery_experiment
import experiments.experiment_34r_stage1_rectangular_C_breakpoint as rectangular_experiment
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, relative
from src.ssm.ssm_varx_p_simulator import build_var_companion_matrix
from src.ssm.vb_ard_varx_ssm_support_mask import HybridVBARDVARXSSMKnownCWithBQSupportMask
from src.stats.louis_missing_information import (
    estimate_missing_information_numba,
    prior_precision_for_row,
    sample_companion_trajectories_ffbs,
    stabilized_louis_covariance,
)
from src.stats.lrvb_stageB_smoother_feedback import (
    global_index,
    project_selected_covariance,
    sensitivity_column,
)

CHI2_95_DF2 = 5.991464547107979
EPS = 1e-4
NUMERIC_EPS = 1e-12
LAMBDA_GRID = np.round(np.arange(0.0, 1.5001, 0.1), 10)
METHODS = (
    "louis_only",
    "full_stageb_reference",
    "random_budget_stageb",
    "snr_budget_stageb",
    "observability_budget_stageb",
    "snr_observability_budget_stageb",
    "global_0p75_budget_stageb",
    "observability_adaptive_budget_stageb",
    "snr_observability_adaptive_budget_stageb",
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Exercise the complete pipeline with a tiny problem.")
    return parser.parse_args()


class Progress:
    def __init__(self, total: int):
        self.total = max(int(total), 0)
        self.done = 0
        self.started = time.perf_counter()

    def update(self, label: str):
        self.done += 1
        frac = self.done / max(self.total, 1)
        filled = round(30 * frac)
        eta = (time.perf_counter() - self.started) / max(self.done, 1) * (self.total - self.done)
        print(
            f"\r34U-A [{'#' * filled}{'-' * (30-filled)}] {self.done}/{self.total} "
            f"{100*frac:5.1f}% ETA {eta/60:6.1f}m {label[:46]}",
            end="\n" if self.done >= self.total else "",
            flush=True,
        )


def configuration(smoke: bool) -> dict:
    smoke = bool(smoke or os.environ.get("EXPERIMENT_34U_SMOKE", "0") == "1")
    # M=10 keeps all four observability quartiles populated in permissive-2K
    # while making the end-to-end Stage-B smoke test tractable.
    mx = 10 if smoke else 20
    config = {
        "experiment": "34U-A_fullrun_observability_calibration",
        "smoke_test": smoke,
        "M_x": mx,
        "M_y_values": [15, 8] if smoke else [40, 15],
        "T": 150 if smoke else 1000,
        "na": 2,
        "nb": 3,
        "candidate_support_regimes": ["full_candidate", "permissive_2K"],
        "calibration_replicates": env_int("EXPERIMENT_34U_CAL_REPS", 1 if smoke else 8),
        "evaluation_replicates": env_int("EXPERIMENT_34U_EVAL_REPS", 1 if smoke else 12),
        "workers": max(1, env_int("EXPERIMENT_34U_WORKERS", 2)),
        "inner_threads": max(1, env_int("EXPERIMENT_34U_INNER_THREADS", 1)),
        "benchmark_group_budget": env_int("EXPERIMENT_34U_BENCHMARK_GROUPS", 8 if smoke else 64),
        "stageb_budget_fraction": env_float("EXPERIMENT_34U_STAGEB_BUDGET_FRACTION", 0.50),
        "random_allocation_repeats": env_int("EXPERIMENT_34U_RANDOM_REPEATS", 2 if smoke else 10),
        "VB_MAX_ITER": env_int("EXPERIMENT_34U_VB_MAX_ITER", 6 if smoke else 75),
        "N_FFBS_SAMPLES": env_int("EXPERIMENT_34U_N_FFBS", 4 if smoke else 50),
        "PERTURBED_VB_MAX_ITER": env_int("EXPERIMENT_34U_PERTURB_MAX_ITER", 5 if smoke else 30),
        "PERTURBED_RESCUE_MAX_ITER": env_int("EXPERIMENT_34U_PERTURB_RESCUE_MAX_ITER", 7 if smoke else 50),
        "N_FREQUENCIES": env_int("EXPERIMENT_34U_N_FREQS", 32 if smoke else 128),
        "finite_difference_epsilon": EPS,
        "lambda_grid": LAMBDA_GRID.tolist(),
        "lambda_quartile_primary_statistic": "80th percentile of calibration oracle_lambda_min, censored values treated as 1.5",
        "ridge_penalty": env_float("EXPERIMENT_34U_RIDGE_PENALTY", 1e-3),
        "Louis_eta": 0.70,
        "Louis_tau": 0.90,
        "Q_shrinkage_rho": 0.25,
        "Q_update_damping": 0.50,
        "a0": 1e-3,
        "b0": 1e-3,
        "A_center": "A_VB",
        "primary_observability": "edge_o_inst_product = J_ii * J_jj, J=C^T R^{-1} C",
        "primary_spectral_score": "transfer_spectral_gc_diagQ",
        "calibration_network_ids": list(range(0, env_int("EXPERIMENT_34U_CAL_REPS", 1 if smoke else 8))),
        "evaluation_network_ids": list(range(10000, 10000 + env_int("EXPERIMENT_34U_EVAL_REPS", 1 if smoke else 12))),
        "truth_used_by_deployable_allocator": False,
    }
    if not 0 < config["stageb_budget_fraction"] <= 1:
        raise ValueError("EXPERIMENT_34U_STAGEB_BUDGET_FRACTION must be in (0,1].")
    root = Path(os.environ.get("EXPERIMENT_34U_RESULTS_DIR", "results/experiment_34u_a_fullrun_observability_calibration"))
    config["results_directory"] = str(root / "smoke_test" if smoke and "EXPERIMENT_34U_RESULTS_DIR" not in os.environ else root)
    return config


def atomic_json(value: dict, path: Path):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temp, path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_or_validate_config(config: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_config.json"
    final_marker = output / "_COMPLETED.json"
    if final_marker.exists():
        raise FileExistsError(f"Completed result directory already exists: {output}. Set EXPERIMENT_34U_RESULTS_DIR to a new directory.")
    if path.exists():
        with path.open(encoding="utf8") as handle:
            old = json.load(handle)
        protected = [
            "smoke_test", "M_x", "M_y_values", "T", "candidate_support_regimes",
            "calibration_replicates", "evaluation_replicates", "benchmark_group_budget",
            "stageb_budget_fraction", "lambda_grid", "finite_difference_epsilon",
        ]
        differences = [key for key in protected if old.get(key) != config.get(key)]
        if differences:
            raise ValueError(f"Checkpoint configuration mismatch for: {differences}")
    elif any(item.name != "smoke_test" for item in output.iterdir()):
        raise FileExistsError(f"Non-empty result directory has no valid experiment_config.json: {output}")
    atomic_json(config, path)


def metadata(config: dict, split: str, my: int, support: str, replicate: int) -> dict:
    network = replicate if split == "calibration" else 10000 + replicate
    return {
        "experiment_name": config["experiment"],
        "split": split,
        "M_x": config["M_x"],
        "M_y": my,
        "observation_ratio": my / config["M_x"],
        "T": config["T"],
        "C_family": "gaussian_isotropic",
        "C_MODE": "gaussian_isotropic",
        "method": "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_C_known_R",
        "candidate_support": support,
        "true_network_id": network,
        "replicate_id": replicate,
        "seed_group": "CALIBRATION" if split == "calibration" else "EVALUATION",
        "run_id": f"{split}_Mx{config['M_x']}_My{my}_{support}_net{network}_rep{replicate}",
    }


def simulate(config: dict, meta: dict):
    return rectangular_experiment.simulate(
        config["M_x"], meta["M_y"], config["T"], "gaussian_isotropic",
        meta["true_network_id"], meta["replicate_id"],
    )


def paired_candidate_masks(config: dict, network: int, replicate: int) -> dict[str, np.ndarray]:
    """Exact generalized form of the validated 34S paired permissive-mask rule."""
    mx, T = config["M_x"], config["T"]
    nseed = rectangular_experiment.BASE_SEED + mx * 100000 + network * 1000
    _, _, truth = louis_experiment.fixed_network(mx, nseed)
    truth = np.asarray(truth, bool)
    np.fill_diagonal(truth, False)
    true_pairs = list(zip(*np.where(truth)))
    false_pairs = [(i, j) for i in range(mx) for j in range(mx) if i != j and not truth[i, j]]
    source_scores = []
    for my in config["M_y_values"]:
        seed = nseed + my * 10000 + T + replicate
        C = rectangular_experiment.make_C(my, mx, "gaussian_isotropic", seed)
        R = 0.6 * np.eye(my)
        values = np.diag(C.T @ np.linalg.solve(R, C))
        source_scores.append((values - values.min()) / (values.max() - values.min() + NUMERIC_EPS))
    source_score = np.mean(source_scores, axis=0)
    false_score = np.asarray([(source_score[i] + source_score[j]) / 2 for i, j in false_pairs])
    bins = np.searchsorted(np.quantile(false_score, [0.25, 0.50, 0.75]), false_score, side="right")
    rng = np.random.default_rng(6100000 + network * 1000 + replicate)
    quartiles = []
    for quartile in range(4):
        values = [pair for pair, bin_id in zip(false_pairs, bins) if bin_id == quartile]
        rng.shuffle(values)
        quartiles.append(values)
    ordered = []
    while any(quartiles):
        for values in quartiles:
            if values:
                ordered.append(values.pop())
    full = np.ones((mx, mx), dtype=bool)
    permissive = np.eye(mx, dtype=bool) | truth
    K = len(true_pairs)
    needed = max(0, min(2 * K, mx * (mx - 1)) - K)
    for target, source in ordered[:needed]:
        permissive[target, source] = True
    if not np.all(permissive[truth]):
        raise RuntimeError("The permissive_2K mask excluded a true edge.")
    return {"full_candidate": full, "permissive_2K": permissive}


def fit_model(config: dict, data: dict, mask: np.ndarray, seed: int):
    return HybridVBARDVARXSSMKnownCWithBQSupportMask(
        2, 3, data["C"], data["Q"], data["R"], candidate_mask=mask,
        B_update_mode="free", B_ridge_lambda=0.0, base_B_ridge=0.0,
        estimate_Q=True, Q_update_mode="diag_shrink_scalar", Q_floor_mode="none",
        Q_floor_value=0.0, Q_shrinkage_rho=0.25, Q_update_damping=0.5,
        include_A_posterior_uncertainty_in_Q=True, max_iter=config["VB_MAX_ITER"],
        tol_objective=1e-4, tol_A_change=1e-4, tol_B_change=1e-4,
        tol_Q_change=1e-4, tol_alpha_change=1e-4, a0=1e-3, b0=1e-3,
        diagonal_prior_precision=1e-4, posterior_jitter=1e-8,
        random_state=seed,
    ).fit(data["y"], data["u"])


def compute_louis(config: dict, model, data: dict, seed: int):
    samples = sample_companion_trajectories_ffbs(
        model.smooth_result_, model.F, config["N_FFBS_SAMPLES"], seed,
    )
    beta, M = model._pack_A(), model.n_states
    priors = np.asarray([
        prior_precision_for_row(M, 2, row, model.alpha_mean_, model.diagonal_prior_precision)
        for row in range(M)
    ])
    missing = estimate_missing_information_numba(
        samples, data["u"], beta, model.B_matrices, model.Q, 2, 3, priors,
    )
    result = []
    for target, (current, item) in enumerate(zip(model.A_row_covariances_, missing)):
        allowed = model._allowed_columns(target)
        block, _, _ = stabilized_louis_covariance(
            np.linalg.pinv(current[np.ix_(allowed, allowed)]),
            item["missing_information"][np.ix_(allowed, allowed)], 0.70, 0.90,
        )
        full = np.zeros_like(current)
        full[np.ix_(allowed, allowed)] = block
        result.append(full)
    return result


def dynamic_observability(A: np.ndarray, C: np.ndarray, R: np.ndarray, horizon: int = 10):
    F = build_var_companion_matrix(A)
    C_aug = np.hstack([C, np.zeros_like(C)])
    base = C_aug.T @ np.linalg.solve(R, C_aug)
    gramian, power = np.zeros_like(F), np.eye(len(F))
    for _ in range(horizon + 1):
        gramian += power.T @ base @ power
        power = power @ F
    values = np.diag(gramian)[: C.shape[1]]
    return (values - values.min()) / (values.max() - values.min() + NUMERIC_EPS)


def safe_covariance(block: np.ndarray) -> tuple[np.ndarray, dict]:
    symmetric = 0.5 * (np.asarray(block, float) + np.asarray(block, float).T)
    before = np.linalg.eigvalsh(symmetric)
    maximum = max(float(before.max()), NUMERIC_EPS)
    floor = max(NUMERIC_EPS, 1e-8 * maximum)
    values, vectors = np.linalg.eigh(symmetric)
    after_values = np.maximum(values, floor)
    stable = vectors @ np.diag(after_values) @ vectors.T
    stable = 0.5 * (stable + stable.T)
    return stable, {
        "min_eig_before": float(before.min()),
        "min_eig_after": float(after_values.min()),
        "condition_number": float(after_values.max() / after_values.min()),
        "stabilization_applied": bool(np.any(values < floor)),
    }


def group_stats(center, truth, covariance) -> dict:
    covariance, diagnostics = safe_covariance(covariance)
    difference = np.asarray(center) - np.asarray(truth)
    try:
        D2 = float(difference @ np.linalg.solve(covariance, difference))
    except np.linalg.LinAlgError:
        D2 = float(difference @ np.linalg.pinv(covariance) @ difference)
    diagonal = np.maximum(np.diag(covariance), 0.0)
    coefficient_cover = np.abs(difference) <= 1.96 * np.sqrt(diagonal)
    sign, logdet = np.linalg.slogdet(covariance)
    determinant = float(np.exp(logdet)) if sign > 0 else np.nan
    return {
        "D2": D2,
        "group_covered_95": bool(D2 <= CHI2_95_DF2),
        "coefficient_coverage_mean": float(coefficient_cover.mean()),
        "coefficient_coverage_lag1": bool(coefficient_cover[0]),
        "coefficient_coverage_lag2": bool(coefficient_cover[1]),
        "mean_ci_width": float(np.mean(3.92 * np.sqrt(diagonal))),
        "median_ci_width": float(np.median(3.92 * np.sqrt(diagonal))),
        "standardized_abs_error_mean": float(np.mean(np.abs(difference) / np.maximum(np.sqrt(diagonal), NUMERIC_EPS))),
        "cov_trace": float(np.trace(covariance)),
        "cov_determinant": determinant,
        "ellipse_area_95": float(np.pi * CHI2_95_DF2 * np.sqrt(determinant)) if np.isfinite(determinant) else np.nan,
        **diagnostics,
    }


def spread_sample(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if count <= 0 or not len(frame):
        return frame.iloc[0:0]
    if len(frame) <= count:
        return frame.copy()
    ordered = frame.sort_values("louis_group_snr").reset_index(drop=True)
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, count)).astype(int))
    selected = ordered.iloc[positions]
    if len(selected) < count:
        remaining = ordered.drop(selected.index)
        selected = pd.concat([selected, remaining.sample(count - len(selected), random_state=int(rng.integers(2**31 - 1)))])
    return selected.iloc[:count]


def benchmark_groups(config: dict, model, data: dict, mask: np.ndarray, louis, meta: dict, seed: int):
    M = model.n_states
    J = data["C"].T @ np.linalg.solve(data["R"], data["C"])
    node_o = np.diag(J)
    Ahat, Atrue = np.asarray(model.A_mean_matrices_), np.asarray(data["A"])
    dynamic = dynamic_observability(Ahat, data["C"], data["R"], 10)
    rows = []
    for target in range(M):
        for source in range(M):
            if target == source or not mask[target, source]:
                continue
            columns = [source, M + source]
            center = Ahat[:, target, source]
            block, _ = safe_covariance(louis[target][np.ix_(columns, columns)])
            snr = float(np.sqrt(max(center @ np.linalg.pinv(block) @ center, 0.0)))
            raw = float(node_o[target] * node_o[source])
            rows.append({
                **meta, "target": target, "source": source,
                "true_edge": bool(data["mask"][target, source]),
                "louis_group_snr": snr,
                "edge_o_inst_product": raw,
                "log_edge_o_inst_product": float(np.log(raw + NUMERIC_EPS)),
                "edge_o_dyn_estA_mean_K10": float((dynamic[target] + dynamic[source]) / 2),
                "estimated_group_norm": float(np.linalg.norm(center)),
                "true_group_norm": float(np.linalg.norm(Atrue[:, target, source])),
                "group_error_norm": float(np.linalg.norm(center - Atrue[:, target, source])),
            })
    candidates = pd.DataFrame(rows)
    log_values = candidates["log_edge_o_inst_product"]
    candidates["normalized_edge_o_inst_product"] = (log_values - log_values.mean()) / max(log_values.std(ddof=0), NUMERIC_EPS)
    candidates["observability_quartile"] = pd.qcut(
        candidates["edge_o_inst_product"].rank(method="first"), 4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )
    budget = min(config["benchmark_group_budget"], len(candidates))
    base, remainder = divmod(budget, 4)
    rng, chosen = np.random.default_rng(seed), []
    for q_index, quartile in enumerate(["Q1_low", "Q2", "Q3", "Q4_high"]):
        group = candidates[candidates.observability_quartile == quartile]
        q_budget = base + int(q_index < remainder)
        active_target = min((q_budget + 1) // 2, int(group.true_edge.sum()))
        active = spread_sample(group[group.true_edge], active_target, rng).copy()
        active["benchmark_selection_reason"] = f"truth_stratified_{quartile}_active_snr_spread"
        zero_needed = q_budget - len(active)
        zero = spread_sample(group[~group.true_edge], zero_needed, rng).copy()
        zero["benchmark_selection_reason"] = f"truth_stratified_{quartile}_zero_snr_spread_or_fill"
        chosen.extend([active, zero])
    selected = pd.concat(chosen, ignore_index=True)
    if selected.duplicated(["target", "source"]).any():
        raise RuntimeError("Benchmark selection produced duplicated directed edges.")
    selected["benchmark_truth_used_for_sampling_only"] = True
    return selected


def perturb_one(arguments):
    model, y, u, index, max_iter, rescue_iter = arguments
    started = time.perf_counter()
    column, plus, minus = sensitivity_column(
        model, y, u, index, EPS, max_iter, 1e-4, reestimate_B=True, reestimate_Q=True,
    )
    rescue = not (plus.converged and minus.converged)
    if rescue and rescue_iter > max_iter:
        column, plus, minus = sensitivity_column(
            model, y, u, index, EPS, rescue_iter, 1e-4, reestimate_B=True, reestimate_Q=True,
        )
    return index, column, {
        "direction_index": index,
        "plus_converged": plus.converged,
        "minus_converged": minus.converged,
        "plus_n_iter": plus.n_iter,
        "minus_n_iter": minus.n_iter,
        "rescue_used": rescue,
        "finite_difference_valid": bool(np.all(np.isfinite(column))),
        "runtime_seconds": time.perf_counter() - started,
        "warning_flag": ";".join(filter(None, [plus.warning_flag, minus.warning_flag])),
    }


def stageb_checkpoint_path(output: Path, meta: dict, index: int) -> Path:
    directory = output / "stageB_column_checkpoints" / meta["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"direction_{int(index)}.npy"


def stageb_diagnostic_path(output: Path, meta: dict, index: int) -> Path:
    return stageb_checkpoint_path(output, meta, index).with_suffix(".json")


def atomic_array(path: Path, value: np.ndarray):
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, np.asarray(value, float))
    os.replace(temp, path)


def run_stageb(config: dict, output: Path, model, data: dict, selected: pd.DataFrame, meta: dict):
    M = model.n_states
    indices = [global_index(int(row.target), lag, int(row.source), M, 2) for row in selected.itertuples() for lag in range(2)]
    columns = {}
    for index in indices:
        path = stageb_checkpoint_path(output, meta, index)
        if path.exists():
            columns[index] = np.load(path)
    pending = [index for index in indices if index not in columns]
    diagnostics, started = [], time.perf_counter()
    for index in indices:
        diagnostic_path = stageb_diagnostic_path(output, meta, index)
        if diagnostic_path.exists():
            with diagnostic_path.open(encoding="utf8") as handle:
                diagnostics.append({**meta, **json.load(handle)})
    if pending:
        print(f"\n{meta['run_id']}: Stage-B directions {len(columns)}/{len(indices)} complete", flush=True)
        with ProcessPoolExecutor(max_workers=min(config["workers"], len(pending))) as executor:
            futures = {
                executor.submit(
                    perturb_one,
                    (model, data["y"], data["u"], index, config["PERTURBED_VB_MAX_ITER"], config["PERTURBED_RESCUE_MAX_ITER"]),
                ): index
                for index in pending
            }
            completed = len(columns)
            for future in as_completed(futures):
                index, column, diagnostic = future.result()
                if np.all(np.isfinite(column)):
                    columns[index] = column
                    atomic_array(stageb_checkpoint_path(output, meta, index), column)
                    atomic_json(diagnostic, stageb_diagnostic_path(output, meta, index))
                diagnostics.append({**meta, **diagnostic})
                completed += 1
                width = 24
                filled = round(width * completed / len(indices))
                print(f"\r  Stage-B [{'#'*filled}{'-'*(width-filled)}] {completed}/{len(indices)}", end="", flush=True)
        print()
    valid = [index for index in indices if index in columns and np.all(np.isfinite(columns[index]))]
    if len(valid) != len(indices):
        raise FloatingPointError(f"Only {len(valid)}/{len(indices)} Stage-B directions are valid.")
    matrix = np.column_stack([columns[index] for index in valid])
    raw, projected, projection = project_selected_covariance(matrix, valid)
    position = {index: offset for offset, index in enumerate(valid)}
    measured = sum(float(row.get("runtime_seconds", 0.0)) for row in diagnostics)
    stage_seconds = measured if measured > 0 else time.perf_counter() - started
    return valid, position, raw, projected, projection, pd.DataFrame(diagnostics), stage_seconds


def safe_curve(truth, score) -> dict:
    metrics, _ = louis_experiment.safe_curve(np.asarray(truth, bool), np.asarray(score, float))
    return metrics


def point_and_spectral_metrics(config: dict, model, data: dict, mask: np.ndarray, louis, meta: dict):
    Ahat, Atrue, M = np.asarray(model.A_mean_matrices_), np.asarray(data["A"]), model.n_states
    off = ~np.eye(M, dtype=bool)
    pairs = [(target, source) for target in range(M) for source in range(M) if target != source]
    truth = np.asarray([data["mask"][target, source] for target, source in pairs], bool)
    A_score = np.asarray([np.linalg.norm(Ahat[:, target, source]) for target, source in pairs])
    A_metrics = safe_curve(truth, A_score)
    K = int(truth.sum())
    top = np.argsort(-A_score)[:K]
    A_detected = np.zeros(len(pairs), dtype=bool); A_detected[top] = True
    A_top_precision = float(truth[top].mean()) if K else np.nan
    A_top_recall = float(truth[top].sum() / K) if K else np.nan
    latent = recovery_experiment.latent_metrics(model, data)
    point = {
        **meta,
        "smoothed_state_corr": latent["smoothed_state_correlation"],
        "smoothed_state_MSE": latent["smoothed_state_MSE"],
        "A_relative_error": relative(Ahat, Atrue),
        "A_offdiag_relative_error": relative(Ahat[:, off], Atrue[:, off]),
        "B_correlation": correlations(np.asarray(model.B_matrices).ravel(), np.asarray(data["B"]).ravel()),
        "B_relative_error": relative(np.asarray(model.B_matrices), np.asarray(data["B"])),
        "Q_relative_error": relative(model.Q, data["Q"]),
        "A_ROC_AUC": A_metrics.get("ROC_AUC", np.nan),
        "A_PR_AUC": A_metrics.get("AUPRC", np.nan),
        "A_topK_precision": A_top_precision,
        "A_topK_recall": A_top_recall,
        "posterior_center_unchanged": True,
    }
    old_M = spectral_experiment.M
    spectral_experiment.M = M
    bands, summaries = None, None
    try:
        _, bands, _, _, _ = spectral_experiment.compute_spectral_tables(
            model.A_mean_matrices_, model.B_matrices, model.Q,
            np.linspace(0, np.pi, config["N_FREQUENCIES"]), data, meta, "estimated_AQ",
        )
        summaries, _ = spectral_experiment.metric_rows(bands)
    finally:
        spectral_experiment.M = old_M
    for key, value in meta.items():
        summaries[key] = value
    primary = bands[(bands.score_type == "transfer_spectral_gc_diagQ") & (bands.band_name == "full_band")]
    lookup = primary.set_index(["target", "source"]).integrated_score
    spectral_scores = np.asarray([lookup.get(pair, 0.0) for pair in pairs])
    spectral_metrics = safe_curve(truth, spectral_scores)
    top = np.argsort(-spectral_scores)[:K]
    spectral_detected = np.zeros(len(pairs), dtype=bool); spectral_detected[top] = True
    point.update({
        "spectral_ROC_AUC": spectral_metrics.get("ROC_AUC", np.nan),
        "spectral_PR_AUC": spectral_metrics.get("AUPRC", np.nan),
        "spectral_topK_precision": float(truth[top].mean()) if K else np.nan,
        "spectral_topK_recall": float(truth[top].sum() / K) if K else np.nan,
    })
    edge_scores = {(target, source): (A_score[index], spectral_scores[index], A_detected[index], spectral_detected[index]) for index, (target, source) in enumerate(pairs)}
    return point, summaries, edge_scores


def build_reference_rows(config: dict, selected: pd.DataFrame, model, data: dict, louis, valid, position, stage_cov, projection, edge_scores, meta: dict):
    M = model.n_states
    global_vb = stageb_experiment.o.global_covariance_block(model.A_row_covariances_)
    global_louis = stageb_experiment.o.global_covariance_block(louis)
    Ahat, Atrue = np.asarray(model.A_mean_matrices_), np.asarray(data["A"])
    rows = []
    for row in selected.to_dict("records"):
        target, source = int(row["target"]), int(row["source"])
        indices = [global_index(target, lag, source, M, 2) for lag in range(2)]
        positions = [position[index] for index in indices]
        beta_hat, beta_true = Ahat[:, target, source], Atrue[:, target, source]
        blocks = {
            "VB": global_vb[np.ix_(indices, indices)],
            "Louis": global_louis[np.ix_(indices, indices)],
            "StageB": stage_cov[np.ix_(positions, positions)],
        }
        stable = {name: safe_covariance(block)[0] for name, block in blocks.items()}
        stats = {name: group_stats(beta_hat, beta_true, block) for name, block in stable.items()}
        lambda_found = np.nan
        for value in LAMBDA_GRID:
            candidate, _ = safe_covariance(stable["Louis"] + value * (stable["StageB"] - stable["Louis"]))
            if group_stats(beta_hat, beta_true, candidate)["group_covered_95"]:
                lambda_found = float(value)
                break
        trace_louis, trace_stage = stats["Louis"]["cov_trace"], stats["StageB"]["cov_trace"]
        common = {
            **row,
            "beta_hat_lag1": beta_hat[0], "beta_hat_lag2": beta_hat[1],
            "beta_true_lag1": beta_true[0], "beta_true_lag2": beta_true[1],
            "A_model_score": edge_scores[(target, source)][0],
            "spectral_transfer_full_band_score": edge_scores[(target, source)][1],
            "A_detected_topK": bool(edge_scores[(target, source)][2]),
            "spectral_detected_topK": bool(edge_scores[(target, source)][3]),
            "delta_trace": trace_stage - trace_louis,
            "relative_stageb_inflation": (trace_stage - trace_louis) / max(trace_louis, NUMERIC_EPS),
            "delta_logdet": np.log(max(stats["StageB"]["cov_determinant"], NUMERIC_EPS)) - np.log(max(stats["Louis"]["cov_determinant"], NUMERIC_EPS)),
            "stageb_rescue": bool((not stats["Louis"]["group_covered_95"]) and stats["StageB"]["group_covered_95"]),
            "oracle_lambda_min": lambda_found,
            "oracle_lambda_censored": bool(not np.isfinite(lambda_found)),
            "stageB_projection_min_eigenvalue_raw": projection.get("min_eigenvalue_raw", np.nan),
            "stageB_projection_applied": projection.get("psd_projection_used", False),
        }
        for covariance_name, block in stable.items():
            prefix = covariance_name.lower()
            common.update({f"Sigma_{prefix}_{i}{j}": float(block[i, j]) for i in range(2) for j in range(2)})
            common.update({f"{prefix}_{key}": value for key, value in stats[covariance_name].items()})
        rows.append(common)
    return pd.DataFrame(rows)


def fit_ridge(X: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    penalty_matrix = penalty * np.eye(X.shape[1])
    penalty_matrix[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty_matrix, X.T @ y)


def fit_calibration_models(config: dict, calibration: pd.DataFrame):
    models, rows = {}, []
    keys = ["M_y", "candidate_support"]
    for values, group in calibration.groupby(keys, dropna=False):
        my, support = values
        target = group.oracle_lambda_min.fillna(1.5).clip(0, 1.5).to_numpy(float)
        snr = group.louis_group_snr.to_numpy(float)
        log_o = group.log_edge_o_inst_product.to_numpy(float)
        means = {"snr": float(np.mean(snr)), "log_o": float(np.mean(log_o))}
        scales = {"snr": float(max(np.std(snr), NUMERIC_EPS)), "log_o": float(max(np.std(log_o), NUMERIC_EPS))}
        z_snr, z_log = (snr - means["snr"]) / scales["snr"], (log_o - means["log_o"]) / scales["log_o"]
        designs = {
            "observability_only": np.c_[np.ones(len(group)), z_log],
            "snr_only": np.c_[np.ones(len(group)), z_snr],
            "snr_plus_observability": np.c_[np.ones(len(group)), z_snr, z_log, z_snr * z_log],
        }
        quartile_rules = {}
        for quartile, q_group in group.groupby("observability_quartile", observed=False):
            q_target = q_group.oracle_lambda_min.fillna(1.5).clip(0, 1.5)
            quartile_rules[str(quartile)] = {"median": float(q_target.median()), "coverage_targeted_q80": float(q_target.quantile(0.80))}
        fitted = {name: fit_ridge(X, target, config["ridge_penalty"]) for name, X in designs.items()}
        model = {"means": means, "scales": scales, "coefficients": fitted, "quartile_rules": quartile_rules}
        models[(int(my), str(support))] = model
        rows.extend([
            {"M_y": my, "candidate_support": support, "model_name": name,
             "feature_names": ";".join(("intercept", "z_log_observability") if name == "observability_only" else (("intercept", "z_snr") if name == "snr_only" else ("intercept", "z_snr", "z_log_observability", "interaction"))),
             "coefficients": ";".join(map(str, coefficients)), "snr_mean": means["snr"], "snr_scale": scales["snr"],
             "log_observability_mean": means["log_o"], "log_observability_scale": scales["log_o"],
             "ridge_penalty": config["ridge_penalty"], "quartile_rules_json": json.dumps(quartile_rules), "n_training_edges": len(group)}
            for name, coefficients in fitted.items()
        ])
        for quartile, rules in quartile_rules.items():
            rows.append({"M_y": my, "candidate_support": support, "model_name": "observability_quartile_rule",
                         "feature_names": quartile, "coefficients": rules["coverage_targeted_q80"],
                         "quartile_median_lambda": rules["median"], "quartile_q80_lambda": rules["coverage_targeted_q80"],
                         "snr_mean": means["snr"], "snr_scale": scales["snr"],
                         "log_observability_mean": means["log_o"], "log_observability_scale": scales["log_o"],
                         "ridge_penalty": np.nan, "quartile_rules_json": json.dumps(quartile_rules), "n_training_edges": len(group)})
    return models, pd.DataFrame(rows)


def predict(model: dict, frame: pd.DataFrame, model_name: str) -> np.ndarray:
    z_snr = (frame.louis_group_snr.to_numpy(float) - model["means"]["snr"]) / model["scales"]["snr"]
    z_log = (frame.log_edge_o_inst_product.to_numpy(float) - model["means"]["log_o"]) / model["scales"]["log_o"]
    if model_name == "observability_only":
        X = np.c_[np.ones(len(frame)), z_log]
    elif model_name == "snr_only":
        X = np.c_[np.ones(len(frame)), z_snr]
    else:
        X = np.c_[np.ones(len(frame)), z_snr, z_log, z_snr * z_log]
    return np.clip(X @ model["coefficients"][model_name], 0.0, 1.5)


def covariance_from_row(row: pd.Series, name: str) -> np.ndarray:
    prefix = name.lower()
    return np.asarray([[row[f"Sigma_{prefix}_00"], row[f"Sigma_{prefix}_01"]], [row[f"Sigma_{prefix}_10"], row[f"Sigma_{prefix}_11"]]], float)


def evaluation_rows(config: dict, reference: pd.DataFrame, model: dict, seed: int, stage_seconds: float, baseline_seconds: float):
    frame = reference.copy()
    predictions = {
        name: predict(model, frame, name)
        for name in ("observability_only", "snr_only", "snr_plus_observability")
    }
    n_budget = max(1, int(np.floor(config["stageb_budget_fraction"] * len(frame))))
    allocations = []
    allocations.append(("louis_only", 0, np.zeros(len(frame), bool), np.zeros(len(frame)), 0))
    allocations.append(("full_stageb_reference", 0, np.ones(len(frame), bool), np.ones(len(frame)), 0))
    rng = np.random.default_rng(seed)
    for repeat in range(config["random_allocation_repeats"]):
        chosen = rng.choice(len(frame), n_budget, replace=False)
        mask = np.zeros(len(frame), bool); mask[chosen] = True
        allocations.append(("random_budget_stageb", repeat, mask, np.ones(len(frame)), n_budget))
    for method, predictor in (
        ("snr_budget_stageb", "snr_only"),
        ("observability_budget_stageb", "observability_only"),
        ("snr_observability_budget_stageb", "snr_plus_observability"),
    ):
        chosen = np.argsort(-predictions[predictor])[:n_budget]
        mask = np.zeros(len(frame), bool); mask[chosen] = True
        allocations.append((method, 0, mask, np.ones(len(frame)), n_budget))
    chosen_obs = np.argsort(-predictions["observability_only"])[:n_budget]
    chosen_hybrid = np.argsort(-predictions["snr_plus_observability"])[:n_budget]
    for method, chosen, lambdas in (
        ("global_0p75_budget_stageb", chosen_hybrid, np.full(len(frame), 0.75)),
        ("observability_adaptive_budget_stageb", chosen_obs, predictions["observability_only"]),
        ("snr_observability_adaptive_budget_stageb", chosen_hybrid, predictions["snr_plus_observability"]),
    ):
        mask = np.zeros(len(frame), bool); mask[chosen] = True
        allocations.append((method, 0, mask, lambdas, n_budget))
    output = []
    for method, repeat, allocated, lambdas, count in allocations:
        for index, (_, row) in enumerate(frame.iterrows()):
            louis = covariance_from_row(row, "Louis")
            stage = covariance_from_row(row, "StageB")
            used_stageb = bool(allocated[index])
            lam = float(lambdas[index]) if used_stageb else 0.0
            covariance, _ = safe_covariance(louis + lam * (stage - louis))
            beta_hat = np.asarray([row.beta_hat_lag1, row.beta_hat_lag2])
            beta_true = np.asarray([row.beta_true_lag1, row.beta_true_lag2])
            stats = group_stats(beta_hat, beta_true, covariance)
            prospective_stage_seconds = stage_seconds * count / max(len(frame), 1)
            output.append({
                **row.to_dict(), "covariance_method": method, "allocation_repeat": repeat,
                "received_stageB": used_stageb, "lambda_pred": lam,
                "predicted_lambda_observability": predictions["observability_only"][index],
                "predicted_lambda_snr": predictions["snr_only"][index],
                "predicted_lambda_snr_observability": predictions["snr_plus_observability"][index],
                "number_stageb_groups": count, "number_stageb_directions": 2 * count,
                "total_stageb_fits": 4 * count,
                "stageb_runtime_sec": prospective_stage_seconds,
                "full_reference_stageb_runtime_sec": stage_seconds,
                "total_runtime_sec": baseline_seconds + prospective_stage_seconds,
                **stats,
            })
    return pd.DataFrame(output)


def completed_keys(runtime: pd.DataFrame) -> set:
    if not len(runtime):
        return set()
    success = runtime[runtime.run_status == "success"]
    return set(zip(success.split, success.M_y.astype(int), success.candidate_support, success.replicate_id.astype(int)))


def append_checkpoint(path: Path, new_frame: pd.DataFrame, keys: list[str]):
    old = read_csv(path)
    combined = pd.concat([old, new_frame], ignore_index=True, sort=False)
    subset = [key for key in keys if key in combined]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    atomic_csv(combined, str(path))
    return combined


def run_one(config: dict, output: Path, meta: dict):
    started = time.perf_counter()
    data, seed = simulate(config, meta)
    masks = paired_candidate_masks(config, meta["true_network_id"], meta["replicate_id"])
    mask = masks[meta["candidate_support"]]
    if not np.all(mask[data["mask"]]):
        raise RuntimeError("Candidate mask excludes true edges.")
    vb_started = time.perf_counter()
    model = fit_model(config, data, mask, seed + 500)
    vb_seconds = time.perf_counter() - vb_started
    louis_started = time.perf_counter()
    louis = compute_louis(config, model, data, seed + 700)
    louis_seconds = time.perf_counter() - louis_started
    selected = benchmark_groups(config, model, data, mask, louis, meta, seed + 900)
    valid, position, raw, stage, projection, perturbations, stage_seconds = run_stageb(config, output, model, data, selected, meta)
    point, spectral, edge_scores = point_and_spectral_metrics(config, model, data, mask, louis, meta)
    reference = build_reference_rows(config, selected, model, data, louis, valid, position, stage, projection, edge_scores, meta)
    runtime = pd.DataFrame([{**meta,
        "baseline_VB_runtime_seconds": vb_seconds, "louis_runtime_seconds": louis_seconds,
        "stageb_runtime_seconds": stage_seconds, "total_runtime_seconds": time.perf_counter() - started,
        "n_benchmark_groups": len(selected), "n_stageb_directions": len(valid),
        "n_perturbed_fits": 2 * len(valid), "n_workers": config["workers"],
        "run_status": "success", "error_type": "", "error_message": "",
    }])
    return reference, pd.DataFrame([point]), spectral, runtime, perturbations


def run_split(config: dict, output: Path, split: str, model_lookup=None):
    runtime_path = output / "runtime_summary_partial.csv"
    runtime = read_csv(runtime_path)
    done = completed_keys(runtime)
    count = config["calibration_replicates"] if split == "calibration" else config["evaluation_replicates"]
    tasks = [
        metadata(config, split, my, support, replicate)
        for my in config["M_y_values"]
        for support in config["candidate_support_regimes"]
        for replicate in range(count)
    ]
    pending = [meta for meta in tasks if (split, meta["M_y"], meta["candidate_support"], meta["replicate_id"]) not in done]
    progress = Progress(len(pending))
    for meta in pending:
        try:
            reference, point, spectral, run_runtime, perturbations = run_one(config, output, meta)
            if split == "calibration":
                append_checkpoint(output / "calibration_edge_rows_partial.csv", reference,
                                  ["run_id", "target", "source"])
            else:
                model = model_lookup[(int(meta["M_y"]), str(meta["candidate_support"]))]
                evaluated = evaluation_rows(
                    config, reference, model, 8800000 + meta["true_network_id"],
                    float(run_runtime.stageb_runtime_seconds.iloc[0]),
                    float(run_runtime.baseline_VB_runtime_seconds.iloc[0] + run_runtime.louis_runtime_seconds.iloc[0]),
                )
                append_checkpoint(output / "evaluation_edge_rows_partial.csv", evaluated,
                                  ["run_id", "target", "source", "covariance_method", "allocation_repeat"])
            append_checkpoint(output / "network_recovery_summary_partial.csv", point, ["run_id"])
            append_checkpoint(output / "spectral_gc_summary_partial.csv", spectral,
                              ["run_id", "score_type", "band_name"])
            append_checkpoint(output / "stageB_perturbation_diagnostics_partial.csv", perturbations,
                              ["run_id", "direction_index"])
            append_checkpoint(runtime_path, run_runtime, ["run_id"])
            progress.update(f"{split} My={meta['M_y']} {meta['candidate_support']}")
        except Exception as error:
            failed = pd.DataFrame([{**meta, "run_status": "failed", "error_type": type(error).__name__,
                                    "error_message": str(error), "traceback": traceback.format_exc(),
                                    "total_runtime_seconds": np.nan}])
            append_checkpoint(runtime_path, failed, ["run_id"])
            progress.update(f"FAILED {split} My={meta['M_y']} {meta['candidate_support']}")


def predictive_diagnostics(evaluation: pd.DataFrame) -> pd.DataFrame:
    base = evaluation[evaluation.covariance_method == "louis_only"].drop_duplicates(["run_id", "target", "source"])
    rows = []
    for values, group in base.groupby(["M_y", "candidate_support"], dropna=False):
        target_oracle = group.oracle_lambda_min.fillna(1.5).to_numpy(float)
        targets = {
            "relative_stageb_inflation": group.relative_stageb_inflation.to_numpy(float),
            "oracle_lambda_min": target_oracle,
        }
        for model_name, column in (
            ("observability_only", "predicted_lambda_observability"),
            ("snr_only", "predicted_lambda_snr"),
            ("snr_plus_observability", "predicted_lambda_snr_observability"),
        ):
            prediction = group[column].to_numpy(float)
            for target_name, target in targets.items():
                residual = target - prediction
                denominator = np.sum((target - target.mean()) ** 2)
                rows.append({
                    "M_y": values[0], "candidate_support": values[1], "prediction_model": model_name,
                    "target": target_name, "R2": 1 - np.sum(residual**2) / max(denominator, NUMERIC_EPS),
                    "Spearman": correlations(rankdata(target), rankdata(prediction)),
                    "MAE": float(np.mean(np.abs(residual))), "ROC_AUC": np.nan, "n_edges": len(group),
                })
            rescue_metrics = safe_curve(group.stageb_rescue, prediction)
            rows.append({"M_y": values[0], "candidate_support": values[1], "prediction_model": model_name,
                         "target": "stageb_rescue", "R2": np.nan, "Spearman": np.nan, "MAE": np.nan,
                         "ROC_AUC": rescue_metrics.get("ROC_AUC", np.nan), "n_edges": len(group)})
    return pd.DataFrame(rows)


def evaluation_replicate_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "M_y", "candidate_support", "covariance_method", "allocation_repeat"]
    rows = []
    for values, group in evaluation.groupby(keys, dropna=False):
        active, zero = group[group.true_edge], group[~group.true_edge]
        louis_coverage = group.louis_group_covered_95.mean()
        row = {**dict(zip(keys, values)),
            "active_coeff_coverage": active.coefficient_coverage_mean.mean(),
            "zero_coeff_coverage": zero.coefficient_coverage_mean.mean(),
            "active_group_coverage": active.group_covered_95.mean(),
            "zero_group_coverage": zero.group_covered_95.mean(),
            "all_group_coverage": group.group_covered_95.mean(),
            "n_benchmark_groups": len(group),
            "mean_cov_trace": group.cov_trace.mean(), "mean_ellipse_area": group.ellipse_area_95.mean(),
            "n_stageb_directions": group.number_stageb_directions.iloc[0],
            "total_stageb_fits": group.total_stageb_fits.iloc[0],
            "stageb_runtime_sec": group.stageb_runtime_sec.iloc[0], "total_runtime_sec": group.total_runtime_sec.iloc[0],
            "coverage_gain_vs_louis": group.group_covered_95.mean() - louis_coverage,
            "coverage_gain_per_stageb_direction": (group.group_covered_95.mean() - louis_coverage) / max(group.number_stageb_directions.iloc[0], 1),
            "runtime_normalized_coverage_gain": (group.group_covered_95.mean() - louis_coverage) / max(group.stageb_runtime_sec.iloc[0], NUMERIC_EPS),
        }
        for quartile in ["Q1_low", "Q2", "Q3", "Q4_high"]:
            q = group[group.observability_quartile == quartile]
            row[f"{quartile}_active_group_coverage"] = q[q.true_edge].group_covered_95.mean()
            row[f"{quartile}_zero_group_coverage"] = q[~q.true_edge].group_covered_95.mean()
            row[f"{quartile}_fraction_receiving_stageB"] = q.received_stageB.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summaries(evaluation: pd.DataFrame):
    replicate = evaluation_replicate_summary(evaluation)
    group_keys = ["M_y", "candidate_support", "covariance_method"]
    numeric = [column for column in replicate if column not in [*group_keys, "run_id", "allocation_repeat"]]
    method = replicate.groupby(group_keys, dropna=False)[numeric].mean().reset_index()
    method["stageb_budget_fraction"] = method.n_stageb_directions / np.maximum(2 * method.n_benchmark_groups, 1)
    uncertainty = evaluation.groupby(
        ["M_y", "candidate_support", "covariance_method", "true_edge"], dropna=False
    ).agg(
        coefficient_coverage_95=("coefficient_coverage_mean", "mean"),
        group_coverage_95=("group_covered_95", "mean"), mean_ci_width=("mean_ci_width", "mean"),
        median_ci_width=("median_ci_width", "median"), mean_standardized_abs_error=("standardized_abs_error_mean", "mean"),
        mean_cov_trace=("cov_trace", "mean"), mean_ellipse_area=("ellipse_area_95", "mean"), n_edges=("source", "size"),
    ).reset_index()
    quartile = evaluation.groupby(
        ["M_y", "candidate_support", "covariance_method", "observability_quartile", "true_edge"], observed=False, dropna=False
    ).agg(
        group_coverage_95=("group_covered_95", "mean"), coefficient_coverage_95=("coefficient_coverage_mean", "mean"),
        mean_cov_trace=("cov_trace", "mean"), mean_ellipse_area=("ellipse_area_95", "mean"),
        mean_A_group_error=("group_error_norm", "mean"), fraction_receiving_stageB=("received_stageB", "mean"),
        A_detection_rate=("A_detected_topK", "mean"), spectral_detection_rate=("spectral_detected_topK", "mean"),
        fraction_stageB_rescued=("stageb_rescue", "mean"), n_edges=("source", "size"),
    ).reset_index()
    allocation = evaluation.groupby(
        ["M_y", "candidate_support", "covariance_method", "observability_quartile"], observed=False, dropna=False
    ).agg(fraction_receiving_stageB=("received_stageB", "mean"), n_edges=("source", "size"),
          mean_predicted_lambda=("lambda_pred", "mean"), mean_stageb_runtime_sec=("stageb_runtime_sec", "mean")).reset_index()
    return replicate, method, uncertainty, quartile, allocation


def decision_summary(method: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in method.groupby(["M_y", "candidate_support"], dropna=False):
        lookup = group.set_index("covariance_method")
        def value(method_name, column):
            return lookup.loc[method_name, column] if method_name in lookup.index else np.nan
        louis = value("louis_only", "active_group_coverage")
        random = value("random_budget_stageb", "active_group_coverage")
        snr = value("snr_budget_stageb", "active_group_coverage")
        hybrid = value("snr_observability_budget_stageb", "active_group_coverage")
        full = value("full_stageb_reference", "active_group_coverage")
        q1_louis = value("louis_only", "Q1_low_active_group_coverage")
        q1_hybrid = value("snr_observability_budget_stageb", "Q1_low_active_group_coverage")
        q4_louis = value("louis_only", "Q4_high_active_group_coverage")
        q4_hybrid = value("snr_observability_budget_stageb", "Q4_high_active_group_coverage")
        d = diagnostics[(diagnostics.M_y == values[0]) & (diagnostics.candidate_support == values[1]) & (diagnostics.target == "relative_stageb_inflation")]
        pred = d.set_index("prediction_model") if len(d) else pd.DataFrame()
        obs_r2 = pred.loc["observability_only", "R2"] if "observability_only" in pred.index else np.nan
        snr_r2 = pred.loc["snr_only", "R2"] if "snr_only" in pred.index else np.nan
        hybrid_r2 = pred.loc["snr_plus_observability", "R2"] if "snr_plus_observability" in pred.index else np.nan
        rows.append({
            "M_y": values[0], "candidate_support": values[1],
            "observability_predicts_stageb_inflation": bool(np.isfinite(obs_r2) and obs_r2 > 0.05),
            "snr_predicts_stageb_inflation": bool(np.isfinite(snr_r2) and snr_r2 > 0.05),
            "snr_plus_obs_improves_prediction": bool(np.isfinite(hybrid_r2) and hybrid_r2 > max(obs_r2, snr_r2) + 0.02),
            "obs_budget_beats_random_budget": value("observability_budget_stageb", "active_group_coverage") > random,
            "hybrid_budget_beats_random_budget": hybrid > random,
            "hybrid_budget_beats_snr_only_budget": hybrid > snr,
            "hybrid_improves_Q1_active_coverage_vs_louis": q1_hybrid > q1_louis,
            "hybrid_reduces_Q1_Q4_coverage_gap": abs(q1_hybrid - q4_hybrid) < abs(q1_louis - q4_louis),
            "hybrid_approaches_full_stageb_at_lower_cost": bool(abs(hybrid - full) <= 0.03 and value("snr_observability_budget_stageb", "stageb_runtime_sec") < value("full_stageb_reference", "stageb_runtime_sec")),
            "delta_active_group_coverage_vs_louis": hybrid - louis,
            "delta_Q1_active_group_coverage_vs_louis": q1_hybrid - q1_louis,
            "delta_vs_random_budget": hybrid - random,
            "delta_vs_snr_budget": hybrid - snr,
            "runtime_ratio_vs_full_stageb": value("snr_observability_budget_stageb", "stageb_runtime_sec") / max(value("full_stageb_reference", "stageb_runtime_sec"), NUMERIC_EPS),
            "observability_inflation_R2": obs_r2, "snr_inflation_R2": snr_r2, "hybrid_inflation_R2": hybrid_r2,
        })
    result = pd.DataFrame(rows)
    if len(result):
        hybrid_success = result.hybrid_budget_beats_random_budget.mean() >= 0.75 and result.delta_active_group_coverage_vs_louis.mean() > 0
        hybrid_increment = result.snr_plus_obs_improves_prediction.mean() >= 0.5 and result.hybrid_budget_beats_snr_only_budget.mean() >= 0.5
        obs_alone = result.obs_budget_beats_random_budget.mean() >= 0.75
        conclusion = ("A_observability_adds_substantial_value" if obs_alone and hybrid_success else
                      "B_observability_weak_alone_but_hybrid_useful" if hybrid_success and hybrid_increment else
                      "C_observability_explains_difficulty_but_not_allocation" if result.observability_predicts_stageb_inflation.any() else
                      "D_deployable_predictors_insufficient_use_stronger_sparse_geometry")
        result["scientific_conclusion"] = conclusion
    return result


def make_plots(output: Path, calibration: pd.DataFrame, evaluation: pd.DataFrame, method: pd.DataFrame, quartile: pd.DataFrame, allocation: pd.DataFrame):
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(plot_dir / name, dpi=160); plt.close()
    plt.scatter(calibration.log_edge_o_inst_product, calibration.relative_stageb_inflation, s=7, alpha=0.35)
    plt.xlabel("log edge observability"); plt.ylabel("relative Stage-B inflation"); save("01_inflation_vs_log_observability.png")
    plt.scatter(calibration.louis_group_snr, calibration.relative_stageb_inflation, s=7, alpha=0.35)
    plt.xlabel("Louis group-SNR"); plt.ylabel("relative Stage-B inflation"); save("02_inflation_vs_louis_snr.png")
    scatter = plt.scatter(calibration.louis_group_snr, calibration.log_edge_o_inst_product, c=calibration.relative_stageb_inflation, s=9)
    plt.colorbar(scatter, label="relative Stage-B inflation"); plt.xlabel("Louis group-SNR"); plt.ylabel("log observability"); save("03_snr_observability_inflation.png")
    active = quartile[quartile.true_edge]
    for method_name, group in active.groupby("covariance_method"):
        values = group.groupby("observability_quartile", observed=False).group_coverage_95.mean().reindex(["Q1_low", "Q2", "Q3", "Q4_high"])
        plt.plot(range(4), values, marker="o", label=method_name)
    plt.xticks(range(4), ["Q1", "Q2", "Q3", "Q4"]); plt.ylabel("active group coverage"); plt.legend(fontsize=6); save("04_active_coverage_by_observability_quartile.png")
    for method_name, group in allocation.groupby("covariance_method"):
        values = group.groupby("observability_quartile", observed=False).fraction_receiving_stageB.mean().reindex(["Q1_low", "Q2", "Q3", "Q4_high"])
        plt.plot(range(4), values, marker="o", label=method_name)
    plt.xticks(range(4), ["Q1", "Q2", "Q3", "Q4"]); plt.ylabel("fraction receiving Stage-B"); plt.legend(fontsize=6); save("05_budget_by_observability_quartile.png")
    for method_name, group in method.groupby("covariance_method"):
        plt.scatter(group.stageb_runtime_sec, group.active_group_coverage, label=method_name)
    plt.xlabel("prospective Stage-B seconds"); plt.ylabel("active group coverage"); plt.legend(fontsize=6); save("06_coverage_vs_cost.png")
    base = evaluation[evaluation.covariance_method == "louis_only"].drop_duplicates(["run_id", "target", "source"])
    plt.scatter(base.oracle_lambda_min.fillna(1.5), base.predicted_lambda_snr_observability, s=8, alpha=0.4)
    plt.xlabel("oracle lambda (censored=1.5)"); plt.ylabel("held-out predicted lambda"); save("07_predicted_vs_oracle_lambda.png")
    gaps = method.copy(); gaps["Q1_Q4_active_coverage_gap"] = gaps.Q1_low_active_group_coverage - gaps.Q4_high_active_group_coverage
    gaps.groupby("covariance_method").Q1_Q4_active_coverage_gap.mean().plot.bar()
    plt.ylabel("Q1 - Q4 active coverage"); plt.xticks(rotation=25, ha="right"); save("08_Q1_Q4_coverage_gap.png")


def finalize(config: dict, output: Path):
    calibration = read_csv(output / "calibration_edge_rows_partial.csv")
    evaluation = read_csv(output / "evaluation_edge_rows_partial.csv")
    if not len(calibration) or not len(evaluation):
        raise RuntimeError("Calibration or evaluation checkpoint is empty; finalization is unsafe.")
    models, model_summary = fit_calibration_models(config, calibration)
    diagnostics = predictive_diagnostics(evaluation)
    replicate, method, uncertainty, quartile, allocation = summaries(evaluation)
    decision = decision_summary(method, diagnostics)
    network = read_csv(output / "network_recovery_summary_partial.csv")
    spectral = read_csv(output / "spectral_gc_summary_partial.csv")
    runtime = read_csv(output / "runtime_summary_partial.csv")
    outputs = {
        "calibration_edge_rows.csv": calibration,
        "calibration_model_summary.csv": model_summary,
        "calibration_prediction_diagnostics.csv": diagnostics,
        "evaluation_edge_rows.csv": evaluation,
        "evaluation_replicate_summary.csv": replicate,
        "method_comparison_summary.csv": method,
        "observability_quartile_summary.csv": quartile,
        "stageb_allocation_summary.csv": allocation,
        "uncertainty_calibration_summary.csv": uncertainty,
        "network_recovery_summary.csv": network,
        "spectral_gc_summary.csv": spectral,
        "runtime_summary.csv": runtime,
        "decision_summary.csv": decision,
    }
    for name, frame in outputs.items():
        atomic_csv(frame, str(output / name))
    make_plots(output, calibration, evaluation, method, quartile, allocation)
    atomic_json({"completed": True, "completed_at_unix": time.time(), "n_calibration_edges": len(calibration),
                 "n_evaluation_rows": len(evaluation)}, output / "_COMPLETED.json")
    print("\n34U-A complete. Main conclusion:", decision.scientific_conclusion.iloc[0] if len(decision) else "unavailable")


def main():
    args = parse_args()
    config = configuration(args.smoke)
    output = Path(config["results_directory"])
    write_or_validate_config(config, output)
    run_split(config, output, "calibration")
    calibration = read_csv(output / "calibration_edge_rows_partial.csv")
    if not len(calibration):
        raise RuntimeError("No successful calibration runs; evaluation cannot begin.")
    models, model_summary = fit_calibration_models(config, calibration)
    atomic_csv(model_summary, str(output / "calibration_model_summary_partial.csv"))
    run_split(config, output, "evaluation", models)
    finalize(config, output)


if __name__ == "__main__":
    main()
