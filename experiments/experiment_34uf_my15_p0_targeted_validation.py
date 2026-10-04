"""Experiment 34UF: targeted My=15 P0 Stage-B diagnostic validation.

This is the final selector-budget experiment.  It preserves the canonical
34R/34U-A/34UC estimator and legacy Stage-B map, compares the same P0 model
under frozen-My40 and My15-development calibration, and treats networks as
the independent units.  Confirmation networks are inaccessible until the
development design has been written atomically to ``frozen_design.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UF_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import experiments.experiment_34u_a_fullrun_observability_calibration as ua
import experiments.experiment_34uc_greedy_selector_validation as uc
import experiments.experiment_34ue_historical_multidomain_greedy_validation as ue
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.diagnostic_subset_design import conclusion_vector


EXPERIMENT = "34UF_targeted_My15_P0_StageB_diagnostic_validation"
OUTPUT_ROOT = Path("results/experiment_34uf_my15_p0_targeted_validation")
HISTORICAL_34UC = Path("results/experiment_34uc_greedy_selector_validation/prospective_validation_5refs")
HISTORICAL_34UA = Path("results/experiment_34u_a_fullrun_observability_calibration")
ALLOW_NEW_STAGEB = True
ALLOW_FULL_380_STAGEB = False
MAX_STAGEB_GROUPS_PER_NETWORK = 90
EPS = 1e-12
K_GRID = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 63, 70, 80, 90]
RAW_FEATURES = list(ue.RAW_FEATURES)
FEATURES = list(ue.FEATURES)
PRIMARY_CATEGORIES = [
    "active_coverage_conclusion", "zero_coverage_conclusion", "rescue_conclusion",
    "inflation_conclusion", "numerical_stability_conclusion",
]


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UF_SMOKE", "0") == "1")
    output = Path(args.results_dir) if args.results_dir else (OUTPUT_ROOT / "smoke_test" if smoke else OUTPUT_ROOT)
    candidate_count = 6 if smoke else 90
    k_grid = [2, 4, 6] if smoke else K_GRID
    return {
        "experiment": EXPERIMENT, "smoke_test": smoke, "results_directory": str(output),
        "M_x": 20, "M_y": 15, "T": 120 if smoke else 1000, "na": 2, "nb": 3,
        "C_family": "gaussian_isotropic", "C_known": True, "R_fixed_true": True,
        "B_update_mode": "free", "B_ridge_lambda": 0.0,
        "Q_update_mode": "diag_shrink_scalar", "Q_shrinkage_rho": 0.25,
        "Q_update_damping": 0.50, "include_A_posterior_uncertainty_in_Q": True,
        "a0": 1e-3, "b0": 1e-3, "diagonal_prior_precision": 1e-4,
        "posterior_jitter": 1e-8, "VB_MAX_ITER": 6 if smoke else 75,
        "N_FFBS_SAMPLES": 4 if smoke else 50,
        "finite_difference_epsilon": 1e-4,
        "PERTURBED_VB_MAX_ITER": 3 if smoke else 30,
        "PERTURBED_RESCUE_MAX_ITER": 4 if smoke else 50,
        "N_FREQUENCIES": 32 if smoke else 128,
        "A_center": "A_VB", "Louis_eta": 0.70, "Louis_tau": 0.90,
        "development_network_ids": [35000] if smoke else list(range(35000, 35006)),
        "confirmation_network_ids": [] if smoke else list(range(35100, 35106)),
        "candidate_pool_size": candidate_count,
        "max_stageb_groups_per_network": MAX_STAGEB_GROUPS_PER_NETWORK,
        "ALLOW_NEW_STAGEB": ALLOW_NEW_STAGEB, "ALLOW_FULL_380_STAGEB": ALLOW_FULL_380_STAGEB,
        "K_GRID": k_grid, "ridge_penalty": 1e-3,
        "random_repeats": env_int("EXPERIMENT_34UF_RANDOM_REPEATS", 5 if smoke else 500),
        "network_bootstrap_repeats": env_int("EXPERIMENT_34UF_BOOTSTRAP_REPEATS", 100 if smoke else 5000),
        "network_workers": max(1, env_int("EXPERIMENT_34UF_NETWORK_WORKERS", 1)),
        "effective_network_workers": 1,
        "network_parallelism_note": "network loop is serialized to preserve per-edge atomic CSV checkpoints; Stage-B directions are parallel",
        "stageB_workers": max(1, env_int("EXPERIMENT_34UF_STAGEB_WORKERS", 2)),
        "inner_threads": max(1, env_int("EXPERIMENT_34UF_INNER_THREADS", 1)),
        "master_seed": 3406001,
        "feature_set": RAW_FEATURES,
        "feature_transforms": {name: ("identity" if name == "logdet_Louis_cov" else "log1p(max(x,0))") for name in RAW_FEATURES},
        "stageB_need_target": "log1p(max((trace(Sigma_SB)-trace(Sigma_L))/max(trace(Sigma_L),1e-12),0))",
        "development_thresholds": {
            "zero_primary_reversals": True, "stability_matches_all": True,
            "mean_primary_categorical_agreement": 0.95, "minimum_network_categorical_agreement": 0.80,
            "mean_retained_active_gain": 0.90, "minimum_retained_active_gain": 0.70,
            "mean_absolute_zero_coverage_delta": 0.02, "mean_correction_capture": 0.80,
            "mean_Q1_retained_active_gain": 0.70,
        },
        "confirmation_thresholds": {
            "zero_primary_reversals": True, "stability_matches_all": True,
            "mean_primary_categorical_agreement": 0.95, "networks_agreement_ge_0p80": 5,
            "mean_retained_active_gain": 0.90, "maximum_networks_gain_below_0p70": 1,
            "mean_absolute_zero_coverage_delta": 0.02, "mean_correction_capture": 0.80,
            "mean_Q1_retained_active_gain": 0.70,
        },
        "historical_34UC": str(HISTORICAL_34UC), "historical_34UA": str(HISTORICAL_34UA),
        "matrixfree_LRVB_used": False, "full_380_stageB_run": False,
        "independent_unit": "network",
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def initialize(config):
    if not ALLOW_NEW_STAGEB or not config["ALLOW_NEW_STAGEB"]:
        raise RuntimeError("34UF requires the explicitly enabled legacy Stage-B reference path.")
    if config["candidate_pool_size"] > MAX_STAGEB_GROUPS_PER_NETWORK or ALLOW_FULL_380_STAGEB:
        raise RuntimeError("34UF hard guard forbids more than 90 Stage-B groups per network.")
    if not config["smoke_test"] and os.environ.get("EXPERIMENT_34UF_RUN_FULL", "0") != "1":
        raise RuntimeError("The 12-network Stage-B run is opt-in. Set EXPERIMENT_34UF_RUN_FULL=1.")
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_config.json"
    protected = [
        "smoke_test", "M_x", "M_y", "T", "development_network_ids", "confirmation_network_ids",
        "candidate_pool_size", "K_GRID", "ridge_penalty", "finite_difference_epsilon",
        "PERTURBED_VB_MAX_ITER", "PERTURBED_RESCUE_MAX_ITER", "random_repeats",
    ]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Existing 34UF checkpoint configuration differs: {changed}")
    else:
        # The smoke test intentionally lives below the full-run output root.
        # Its presence must not make a first full run look like an ambiguous
        # partial run, but every other pre-existing item remains protected.
        unexpected = [item for item in output.iterdir() if item.name != "smoke_test"]
        if unexpected:
            names = ", ".join(sorted(item.name for item in unexpected))
            raise FileExistsError(
                f"Non-empty 34UF output has no compatible experiment_config.json: "
                f"{output}. Unexpected items: {names}"
            )
    atomic_json(config, path); (output / "plots").mkdir(exist_ok=True)
    return output


class Progress:
    def __init__(self, total_networks):
        self.total_networks = total_networks; self.started = time.perf_counter()
    def edge(self, network_number, run_id, edge_done, edge_total, directions_done):
        fraction = edge_done / max(edge_total, 1); filled = round(24 * fraction)
        elapsed = (time.perf_counter() - self.started) / 60
        remaining = max(edge_total - edge_done, 0)
        print(
            f"\r34UF net {network_number}/{self.total_networks} "
            f"[{'#' * filled}{'-' * (24-filled)}] edge {edge_done}/{edge_total} "
            f"dirs {directions_done}/{2*edge_total} elapsed {elapsed:6.1f}m remaining_edges {remaining} {run_id[:24]}",
            end="\n" if edge_done >= edge_total else "", flush=True,
        )
    def phase(self, label):
        elapsed = (time.perf_counter() - self.started) / 60
        print(f"\r34UF {label} elapsed {elapsed:6.1f}m", end="\n", flush=True)


def source_config(config):
    source = json.loads((HISTORICAL_34UA / "experiment_config.json").read_text(encoding="utf8"))
    source.update({
        "M_x": config["M_x"], "T": config["T"], "VB_MAX_ITER": config["VB_MAX_ITER"],
        "N_FFBS_SAMPLES": config["N_FFBS_SAMPLES"], "N_FREQUENCIES": config["N_FREQUENCIES"],
        "PERTURBED_VB_MAX_ITER": config["PERTURBED_VB_MAX_ITER"],
        "PERTURBED_RESCUE_MAX_ITER": config["PERTURBED_RESCUE_MAX_ITER"],
        "workers": config["stageB_workers"], "ridge_penalty": config["ridge_penalty"],
    })
    return source


def load_current_allocator(source):
    edges = read_csv(HISTORICAL_34UA / "calibration_edge_rows_partial.csv")
    runtime = read_csv(HISTORICAL_34UA / "runtime_summary_partial.csv")
    success = set(runtime.loc[runtime.run_status.eq("success"), "run_id"]) if len(runtime) else set(edges.run_id)
    calibration = edges[(edges.M_y.eq(40)) & edges.candidate_support.eq("full_candidate") & edges.run_id.isin(success)]
    models, _ = ua.fit_calibration_models(source, calibration)
    if (40, "full_candidate") not in models:
        raise RuntimeError("Canonical 34UC current-ordering allocator could not be reconstructed.")
    return models[(40, "full_candidate")]


def metadata(config, network_id, phase):
    replicate = 0
    return {
        "experiment_name": config["experiment"], "split": phase, "phase": phase,
        "M_x": config["M_x"], "M_y": config["M_y"], "observation_ratio": config["M_y"] / config["M_x"],
        "T": config["T"], "C_family": config["C_family"], "C_MODE": config["C_family"],
        "method": "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_C_known_R",
        "candidate_support": "deployable_current_top90", "true_network_id": int(network_id),
        "replicate_id": replicate, "seed_group": f"34UF_{phase.upper()}",
        "run_id": f"34UF_{phase}_Mx20_My15_top{config['candidate_pool_size']}_net{network_id}_rep0",
    }


def frozen40_state():
    edges = read_csv(HISTORICAL_34UC / "new_reference_edge_rows_partial.csv")
    runtime = read_csv(HISTORICAL_34UC / "new_reference_runtime_partial.csv")
    success = set(runtime.loc[runtime.run_status.eq("success"), "run_id"])
    edges = edges[edges.run_id.isin(success)].copy()
    panel = ue.transform_features(ue.normalize_wide(edges, "34UC", "primary", "deployable_reference_90"))
    if panel.run_id.nunique() != 5:
        raise RuntimeError(f"P0-FROZEN40 requires five complete 34UC networks; found {panel.run_id.nunique()}.")
    state = ue.fit_selector_state(panel, {"purpose_ridge": 1e-3, "logdet_delta": 1e-3})
    return state, panel


def alias_reference_features(frame):
    result = frame.copy()
    aliases = {
        "target_i": "target", "source_j": "source", "true_active": "true_edge",
        "A_group_norm": "estimated_group_norm", "VB_cov_trace": "vb_cov_trace",
        "Louis_cov_trace": "louis_cov_trace", "logdet_Louis_cov": "log_louis_cov_determinant",
        "Louis_group_SNR": "louis_group_snr", "observability_product": "edge_o_inst_product",
        "transfer_spectral_score": "spectral_transfer_full_band_score",
        "stageB_relative_inflation": "relative_stageb_inflation",
    }
    for new, old in aliases.items():
        if old in result: result[new] = result[old]
    result["edge_id"] = result.target.astype(int).astype(str) + "<-" + result.source.astype(int).astype(str)
    for raw, transformed in zip(RAW_FEATURES, FEATURES):
        values = pd.to_numeric(result[raw], errors="coerce")
        result[transformed] = values if raw == "logdet_Louis_cov" else np.log1p(np.maximum(values, 0.0))
    if "stageB_relative_inflation" in result:
        result["stageB_need_target"] = np.log1p(np.maximum(result.stageB_relative_inflation, 0.0))
    return result


def edge_checkpoint_path(output, meta, target, source):
    directory = Path(output) / "edge_group_checkpoints" / meta["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"edge_{int(target)}_{int(source)}.npz"


def network_marker(output, run_id):
    directory = Path(output) / "network_complete"; directory.mkdir(exist_ok=True)
    return directory / f"{run_id}.json"


def build_one_edge_reference(source, selected_row, model, data, louis, columns, edge_scores, meta):
    target, source_node = int(selected_row.target), int(selected_row.source)
    indices = [ua.global_index(target, lag, source_node, model.n_states, 2) for lag in range(2)]
    matrix = np.column_stack([columns[index] for index in indices])
    raw = matrix[np.ix_(indices, range(2))]
    stage, diagnostics = ua.safe_covariance(raw)
    projection = {"min_eigenvalue_raw": float(np.linalg.eigvalsh(0.5 * (raw + raw.T)).min()),
                  "psd_projection_used": diagnostics["stabilization_applied"]}
    return ua.build_reference_rows(
        source, pd.DataFrame([selected_row._asdict()]), model, data, louis,
        indices, {index: offset for offset, index in enumerate(indices)}, stage,
        projection, edge_scores, meta,
    )


def compute_network_reference(config, source, allocator, output, meta, network_number, progress):
    if network_marker(output, meta["run_id"]).exists():
        return
    started = time.perf_counter(); data, seed = ua.simulate(source, meta)
    if data["C"].shape != (config["M_y"], config["M_x"]):
        raise RuntimeError(f"Rectangular C has shape {data['C'].shape}, expected {(config['M_y'], config['M_x'])}.")
    full_mask = np.ones((config["M_x"], config["M_x"]), bool)
    vb_start = time.perf_counter(); model = ua.fit_model(source, data, full_mask, seed + 500)
    vb_seconds = time.perf_counter() - vb_start
    louis_start = time.perf_counter(); louis = ua.compute_louis(source, model, data, seed + 700)
    louis_seconds = time.perf_counter() - louis_start
    _, _, edge_scores = ua.point_and_spectral_metrics(source, model, data, full_mask, louis, meta)
    candidates = uc.all_deployable_candidates(model, data, louis, meta, edge_scores, allocator)
    if len(candidates) != config["M_x"] * (config["M_x"] - 1):
        raise RuntimeError(f"Candidate universe has {len(candidates)} groups, expected 380.")
    candidates["source_row_order"] = np.arange(1, len(candidates) + 1)
    ua.append_checkpoint(output / "cheap_full380_partial.csv", candidates, ["run_id", "target", "source"])
    selected = candidates.head(config["candidate_pool_size"]).copy()
    if selected.target.eq(selected.source).any() or len(selected) > MAX_STAGEB_GROUPS_PER_NETWORK:
        raise RuntimeError("Stage-B pool violated the off-diagonal 90-group hard guard.")
    columns = {}
    for row in selected.itertuples(index=False):
        target, source_node = int(row.target), int(row.source)
        indices = [ua.global_index(target, lag, source_node, model.n_states, 2) for lag in range(2)]
        for index in indices:
            path = uc.direction_checkpoint_path(output, meta, index)
            if path.exists(): columns[index] = np.load(path)
        pending = [index for index in indices if index not in columns]
        edge_started = time.perf_counter(); diagnostics = []
        if pending:
            with ProcessPoolExecutor(max_workers=min(config["stageB_workers"], len(pending))) as pool:
                futures = {
                    pool.submit(ua.perturb_one, (model, data["y"], data["u"], index,
                                                config["PERTURBED_VB_MAX_ITER"], config["PERTURBED_RESCUE_MAX_ITER"])): index
                    for index in pending
                }
                for future in as_completed(futures):
                    index, column, diagnostic = future.result()
                    if not np.all(np.isfinite(column)):
                        raise FloatingPointError(f"Invalid Stage-B sensitivity direction {index}.")
                    columns[index] = column; ua.atomic_array(uc.direction_checkpoint_path(output, meta, index), column)
                    ua.atomic_json(diagnostic, uc.direction_diagnostic_path(output, meta, index))
                    diagnostics.append({**meta, "target": target, "source": source_node, **diagnostic})
        if not all(index in columns for index in indices):
            raise RuntimeError(f"Incomplete Stage-B edge checkpoint {target}<-{source_node}.")
        local = build_one_edge_reference(source, row, model, data, louis, columns, edge_scores, meta)
        ua.append_checkpoint(output / "reference_edge_rows_partial.csv", local, ["run_id", "target", "source"])
        if diagnostics:
            ua.append_checkpoint(output / "perturbation_diagnostics_partial.csv", pd.DataFrame(diagnostics), ["run_id", "direction_index"])
        edge_marker = edge_checkpoint_path(output, meta, target, source_node)
        ua.atomic_array(edge_marker, np.column_stack([columns[index] for index in indices]))
        completed_edges = sum(edge_checkpoint_path(output, meta, int(r.target), int(r.source)).exists() for r in selected.itertuples())
        runtime_row = pd.DataFrame([{**meta, "component": "edge_group", "target": target, "source": source_node,
                                     "edge_runtime_seconds": time.perf_counter() - edge_started,
                                     "completed_edge_groups": completed_edges, "total_edge_groups": len(selected),
                                     "run_status": "edge_complete"}])
        ua.append_checkpoint(output / "reference_runtime_partial.csv", runtime_row, ["run_id", "component", "target", "source"])
        progress.edge(network_number, meta["run_id"], completed_edges, len(selected), 2 * completed_edges)
    all_indices = [ua.global_index(int(row.target), lag, int(row.source), model.n_states, 2)
                   for row in selected.itertuples() for lag in range(2)]
    if len(all_indices) > 2 * MAX_STAGEB_GROUPS_PER_NETWORK:
        raise RuntimeError("Full-380 Stage-B guard triggered.")
    matrix = np.column_stack([columns[index] for index in all_indices])
    _, stage, projection = ua.project_selected_covariance(matrix, all_indices)
    position = {index: offset for offset, index in enumerate(all_indices)}
    reference = ua.build_reference_rows(source, selected, model, data, louis, all_indices, position, stage, projection, edge_scores, meta)
    reference["evidence_stage"] = "prospective_My15"; reference["candidate_pool_truth_stratified"] = False
    ua.append_checkpoint(output / "reference_edge_rows_partial.csv", reference, ["run_id", "target", "source"])
    diagnostics = []
    pair_for_index = {
        ua.global_index(int(row.target), lag, int(row.source), model.n_states, 2): (int(row.target), int(row.source))
        for row in selected.itertuples() for lag in range(2)
    }
    for index in all_indices:
        path = uc.direction_diagnostic_path(output, meta, index)
        if path.exists():
            target, source_node = pair_for_index[index]
            diagnostics.append({**meta, "target": target, "source": source_node,
                                **json.loads(path.read_text(encoding="utf8"))})
    if diagnostics:
        ua.append_checkpoint(output / "perturbation_diagnostics_partial.csv", pd.DataFrame(diagnostics), ["run_id", "direction_index"])
    runtime = pd.DataFrame([{**meta, "component": "network", "target": np.nan, "source": np.nan,
                             "baseline_VB_runtime_seconds": vb_seconds, "louis_runtime_seconds": louis_seconds,
                             "stageB_runtime_seconds": sum(float(item.get("runtime_seconds", 0.0)) for item in diagnostics),
                             "total_runtime_seconds": time.perf_counter() - started,
                             "completed_edge_groups": len(selected), "total_edge_groups": len(selected),
                             "n_stageb_directions": len(all_indices), "run_status": "success"}])
    ua.append_checkpoint(output / "reference_runtime_partial.csv", runtime, ["run_id", "component", "target", "source"])
    atomic_json({"completed": True, "run_id": meta["run_id"], "edge_groups": len(selected),
                 "directions": len(all_indices), "completed_at": time.time()}, network_marker(output, meta["run_id"]))


def run_network_phase(config, source, allocator, output, network_ids, phase, progress, offset=0):
    manifest_rows = []
    for index, network_id in enumerate(network_ids, 1):
        meta = metadata(config, network_id, phase); status = "pending"
        try:
            compute_network_reference(config, source, allocator, output, meta, offset + index, progress)
            status = "success" if network_marker(output, meta["run_id"]).exists() else "incomplete"
        except Exception as error:
            status = "failed"
            failure = pd.DataFrame([{**meta, "component": "network", "target": np.nan, "source": np.nan,
                                     "run_status": "failed", "error_type": type(error).__name__,
                                     "error_message": str(error), "traceback": traceback.format_exc()}])
            ua.append_checkpoint(output / "reference_runtime_partial.csv", failure, ["run_id", "component", "target", "source"])
        manifest_rows.append({**meta, "completion_status": status,
                              "network_complete_marker": str(network_marker(output, meta["run_id"])),
                              "candidate_groups": config["candidate_pool_size"],
                              "StageB_reference_is_full_380": False})
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(manifest, str(output / f"{phase}_network_manifest.csv"))
    return manifest


def hybrid_frame(reference, selected_indices):
    frame = reference.copy().reset_index(drop=True); selected = np.zeros(len(frame), bool)
    selected[np.asarray(selected_indices, int)] = True
    for column in list(frame.columns):
        if column.startswith("Sigma_stageb_"):
            louis = column.replace("Sigma_stageb_", "Sigma_louis_")
            frame.loc[~selected, column] = frame.loc[~selected, louis]
        elif column.startswith("stageb_"):
            louis = column.replace("stageb_", "louis_")
            if louis in frame: frame.loc[~selected, column] = frame.loc[~selected, louis]
    frame.loc[~selected, "relative_stageb_inflation"] = 0.0
    frame.loc[~selected, "delta_trace"] = 0.0
    frame["selected_stageB"] = selected
    return frame


def primary_agreement(reference_vector, hybrid_vector):
    matches = [float(reference_vector[name] == hybrid_vector[name])
               for name in PRIMARY_CATEGORIES if pd.notna(reference_vector[name]) and pd.notna(hybrid_vector[name])]
    reversal = {reference_vector.get("active_coverage_conclusion"), hybrid_vector.get("active_coverage_conclusion")} == {"BENEFICIAL", "HARMFUL"}
    return float(np.mean(matches)) if matches else np.nan, bool(reversal), bool(
        reference_vector.get("numerical_stability_conclusion") == hybrid_vector.get("numerical_stability_conclusion")
    )


def replay_row(group, order, k, variant, phase, diagnostics):
    group = group.reset_index(drop=True); selected = list(map(int, order[:min(k, len(group))]))
    full_vector = conclusion_vector(group, diagnostics)
    hybrid = hybrid_frame(group, selected)
    selected_pairs = set(zip(hybrid.iloc[selected].target.astype(int), hybrid.iloc[selected].source.astype(int)))
    diag = diagnostics[[
        (int(t), int(s)) in selected_pairs for t, s in zip(diagnostics.target, diagnostics.source)
    ]] if len(diagnostics) and {"target", "source"}.issubset(diagnostics) else diagnostics.iloc[:0]
    vector = conclusion_vector(hybrid, diag)
    agreement, reversal, stability = primary_agreement(full_vector, vector)
    metrics = ue.replay_metrics(alias_reference_features(group), selected + [i for i in range(len(group)) if i not in set(selected)], k / len(group))
    active = group.true_edge.astype(bool).to_numpy(); selected_mask = np.zeros(len(group), bool); selected_mask[selected] = True
    actual = np.log1p(np.maximum(group.relative_stageb_inflation.to_numpy(float), 0.0))
    predicted = group.predicted_stageB_need.to_numpy(float)
    positive_mass = np.maximum(group.delta_trace.to_numpy(float), 0.0)
    output = {
        "phase": phase, "run_id": group.run_id.iloc[0], "selector_variant": variant,
        "k": int(k), "N": len(group), "rho": k / len(group),
        "primary_categorical_agreement": agreement, "primary_usefulness_reversal": reversal,
        "stability_agreement": stability,
        "retained_active_gain": metrics["retained_active_gain"],
        "zero_coverage_delta_vs_full": metrics["zero_coverage_delta_vs_full"],
        "absolute_zero_coverage_delta_vs_full": abs(metrics["zero_coverage_delta_vs_full"]),
        "relative_Frobenius_error": metrics["relative_Frobenius_error"],
        "trace_relative_error": metrics["trace_relative_error"],
        "mean_abs_logdet_error": metrics["mean_abs_logdet_error"],
        "positive_correction_capture": metrics["correction_capture"],
        "signed_correction_capture": metrics["signed_correction_capture"],
        "covariance_inflation_correlation": metrics["inflation_correlation"],
        "Q1_retained_active_gain": metrics["Q1_retained_active_gain"],
        "active_edge_selected_fraction": float(selected_mask[active].mean()) if active.any() else np.nan,
        "zero_edge_selected_fraction": float(selected_mask[~active].mean()) if (~active).any() else np.nan,
        "Louis_rescue_rate_full": full_vector.get("rescue_rate", np.nan),
        "Louis_rescue_rate_hybrid": vector.get("rescue_rate", np.nan),
        "rescue_rate_error": abs(float(vector.get("rescue_rate", np.nan)) - float(full_vector.get("rescue_rate", np.nan))),
        "predicted_actual_need_spearman": spearmanr(predicted, actual).statistic if np.ptp(predicted) > EPS and np.ptp(actual) > EPS else np.nan,
        "predicted_actual_need_kendall": kendalltau(predicted, actual).statistic if np.ptp(predicted) > EPS and np.ptp(actual) > EPS else np.nan,
        "topk_stageB_need_capture": float(actual[selected].sum() / max(actual.sum(), EPS)),
        "topk_positive_trace_capture": float(positive_mass[selected].sum() / max(positive_mass.sum(), EPS)),
    }
    quartile = pd.qcut(group.edge_o_inst_product.rank(method="first"), 4, labels=False).to_numpy(int)
    louis_cover = group.louis_group_covered_95.astype(bool).to_numpy()
    stage_cover = group.stageb_group_covered_95.astype(bool).to_numpy()
    hybrid_cover = np.where(selected_mask, stage_cover, louis_cover)
    for q_index in range(4):
        mask = active & (quartile == q_index)
        if not mask.any():
            output[f"Q{q_index + 1}_retained_active_gain"] = np.nan
            continue
        denominator = stage_cover[mask].mean() - louis_cover[mask].mean()
        output[f"Q{q_index + 1}_retained_active_gain"] = (
            (hybrid_cover[mask].mean() - louis_cover[mask].mean()) / denominator
            if denominator > EPS else np.nan
        )
    return output


def aggregate_budget(rows, config):
    frame = pd.DataFrame(rows); summaries = []
    rng = np.random.default_rng(config["master_seed"])
    metrics = [
        "primary_categorical_agreement", "retained_active_gain", "zero_coverage_delta_vs_full",
        "absolute_zero_coverage_delta_vs_full",
        "positive_correction_capture", "Q1_retained_active_gain", "relative_Frobenius_error",
        "trace_relative_error", "mean_abs_logdet_error", "rescue_rate_error",
        "predicted_actual_need_spearman", "predicted_actual_need_kendall", "topk_stageB_need_capture",
    ]
    for keys, group in frame.groupby(["phase", "selector_variant", "k", "N", "rho"], dropna=False):
        row = dict(zip(["phase", "selector_variant", "k", "N", "rho"], keys))
        row["n_networks"] = group.run_id.nunique()
        row["number_primary_reversals"] = int(group.primary_usefulness_reversal.sum())
        row["stability_match_fraction"] = float(group.stability_agreement.mean())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row.update({f"{metric}_n": len(values),
                        f"{metric}_mean": np.mean(values) if len(values) else np.nan,
                        f"{metric}_median": np.median(values) if len(values) else np.nan,
                        f"{metric}_min": np.min(values) if len(values) else np.nan,
                        f"{metric}_max": np.max(values) if len(values) else np.nan})
            if len(values):
                sample = rng.choice(values, (config["network_bootstrap_repeats"], len(values)), replace=True).mean(axis=1)
                row[f"{metric}_network_bootstrap_q025"] = np.quantile(sample, .025)
                row[f"{metric}_network_bootstrap_q975"] = np.quantile(sample, .975)
        summaries.append(row)
    return frame, pd.DataFrame(summaries)


def passes_development(row):
    q1 = row.get("Q1_retained_active_gain_mean", np.nan)
    return bool(
        row.number_primary_reversals == 0 and row.stability_match_fraction >= 1 - EPS
        and row.primary_categorical_agreement_n == row.n_networks
        and row.retained_active_gain_n == row.n_networks
        and row.primary_categorical_agreement_mean >= .95 - EPS
        and row.primary_categorical_agreement_min >= .80 - EPS
        and row.retained_active_gain_mean >= .90 - EPS and row.retained_active_gain_min >= .70 - EPS
        and row.absolute_zero_coverage_delta_vs_full_mean <= .02 + EPS
        and row.positive_correction_capture_mean >= .80 - EPS
        and (not np.isfinite(q1) or q1 >= .70 - EPS)
    )


def fit_cal15_cv(development, config):
    predictions = []; coefficients = []
    run_ids = sorted(development.run_id.unique())
    for fold, heldout in enumerate(run_ids):
        train = development[~development.run_id.eq(heldout)]; test = development[development.run_id.eq(heldout)].copy()
        if not len(train):
            train = development.copy()
        state = ue.fit_selector_state(train, {"purpose_ridge": config["ridge_penalty"], "logdet_delta": 1e-3})
        _, raw, score = ue.predict_state(test, state)
        for index, row in enumerate(test.itertuples()):
            predictions.append({"fold": fold, "heldout_run_id": heldout, "run_id": heldout,
                                "target": row.target, "source": row.source,
                                "predicted_stageB_need": raw[index], "purpose_score": score[index],
                                "actual_stageB_need": row.stageB_need_target})
        for term, value in zip(["intercept", *FEATURES], state["coefficients"]):
            coefficients.append({"fold": fold, "heldout_run_id": heldout, "term": term, "coefficient": value,
                                 "ridge_penalty": config["ridge_penalty"], "training_run_ids": "|".join(sorted(train.run_id.unique()))})
    return pd.DataFrame(predictions), pd.DataFrame(coefficients)


def score_by_predictions(reference, predictions):
    keys = ["run_id", "target", "source"]
    return reference.drop(columns=["predicted_stageB_need", "purpose_score"], errors="ignore").merge(
        predictions[keys + ["predicted_stageB_need", "purpose_score"]], on=keys, how="inner", validate="one_to_one"
    )


def deterministic_replay(reference, diagnostics, config, phase, frozen_state, cal_predictions=None):
    rows = []; order_rows = []
    for run_id, base in reference.groupby("run_id", sort=False):
        base = base.reset_index(drop=True)
        variants = {}
        Z, raw, q = ue.predict_state(base, frozen_state)
        variants["P0_FROZEN40"] = (raw, q)
        if cal_predictions is not None:
            lookup = cal_predictions[cal_predictions.run_id.eq(run_id)].set_index(["target", "source"])
            raw_cal = np.asarray([lookup.loc[(int(r.target), int(r.source)), "predicted_stageB_need"] for r in base.itertuples()])
            variants["P0_CAL15"] = (raw_cal, np.maximum(raw_cal, 0.0))
        current = np.argsort(base.source_row_order.to_numpy(float), kind="stable")
        variants["D0_CURRENT"] = (np.nan * np.ones(len(base)), -base.source_row_order.to_numpy(float))
        for variant, (predicted, score) in variants.items():
            order = list(map(int, current if variant == "D0_CURRENT" else np.argsort(-score, kind="stable")))
            scored = base.copy(); scored["predicted_stageB_need"] = predicted
            for rank, index in enumerate(order, 1):
                edge = scored.iloc[index]
                order_rows.append({"phase": phase, "run_id": run_id, "selector_variant": variant, "rank": rank,
                                   "target": int(edge.target), "source": int(edge.source),
                                   "predicted_stageB_need": predicted[index], "true_edge_evaluation_only": bool(edge.true_edge)})
            diag = diagnostics[diagnostics.run_id.eq(run_id)]
            for k in config["K_GRID"]:
                if k <= len(scored): rows.append(replay_row(scored, order, k, variant, phase, diag))
    return rows, pd.DataFrame(order_rows)


def choose_design(summary, development, config):
    passing = {}
    for variant in ["P0_FROZEN40", "P0_CAL15"]:
        candidates = summary[(summary.selector_variant.eq(variant)) & summary.apply(passes_development, axis=1)]
        passing[variant] = int(candidates.k.min()) if len(candidates) else np.nan
    frozen_k, cal_k = passing["P0_FROZEN40"], passing["P0_CAL15"]
    if np.isfinite(frozen_k) and (not np.isfinite(cal_k) or frozen_k <= cal_k + 4):
        selected, k = "P0_FROZEN40", int(frozen_k)
    elif np.isfinite(cal_k):
        selected, k = "P0_CAL15", int(cal_k)
    elif np.isfinite(frozen_k):
        selected, k = "P0_FROZEN40", int(frozen_k)
    else:
        selected, k = "P0_FROZEN40", int(max(config["K_GRID"]))
    final_cal = ue.fit_selector_state(development, {"purpose_ridge": config["ridge_penalty"], "logdet_delta": 1e-3})
    return selected, k, passing, final_cal


def state_json(state):
    return {"feature_mean": state["mean"].tolist(), "feature_scale": state["scale"].tolist(),
            "ridge_coefficients": state["coefficients"].tolist(), "ridge_penalty": 1e-3}


def random_controls(reference, diagnostics, config, phase):
    rows = []
    for run_number, (run_id, group) in enumerate(reference.groupby("run_id", sort=False)):
        group = group.reset_index(drop=True); diag = diagnostics[diagnostics.run_id.eq(run_id)]
        purpose = group.predicted_stageB_need.to_numpy(float)
        strata = pd.qcut(pd.Series(purpose).rank(method="first"), 4, labels=False).to_numpy(int)
        for repeat in range(config["random_repeats"]):
            rng = np.random.default_rng(config["master_seed"] + 10000 * run_number + repeat)
            orders = {"U0_UNIFORM_RANDOM": list(map(int, rng.permutation(len(group)))),
                      "D1_STRATIFIED_RANDOM": ue.proportional_random_order(strata, rng)}
            for variant, order in orders.items():
                for k in config["K_GRID"]:
                    if k <= len(group):
                        row = replay_row(group, order, k, variant, phase, diag); row["random_repeat"] = repeat; rows.append(row)
    raw = pd.DataFrame(rows); summaries = []
    if not len(raw): return raw, pd.DataFrame()
    for keys, group in raw.groupby(["phase", "run_id", "selector_variant", "k", "N", "rho"]):
        row = dict(zip(["phase", "run_id", "selector_variant", "k", "N", "rho"], keys))
        row["random_repeats"] = config["random_repeats"]
        for metric in ["primary_categorical_agreement", "retained_active_gain", "positive_correction_capture", "zero_coverage_delta_vs_full"]:
            values = group[metric]
            row.update({f"{metric}_mean": values.mean(), f"{metric}_median": values.median(),
                        f"{metric}_q05": values.quantile(.05), f"{metric}_q95": values.quantile(.95)})
        summaries.append(row)
    return raw, pd.DataFrame(summaries)


def confirmation_pass(rows):
    frame = pd.DataFrame(rows)
    q1 = frame.Q1_retained_active_gain.dropna()
    return bool(
        frame[["primary_categorical_agreement", "retained_active_gain", "zero_coverage_delta_vs_full",
               "positive_correction_capture"]].notna().all().all()
        and
        not frame.primary_usefulness_reversal.any() and frame.stability_agreement.all()
        and frame.primary_categorical_agreement.mean() >= .95 - EPS
        and (frame.primary_categorical_agreement >= .80 - EPS).sum() >= 5
        and frame.retained_active_gain.mean() >= .90 - EPS
        and (frame.retained_active_gain < .70 - EPS).sum() <= 1
        and frame.zero_coverage_delta_vs_full.abs().mean() <= .02 + EPS
        and frame.positive_correction_capture.mean() >= .80 - EPS
        and (not len(q1) or q1.mean() >= .70 - EPS)
    )


def feature_shift(my40, my15):
    rows = []
    for raw, feature in zip(RAW_FEATURES, FEATURES):
        a = my40[feature].to_numpy(float); b = my15[feature].to_numpy(float); scale = max(np.std(a), EPS)
        rows.append({"feature": raw, "transform": "identity" if raw == "logdet_Louis_cov" else "log1p(max(x,0))",
                     "My40_mean": np.mean(a), "My40_std": np.std(a), "My15_mean": np.mean(b), "My15_std": np.std(b),
                     "standardized_mean_shift_My15_minus_My40": (np.mean(b) - np.mean(a)) / scale})
    return pd.DataFrame(rows)


def summaries_and_plots(output, development_rows, confirmation_rows, random_summary, shift, decision):
    dev = pd.DataFrame(development_rows); conf = pd.DataFrame(confirmation_rows)
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    items = [
        ("primary_categorical_agreement", "01_categorical_agreement_vs_k.png", "diagnostic categorical agreement"),
        ("retained_active_gain", "02_retained_active_gain_vs_k.png", "retained active gain"),
        ("positive_correction_capture", "03_correction_capture_vs_k.png", "correction capture"),
        ("zero_coverage_delta_vs_full", "04_zero_coverage_delta_vs_k.png", "zero coverage delta"),
        ("relative_Frobenius_error", "05_covariance_error_vs_k.png", "covariance Frobenius error"),
        ("Q1_retained_active_gain", "06_Q1_gain_vs_k.png", "Q1 retained gain"),
    ]
    for metric, filename, ylabel in items:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for variant, group in dev.groupby("selector_variant"):
            curve = group.groupby("k")[metric].mean(); ax.plot(curve.index, curve, marker="o", label=variant)
        ax.set(xlabel="Stage-B edge groups k", ylabel=ylabel); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ["P0_FROZEN40", "P0_CAL15"]:
        group = dev[dev.selector_variant.eq(variant)].groupby("k").primary_categorical_agreement.mean()
        ax.plot(group.index, group, marker="o", label=variant)
    ax.legend(); ax.set(xlabel="k", ylabel="categorical agreement"); fig.tight_layout(); fig.savefig(plots / "07_frozen40_vs_cal15.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected_k = int(decision.selected_k.iloc[0])
    deterministic = dev[dev.k.eq(selected_k)].groupby("selector_variant").retained_active_gain.mean()
    if len(random_summary):
        random_at_k = random_summary[random_summary.k.eq(selected_k)].groupby("selector_variant").retained_active_gain_median.mean()
        deterministic = pd.concat([deterministic, random_at_k])
    deterministic.plot.bar(ax=ax); ax.set_ylabel("retained active gain"); fig.tight_layout(); fig.savefig(plots / "08_p0_vs_controls.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    scores = read_csv(output / "p0_frozen40_scores.csv")
    if len(scores): ax.scatter(scores.predicted_stageB_need, scores.stageB_need_target, alpha=.5)
    ax.set(xlabel="predicted Stage-B need", ylabel="actual Stage-B need"); fig.tight_layout(); fig.savefig(plots / "09_predicted_vs_actual_need.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(shift.feature, shift.standardized_mean_shift_My15_minus_My40); ax.tick_params(axis="x", rotation=45); ax.set_ylabel("standardized My15-My40 mean shift"); fig.tight_layout(); fig.savefig(plots / "10_feature_shift.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = dev.pivot_table(index="run_id", columns="k", values="primary_categorical_agreement", aggfunc="mean")
    if len(pivot):
        image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1); fig.colorbar(image, ax=ax); ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), range(1, len(pivot.index)+1))
    ax.set(xlabel="k", ylabel="development network"); fig.tight_layout(); fig.savefig(plots / "11_network_diagnostic_heatmap.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, group in dev.groupby("selector_variant"):
        curve = group.groupby("k").primary_categorical_agreement.mean(); ax.plot(curve.index / 90, curve, marker="o", label=variant)
    ax.set(xlabel="Stage-B cost fraction k/90", ylabel="diagnostic agreement"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "12_cost_fidelity_pareto.png", dpi=150); plt.close(fig)


def ensure_required_outputs(output):
    schemas = {
        "reference_edge_rows.csv": [], "reference_runtime.csv": [], "perturbation_diagnostics.csv": [],
        "p0_frozen40_scores.csv": [], "p0_cal15_cv_predictions.csv": [], "p0_cal15_coefficients.csv": [],
        "selector_orderings.csv": [], "budget_mapping.csv": [], "development_budget_curves.csv": [],
        "development_network_summary.csv": [], "development_decision_summary.csv": [],
        "confirmation_budget_results.csv": [], "confirmation_network_summary.csv": [],
        "confirmation_decision_summary.csv": [], "random_baseline_summary.csv": [],
        "observability_quartile_summary.csv": [], "correction_capture_summary.csv": [],
        "covariance_fidelity_summary.csv": [], "feature_shift_my40_vs_my15.csv": [],
        "runtime_summary.csv": [], "decision_summary.csv": [],
    }
    partial_map = {
        "reference_edge_rows.csv": "reference_edge_rows_partial.csv",
        "reference_runtime.csv": "reference_runtime_partial.csv",
        "perturbation_diagnostics.csv": "perturbation_diagnostics_partial.csv",
    }
    for final, partial in partial_map.items():
        frame = read_csv(output / partial); atomic_csv(frame, str(output / final))
    for name, columns in schemas.items():
        path = output / name
        if not path.exists(): atomic_csv(pd.DataFrame(columns=columns), str(path))


def main():
    args = parse_args(); config = configuration(args); output = initialize(config); started = time.perf_counter()
    forbidden = {"true_edge", "true_active", "true_A", "true_group_norm", "oracle_support", "oracle_lambda"}
    if forbidden.intersection(FEATURES) or forbidden.intersection(RAW_FEATURES):
        raise RuntimeError("Truth leakage detected in the P0 selector feature schema.")
    total = len(config["development_network_ids"]) + len(config["confirmation_network_ids"]); progress = Progress(total)
    source = source_config(config); allocator = load_current_allocator(source); frozen_state, my40 = frozen40_state()
    if config["smoke_test"]: atomic_json({"completed": True, "phase": 0}, output / "_PHASE0_SMOKE_STARTED.json")

    dev_manifest = run_network_phase(config, source, allocator, output, config["development_network_ids"], "development", progress)
    if not dev_manifest.completion_status.eq("success").all():
        ensure_required_outputs(output); progress.phase("development incomplete; resume safe"); return
    atomic_json({"completed": True, "networks": len(dev_manifest)}, output / "_PHASE1_DEVELOPMENT_REFERENCES_COMPLETED.json")
    references = alias_reference_features(read_csv(output / "reference_edge_rows_partial.csv"))
    diagnostics = read_csv(output / "perturbation_diagnostics_partial.csv")
    development = references[references.phase.eq("development")].copy()
    _, frozen_raw, frozen_q = ue.predict_state(development, frozen_state)
    frozen_scores = development[["run_id", "target", "source", "stageB_need_target"]].copy()
    frozen_scores["predicted_stageB_need"] = frozen_raw; frozen_scores["purpose_score"] = frozen_q
    atomic_csv(frozen_scores, str(output / "p0_frozen40_scores.csv"))
    cal_predictions, cal_coefficients = fit_cal15_cv(development, config)
    atomic_csv(cal_predictions, str(output / "p0_cal15_cv_predictions.csv")); atomic_csv(cal_coefficients, str(output / "p0_cal15_coefficients.csv"))
    dev_frozen = score_by_predictions(development, frozen_scores)
    dev_cal = score_by_predictions(development, cal_predictions)
    dev_rows_frozen, orders_frozen = deterministic_replay(dev_frozen, diagnostics, config, "development", frozen_state)
    dev_rows_cal, orders_cal = deterministic_replay(dev_cal, diagnostics, config, "development", frozen_state, cal_predictions)
    dev_rows_cal = [row for row in dev_rows_cal if row["selector_variant"] == "P0_CAL15"]
    development_rows = dev_rows_frozen + dev_rows_cal
    dev_network, dev_summary = aggregate_budget(development_rows, config)
    selected_variant, selected_k, passing, final_cal_state = choose_design(dev_summary, development, config)
    final_rows = [
        {"fold": "final_all_development", "heldout_run_id": "", "term": term, "coefficient": value,
         "ridge_penalty": config["ridge_penalty"], "training_run_ids": "|".join(sorted(development.run_id.unique()))}
        for term, value in zip(["intercept", *FEATURES], final_cal_state["coefficients"])
    ]
    cal_coefficients = pd.concat([cal_coefficients, pd.DataFrame(final_rows)], ignore_index=True)
    atomic_csv(cal_coefficients, str(output / "p0_cal15_coefficients.csv"))
    selected_state = frozen_state if selected_variant == "P0_FROZEN40" else final_cal_state
    development_decision = pd.DataFrame([{"selected_p0_variant": selected_variant, "selected_k": selected_k,
                                          "selected_rho": selected_k / config["candidate_pool_size"],
                                          "frozen40_required_k": passing["P0_FROZEN40"],
                                          "cal15_required_k": passing["P0_CAL15"],
                                          "selected_k_le_40": selected_k <= 40}])
    atomic_csv(dev_network, str(output / "development_budget_curves.csv")); atomic_csv(dev_summary, str(output / "development_network_summary.csv")); atomic_csv(development_decision, str(output / "development_decision_summary.csv"))
    atomic_json({"completed": True}, output / "_PHASE2_DEVELOPMENT_REPLAY_COMPLETED.json")
    frozen_design = {
        "selected_p0_variant": selected_variant, "selected_k": selected_k,
        "selected_rho": selected_k / config["candidate_pool_size"], "feature_names": FEATURES,
        "raw_feature_names": RAW_FEATURES, "feature_transforms": config["feature_transforms"],
        "target_definition": config["stageB_need_target"], "ridge_penalty": config["ridge_penalty"],
        "model": state_json(selected_state), "decision_thresholds": config["development_thresholds"],
        "development_network_ids": config["development_network_ids"], "confirmation_locked_during_design": True,
        "created_at": time.time(),
    }
    atomic_json(frozen_design, output / "frozen_design.json"); atomic_json({"completed": True}, output / "_PHASE3_DESIGN_FROZEN.json")

    confirmation_rows = []; confirmation_orders = pd.DataFrame(); conf_manifest = pd.DataFrame()
    conf_network = pd.DataFrame(); conf_summary = pd.DataFrame()
    if config["confirmation_network_ids"]:
        if not (output / "frozen_design.json").exists(): raise RuntimeError("Confirmation lock violation: frozen_design.json absent.")
        conf_manifest = run_network_phase(config, source, allocator, output, config["confirmation_network_ids"], "confirmation", progress, len(config["development_network_ids"]))
        if conf_manifest.completion_status.eq("success").all():
            atomic_json({"completed": True, "networks": len(conf_manifest)}, output / "_PHASE4_CONFIRMATION_REFERENCES_COMPLETED.json")
            references = alias_reference_features(read_csv(output / "reference_edge_rows_partial.csv")); confirmation = references[references.phase.eq("confirmation")].copy()
            diagnostics = read_csv(output / "perturbation_diagnostics_partial.csv")
            _, raw, q = ue.predict_state(confirmation, selected_state)
            conf_scores = confirmation[["run_id", "target", "source", "stageB_need_target"]].copy(); conf_scores["predicted_stageB_need"] = raw; conf_scores["purpose_score"] = q
            confirmation_scored = score_by_predictions(confirmation, conf_scores)
            all_rows, confirmation_orders = deterministic_replay(confirmation_scored, diagnostics, config, "confirmation", selected_state)
            confirmation_rows = [row for row in all_rows if row["selector_variant"] == "P0_FROZEN40"]
            for row in confirmation_rows: row["selector_variant"] = selected_variant
            conf_network, conf_summary = aggregate_budget(confirmation_rows, config)
            atomic_csv(conf_network, str(output / "confirmation_budget_results.csv")); atomic_csv(conf_summary, str(output / "confirmation_network_summary.csv"))
            selected_rows = conf_network[conf_network.k.eq(selected_k)]
            confirmed = confirmation_pass(selected_rows)
            confirmation_decision = pd.DataFrame([{"selected_p0_variant": selected_variant, "selected_k": selected_k,
                                                    "p0_my15_confirmed": confirmed, "n_confirmation_networks": selected_rows.run_id.nunique()}])
            atomic_csv(confirmation_decision, str(output / "confirmation_decision_summary.csv")); atomic_json({"completed": True, "confirmed": confirmed}, output / "_PHASE5_FINAL_DECISION_COMPLETED.json")
        else:
            ensure_required_outputs(output); progress.phase("confirmation incomplete; resume safe"); return
    else:
        confirmed = bool(config["smoke_test"])
        atomic_csv(pd.DataFrame(columns=dev_manifest.columns), str(output / "confirmation_network_manifest.csv"))
        atomic_csv(pd.DataFrame([{"smoke_only": True, "p0_my15_confirmed": False}]), str(output / "confirmation_decision_summary.csv"))

    selected_dev = dev_network[(dev_network.selector_variant.eq(selected_variant)) & dev_network.k.eq(selected_k)]
    selected_conf = conf_network[conf_network.k.eq(selected_k)] if len(conf_network) else pd.DataFrame()
    selected_development = score_by_predictions(development, frozen_scores if selected_variant == "P0_FROZEN40" else cal_predictions)
    random_raw, random_summary = random_controls(selected_development, diagnostics, config, "development")
    atomic_csv(random_summary, str(output / "random_baseline_summary.csv"))
    shift = feature_shift(my40, development); atomic_csv(shift, str(output / "feature_shift_my40_vs_my15.csv"))
    orderings = pd.concat([orders_frozen, orders_cal[orders_cal.selector_variant.eq("P0_CAL15")], confirmation_orders], ignore_index=True, sort=False)
    atomic_csv(orderings, str(output / "selector_orderings.csv"))
    atomic_csv(pd.DataFrame([{"k": k, "N": config["candidate_pool_size"], "rho": k / config["candidate_pool_size"]} for k in config["K_GRID"]]), str(output / "budget_mapping.csv"))
    atomic_csv(pd.concat([dev_network, selected_conf], ignore_index=True, sort=False), str(output / "correction_capture_summary.csv"))
    atomic_csv(pd.concat([dev_network, selected_conf], ignore_index=True, sort=False), str(output / "covariance_fidelity_summary.csv"))
    atomic_csv(pd.concat([dev_network, selected_conf], ignore_index=True, sort=False), str(output / "observability_quartile_summary.csv"))
    conf_selected = selected_conf
    p0_current = selected_dev.retained_active_gain.mean() >= dev_network[(dev_network.selector_variant.eq("D0_CURRENT")) & dev_network.k.eq(selected_k)].retained_active_gain.mean() if len(selected_dev) else False
    uniform = random_summary[(random_summary.selector_variant.eq("U0_UNIFORM_RANDOM")) & random_summary.k.eq(selected_k)]
    stratified = random_summary[(random_summary.selector_variant.eq("D1_STRATIFIED_RANDOM")) & random_summary.k.eq(selected_k)]
    decision = pd.DataFrame([{
        "n_development_networks": dev_network.run_id.nunique(), "n_confirmation_networks": conf_selected.run_id.nunique() if len(conf_selected) else 0,
        "selected_p0_variant": selected_variant, "selected_k": selected_k, "selected_rho": selected_k / config["candidate_pool_size"], "selected_k_le_40": selected_k <= 40,
        "development_mean_categorical_agreement": selected_dev.primary_categorical_agreement.mean(), "development_min_categorical_agreement": selected_dev.primary_categorical_agreement.min(),
        "development_mean_retained_active_gain": selected_dev.retained_active_gain.mean(), "development_min_retained_active_gain": selected_dev.retained_active_gain.min(),
        "confirmation_mean_categorical_agreement": conf_selected.primary_categorical_agreement.mean() if len(conf_selected) else np.nan,
        "confirmation_min_categorical_agreement": conf_selected.primary_categorical_agreement.min() if len(conf_selected) else np.nan,
        "confirmation_number_primary_reversals": int(conf_selected.primary_usefulness_reversal.sum()) if len(conf_selected) else 0,
        "confirmation_stability_match_fraction": conf_selected.stability_agreement.mean() if len(conf_selected) else np.nan,
        "confirmation_mean_retained_active_gain": conf_selected.retained_active_gain.mean() if len(conf_selected) else np.nan,
        "confirmation_min_retained_active_gain": conf_selected.retained_active_gain.min() if len(conf_selected) else np.nan,
        "confirmation_zero_coverage_delta": conf_selected.zero_coverage_delta_vs_full.abs().mean() if len(conf_selected) else np.nan,
        "confirmation_correction_capture": conf_selected.positive_correction_capture.mean() if len(conf_selected) else np.nan,
        "confirmation_Q1_retained_gain": conf_selected.Q1_retained_active_gain.mean() if len(conf_selected) else np.nan,
        "frozen40_transfer_successful": bool(np.isfinite(passing["P0_FROZEN40"])), "my15_recalibration_needed": selected_variant == "P0_CAL15",
        "p0_beats_current_ordering": bool(p0_current),
        "p0_beats_uniform_random": bool(len(uniform) and selected_dev.retained_active_gain.mean() > uniform.retained_active_gain_median.mean()),
        "p0_beats_stratified_random": bool(len(stratified) and selected_dev.retained_active_gain.mean() > stratified.retained_active_gain_median.mean()),
        "diagnostic_budget_below_40_supported": bool(confirmed and selected_k <= 40 and not config["smoke_test"]), "p0_my15_confirmed": bool(confirmed and not config["smoke_test"]),
        "selector_ready_to_freeze": bool(confirmed and not config["smoke_test"]), "ready_to_move_to_lrvb_cost_reduction": bool(not config["smoke_test"] and (confirmed or selected_k >= 63)),
        "evidence_limitation": "90-edge truth-free deployable reference pool; not the full 380-edge network" + ("; smoke test only" if config["smoke_test"] else ""),
    }])
    atomic_csv(decision, str(output / "decision_summary.csv"))
    if confirmed and not config["smoke_test"]:
        full = read_csv(output / "cheap_full380_partial.csv")
        full = full[full.phase.eq("confirmation")].copy()
        full = alias_reference_features(full)
        _, prediction, score = ue.predict_state(full, selected_state); full["predicted_stageB_need"] = prediction; full["purpose_score"] = score
        full["p0_rank"] = full.groupby("run_id").purpose_score.rank(method="first", ascending=False).astype(int)
        atomic_csv(full, str(output / "full380_p0_rankings.csv")); atomic_json({"completed": True}, output / "_PHASE6_FULL380_CHEAP_RANKING_COMPLETED.json")
    ensure_required_outputs(output)
    runtime = read_csv(output / "reference_runtime_partial.csv")
    runtime_summary = pd.DataFrame([{"experiment_runtime_seconds_this_invocation": time.perf_counter() - started,
                                     "completed_development_networks": int(dev_manifest.completion_status.eq("success").sum()),
                                     "completed_confirmation_networks": int(conf_manifest.completion_status.eq("success").sum()) if len(conf_manifest) else 0,
                                     "network_workers": config["network_workers"], "stageB_workers": config["stageB_workers"],
                                     "inner_threads": config["inner_threads"], "new_stageB_group_limit_per_network": MAX_STAGEB_GROUPS_PER_NETWORK,
                                     "full_380_stageB_run": False}])
    atomic_csv(runtime_summary, str(output / "runtime_summary.csv"))
    summaries_and_plots(output, development_rows, confirmation_rows, random_summary, shift, decision)
    complete = bool(config["smoke_test"] or (len(conf_manifest) and conf_manifest.completion_status.eq("success").all()))
    completion = {"completed": complete, "smoke_test": config["smoke_test"], "selected_p0_variant": selected_variant,
                  "selected_k": selected_k, "selected_rho": selected_k / config["candidate_pool_size"],
                  "p0_my15_confirmed": bool(decision.p0_my15_confirmed.iloc[0]), "full_380_stageB_run": False,
                  "checkpoint_cleanup_pending": False, "completed_at": time.time()}
    atomic_json(completion, output / "_COMPLETED.json")
    if complete:
        for name in ("ckpt", "edge_group_checkpoints"):
            checkpoint = (output / name).resolve()
            if checkpoint.exists() and checkpoint.parent == output.resolve():
                try:
                    shutil.rmtree(checkpoint)
                except OSError:
                    completion["checkpoint_cleanup_pending"] = bool(
                        checkpoint.exists() and any(item.is_file() for item in checkpoint.rglob("*"))
                    )
        atomic_json(completion, output / "_COMPLETED.json")
    if config["smoke_test"]: atomic_json({"completed": True}, output / "_PHASE0_SMOKE_COMPLETED.json")
    progress.phase("complete")


if __name__ == "__main__":
    main()
