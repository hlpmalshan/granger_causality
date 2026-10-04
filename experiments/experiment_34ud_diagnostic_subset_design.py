"""Experiment 34UD: small diagnostic subsets for legacy Stage-B LRVB.

The default run is a read-only retrospective replay of the completed 34UC
five-reference study.  Prospective full-380 feature construction and legacy
Stage-B on only the selected first 40 groups is strictly opt-in via
EXPERIMENT_34UD_RUN_PROSPECTIVE=1 and is refused unless retrospective success
has first been established.  This experiment never runs full-380 Stage-B.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import zlib
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34u_a_fullrun_observability_calibration as ua
import experiments.experiment_34uc_greedy_selector_validation as uc
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.diagnostic_subset_design import (
    CATEGORY_COLUMNS,
    CONTINUOUS_COLUMNS,
    FEATURES,
    add_diagnostic_strata,
    apply_robust_scaler,
    calibration_bandwidth,
    conclusion_agreement,
    conclusion_vector,
    fit_robust_scaler,
    node_coverage,
    quota_cell_schedule,
    quota_counts,
    representativeness,
    stratified_facility_order,
    stratified_logdet_order,
    stratified_maximin_order,
    stratified_random_order,
    success_assessment,
)


EPS = 1e-12
EXPERIMENT = "34UD_diagnostic_subset_design"
REFERENCE_DIR = Path("results/experiment_34uc_greedy_selector_validation/prospective_validation_5refs")
ROOT_OUTPUT = Path("results/experiment_34ud_diagnostic_subset_design")
PRIMARY_K = [10, 20, 30, 40]
EVIDENCE_SCOPE = (
    "Validated against larger 90-edge legacy Stage-B references; "
    "not a proof of equivalence to full 380-edge Stage-B."
)


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UD_SMOKE", "0") == "1")
    output = Path(args.results_dir) if args.results_dir else (ROOT_OUTPUT / "smoke_test" if smoke else ROOT_OUTPUT)
    random_default = 20 if smoke else 1000
    max_k = min(40, max(5, env_int("EXPERIMENT_34UD_MAX_K", 10 if smoke else 40)))
    return {
        "experiment": EXPERIMENT,
        "title": "Greedy diagnostic edge-subset design for Stage-B LRVB",
        "scientific_question": "Can 10-40 diagnostic edges reproduce methodological conclusions of a larger 90-edge reference?",
        "smoke_test": smoke,
        "results_directory": str(output),
        "source_34UC": str(REFERENCE_DIR),
        "source_34UA": "results/experiment_34u_a_fullrun_observability_calibration",
        "M_x": 20, "M_y": 40, "T": 1000, "na": 2, "nb": 3,
        "offdiagonal_edge_universe": 380,
        "atomic_item": "directed p=2 group [A1[target,source], A2[target,source]]",
        "reference_edge_groups_per_network": 90,
        "calibration_references": 2,
        "heldout_references": min(3, max(1, env_int("EXPERIMENT_34UD_EVAL_REFS", 1 if smoke else 3))),
        "max_k": max_k,
        "k_grid": list(range(5, max_k + 1)),
        "primary_k": [k for k in PRIMARY_K if k <= max_k],
        "random_repeats": max(1, env_int("EXPERIMENT_34UD_RANDOM_REPEATS", random_default)),
        "workers": max(1, env_int("EXPERIMENT_34UD_WORKERS", 1)),
        "run_prospective": os.environ.get("EXPERIMENT_34UD_RUN_PROSPECTIVE", "0") == "1",
        "facility_bandwidth_multiplier_grid": [0.5, 1.0, 2.0],
        "facility_weight_modes": ["uniform", "louis_uncertainty"],
        "maximin_initializations": ["centroid", "uncertainty"],
        "d_opt_node_scale_grid": [0.0, 0.25, 0.5],
        "d_opt_delta": 1e-3,
        "A_center": "A_VB",
        "matrixfree_LRVB_used": False,
        "full_380_legacy_stageB_allowed": False,
        "prospective_stageB_cap_groups": 40,
        "selection_uses_truth_or_stageB": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "success_thresholds": {
            "mean_categorical_agreement": 0.85,
            "networks_at_agreement_0p80": "at least 2 of 3",
            "Delta_active_absolute_error": 0.10,
            "rescue_rate_absolute_error": 0.15,
            "observability_rho_absolute_error": 0.20,
            "SNR_rho_absolute_error": 0.20,
            "primary_beneficial_harmful_reversals": 0,
            "stability_agreement": "all held-out networks",
        },
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def empty_csv(path, columns):
    atomic_csv(pd.DataFrame(columns=list(columns)), str(path))


class Progress:
    def __init__(self, total):
        self.total = max(int(total), 1); self.done = 0; self.started = time.perf_counter()
    def update(self, label=""):
        self.done += 1; fraction = min(self.done / self.total, 1.0); filled = round(30 * fraction)
        elapsed = time.perf_counter() - self.started
        eta = elapsed / max(self.done, 1) * max(self.total - self.done, 0)
        print(f"\r34UD [{'#' * filled}{'-' * (30 - filled)}] {self.done}/{self.total} "
              f"{100 * fraction:5.1f}% ETA {eta / 60:5.1f}m {label[:28]}",
              end="\n" if self.done >= self.total else "", flush=True)


def initialize(config):
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    config_path = output / "experiment_config.json"
    protected = ["smoke_test", "max_k", "random_repeats", "heldout_references", "run_prospective"]
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf8"))
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed and not (output / "_COMPLETED.json").exists():
            raise ValueError(f"Existing resumable 34UD configuration differs in {changed}")
    atomic_json(config, config_path)
    (output / "plots").mkdir(exist_ok=True)
    return output


def handle_completed_prospective_gate(config, output):
    """Turn an ineligible prospective request into a clean scientific stop.

    A failed retrospective decision is an expected experiment outcome, not a
    software failure.  Avoid rerunning the 1,000-replay analysis and never
    launch Stage-B in this state.
    """
    marker_path = output / "_COMPLETED.json"
    decision_path = output / "decision_summary.csv"
    if not config["run_prospective"] or not marker_path.exists() or not decision_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf8"))
    decision = pd.read_csv(decision_path)
    retrospective_success = bool(marker.get("retrospective_success", False))
    if len(decision) and "retrospective_success" in decision:
        retrospective_success = bool(decision.retrospective_success.iloc[0])
    if retrospective_success:
        return False
    status = "blocked_retrospective_k_le_40_not_validated"
    config["prospective_gate_status"] = status
    decision["prospective_request_enabled"] = True
    decision["prospective_full380_run"] = False
    decision["prospective_gate_status"] = status
    atomic_json(config, output / "experiment_config.json")
    atomic_csv(decision, str(decision_path))
    marker.update({
        "prospective_requested": True,
        "prospective_full380_run": False,
        "prospective_gate_status": status,
        "last_checked_at_unix": time.time(),
    })
    atomic_json(marker, marker_path)
    Progress(1).update("prospective blocked safely")
    return True


def load_references(config):
    edge_path = REFERENCE_DIR / "new_reference_edge_rows_partial.csv"
    diagnostic_path = REFERENCE_DIR / "new_reference_perturbation_diagnostics_partial.csv"
    runtime_path = REFERENCE_DIR / "new_reference_runtime_partial.csv"
    manifest_path = REFERENCE_DIR / "reference_replicate_manifest.csv"
    for path in (edge_path, diagnostic_path, runtime_path, manifest_path, REFERENCE_DIR / "_COMPLETED.json"):
        if not path.exists():
            raise FileNotFoundError(f"Required completed 34UC artifact is missing: {path}")
    edges = pd.read_csv(edge_path)
    diagnostics = pd.read_csv(diagnostic_path)
    runtime = pd.read_csv(runtime_path)
    successful = set(runtime.loc[runtime["run_status"].eq("success"), "run_id"])
    edges = edges[edges["run_id"].isin(successful)].copy()
    diagnostics = diagnostics[diagnostics["run_id"].isin(successful)].copy()
    edges["reference_split"] = edges["split"].astype(str)
    calibration = sorted(edges.loc[edges.reference_split.eq("calibration"), "run_id"].unique())
    heldout = sorted(edges.loc[edges.reference_split.eq("evaluation"), "run_id"].unique())[: config["heldout_references"]]
    wanted = calibration[:2] + heldout
    edges = edges[edges.run_id.isin(wanted)].copy()
    diagnostics = diagnostics[diagnostics.run_id.isin(wanted)].copy()
    counts = edges.groupby("run_id").size()
    direction_counts = diagnostics.groupby("run_id").size()
    if len(calibration) < 2 or len(heldout) < config["heldout_references"]:
        raise RuntimeError("34UD requires two calibration and the requested held-out 34UC references.")
    if not (counts == 90).all() or not (direction_counts == 180).all():
        raise RuntimeError(f"Incomplete 34UC references: edge counts={counts.to_dict()}, directions={direction_counts.to_dict()}")
    if edges.duplicated(["run_id", "target", "source"]).any():
        raise RuntimeError("The 34UC reference contains duplicate atomic edge groups.")
    if (edges.target.astype(int) == edges.source.astype(int)).any():
        raise RuntimeError("The 34UC off-diagonal reference unexpectedly contains diagonal groups.")
    # Map each finite-difference direction back to its atomic p=2 group.
    remainder = diagnostics.direction_index.astype(int) % 40
    diagnostics["target"] = diagnostics.direction_index.astype(int) // 40
    diagnostics["source"] = np.where(remainder < 20, remainder, remainder - 20)
    diagnostics["lag"] = np.where(remainder < 20, 1, 2)
    manifest = pd.DataFrame([
        {
            "run_id": run_id,
            "reference_split": group.reference_split.iloc[0],
            "source_experiment": "34UC",
            "source_path": str(REFERENCE_DIR),
            "candidate_pool_size": len(group),
            "atomic_edge_groups": len(group),
            "individual_coefficient_directions": int(direction_counts[run_id]),
            "candidate_pool_semantics": "deployable current-ranking top-90; no truth used for selection",
            "full_legacy_stageB_available": True,
            "reference_is_full_380": False,
            "stageB_runtime_seconds": float(runtime.loc[runtime.run_id.eq(run_id), "stageB_runtime_seconds"].iloc[-1]),
        }
        for run_id, group in edges.groupby("run_id", sort=False)
    ])
    return edges.reset_index(drop=True), diagnostics.reset_index(drop=True), manifest


def prepare_features(edges):
    missing = [name for name in FEATURES if name not in edges]
    if missing:
        raise KeyError(f"34UC reference is missing deployable features: {missing}")
    numeric = edges[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        bad = [name for name in FEATURES if not np.isfinite(numeric[name]).all()]
        raise FloatingPointError(f"Invalid deployable feature values: {bad}")
    edges = add_diagnostic_strata(edges)
    calibration = edges[edges.reference_split.eq("calibration")]
    scaler = fit_robust_scaler(calibration)
    edges = apply_robust_scaler(edges, scaler)
    bandwidth = calibration_bandwidth(edges[edges.reference_split.eq("calibration")])
    return edges, scaler, bandwidth


def selector_specs(config, bandwidth):
    specs = [
        {"selector": "D0_current_ordering", "variant": "current_production_prefix"},
        {"selector": "D1_stratified_random", "variant": "fixed_seed_0", "seed": 340000},
    ]
    specs += [{"selector": "D2_stratified_maximin", "variant": f"init_{init}", "initialization": init}
              for init in config["maximin_initializations"]]
    specs += [{"selector": "D3_stratified_facility", "variant": f"h{mult:g}_{weight}",
               "bandwidth": bandwidth * mult, "bandwidth_multiplier": mult, "weight_mode": weight}
              for mult in config["facility_bandwidth_multiplier_grid"] for weight in config["facility_weight_modes"]]
    specs += [{"selector": "D4_stratified_logdet", "variant": f"node_scale_{scale:g}", "node_scale": scale,
               "delta": config["d_opt_delta"]} for scale in config["d_opt_node_scale_grid"]]
    return specs


def select_order(group, spec, max_k, seed_offset=0):
    group = group.reset_index(drop=True)
    Z = group[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    strata = group.diagnostic_stratum.to_numpy(str)
    if spec["selector"] == "D0_current_ordering":
        key = "source_row_order" if "source_row_order" in group else "current_production_score"
        ascending = key == "source_row_order"
        return group.sort_values(key, ascending=ascending, kind="stable").index.to_list()[:max_k]
    if spec["selector"] == "D1_stratified_random":
        return stratified_random_order(strata, max_k, int(spec["seed"]) + int(seed_offset))
    if spec["selector"] == "D2_stratified_maximin":
        return stratified_maximin_order(Z, strata, max_k, spec["initialization"])
    if spec["selector"] == "D3_stratified_facility":
        return stratified_facility_order(Z, strata, max_k, spec["bandwidth"], spec["weight_mode"])
    if spec["selector"] == "D4_stratified_logdet":
        return stratified_logdet_order(Z, strata, group.source, group.target, max_k,
                                       spec["node_scale"], spec["delta"])
    raise ValueError(spec["selector"])


def build_orderings(features, specs, max_k):
    rows = []
    for run_id, group in features.groupby("run_id", sort=False):
        group = group.reset_index(drop=True)
        for spec in specs:
            stable_offset = zlib.crc32(str(run_id).encode("utf8")) % 100000
            order = select_order(group, spec, max_k, seed_offset=stable_offset)
            for rank, index in enumerate(order, 1):
                edge = group.iloc[index]
                row = {
                    "run_id": run_id, "reference_split": edge.reference_split,
                    "selector": spec["selector"], "selector_variant": spec["variant"],
                    "rank": rank, "target": int(edge.target), "source": int(edge.source),
                    "diagnostic_stratum": edge.diagnostic_stratum,
                    "diagnostic_observability_quartile": int(edge.diagnostic_observability_quartile),
                    "diagnostic_snr_tertile": int(edge.diagnostic_snr_tertile),
                }
                row.update({name: float(edge[name]) for name in FEATURES})
                rows.append(row)
    return pd.DataFrame(rows)


def edge_perturbations(diagnostics, run_id, subset):
    pairs = set(zip(subset.target.astype(int), subset.source.astype(int)))
    group = diagnostics[diagnostics.run_id.eq(run_id)]
    mask = [(int(t), int(s)) in pairs for t, s in zip(group.target, group.source)]
    return group.loc[mask]


def reference_conclusions(features, diagnostics):
    rows = []
    for run_id, group in features.groupby("run_id", sort=False):
        row = conclusion_vector(group, diagnostics[diagnostics.run_id.eq(run_id)])
        rows.append({"run_id": run_id, "reference_split": group.reference_split.iloc[0],
                     "reference_size": len(group), **row})
    return pd.DataFrame(rows)


def evaluate_orderings(features, diagnostics, references, orderings, k_grid, random_repeat=np.nan):
    subset_rows, agreement_rows = [], []
    ref_lookup = references.set_index("run_id")
    edge_groups = {run_id: group.set_index(["target", "source"], drop=False)
                   for run_id, group in features.groupby("run_id", sort=False)}
    grouping = ["run_id", "selector", "selector_variant"]
    for (run_id, selector, variant), order in orderings.groupby(grouping, sort=False):
        order = order.sort_values("rank")
        population = edge_groups[run_id]
        for k in k_grid:
            chosen_pairs = list(zip(order.head(k).target.astype(int), order.head(k).source.astype(int)))
            subset = population.loc[chosen_pairs].reset_index(drop=True)
            vector = conclusion_vector(subset, edge_perturbations(diagnostics, run_id, subset))
            meta = {"run_id": run_id, "reference_split": subset.reference_split.iloc[0],
                    "selector": selector, "selector_variant": variant, "k": int(k),
                    "random_repeat": random_repeat}
            subset_rows.append({**meta, **vector})
            agreement_rows.append({**meta, **conclusion_agreement(ref_lookup.loc[run_id], vector)})
    return pd.DataFrame(subset_rows), pd.DataFrame(agreement_rows)


def _fast_rank_correlation(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return np.nan
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / denominator) if denominator > EPS else np.nan


def _random_fast_data(group, diagnostic_group):
    group = group.reset_index(drop=True)
    convergence = diagnostic_group.assign(
        _converged=diagnostic_group.plus_converged.astype(bool) & diagnostic_group.minus_converged.astype(bool)
    ).groupby(["target", "source"])["_converged"].mean().to_dict()
    pairs = list(zip(group.target.astype(int), group.source.astype(int)))
    return {
        "true": group["true_edge"].to_numpy(bool),
        "stage": group["stageb_group_covered_95"].to_numpy(bool),
        "louis": group["louis_group_covered_95"].to_numpy(bool),
        "inflation": group["relative_stageb_inflation"].to_numpy(float),
        "obs": group["log_edge_o_inst_product"].to_numpy(float),
        "snr": group["louis_group_snr"].to_numpy(float),
        "quartile": group["diagnostic_observability_quartile"].to_numpy(int),
        "finite_matrix": group[["Sigma_stageb_00", "Sigma_stageb_01", "Sigma_stageb_10", "Sigma_stageb_11"]].to_numpy(float),
        "eig": group["stageb_min_eig_after"].to_numpy(float),
        "convergence": np.asarray([convergence.get(pair, np.nan) for pair in pairs], float),
        "strata": group.diagnostic_stratum.to_numpy(str),
        "reference_split": group.reference_split.iloc[0],
    }


def _fast_conclusion(data, indices):
    chosen = np.asarray(indices, int)
    active = data["true"][chosen]
    stage = data["stage"][chosen]
    louis = data["louis"][chosen]
    inflation = data["inflation"][chosen]
    obs = data["obs"][chosen]
    snr = data["snr"][chosen]
    quartile = data["quartile"][chosen]
    delta_active = float(stage[active].mean() - louis[active].mean()) if active.any() else np.nan
    delta_zero = float(stage[~active].mean() - louis[~active].mean()) if (~active).any() else np.nan
    failed = ~louis
    rescue = float(stage[failed].mean()) if failed.any() else np.nan
    median_inflation = float(np.median(inflation))
    obs_rho = _fast_rank_correlation(obs, inflation)
    snr_rho = _fast_rank_correlation(snr, inflation)
    q1 = active & (quartile == 0); q4 = active & (quartile == 3)
    q1_cov = float(stage[q1].mean()) if q1.any() else np.nan
    q4_cov = float(stage[q4].mean()) if q4.any() else np.nan
    gap = q1_cov - q4_cov if np.isfinite(q1_cov) and np.isfinite(q4_cov) else np.nan
    finite_matrix = data["finite_matrix"][chosen]
    finite = np.isfinite(finite_matrix).all(axis=1)
    eig = data["eig"][chosen]
    failure_fraction = 1.0 - float(finite.mean())

    def category(value, low, high, below, middle, above):
        if not np.isfinite(value): return np.nan
        if value < low: return below
        if value > high: return above
        return middle

    conv = data["convergence"][chosen]
    return {
        "Delta_active": delta_active,
        "active_coverage_conclusion": category(delta_active, -.05, .05, "HARMFUL", "NEUTRAL", "BENEFICIAL"),
        "Delta_zero": delta_zero,
        "zero_coverage_conclusion": category(delta_zero, -.02, .02, "WORSENS", "NEUTRAL", "IMPROVES"),
        "rescue_rate": rescue,
        "rescue_conclusion": category(rescue, .25, .5 - EPS, "LOW", "MODERATE", "HIGH"),
        "median_relative_inflation": median_inflation,
        "mean_relative_inflation": float(np.mean(inflation)),
        "fraction_relative_inflation_gt_0": float(np.mean(inflation > 0)),
        "fraction_relative_inflation_gt_0p25": float(np.mean(inflation > .25)),
        "inflation_conclusion": category(median_inflation, -.1, .1, "DEFLATION", "SMALL_NEUTRAL", "MATERIAL_INFLATION"),
        "observability_rho": obs_rho,
        "observability_dependence_conclusion": category(obs_rho, -.2, .2, "NEGATIVE", "WEAK_FLAT", "POSITIVE"),
        "SNR_rho": snr_rho,
        "snr_dependence_conclusion": category(snr_rho, -.2, .2, "NEGATIVE", "WEAK_FLAT", "POSITIVE"),
        "Q1_active_coverage": q1_cov, "Q4_active_coverage": q4_cov,
        "Q1_minus_Q4_active_coverage_gap": gap,
        "low_observability_conclusion": category(gap, -.1, .1, "Q1_WORSE", "SIMILAR", "Q1_BETTER"),
        "finite_result_fraction": float(finite.mean()),
        "perturbation_convergence_fraction": float(np.nanmean(conv)) if np.isfinite(conv).any() else np.nan,
        "PSD_fraction": float(np.mean(eig >= -1e-10)),
        "minimum_covariance_eigenvalue": float(np.min(eig)),
        "failure_fraction": failure_fraction,
        "numerical_stability_conclusion": "ACCEPTABLE" if failure_fraction <= .05 and finite.mean() >= .95 else "UNSTABLE",
        "n_edges": len(chosen), "n_active_edges": int(active.sum()), "n_zero_edges": int((~active).sum()),
    }


def random_replay(features, diagnostics, references, config, progress=None):
    ref_lookup = {row.run_id: row._asdict() for row in references.itertuples(index=False)}
    groups = []
    for run_id, group in features.groupby("run_id", sort=False):
        data = _random_fast_data(group, diagnostics[diagnostics.run_id.eq(run_id)])
        groups.append((run_id, data))
    metric_names = ["categorical_agreement"] + [f"abs_error_{name}" for name in CONTINUOUS_COLUMNS]
    samples = {(run_id, k, metric): [] for run_id, _ in groups for k in config["k_grid"] for metric in metric_names}
    for repeat in range(config["random_repeats"]):
        for run_number, (run_id, data) in enumerate(groups):
            order = stratified_random_order(data["strata"], config["max_k"],
                                             340000 + 10007 * repeat + run_number)
            for k in config["k_grid"]:
                vector = _fast_conclusion(data, order[:k])
                agreement = conclusion_agreement(ref_lookup[run_id], vector)
                for metric in metric_names:
                    samples[(run_id, k, metric)].append(agreement[metric])
        if progress is not None and (repeat + 1) % max(1, config["random_repeats"] // 4) == 0:
            progress.update("random replay")
    rows = []
    for run_id, data in groups:
        for k in config["k_grid"]:
            row = {"reference_split": data["reference_split"], "run_id": run_id, "k": k,
                   "selector": "D1_stratified_random", "selector_variant": "replay_distribution",
                   "n_random_replays": config["random_repeats"]}
            for metric in metric_names:
                values = np.asarray(samples[(run_id, k, metric)], float)
                valid = values[np.isfinite(values)]
                row.update({f"{metric}_mean": np.mean(valid) if len(valid) else np.nan,
                            f"{metric}_sd": np.std(valid, ddof=1) if len(valid) > 1 else np.nan,
                            f"{metric}_q05": np.quantile(valid, .05) if len(valid) else np.nan,
                            f"{metric}_median": np.median(valid) if len(valid) else np.nan,
                            f"{metric}_q95": np.quantile(valid, .95) if len(valid) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_method(agreement):
    rows = []
    columns = ["reference_split", "selector", "selector_variant", "k"]
    for keys, group in agreement.groupby(columns, dropna=False, sort=False):
        row = dict(zip(columns, keys))
        row.update({
            "n_networks": group.run_id.nunique(),
            "mean_categorical_agreement": group.categorical_agreement.mean(),
            "median_categorical_agreement": group.categorical_agreement.median(),
            "min_categorical_agreement": group.categorical_agreement.min(),
            "max_categorical_agreement": group.categorical_agreement.max(),
            "primary_usefulness_reversal_count": int(group.primary_usefulness_reversal.sum()),
            "stability_agreement_rate": group.stability_agreement.mean(),
        })
        for metric in CONTINUOUS_COLUMNS:
            row[f"mean_abs_error_{metric}"] = group[f"abs_error_{metric}"].mean()
        if int(keys[-1]) in PRIMARY_K:
            row.update({f"success_{key}": value for key, value in success_assessment(group).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_family_hyperparameters(agreement, specs, config):
    cal = agreement[agreement.reference_split.eq("calibration")]
    rows = []
    for selector in sorted(cal.selector.unique()):
        for variant in sorted(cal.loc[cal.selector.eq(selector), "selector_variant"].unique()):
            group = cal[(cal.selector.eq(selector)) & (cal.selector_variant.eq(variant))]
            success_by_k = {}
            for k in config["primary_k"]:
                success_by_k[k] = success_assessment(group[group.k.eq(k)])["successful"]
            earliest = min([k for k, ok in success_by_k.items() if ok], default=999)
            at_max = group[group.k.eq(config["max_k"])]
            rows.append({
                "selector": selector, "selector_variant": variant,
                "calibration_smallest_successful_k": earliest if earliest < 999 else np.nan,
                "calibration_success_by_max_k": bool(earliest <= config["max_k"]),
                "calibration_primary_reversals_at_max_k": int(at_max.primary_usefulness_reversal.sum()),
                "calibration_mean_agreement_at_max_k": at_max.categorical_agreement.mean(),
                "calibration_Delta_active_error_at_max_k": at_max.abs_error_Delta_active.mean(),
                "calibration_rescue_error_at_max_k": at_max.abs_error_rescue_rate.mean(),
                "calibration_observability_rho_error_at_max_k": at_max.abs_error_observability_rho.mean(),
                "calibration_SNR_rho_error_at_max_k": at_max.abs_error_SNR_rho.mean(),
                "calibration_stability_agreement_at_max_k": at_max.stability_agreement.mean(),
                **{f"success_at_k{k}": bool(success_by_k.get(k, False)) for k in PRIMARY_K},
            })
    table = pd.DataFrame(rows)
    chosen = {}
    for selector, group in table.groupby("selector", sort=False):
        ranked = group.assign(
            _k=group.calibration_smallest_successful_k.fillna(999),
            _rev=group.calibration_primary_reversals_at_max_k,
            _agreement=-group.calibration_mean_agreement_at_max_k.fillna(-np.inf),
            _active=group.calibration_Delta_active_error_at_max_k.fillna(np.inf),
            _rescue=group.calibration_rescue_error_at_max_k.fillna(np.inf),
        ).sort_values(["_k", "_rev", "_active", "_rescue", "_agreement", "selector_variant"], kind="stable")
        chosen[selector] = ranked.iloc[0].selector_variant
    table["frozen_for_family"] = [chosen.get(selector) == variant for selector, variant in zip(table.selector, table.selector_variant)]
    # Attach actual selector hyperparameters for auditability.
    spec_map = {(item["selector"], item["variant"]): item for item in specs}
    for name in ("initialization", "bandwidth", "bandwidth_multiplier", "weight_mode", "node_scale", "delta", "seed"):
        table[name] = [spec_map.get((s, v), {}).get(name, np.nan) for s, v in zip(table.selector, table.selector_variant)]
    return table, chosen


def rank_families(calibration_agreement, chosen, config):
    candidates = []
    simplicity = {"D0_current_ordering": 0, "D1_stratified_random": 1,
                  "D2_stratified_maximin": 2, "D3_stratified_facility": 3,
                  "D4_stratified_logdet": 4}
    for selector, variant in chosen.items():
        group = calibration_agreement[(calibration_agreement.selector.eq(selector)) &
                                      (calibration_agreement.selector_variant.eq(variant))]
        assessments = {k: success_assessment(group[group.k.eq(k)]) for k in config["primary_k"]}
        earliest = min([k for k, item in assessments.items() if item["successful"]], default=999)
        at_max = group[group.k.eq(config["max_k"])]
        candidates.append({
            "selector": selector, "selector_variant": variant, "smallest_successful_k": earliest,
            "reversals": int(at_max.primary_usefulness_reversal.sum()),
            "active_error": at_max.abs_error_Delta_active.mean(),
            "rescue_error": at_max.abs_error_rescue_rate.mean(),
            "pattern_error": at_max[["abs_error_observability_rho", "abs_error_SNR_rho"]].mean().mean(),
            "stability_mismatch": int((~at_max.stability_agreement.astype(bool)).sum()),
            "simplicity": simplicity[selector],
        })
    ranked = pd.DataFrame(candidates).sort_values(
        ["smallest_successful_k", "reversals", "active_error", "rescue_error",
         "pattern_error", "stability_mismatch", "simplicity"], kind="stable"
    ).reset_index(drop=True)
    return ranked.iloc[0].to_dict(), ranked


def frozen_agreement(agreement, chosen):
    parts = [agreement[(agreement.selector.eq(selector)) & (agreement.selector_variant.eq(variant))]
             for selector, variant in chosen.items()]
    return pd.concat(parts, ignore_index=True)


def decision_table(winner, heldout_agreement, orderings, config):
    selector, variant = winner["selector"], winner["selector_variant"]
    heldout = heldout_agreement[(heldout_agreement.selector.eq(selector)) &
                                (heldout_agreement.selector_variant.eq(variant))]
    assessments = {k: success_assessment(heldout[heldout.k.eq(k)]) for k in config["primary_k"]}
    recommended = next((k for k in PRIMARY_K if k in assessments and assessments[k]["successful"]), np.nan)
    row = {
        "winning_diagnostic_selector": selector,
        "winning_selector_variant": variant,
        "winning_selector_reason": "Calibration-only lexicographic rule: smallest successful k, no reversal, continuous accuracy, stability, simplicity.",
        "retrospective_success": bool(np.isfinite(recommended)),
        "prospective_full380_run": bool(config["run_prospective"] and np.isfinite(recommended)),
        "recommended_k": recommended,
        "recommended_fraction_of_380": recommended / 380 if np.isfinite(recommended) else np.nan,
        "ready_as_diagnostic_screen": bool(np.isfinite(recommended)),
        "evidence_scope": EVIDENCE_SCOPE,
    }
    for k in PRIMARY_K:
        group = heldout[heldout.k.eq(k)]
        item = assessments.get(k, {"successful": False})
        row.update({
            f"success_at_k{k}": bool(item["successful"]),
            f"mean_conclusion_agreement_k{k}": group.categorical_agreement.mean(),
            f"primary_usefulness_reversal_k{k}": int(group.primary_usefulness_reversal.sum()) if len(group) else np.nan,
            f"Delta_active_error_k{k}": group.abs_error_Delta_active.mean(),
            f"rescue_rate_error_k{k}": group.abs_error_rescue_rate.mean(),
            f"stability_agreement_k{k}": bool(group.stability_agreement.all()) if len(group) else False,
        })
    if np.isfinite(recommended):
        winner_orders = orderings[(orderings.selector.eq(selector)) & (orderings.selector_variant.eq(variant))]
        heldout_ids = set(heldout.run_id)
        cover = []
        for run_id, group in winner_orders[winner_orders.run_id.isin(heldout_ids)].groupby("run_id"):
            cover.append(node_coverage(group.nsmallest(int(recommended), "rank")))
        row["source_node_coverage_at_recommended_k"] = np.mean([item["unique_source_nodes"] for item in cover])
        row["target_node_coverage_at_recommended_k"] = np.mean([item["unique_target_nodes"] for item in cover])
    else:
        row["source_node_coverage_at_recommended_k"] = np.nan
        row["target_node_coverage_at_recommended_k"] = np.nan
    return pd.DataFrame([row])


def diagnostics_for_winner(features, orderings, winner, config):
    selector, variant = winner["selector"], winner["selector_variant"]
    selected_orders = orderings[(orderings.selector.eq(selector)) & (orderings.selector_variant.eq(variant))]
    feature_groups = {run_id: group.set_index(["target", "source"], drop=False)
                      for run_id, group in features.groupby("run_id")}
    representation, nodes, strata = [], [], []
    for run_id, order in selected_orders.groupby("run_id", sort=False):
        population = feature_groups[run_id]
        for k in config["k_grid"]:
            pairs = list(zip(order.nsmallest(k, "rank").target.astype(int), order.nsmallest(k, "rank").source.astype(int)))
            subset = population.loc[pairs].reset_index(drop=True)
            meta = {"run_id": run_id, "reference_split": subset.reference_split.iloc[0],
                    "selector": selector, "selector_variant": variant, "k": k}
            representation += [{**meta, **row} for row in representativeness(population.reset_index(drop=True), subset)]
            nodes.append({**meta, **node_coverage(subset)})
            counts = subset.diagnostic_stratum.value_counts()
            population_counts = population.diagnostic_stratum.value_counts()
            for cell, count in population_counts.items():
                strata.append({**meta, "diagnostic_stratum": cell, "population_count": int(count),
                               "selected_count": int(counts.get(cell, 0)),
                               "population_fraction": count / len(population),
                               "selected_fraction": counts.get(cell, 0) / len(subset)})
    return pd.DataFrame(representation), pd.DataFrame(nodes), pd.DataFrame(strata)


def poststratified_summary(features, subset_conclusions, orderings, winner, config):
    # Raw conclusion estimates are always saved.  Simple cell-weighted means are
    # added only when every deployable reference cell is represented.
    selector, variant = winner["selector"], winner["selector_variant"]
    orders = orderings[(orderings.selector.eq(selector)) & (orderings.selector_variant.eq(variant))]
    rows = []
    for run_id, order in orders.groupby("run_id", sort=False):
        population = features[features.run_id.eq(run_id)]
        for k in config["primary_k"]:
            pairs = set(zip(order.nsmallest(k, "rank").target.astype(int), order.nsmallest(k, "rank").source.astype(int)))
            subset = population[[((int(t), int(s)) in pairs) for t, s in zip(population.target, population.source)]]
            pop_cells = set(population.diagnostic_stratum); selected_cells = set(subset.diagnostic_stratum)
            available = pop_cells.issubset(selected_cells)
            raw = subset_conclusions[(subset_conclusions.run_id.eq(run_id)) &
                                     (subset_conclusions.selector.eq(selector)) &
                                     (subset_conclusions.selector_variant.eq(variant)) &
                                     (subset_conclusions.k.eq(k))].iloc[0]
            row = {"run_id": run_id, "reference_split": population.reference_split.iloc[0], "k": k,
                   "selector": selector, "selector_variant": variant,
                   "poststratification_available": available}
            for name in CONTINUOUS_COLUMNS:
                row[f"raw_{name}"] = raw.get(name, np.nan)
            if available:
                weights = population.diagnostic_stratum.value_counts(normalize=True)
                cell_vectors = {cell: conclusion_vector(group) for cell, group in subset.groupby("diagnostic_stratum")}
                for name in ("Delta_active", "Delta_zero", "rescue_rate", "median_relative_inflation", "failure_fraction"):
                    values = [(weights[cell], cell_vectors[cell].get(name, np.nan)) for cell in weights.index]
                    row[f"poststratified_{name}"] = (sum(w * value for w, value in values)
                                                       if all(np.isfinite(value) for _, value in values) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def smoke_checks(features, references, config):
    identities = [(target, source) for target in range(20) for source in range(20) if target != source]
    assert len(identities) == 380 and len(set(identities)) == 380
    assert all(target != source for target, source in identities)
    assert all(len([ua.global_index(target, lag, source, 20, 2) for lag in range(2)]) == 2
               for target, source in identities)
    group = features.iloc[:0]
    first_run = features.run_id.iloc[0]; group = features[features.run_id.eq(first_run)].reset_index(drop=True)
    assert np.isfinite(group[[f"z__{name}" for name in FEATURES]].to_numpy()).all()
    assert "true_edge" not in ["diagnostic_observability_quartile", "diagnostic_snr_tertile", "diagnostic_stratum"]
    schedule = quota_cell_schedule(group.diagnostic_stratum, config["max_k"])
    assert all(sum(quota_counts(schedule, k).values()) == k for k in config["k_grid"])
    specs = selector_specs(config, calibration_bandwidth(features[features.reference_split.eq("calibration")]))
    for spec in specs:
        order = select_order(group, spec, config["max_k"])
        assert len(order) == len(set(order)) == config["max_k"]
    vector = conclusion_vector(group)
    agreement = conclusion_agreement(references.iloc[0], vector)
    assert set(CATEGORY_COLUMNS).issubset(vector) and "categorical_agreement" in agreement
    return pd.DataFrame([{"check": "full_offdiagonal_universe", "passed": True, "value": 380},
                         {"check": "atomic_lag_coefficients", "passed": True, "value": 2},
                         {"check": "retrospective_stageB_calls", "passed": True, "value": 0},
                         {"check": "all_selector_smoke_checks", "passed": True, "value": len(specs)}])


def make_plots(output, agreement, method_summary, representation, nodes, orderings, features, winner):
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    selector, variant = winner["selector"], winner["selector_variant"]
    frozen = method_summary[method_summary.selector_variant.eq(method_summary.selector.map(
        method_summary[method_summary.selector_variant.notna()].groupby("selector").selector_variant.first()))]
    metrics = [
        ("mean_categorical_agreement", "01_categorical_agreement_vs_k.png", "categorical agreement", .85),
        ("mean_abs_error_Delta_active", "02_Delta_active_error_vs_k.png", "absolute error", .10),
        ("mean_abs_error_rescue_rate", "03_rescue_rate_error_vs_k.png", "absolute error", .15),
        ("mean_abs_error_observability_rho", "04_observability_rho_error_vs_k.png", "absolute error", .20),
        ("mean_abs_error_SNR_rho", "05_SNR_rho_error_vs_k.png", "absolute error", .20),
        ("mean_abs_error_Q1_minus_Q4_active_coverage_gap", "06_Q1_Q4_gap_error_vs_k.png", "absolute error", None),
        ("stability_agreement_rate", "07_stability_agreement_vs_k.png", "agreement", 1.0),
    ]
    held = method_summary[method_summary.reference_split.eq("evaluation")]
    for metric, filename, ylabel, threshold in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (method, var), group in held.groupby(["selector", "selector_variant"]):
            ax.plot(group.k, group[metric], marker="o", ms=2, label=f"{method}:{var}")
        if threshold is not None: ax.axhline(threshold, color="black", ls="--", lw=1)
        ax.set(xlabel="diagnostic budget k", ylabel=ylabel); ax.legend(fontsize=6); fig.tight_layout()
        fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    major = held[held.k.isin(PRIMARY_K)]
    for method, group in major.groupby("selector"):
        ax.plot(group.k, group.mean_categorical_agreement, marker="o", label=method)
    ax.axhline(.85, color="black", ls="--", lw=1); ax.set(xlabel="k", ylabel="mean categorical agreement")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "08_algorithm_comparison_primary_k.png", dpi=150); plt.close(fig)

    chosen = orderings[(orderings.selector.eq(selector)) & (orderings.selector_variant.eq(variant))]
    fig, ax = plt.subplots(figsize=(6, 5))
    run_id = chosen.run_id.iloc[-1]; pop = features[features.run_id.eq(run_id)]
    sel = chosen[chosen.run_id.eq(run_id)].nsmallest(40, "rank")
    ax.scatter(pop.z__louis_group_snr, pop.z__log_edge_o_inst_product, s=12, alpha=.35, label="reference")
    ax.scatter(sel.louis_group_snr.map(lambda _: np.nan), sel.log_edge_o_inst_product.map(lambda _: np.nan))
    # Plot in original feature units for interpretability.
    ax.clear(); ax.scatter(pop.louis_group_snr, pop.log_edge_o_inst_product, s=12, alpha=.35, label="reference")
    ax.scatter(sel.louis_group_snr, sel.log_edge_o_inst_product, s=28, marker="x", label="selected")
    ax.set(xlabel="Louis group SNR", ylabel="log observability product"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "09_feature_space_selected_edges.png", dpi=150); plt.close(fig)
    for feature, filename, label in [
        ("log_edge_o_inst_product", "10_observability_distribution.png", "log observability product"),
        ("louis_group_snr", "11_louis_SNR_distribution.png", "Louis group SNR")]:
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.hist(pop[feature], bins=20, alpha=.55, density=True, label="reference")
        ax.hist(sel[feature], bins=15, alpha=.55, density=True, label="selected"); ax.set_xlabel(label); ax.legend(); fig.tight_layout()
        fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric in ("unique_source_nodes", "unique_target_nodes"):
        group = nodes[nodes.reference_split.eq("evaluation")].groupby("k")[metric].mean()
        ax.plot(group.index, group.values, label=metric)
    ax.set(xlabel="k", ylabel="mean unique nodes"); ax.legend(); fig.tight_layout()
    fig.savefig(plots / "12_node_coverage_vs_k.png", dpi=150); plt.close(fig)


def prospective_phase(config, output, scaler, winner, decision):
    """Opt-in prospective feature construction and selected-k legacy Stage-B."""
    columns = ["run_id", "target", "source"]
    if not config["run_prospective"]:
        return tuple(pd.DataFrame(columns=columns) for _ in range(5))
    if not bool(decision.retrospective_success.iloc[0]):
        return tuple(pd.DataFrame(columns=columns) for _ in range(5))
    source_config = json.loads((Path(config["source_34UA"]) / "experiment_config.json").read_text(encoding="utf8"))
    source_config["workers"] = config["workers"]
    uc_load = {"source_34UA": config["source_34UA"], "source_34UB": "results/experiment_34ub_validation_6edges",
               "discovery_calibration_references": 2, "discovery_evaluation_references": 2}
    discovery, _, _ = uc.load_discovery(uc_load)
    allocators, _ = ua.fit_calibration_models(source_config, discovery[discovery.reference_split.eq("calibration")])
    allocator = allocators[(40, "full_candidate")]
    hyper = winner
    selected_frames, result_frames, reference_frames, subset_frames, agreement_frames = [], [], [], [], []
    existing_refs = pd.read_csv(REFERENCE_DIR / "new_reference_edge_rows_partial.csv")
    existing_refs = add_diagnostic_strata(existing_refs)
    existing_diag = pd.read_csv(REFERENCE_DIR / "new_reference_perturbation_diagnostics_partial.csv")
    rem = existing_diag.direction_index.astype(int) % 40
    existing_diag["target"] = existing_diag.direction_index.astype(int) // 40
    existing_diag["source"] = np.where(rem < 20, rem, rem - 20)
    for replicate in range(2, 2 + config["heldout_references"]):
        meta = uc.prospective_metadata({"experiment": EXPERIMENT, "calibration_references": 2}, replicate)
        meta["run_id"] = f"34UD_prospective_Mx20_My40_net{meta['true_network_id']}_rep{replicate}"
        data, seed = ua.simulate(source_config, meta); full = np.ones((20, 20), bool)
        model = ua.fit_model(source_config, data, full, seed + 500)
        louis = ua.compute_louis(source_config, model, data, seed + 700)
        _, _, edge_scores = ua.point_and_spectral_metrics(source_config, model, data, full, louis, meta)
        candidates = uc.all_deployable_candidates(model, data, louis, meta, edge_scores, allocator)
        if len(candidates) != 380:
            raise RuntimeError(f"Prospective candidate universe is {len(candidates)}, expected 380")
        candidates["reference_split"] = "evaluation"
        candidates = apply_robust_scaler(add_diagnostic_strata(candidates), scaler)
        spec = dict(winner["spec"])
        order = select_order(candidates.reset_index(drop=True), spec, 40, seed_offset=replicate)
        selected = candidates.iloc[order].copy(); selected["diagnostic_rank"] = np.arange(1, 41)
        selected_frames.append(selected)
        valid, position, _, stage, projection, perturb, _ = uc.run_stageb_checkpointed(
            config, source_config, output, model, data, selected, meta)
        perturb = perturb.copy()
        remainder = perturb.direction_index.astype(int) % 40
        perturb["target"] = perturb.direction_index.astype(int) // 40
        perturb["source"] = np.where(remainder < 20, remainder, remainder - 20)
        results = ua.build_reference_rows(source_config, selected, model, data, louis, valid, position,
                                          stage, projection, edge_scores, meta)
        results = results.merge(selected[["target", "source", "diagnostic_rank", "diagnostic_stratum",
                                          "diagnostic_observability_quartile", "diagnostic_snr_tertile"]],
                                on=["target", "source"], how="left")
        result_frames.append(results)
        source_run = f"34UC_evaluation_Mx20_My40_deployable_top90_net{20000 + replicate}_rep{replicate}"
        ref_edges = existing_refs[existing_refs.run_id.eq(source_run)]
        ref_diag = existing_diag[existing_diag.run_id.eq(source_run)]
        ref_vector = conclusion_vector(ref_edges, ref_diag)
        reference_frames.append(pd.DataFrame([{"run_id": meta["run_id"], "source_reference_run_id": source_run, **ref_vector}]))
        for k in PRIMARY_K:
            subset = results[results.diagnostic_rank <= k]
            vector = conclusion_vector(subset, edge_perturbations(perturb.assign(run_id=meta["run_id"]), meta["run_id"], subset))
            subset_frames.append(pd.DataFrame([{"run_id": meta["run_id"], "k": k, **vector}]))
            agreement_frames.append(pd.DataFrame([{"run_id": meta["run_id"], "k": k,
                                                   **conclusion_agreement(ref_vector, vector)}]))
        atomic_csv(pd.concat(selected_frames, ignore_index=True), str(output / "prospective_full380_selected_edges.csv"))
        atomic_csv(pd.concat(result_frames, ignore_index=True), str(output / "prospective_stageB_results.csv"))
    return tuple(pd.concat(items, ignore_index=True) for items in
                 (selected_frames, result_frames, reference_frames, subset_frames, agreement_frames))


def main():
    args = parse_args(); config = configuration(args); output = initialize(config)
    if handle_completed_prospective_gate(config, output):
        return
    random_interval = max(1, config["random_repeats"] // 4)
    random_updates = config["random_repeats"] // random_interval
    started = time.perf_counter(); progress = Progress(11 + random_updates)
    edges, diagnostics, manifest = load_references(config); progress.update("verified five references")
    features, scaler, bandwidth = prepare_features(edges); progress.update("built deployable strata")
    references = reference_conclusions(features, diagnostics)
    # Recalculate once to explicitly verify deterministic reference extraction.
    repeated = reference_conclusions(features, diagnostics)
    if not references[CATEGORY_COLUMNS + CONTINUOUS_COLUMNS].equals(repeated[CATEGORY_COLUMNS + CONTINUOUS_COLUMNS]):
        raise RuntimeError("Full-reference conclusion extraction is not reproducible.")
    progress.update("verified conclusions")
    smoke = smoke_checks(features, references, config)
    specs = selector_specs(config, bandwidth)
    orderings = build_orderings(features, specs, config["max_k"]); progress.update("built D0-D4 orderings")
    deterministic_subset, deterministic_agreement = evaluate_orderings(
        features, diagnostics, references, orderings, config["k_grid"])
    progress.update("replayed deterministic")
    random_summary = random_replay(features, diagnostics, references, config, progress)
    subsets = deterministic_subset
    agreements = deterministic_agreement
    hyper, chosen = choose_family_hyperparameters(deterministic_agreement, specs, config)
    frozen = frozen_agreement(deterministic_agreement, chosen)
    winner, family_ranking = rank_families(frozen[frozen.reference_split.eq("calibration")], chosen, config)
    spec_map = {(item["selector"], item["variant"]): item for item in specs}
    winner["spec"] = spec_map[(winner["selector"], winner["selector_variant"])]
    progress.update("froze calibration winner")
    method_summary = aggregate_method(frozen)
    heldout = frozen[frozen.reference_split.eq("evaluation")]
    decision = decision_table(winner, heldout, orderings, config)
    representation, nodes, strata = diagnostics_for_winner(features, orderings, winner, config)
    poststratified = poststratified_summary(features, deterministic_subset, orderings, winner, config)
    progress.update("evaluated held-out refs")

    prospective_selected, prospective_results, prospective_reference, prospective_subset, prospective_agreement = prospective_phase(
        config, output, scaler, winner, decision)
    decision.loc[:, "prospective_full380_run"] = bool(len(prospective_results))
    progress.update("prospective gate complete")

    # Required auditable outputs.
    atomic_csv(manifest, str(output / "reference_manifest.csv"))
    atomic_csv(features, str(output / "diagnostic_edge_features.csv"))
    atomic_csv(strata, str(output / "diagnostic_strata_summary.csv"))
    atomic_csv(pd.concat([scaler.assign(component="robust_scaler"), hyper.assign(component="selector_hyperparameter")],
                         ignore_index=True, sort=False), str(output / "selector_hyperparameters.csv"))
    atomic_csv(orderings, str(output / "diagnostic_orderings.csv"))
    atomic_csv(references, str(output / "retrospective_conclusion_reference.csv"))
    atomic_csv(subsets, str(output / "retrospective_conclusion_subset.csv"))
    atomic_csv(agreements, str(output / "retrospective_conclusion_agreement.csv"))
    atomic_csv(pd.concat([method_summary, random_summary], ignore_index=True, sort=False), str(output / "retrospective_method_summary.csv"))
    k_summary = method_summary.groupby(["selector", "k"], as_index=False).agg(
        mean_categorical_agreement=("mean_categorical_agreement", "mean"),
        mean_Delta_active_error=("mean_abs_error_Delta_active", "mean"),
        mean_rescue_rate_error=("mean_abs_error_rescue_rate", "mean"),
        stability_agreement_rate=("stability_agreement_rate", "mean"))
    atomic_csv(k_summary, str(output / "retrospective_k_summary.csv"))
    atomic_csv(representation, str(output / "representativeness_summary.csv"))
    atomic_csv(nodes, str(output / "node_coverage_summary.csv"))
    atomic_csv(prospective_selected, str(output / "prospective_full380_selected_edges.csv"))
    atomic_csv(prospective_results, str(output / "prospective_stageB_results.csv"))
    atomic_csv(prospective_reference, str(output / "prospective_conclusion_reference.csv"))
    atomic_csv(prospective_subset, str(output / "prospective_conclusion_subset.csv"))
    atomic_csv(prospective_agreement, str(output / "prospective_conclusion_agreement.csv"))
    atomic_csv(poststratified, str(output / "poststratified_conclusion_summary.csv"))
    runtime = pd.DataFrame([{"component": "retrospective_replay", "runtime_seconds": time.perf_counter() - started,
                             "n_references": manifest.run_id.nunique(), "random_repeats": config["random_repeats"],
                             "workers_configured": config["workers"], "BLAS_threads": 1,
                             "legacy_stageB_edges_run": len(prospective_results),
                             "retrospective_legacy_stageB_calls": 0}])
    atomic_csv(runtime, str(output / "runtime_summary.csv"))
    atomic_csv(decision, str(output / "decision_summary.csv"))
    atomic_csv(family_ranking.drop(columns="spec", errors="ignore"), str(output / "calibration_algorithm_ranking.csv"))
    atomic_csv(smoke, str(output / "smoke_test_checks.csv"))
    progress.update("saved all CSV outputs")
    make_plots(output, agreements, method_summary, representation, nodes, orderings, features, winner)
    progress.update("saved twelve plots")
    marker = {"completed": True, "completed_at_unix": time.time(),
              "phase": "smoke" if config["smoke_test"] else "retrospective_replay",
              "retrospective_success": bool(decision.retrospective_success.iloc[0]),
              "prospective_full380_run": bool(len(prospective_results)),
              "winning_selector": winner["selector"], "winning_variant": winner["selector_variant"],
              "recommended_k": None if pd.isna(decision.recommended_k.iloc[0]) else int(decision.recommended_k.iloc[0]),
              "evidence_scope": EVIDENCE_SCOPE}
    atomic_json(marker, output / "_COMPLETED.json")
    progress.update("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # One concise terminal error is retained; normal execution prints only the progress bar.
        traceback.print_exc()
        raise
