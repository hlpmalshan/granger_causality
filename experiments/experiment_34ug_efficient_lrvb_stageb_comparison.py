"""Experiment 34UG: efficient LRVB Stage-B computational comparison.

This is a method-validation experiment.  Legacy reconverged central finite
differences remain the numerical reference; finite differences are never used
inside the new implicit response operators.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle
import time
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UG_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import experiments.experiment_34u_a_fullrun_observability_calibration as ua
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls
from src.stats.exact_implicit_lrvb import (
    HybridFixedPointTangentMap,
    neumann_response,
    solve_gmres,
)
from src.stats.lrvb_stageB_smoother_feedback import global_index
from src.stats.matrixfree_lrvb_hybrid import one_update_map


OUTPUT_ROOT = Path("results/experiment_34ug_efficient_lrvb_stageb_comparison")
EPS = 1e-12
CHI2_95_DF2 = 5.991464547107979
REQUIRED_CSV = [
    "network_manifest.csv", "validation_edge_manifest.csv",
    "map_equivalence_checks.csv", "jvp_validation.csv",
    "explicit_jacobian_diagnostics.csv", "operator_diagnostics.csv",
    "legacy_stageb_reference.csv", "full_implicit_results.csv",
    "reduced_implicit_results.csv", "neumann_results.csv",
    "gmres_diagnostics.csv", "direct_vs_gmres.csv",
    "preconditioner_comparison.csv", "covariance_fidelity.csv",
    "coverage_agreement.csv", "stageb_inflation_agreement.csv",
    "runtime_per_direction.csv", "runtime_per_group.csv",
    "amortized_runtime_projection.csv", "method_summary.csv",
    "decision_summary.csv",
]


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UG_SMOKE", "0") == "1")
    root = Path(args.results_dir or os.environ.get(
        "EXPERIMENT_34UG_RESULTS_DIR",
        str(OUTPUT_ROOT / "smoke_test" if smoke else OUTPUT_ROOT)))
    mx = 4 if smoke else 20
    return {
        "experiment": "34UG_efficient_lrvb_stageb_comparison",
        "smoke_test": smoke, "results_directory": str(root),
        "M_x": mx, "M_y_values": [6, 3] if smoke else [40, 15],
        "T": 60 if smoke else 1000, "na": 2, "nb": 3,
        "C_family": "gaussian_isotropic", "C_known": True, "R_fixed_true": True,
        "B_update_mode": "free", "Q_update_mode": "diag_shrink_scalar",
        "Q_shrinkage_rho": 0.25, "Q_update_damping": 0.5,
        "include_A_posterior_uncertainty_in_Q": True,
        "A_center": "A_VB", "a0": 1e-3, "b0": 1e-3,
        "VB_MAX_ITER": env_int("EXPERIMENT_34UG_VB_MAX_ITER", 6 if smoke else 75),
        "N_FFBS_SAMPLES": env_int("EXPERIMENT_34UG_N_FFBS", 4 if smoke else 50),
        "Louis_eta": 0.7, "Louis_tau": 0.9,
        "finite_difference_epsilon": 1e-4,
        "PERTURBED_VB_MAX_ITER": env_int("EXPERIMENT_34UG_PERTURB_MAX_ITER", 3 if smoke else 30),
        "PERTURBED_RESCUE_MAX_ITER": env_int("EXPERIMENT_34UG_PERTURB_RESCUE_MAX_ITER", 5 if smoke else 50),
        "validation_groups_per_network": env_int("EXPERIMENT_34UG_GROUPS", 2 if smoke else 12),
        "stageA_network_ids": {"6" if smoke else "40": 347040, "3" if smoke else "15": 347015},
        "stageB_network_ids": {"6" if smoke else "40": 347140, "3" if smoke else "15": 347115},
        "run_stageB_after_gate": not smoke,
        "workers": max(1, env_int("EXPERIMENT_34UG_WORKERS", 1)),
        "inner_threads": max(1, env_int("EXPERIMENT_34UG_INNER_THREADS", 1)),
        "jvp_engine": "analytic_tangent", "finite_difference_JVP_deployment": False,
        "jvp_check_epsilons": [1e-3, 3e-4, 1e-4, 3e-5],
        "jvp_check_directions": 1 if smoke else 2,
        "map_equivalence_tolerance": 1e-6,
        "jvp_median_relative_error_gate": 1e-3,
        "jvp_max_relative_error_gate": 1e-2,
        "gmres_rtol_grid": [1e-6, 1e-5, 1e-4],
        "gmres_primary_rtol": 1e-5, "gmres_restart_grid": [20, 50],
        "gmres_maxiter": 20 if smoke else 100,
        "neumann_K": [0, 1, 2, 4, 8, 16],
        "legacy_is_numerical_reference": True,
        "truth_used_by_computational_method": False,
        "stageB_group_hard_cap": 48,
        "full_380_stageB_prohibited": True,
        "numba_accelerated_sufficient_statistics": True,
        "identity_preconditioner": True,
        "block_jacobi_status": "not_constructed; exact block diagonal would cost too many JVPs at n=1260",
        "todos": [
            "Do not implement full structured q_x in 34UG.",
            "Do not replace legacy Stage-B unless all numerical fidelity gates pass.",
        ],
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def atomic_pickle(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def append_rows(path, rows, keys):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    old = read_csv(path)
    result = pd.concat([old, frame], ignore_index=True) if len(old) else frame
    subset = [key for key in keys if key in result]
    if subset:
        result = result.drop_duplicates(subset=subset, keep="last")
    atomic_csv(result, str(path))


class Progress:
    """The only routine terminal output during normal execution."""
    def __init__(self, total):
        self.total = max(int(total), 1); self.done = 0; self.started = time.perf_counter()

    def update(self, label=""):
        self.done += 1; fraction = min(self.done / self.total, 1.0)
        filled = round(30 * fraction)
        eta = (time.perf_counter() - self.started) / max(self.done, 1) * max(self.total - self.done, 0)
        print(f"\r34UG [{'#'*filled}{'-'*(30-filled)}] {self.done}/{self.total} "
              f"{100*fraction:5.1f}% ETA {eta/60:6.1f}m {label[:38]}",
              end="\n" if self.done >= self.total else "", flush=True)

    def complete(self, label="complete"):
        self.done = self.total - 1
        self.update(label)


def initialize(config):
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    completed = output / "_COMPLETED.json"
    if completed.exists():
        return output, True
    path = output / "experiment_config.json"
    protected = ["smoke_test", "M_x", "M_y_values", "T", "VB_MAX_ITER",
                 "N_FFBS_SAMPLES", "validation_groups_per_network",
                 "stageA_network_ids", "stageB_network_ids", "jvp_engine"]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"Checkpoint configuration mismatch: {changed}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Non-empty 34UG directory has no compatible configuration: {output}")
    atomic_json(config, path)
    for name in ("plots", "checkpoints", "legacy_edge_checkpoints", "implicit_edge_checkpoints"):
        (output / name).mkdir(exist_ok=True)
    return output, False


def metadata(config, my, network_id, stage):
    return {"experiment_name": config["experiment"], "validation_stage": stage,
            "M_x": config["M_x"], "M_y": int(my), "T": config["T"],
            "C_family": config["C_family"], "true_network_id": int(network_id),
            "replicate_id": 0, "run_id": f"34UG_{stage}_Mx{config['M_x']}_My{my}_net{network_id}"}


def source_config(config):
    return {**config, "benchmark_group_budget": config["validation_groups_per_network"],
            "N_FREQUENCIES": 32 if config["smoke_test"] else 128,
            "ridge_penalty": 1e-3}


def baseline_path(output, run_id):
    directory = output / "checkpoints" / run_id; directory.mkdir(parents=True, exist_ok=True)
    return directory / "baseline.pkl"


def get_baseline(config, output, meta):
    path = baseline_path(output, meta["run_id"])
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    cfg = source_config(config); data, seed = ua.simulate(cfg, meta)
    full_mask = np.ones((config["M_x"], config["M_x"]), bool)
    started = time.perf_counter(); model = ua.fit_model(cfg, data, full_mask, seed + 500)
    vb_seconds = time.perf_counter() - started
    louis_started = time.perf_counter(); louis = ua.compute_louis(cfg, model, data, seed + 700)
    louis_seconds = time.perf_counter() - louis_started
    payload = {"model": model, "data": data, "louis": louis, "seed": seed,
               "baseline_fit_seconds": vb_seconds, "louis_seconds": louis_seconds}
    atomic_pickle(payload, path)
    return payload


def _rank01(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def select_validation_edges(config, model, data, louis, meta):
    path_rows = []
    J = data["C"].T @ np.linalg.solve(data["R"], data["C"])
    obs = np.diag(J); M = model.n_states
    for target in range(M):
        for source in range(M):
            if target == source:
                continue
            columns = [source, M + source]
            center = np.asarray(model.A_mean_matrices_)[:, target, source]
            vb, _ = ua.safe_covariance(model.A_row_covariances_[target][np.ix_(columns, columns)])
            lc, _ = ua.safe_covariance(louis[target][np.ix_(columns, columns)])
            path_rows.append({**meta, "target": target, "source": source,
                "louis_group_snr": float(np.sqrt(max(center @ np.linalg.pinv(lc) @ center, 0))),
                "observability_product": float(obs[target] * obs[source]),
                "vb_cov_trace": float(np.trace(vb)), "louis_cov_trace": float(np.trace(lc)),
                "cheap_need_proxy_raw": float(np.log1p(max(np.trace(lc) / max(np.trace(vb), EPS) - 1, 0))),
                "truth_used_for_selection": False})
    candidates = pd.DataFrame(path_rows)
    candidates["need_rank"] = _rank01(candidates.cheap_need_proxy_raw)
    candidates["snr_rank"] = _rank01(candidates.louis_group_snr)
    candidates["observability_rank"] = _rank01(candidates.observability_product)
    targets = [(need, snr, obsq) for need in (.1, .5, .9) for snr in (.2, .8) for obsq in (.2, .8)]
    selected = []; unused = set(candidates.index)
    for need, snr, obsq in targets:
        if len(selected) >= config["validation_groups_per_network"] or not unused:
            break
        subset = candidates.loc[list(unused)]
        distance = ((subset.need_rank - need) ** 2 + (subset.snr_rank - snr) ** 2 +
                    (subset.observability_rank - obsq) ** 2)
        index = int(distance.idxmin()); selected.append(index); unused.remove(index)
    if len(selected) < config["validation_groups_per_network"]:
        selected.extend(sorted(unused)[:config["validation_groups_per_network"] - len(selected)])
    result = candidates.loc[selected].copy().reset_index(drop=True)
    # Truth is attached only after the deployable cheap-feature selection has
    # been frozen; it is never available to the selection calculation above.
    result["true_edge"] = [bool(data["mask"][int(t), int(s)])
                           for t, s in zip(result.target, result.source)]
    result["true_group_norm"] = [float(np.linalg.norm(np.asarray(data["A"])[:, int(t), int(s)]))
                                 for t, s in zip(result.target, result.source)]
    result["benchmark_order"] = np.arange(1, len(result) + 1)
    result["selection_basis"] = "truth_free_spread_need_louisSNR_observability"
    return result


def get_edges(config, output, payload, meta):
    all_edges = read_csv(output / "validation_edge_manifest.csv")
    selected = all_edges[all_edges.run_id.eq(meta["run_id"])] if len(all_edges) else pd.DataFrame()
    if len(selected):
        return selected.sort_values("benchmark_order").reset_index(drop=True)
    selected = select_validation_edges(config, payload["model"], payload["data"], payload["louis"], meta)
    append_rows(output / "validation_edge_manifest.csv", selected.to_dict("records"),
                ["run_id", "target", "source"])
    return selected


def validate_map_and_jvp(config, output, payload, meta):
    old_map = read_csv(output / "map_equivalence_checks.csv")
    old_jvp = read_csv(output / "jvp_validation.csv")
    if (len(old_map) and len(old_jvp) and "run_id" in old_map and "run_id" in old_jvp and
            len(old_map[old_map.run_id.eq(meta["run_id"])]) >= 2 and
            len(old_jvp[old_jvp.run_id.eq(meta["run_id"])]) >= 2 * config["jvp_check_directions"] * len(config["jvp_check_epsilons"])):
        return
    model, data = payload["model"], payload["data"]
    full = HybridFixedPointTangentMap(model, data["y"], data["u"], reduced=False)
    reduced = HybridFixedPointTangentMap(model, data["y"], data["u"], reduced=True)
    reference = one_update_map(copy.deepcopy(model), data["y"], data["u"], full.theta0)
    rows = []
    for name, engine, target in (("full", full, reference),
                                 ("reduced", reduced, reference[:full.layout.A_slice.stop])):
        started = time.perf_counter(); functional = engine.map(engine.theta0)
        rows.append({**meta, "map_parameterization": name,
            "theta_dimension": len(engine.theta0),
            "relative_map_difference": float(np.linalg.norm(functional - target) / max(np.linalg.norm(target), EPS)),
            "maximum_absolute_map_difference": float(np.max(np.abs(functional - target))),
            "map_equivalence_pass": bool(np.linalg.norm(functional - target) / max(np.linalg.norm(target), EPS) <= config["map_equivalence_tolerance"]),
            "runtime_seconds": time.perf_counter() - started})
    append_rows(output / "map_equivalence_checks.csv", rows, ["run_id", "map_parameterization"])
    rng = np.random.default_rng(payload["seed"] + 3407); checks = []
    for name, engine in (("full", full), ("reduced", reduced)):
        for direction in range(config["jvp_check_directions"]):
            vector = rng.normal(size=len(engine.theta0)); vector /= np.linalg.norm(vector)
            exact = engine.jvp(vector)
            for epsilon in config["jvp_check_epsilons"]:
                plus = engine.map(engine.theta0 + epsilon * vector)
                minus = engine.map(engine.theta0 - epsilon * vector)
                fd = (plus - minus) / (2 * epsilon)
                denominator = max(np.linalg.norm(fd), EPS)
                checks.append({**meta, "map_parameterization": name, "direction": direction,
                    "epsilon": epsilon,
                    "relative_jvp_error": float(np.linalg.norm(exact - fd) / denominator),
                    "cosine_similarity": float(exact @ fd / max(np.linalg.norm(exact) * np.linalg.norm(fd), EPS)),
                    "norm_ratio": float(np.linalg.norm(exact) / denominator),
                    "jvp_engine": "analytic_tangent", "finite_difference_role": "validation_only"})
    append_rows(output / "jvp_validation.csv", checks,
                ["run_id", "map_parameterization", "direction", "epsilon"])


def validation_passes(config, output, run_ids):
    maps = read_csv(output / "map_equivalence_checks.csv")
    jvp = read_csv(output / "jvp_validation.csv")
    maps = maps[maps.run_id.isin(run_ids)]; jvp = jvp[jvp.run_id.isin(run_ids)]
    map_ok = len(maps) == 2 * len(run_ids) and maps.map_equivalence_pass.astype(bool).all()
    suitable = jvp.groupby(["run_id", "map_parameterization", "direction"], dropna=False).relative_jvp_error.min()
    jvp_ok = (len(suitable) == 2 * config["jvp_check_directions"] * len(run_ids) and
              suitable.median() <= config["jvp_median_relative_error_gate"] and
              suitable.max() <= config["jvp_max_relative_error_gate"])
    return bool(map_ok), bool(jvp_ok)


def direct_sanity(config, output):
    path = output / "direct_vs_gmres.csv"
    if len(read_csv(path)):
        return bool(read_csv(path).iloc[-1].direct_vs_gmres_pass)
    M, T = (2, 30) if config["smoke_test"] else (3, 50)
    rng = np.random.default_rng(347777)
    A = [0.42 * np.eye(M), -0.06 * np.eye(M)]
    B = [rng.normal(0, scale, (M, 1)) for scale in (.25, .12, .04)]
    C = rng.normal(0, 1 / np.sqrt(M + 1), (M + 1, M)); Q = .5 * np.eye(M); R = .6 * np.eye(M + 1)
    u = rng.normal(size=T + 50)
    data = generate_ssm_varx_p_data(A, B, u, Q, R, C=C, burn_in=50,
                                    random_seed=347778, return_augmented=True)
    model = HybridVBARDVARXSSMKnownCWithBQControls(
        2, 3, C, Q, R, B_update_mode="free", B_ridge_lambda=0.,
        estimate_Q=True, Q_update_mode="diag_shrink_scalar", Q_shrinkage_rho=.25,
        Q_update_damping=.5, include_A_posterior_uncertainty_in_Q=True,
        max_iter=5 if config["smoke_test"] else 12, tol_objective=1e-4,
        tol_A_change=1e-4, tol_B_change=1e-4, tol_Q_change=1e-4,
        tol_alpha_change=1e-4, a0=1e-3, b0=1e-3).fit(data["y"], data["u"])
    engine = HybridFixedPointTangentMap(model, data["y"], data["u"], reduced=True)
    dimension = len(engine.theta0); J = np.column_stack([engine.jvp(np.eye(dimension)[i]) for i in range(dimension)])
    rhs = engine.perturbation_rhs(global_index(0, 0, min(1, M - 1), M, 2))
    direct = np.linalg.solve(np.eye(dimension) - J, rhs)
    gmres_value, diagnostic = solve_gmres(engine, rhs, rtol=1e-8, restart=min(50, dimension), maxiter=max(100, dimension))
    error = float(np.linalg.norm(gmres_value - direct) / max(np.linalg.norm(direct), EPS))
    row = {"validation_problem": f"Mx{M}_T{T}_reduced", "theta_dimension": dimension,
           "relative_direct_vs_gmres_error": error, "solver_relative_residual": diagnostic["relative_residual"],
           "gmres_iterations": diagnostic["iterations"], "direct_vs_gmres_pass": bool(error <= 1e-6)}
    atomic_csv(pd.DataFrame([row]), str(path))
    atomic_csv(pd.DataFrame([{"validation_problem": row["validation_problem"], "jacobian_dimension": dimension,
        "finite_J": bool(np.all(np.isfinite(J))), "spectral_radius_J": float(np.max(np.abs(np.linalg.eigvals(J)))), 
        "condition_number_I_minus_J": float(np.linalg.cond(np.eye(dimension) - J))}]),
        str(output / "explicit_jacobian_diagnostics.csv"))
    return row["direct_vs_gmres_pass"]


def legacy_edge(config, output, payload, meta, target, source):
    directory = output / "legacy_edge_checkpoints" / meta["run_id"]
    directory.mkdir(parents=True, exist_ok=True); path = directory / f"edge_{target}_{source}.npz"
    json_path = path.with_suffix(".json")
    if path.exists() and json_path.exists():
        with np.load(path) as item: covariance = item["covariance"]
        return covariance, json.loads(json_path.read_text(encoding="utf8"))
    model, data = payload["model"], payload["data"]; M = model.n_states
    indices = [global_index(target, lag, source, M, 2) for lag in range(2)]
    columns = {}; diagnostics = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(config["workers"], 2)) as pool:
        futures = {pool.submit(ua.perturb_one, (model, data["y"], data["u"], index,
                    config["PERTURBED_VB_MAX_ITER"], config["PERTURBED_RESCUE_MAX_ITER"])): index
                   for index in indices}
        for future in as_completed(futures):
            index, column, diagnostic = future.result()
            columns[index] = column; diagnostics.append(diagnostic)
    raw = np.column_stack([columns[index] for index in indices])[np.ix_(indices, [0, 1])]
    covariance, projection = ua.safe_covariance(raw)
    runtime = time.perf_counter() - started
    row = {**meta, "target": target, "source": source, "method": "L0_legacy",
           "runtime_seconds": runtime, "runtime_per_direction_seconds": runtime / 2,
           "perturbed_vb_fits": 4, "smoother_calls_proxy": int(sum(
               int(d.get("plus_n_iter", 0)) + int(d.get("minus_n_iter", 0)) for d in diagnostics)),
           "both_directions_converged": bool(all(d.get("plus_converged") and d.get("minus_converged") for d in diagnostics)),
           **projection}
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle: np.savez(handle, covariance=covariance, raw=raw)
    os.replace(temporary, path); atomic_json(row, json_path)
    return covariance, row


def covariance_columns(block):
    block = np.asarray(block, float)
    return {"cov_00": block[0, 0], "cov_01": block[0, 1],
            "cov_10": block[1, 0], "cov_11": block[1, 1]}


def implicit_edge(config, output, engine, meta, target, source, parameterization, settings):
    method = "I2_full_exact_implicit_GMRES" if parameterization == "full" else "I3_reduced_exact_implicit_GMRES"
    directory = output / "implicit_edge_checkpoints" / meta["run_id"] / parameterization
    directory.mkdir(parents=True, exist_ok=True); path = directory / f"edge_{target}_{source}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    M = engine.layout.n_states
    indices = [global_index(target, lag, source, M, 2) for lag in range(2)]
    columns, diagnostics = [], []; started = time.perf_counter()
    for lag, index in enumerate(indices):
        rhs_started = time.perf_counter(); rhs = engine.perturbation_rhs(index)
        solution, diagnostic = solve_gmres(engine, rhs, rtol=settings["rtol"],
                                            restart=settings["restart"], maxiter=config["gmres_maxiter"])
        columns.append(solution[:engine.layout.A_slice.stop][indices])
        diagnostics.append({**meta, "method": method, "target": target, "source": source,
            "lag": lag + 1, "direction_index": index, "rhs_setup_seconds": time.perf_counter() - rhs_started - diagnostic["runtime_seconds"],
            **{key: value for key, value in diagnostic.items() if key != "residual_history"},
            "residual_history": json.dumps(diagnostic["residual_history"])})
    raw = np.column_stack(columns); covariance, projection = ua.safe_covariance(raw)
    row = {**meta, "method": method, "map_parameterization": parameterization,
           "target": target, "source": source, "runtime_seconds": time.perf_counter() - started,
           "setup_seconds": 0.0, "jvp_calls": int(sum(item["jvp_calls"] for item in diagnostics)),
           "gmres_iterations": int(sum(item["iterations"] for item in diagnostics)),
           "converged": bool(all(item["converged"] for item in diagnostics)),
           "relative_residual_max": float(max(item["relative_residual"] for item in diagnostics)),
           "rtol": settings["rtol"], "restart": settings["restart"],
           **covariance_columns(covariance), **projection, "gmres_rows": diagnostics}
    atomic_json(row, path)
    return row


def tune_solver(config, output, engine, meta, edge):
    path = output / "solver_settings.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    index = global_index(int(edge.target), 0, int(edge.source), engine.layout.n_states, 2)
    rhs = engine.perturbation_rhs(index); rows = []
    for rtol in config["gmres_rtol_grid"]:
        for restart in config["gmres_restart_grid"]:
            _, diagnostic = solve_gmres(engine, rhs, rtol=rtol, restart=restart,
                                        maxiter=config["gmres_maxiter"])
            rows.append({**meta, "method": "I2_full_exact_implicit_GMRES",
                "preconditioner": "P0_identity", "rtol": rtol, "restart": restart,
                **{key: value for key, value in diagnostic.items() if key != "residual_history"}})
    append_rows(output / "preconditioner_comparison.csv", rows,
                ["run_id", "method", "preconditioner", "rtol", "restart"])
    primary = [row for row in rows if row["rtol"] == config["gmres_primary_rtol"]]
    passing = [row for row in primary if row["converged"]]
    chosen = (min(passing, key=lambda row: row["runtime_seconds"]) if passing else
              min(primary, key=lambda row: row["relative_residual"]))
    settings = {"rtol": config["gmres_primary_rtol"], "restart": int(chosen["restart"]),
                "maxiter": config["gmres_maxiter"], "tuned_on_run_id": meta["run_id"]}
    atomic_json(settings, path); return settings


def operator_diagnostics(config, output, engine, meta, parameterization):
    existing = read_csv(output / "operator_diagnostics.csv")
    if (len(existing) and "run_id" in existing and "map_parameterization" in existing and
            len(existing[(existing.run_id.eq(meta["run_id"])) & existing.map_parameterization.eq(parameterization)])):
        return
    rng = np.random.default_rng(340700 + meta["M_y"]); u = rng.normal(size=len(engine.theta0)); v = rng.normal(size=len(engine.theta0))
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    started = time.perf_counter(); Ju, Jv = engine.jvp(u), engine.jvp(v)
    asym = abs(u @ Jv - v @ Ju) / max(abs(u @ Jv), abs(v @ Ju), EPS)
    power = v.copy(); radii = []
    for _ in range(3 if config["smoke_test"] else 6):
        next_value = engine.jvp(power); norm = np.linalg.norm(next_value)
        radii.append(float(norm)); power = next_value / max(norm, EPS)
    append_rows(output / "operator_diagnostics.csv", [{**meta,
        "map_parameterization": parameterization, "theta_dimension": len(engine.theta0),
        "operator_nonsymmetric": True, "bilinear_asymmetry": float(asym),
        "spectral_radius_power_estimate": radii[-1], "power_norm_sequence": json.dumps(radii),
        "jvp_calls": engine.jvp_calls, "mean_time_per_jvp": engine.elapsed_jvp_seconds / max(engine.jvp_calls, 1),
        "diagnostic_runtime_seconds": time.perf_counter() - started}],
        ["run_id", "map_parameterization"])


def run_core_network(config, output, payload, meta, edges, parameterization, settings, progress):
    model, data = payload["model"], payload["data"]
    engine = HybridFixedPointTangentMap(model, data["y"], data["u"], reduced=parameterization == "reduced")
    operator_diagnostics(config, output, engine, meta, parameterization)
    for edge in edges.itertuples(index=False):
        target, source = int(edge.target), int(edge.source)
        if parameterization == "full":
            legacy, legacy_row = legacy_edge(config, output, payload, meta, target, source)
            legacy_row.update(covariance_columns(legacy)); append_rows(
                output / "legacy_stageb_reference.csv", [legacy_row], ["run_id", "target", "source"])
            append_rows(output / "runtime_per_direction.csv", [{**meta, "method": "L0_legacy",
                "target": target, "source": source, "direction": lag + 1,
                "runtime_seconds": legacy_row["runtime_per_direction_seconds"], "JVP_calls": 0,
                "perturbed_VB_fits": 2} for lag in range(2)],
                ["run_id", "method", "target", "source", "direction"])
        row = implicit_edge(config, output, engine, meta, target, source, parameterization, settings)
        gmres_rows = row.pop("gmres_rows", [])
        destination = "full_implicit_results.csv" if parameterization == "full" else "reduced_implicit_results.csv"
        append_rows(output / destination, [row], ["run_id", "method", "target", "source"])
        append_rows(output / "gmres_diagnostics.csv", gmres_rows,
                    ["run_id", "method", "target", "source", "lag"])
        append_rows(output / "runtime_per_direction.csv", [{**item,
            "JVP_calls": item["jvp_calls"], "perturbed_VB_fits": 0} for item in gmres_rows],
            ["run_id", "method", "target", "source", "lag"])
        progress.update(f"{meta['validation_stage']} My={meta['M_y']} {parameterization}")


def run_neumann_network(config, output, payload, meta, edges, parameterization, progress):
    model, data = payload["model"], payload["data"]
    engine = HybridFixedPointTangentMap(model, data["y"], data["u"], reduced=parameterization == "reduced")
    existing = read_csv(output / "neumann_results.csv")
    for edge in edges.itertuples(index=False):
        target, source = int(edge.target), int(edge.source)
        have = (existing[(existing.run_id.eq(meta["run_id"])) & (existing.target.eq(target)) &
                         (existing.source.eq(source)) & (existing.map_parameterization.eq(parameterization))]
                if len(existing) and all(name in existing for name in
                    ("run_id", "target", "source", "map_parameterization")) else pd.DataFrame())
        if len(have) >= len(config["neumann_K"]):
            progress.update(f"resume Neumann {parameterization}"); continue
        indices = [global_index(target, lag, source, model.n_states, 2) for lag in range(2)]
        by_lag = []; started = time.perf_counter(); before = engine.jvp_calls
        for index in indices:
            rhs = engine.perturbation_rhs(index)
            by_lag.append({k: (value, term_norm) for k, value, term_norm in
                           neumann_response(engine, rhs, config["neumann_K"])})
        rows = []
        for K in config["neumann_K"]:
            if K not in by_lag[0] or K not in by_lag[1]:
                continue
            block = np.column_stack([by_lag[lag][K][0][:engine.layout.A_slice.stop][indices] for lag in range(2)])
            covariance, projection = ua.safe_covariance(block)
            term_norms = [by_lag[lag][K][1] for lag in range(2)]
            rows.append({**meta, "method": f"N1_{parameterization}_K{K}",
                "map_parameterization": parameterization, "target": target, "source": source, "K": K,
                "runtime_seconds": time.perf_counter() - started,
                "jvp_calls": engine.jvp_calls - before, "neumann_term_norm_max": max(term_norms),
                "neumann_sequence_finite": bool(np.all(np.isfinite(term_norms))),
                **covariance_columns(covariance), **projection})
        append_rows(output / "neumann_results.csv", rows,
                    ["run_id", "map_parameterization", "target", "source", "K"])
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        progress.update(f"{meta['validation_stage']} My={meta['M_y']} Neumann {parameterization}")


def block_from_row(row):
    return np.asarray([[row.cov_00, row.cov_01], [row.cov_10, row.cov_11]], float)


def compute_fidelity(output, manifests):
    legacy = read_csv(output / "legacy_stageb_reference.csv")
    methods = pd.concat([read_csv(output / "full_implicit_results.csv"),
                         read_csv(output / "reduced_implicit_results.csv"),
                         read_csv(output / "neumann_results.csv")], ignore_index=True)
    edge_manifest = read_csv(output / "validation_edge_manifest.csv")
    rows = []; coverage = []
    if not len(legacy) or not len(methods):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for item in methods.itertuples(index=False):
        ref = legacy[(legacy.run_id.eq(item.run_id)) & legacy.target.eq(item.target) & legacy.source.eq(item.source)]
        edge = edge_manifest[(edge_manifest.run_id.eq(item.run_id)) & edge_manifest.target.eq(item.target) & edge_manifest.source.eq(item.source)]
        if not len(ref) or not len(edge):
            continue
        L, S = block_from_row(ref.iloc[-1]), block_from_row(item)
        L, _ = ua.safe_covariance(L); S, diag = ua.safe_covariance(S)
        lsign, llog = np.linalg.slogdet(L); ssign, slog = np.linalg.slogdet(S)
        truth = np.asarray([edge.iloc[-1].true_group_norm, 0.0])  # replaced below when available
        # Reconstructing coefficient truth is not necessary for covariance fidelity;
        # coverage is populated by the manifest coefficient columns when present.
        center = np.asarray([edge.iloc[-1].get("center_lag1", np.nan), edge.iloc[-1].get("center_lag2", np.nan)])
        truth = np.asarray([edge.iloc[-1].get("truth_lag1", np.nan), edge.iloc[-1].get("truth_lag2", np.nan)])
        def covered(cov):
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(truth)): return np.nan, np.nan
            delta = center - truth; statistic = float(delta @ np.linalg.pinv(cov) @ delta)
            return statistic, bool(statistic <= CHI2_95_DF2)
        dL, cL = covered(L); dS, cS = covered(S)
        louis_trace = float(edge.iloc[-1].louis_cov_trace)
        rows.append({"run_id": item.run_id, "validation_stage": item.validation_stage,
            "M_y": item.M_y, "target": item.target, "source": item.source, "method": item.method,
            "relative_frobenius_error": float(np.linalg.norm(S - L) / max(np.linalg.norm(L), EPS)),
            "trace_relative_error": float(abs(np.trace(S) - np.trace(L)) / max(abs(np.trace(L)), EPS)),
            "logdet_absolute_error": float(abs(slog - llog)) if lsign > 0 and ssign > 0 else np.nan,
            "legacy_trace": float(np.trace(L)), "method_trace": float(np.trace(S)),
            "legacy_relative_stageB_inflation": float((np.trace(L) - louis_trace) / max(louis_trace, EPS)),
            "method_relative_stageB_inflation": float((np.trace(S) - louis_trace) / max(louis_trace, EPS)),
            "ellipse_area_relative_error": float(abs(np.sqrt(np.linalg.det(S)) - np.sqrt(np.linalg.det(L))) / max(np.sqrt(np.linalg.det(L)), EPS)),
            "finite_covariance": bool(np.all(np.isfinite(S))), "symmetric_covariance": bool(np.allclose(S, S.T)),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(S).min()), "PSD": bool(np.linalg.eigvalsh(S).min() >= -1e-12),
            "condition_number": float(np.linalg.cond(S)), "legacy_D2": dL, "method_D2": dS,
            "legacy_covered": cL, "method_covered": cS,
            "coverage_decision_agreement": bool(cL == cS) if pd.notna(cL) and pd.notna(cS) else np.nan})
        coverage.append({**rows[-1]})
    fidelity = pd.DataFrame(rows); coverage = pd.DataFrame(coverage)
    inflation = []
    if len(fidelity):
        for (method, my), group in fidelity.groupby(["method", "M_y"]):
            if len(group) > 1:
                pearson = pearsonr(group.legacy_relative_stageB_inflation, group.method_relative_stageB_inflation).statistic
                spearman = spearmanr(group.legacy_relative_stageB_inflation, group.method_relative_stageB_inflation).statistic
            else: pearson = spearman = np.nan
            inflation.append({"method": method, "M_y": my, "n_groups": len(group),
                "stageB_inflation_Pearson": pearson, "stageB_inflation_Spearman": spearman})
    return fidelity, coverage, pd.DataFrame(inflation)


def method_summary(fidelity):
    if not len(fidelity): return pd.DataFrame()
    rows = []
    groups = [(method, "all", group) for method, group in fidelity.groupby("method")]
    groups += [(method, my, group) for (method, my), group in fidelity.groupby(["method", "M_y"])]
    for method, my, group in groups:
        labelled = group.coverage_decision_agreement.dropna()
        row = {"method": method, "M_y": my, "n_groups": len(group),
            "finite_fraction": group.finite_covariance.mean(), "PSD_fraction": group.PSD.mean(),
            "median_relative_frobenius_error": group.relative_frobenius_error.median(),
            "p90_relative_frobenius_error": group.relative_frobenius_error.quantile(.9),
            "maximum_relative_frobenius_error": group.relative_frobenius_error.max(),
            "median_trace_relative_error": group.trace_relative_error.median(),
            "maximum_trace_relative_error": group.trace_relative_error.max(),
            "coverage_decision_agreement": labelled.mean() if len(labelled) else np.nan}
        row["numerically_valid_without_inflation_gate"] = bool(
            row["finite_fraction"] == 1 and row["PSD_fraction"] == 1 and
            row["median_relative_frobenius_error"] <= .05 and row["p90_relative_frobenius_error"] <= .10 and
            row["median_trace_relative_error"] <= .05 and
            (pd.isna(row["coverage_decision_agreement"]) or row["coverage_decision_agreement"] >= .95))
        rows.append(row)
    return pd.DataFrame(rows)


def full_gate(summary, inflation, method_prefix):
    candidates = summary[(summary.method.str.startswith(method_prefix)) & summary.M_y.astype(str).eq("all")] if len(summary) else pd.DataFrame()
    if not len(candidates): return False
    row = candidates.iloc[0]
    correlations = inflation[inflation.method.eq(row.method)].stageB_inflation_Spearman.dropna()
    correlation_ok = len(correlations) > 0 and correlations.min() >= .95
    return bool(row.numerically_valid_without_inflation_gate and correlation_ok)


def add_truth_coefficients(edges, payload):
    result = edges.copy(); Ahat = np.asarray(payload["model"].A_mean_matrices_); Atrue = np.asarray(payload["data"]["A"])
    result["center_lag1"] = [Ahat[0, int(t), int(s)] for t, s in zip(result.target, result.source)]
    result["center_lag2"] = [Ahat[1, int(t), int(s)] for t, s in zip(result.target, result.source)]
    result["truth_lag1"] = [Atrue[0, int(t), int(s)] for t, s in zip(result.target, result.source)]
    result["truth_lag2"] = [Atrue[1, int(t), int(s)] for t, s in zip(result.target, result.source)]
    return result


def update_manifest(output, rows):
    append_rows(output / "network_manifest.csv", rows, ["run_id"])


def save_runtime_summaries(output):
    legacy = read_csv(output / "legacy_stageb_reference.csv")
    full = read_csv(output / "full_implicit_results.csv"); reduced = read_csv(output / "reduced_implicit_results.csv")
    frames = [frame for frame in (legacy, full, reduced) if len(frame)]
    groups = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(groups): atomic_csv(groups, str(output / "runtime_per_group.csv"))
    projections = []
    if len(legacy):
        legacy_group = legacy.runtime_seconds.median()
        for method_frame in (full, reduced):
            if not len(method_frame): continue
            method = method_frame.method.iloc[0]; steady = method_frame.runtime_seconds.median(); setup = 0.0
            for N in (1, 10, 40, 63, 90, 380):
                implicit = setup + N * steady; reference = N * legacy_group
                projections.append({"method": method, "N_groups": N, "setup_seconds": setup,
                    "steady_group_seconds": steady, "projected_runtime_seconds": implicit,
                    "legacy_projected_runtime_seconds": reference, "projected_speedup": reference / max(implicit, EPS),
                    "projection_valid": True})
    atomic_csv(pd.DataFrame(projections), str(output / "amortized_runtime_projection.csv"))


def finalize(config, output, stopped_at=None):
    fidelity, coverage, inflation = compute_fidelity(output, read_csv(output / "network_manifest.csv"))
    atomic_csv(fidelity, str(output / "covariance_fidelity.csv")); atomic_csv(coverage, str(output / "coverage_agreement.csv"))
    atomic_csv(inflation, str(output / "stageb_inflation_agreement.csv"))
    summary = method_summary(fidelity); atomic_csv(summary, str(output / "method_summary.csv"))
    save_runtime_summaries(output)
    maps = read_csv(output / "map_equivalence_checks.csv"); jvp = read_csv(output / "jvp_validation.csv")
    direct = read_csv(output / "direct_vs_gmres.csv")
    map_pass = bool(len(maps) and maps.map_equivalence_pass.astype(bool).all())
    suitable = jvp.groupby(["run_id", "map_parameterization", "direction"]).relative_jvp_error.min() if len(jvp) else pd.Series(dtype=float)
    jvp_pass = bool(len(suitable) and suitable.median() <= config["jvp_median_relative_error_gate"] and suitable.max() <= config["jvp_max_relative_error_gate"])
    direct_pass = bool(len(direct) and direct.direct_vs_gmres_pass.astype(bool).all())
    full_valid = full_gate(summary, inflation, "I2_"); reduced_valid = full_gate(summary, inflation, "I3_")
    valid_neumann = summary[(summary.method.str.startswith("N1_")) & summary.M_y.astype(str).eq("all") & summary.numerically_valid_without_inflation_gate.astype(bool)] if len(summary) else pd.DataFrame()
    best_neumann = int(valid_neumann.method.str.extract(r"K(\d+)")[0].astype(float).min()) if len(valid_neumann) else np.nan
    preferred = "none"
    if reduced_valid: preferred = "I3_reduced_exact_implicit_GMRES"
    elif full_valid: preferred = "I2_full_exact_implicit_GMRES"
    if len(valid_neumann): preferred = valid_neumann.sort_values("method").iloc[0].method
    projection = read_csv(output / "amortized_runtime_projection.csv")
    def projected(N, field="projected_speedup"):
        q = projection[(projection.method.eq(preferred)) & projection.N_groups.eq(N)] if len(projection) else pd.DataFrame()
        return q.iloc[0][field] if len(q) else np.nan
    def regime(my, field):
        q = fidelity[(fidelity.M_y.eq(my)) & fidelity.method.eq(preferred)] if len(fidelity) else pd.DataFrame()
        return q[field].median() if len(q) else np.nan
    decision = pd.DataFrame([{"map_equivalence_pass": map_pass, "jvp_validation_pass": jvp_pass,
        "direct_vs_gmres_pass": direct_pass, "full_implicit_valid": full_valid,
        "reduced_implicit_valid": reduced_valid, "best_neumann_K": best_neumann,
        "neumann_valid": bool(len(valid_neumann)), "preferred_method": preferred,
        "My40_median_frobenius_error": regime(40, "relative_frobenius_error"),
        "My15_median_frobenius_error": regime(15, "relative_frobenius_error"),
        "My40_coverage_agreement": regime(40, "coverage_decision_agreement"),
        "My15_coverage_agreement": regime(15, "coverage_decision_agreement"),
        "preferred_method_speedup_first_group": projected(1),
        "preferred_method_speedup_10_groups": projected(10),
        "preferred_method_speedup_40_groups": projected(40),
        "preferred_method_speedup_90_groups": projected(90),
        "preferred_method_speedup_380_groups": projected(380),
        "projected_runtime_90_groups": projected(90, "projected_runtime_seconds"),
        "projected_runtime_380_groups": projected(380, "projected_runtime_seconds"),
        "ready_to_replace_legacy_stageB": bool(map_pass and jvp_pass and direct_pass and preferred != "none"),
        "stopped_at": stopped_at or "completed_all_enabled_stages",
        "recommended_next_experiment": "replace legacy only after independent Stage-B confirmation" if preferred != "none" else "inspect the first failed map/JVP/solver/fidelity layer"}])
    atomic_csv(decision, str(output / "decision_summary.csv"))
    for name in REQUIRED_CSV:
        path = output / name
        if not path.exists(): atomic_csv(pd.DataFrame(), str(path))
    make_plots(output, fidelity, coverage, inflation, read_csv(output / "gmres_diagnostics.csv"),
               read_csv(output / "neumann_results.csv"), jvp, projection)
    atomic_json({"completed": True, "stopped_at": stopped_at,
                 "ready_to_replace_legacy_stageB": bool(decision.iloc[0].ready_to_replace_legacy_stageB),
                 "completed_at": time.time()}, output / "_COMPLETED.json")
    failed_marker = output / "_FAILED.json"
    if failed_marker.exists():
        failed_marker.unlink()


def make_plots(output, fidelity, coverage, inflation, gmres, neumann, jvp, projection):
    plots = output / "plots"
    def save(name, draw):
        fig, ax = plt.subplots(); draw(ax); fig.tight_layout(); fig.savefig(plots / name); plt.close(fig)
    def box(metric, ylabel):
        def draw(ax):
            if len(fidelity):
                groups = [(name, group[metric].dropna()) for name, group in fidelity.groupby("method")]
                if groups: ax.boxplot([g[1] for g in groups], tick_labels=[g[0] for g in groups]); ax.tick_params(axis="x", rotation=75)
            ax.set_ylabel(ylabel)
        return draw
    save("01_covariance_frobenius_error.png", box("relative_frobenius_error", "relative Frobenius error"))
    save("02_trace_error.png", box("trace_relative_error", "trace relative error"))
    save("03_legacy_vs_implicit_inflation.png", lambda ax: (ax.scatter(fidelity.legacy_trace, fidelity.method_trace, s=12) if len(fidelity) else None, ax.set(xlabel="legacy trace", ylabel="implicit trace")))
    save("04_coverage_agreement.png", lambda ax: (coverage.groupby("method").coverage_decision_agreement.mean().plot.bar(ax=ax) if len(coverage) else None, ax.set_ylabel("coverage agreement")))
    def gmres_plot(ax):
        if len(gmres):
            for row in gmres.itertuples():
                try: values = json.loads(row.residual_history)
                except Exception: values = []
                ax.semilogy(np.arange(1, len(values)+1), values, alpha=.3)
        ax.set(xlabel="iteration", ylabel="preconditioned residual")
    save("05_gmres_convergence.png", gmres_plot)
    save("06_neumann_error_vs_K.png", lambda ax: (fidelity[fidelity.method.str.startswith("N1_")].assign(K=lambda x:x.method.str.extract(r"K(\d+)").astype(float)).groupby("K").relative_frobenius_error.median().plot(ax=ax, marker="o") if len(fidelity) else None, ax.set(xlabel="K", ylabel="median covariance error")))
    save("07_jvp_validation_error.png", lambda ax: (ax.loglog(jvp.epsilon, jvp.relative_jvp_error, "o") if len(jvp) else None, ax.set(xlabel="FD epsilon (validation only)", ylabel="relative JVP error")))
    runtime = read_csv(output / "runtime_per_group.csv")
    save("08_runtime_per_edge_group.png", lambda ax: (runtime.groupby("method").runtime_seconds.median().plot.bar(ax=ax) if len(runtime) else None, ax.set_ylabel("seconds")))
    save("09_amortized_speedup.png", lambda ax: ([ax.plot(g.N_groups, g.projected_speedup, marker="o", label=m) for m,g in projection.groupby("method")] if len(projection) else None, ax.set(xlabel="groups", ylabel="speedup"), ax.legend(fontsize=6) if len(projection) else None))
    save("10_My40_vs_My15_fidelity.png", lambda ax: ([ax.boxplot([g.relative_frobenius_error for _,g in fidelity.groupby("M_y")], tick_labels=[str(k) for k,_ in fidelity.groupby("M_y")])] if len(fidelity) else None, ax.set(xlabel="My", ylabel="relative Frobenius error")))
    save("11_full_vs_reduced.png", box("trace_relative_error", "full/reduced trace error"))
    save("12_cost_fidelity_pareto.png", lambda ax: (ax.scatter(runtime.runtime_seconds, [fidelity[fidelity.method.eq(m)].relative_frobenius_error.median() for m in runtime.method]) if len(runtime) and len(fidelity) else None, ax.set(xlabel="seconds/group", ylabel="median fidelity error")))


def main():
    args = parse_args(); config = configuration(args); output, completed = initialize(config)
    if completed:
        Progress(1).update("already complete"); return
    stageA = [metadata(config, int(my), config["stageA_network_ids"][str(my)], "stageA") for my in config["M_y_values"]]
    stageB = [metadata(config, int(my), config["stageB_network_ids"][str(my)], "stageB") for my in config["M_y_values"]]
    total = 1 + len(stageA) * (1 + 4 * config["validation_groups_per_network"])
    if config["run_stageB_after_gate"]: total += len(stageB) * (1 + 4 * config["validation_groups_per_network"])
    progress = Progress(total)
    try:
        direct_ok = direct_sanity(config, output); progress.update("direct solve sanity")
        if not direct_ok:
            finalize(config, output, "direct_vs_gmres"); progress.complete("stopped: direct sanity"); return
        payloads = {}; edges_by_run = {}; manifest_rows = []
        for meta in stageA:
            payload = get_baseline(config, output, meta); payloads[meta["run_id"]] = payload
            edges = add_truth_coefficients(get_edges(config, output, payload, meta), payload)
            append_rows(output / "validation_edge_manifest.csv", edges.to_dict("records"), ["run_id", "target", "source"])
            edges_by_run[meta["run_id"]] = edges
            validate_map_and_jvp(config, output, payload, meta); progress.update(f"validate My={meta['M_y']}")
            manifest_rows.append({**meta, "theta_full_dimension": len(HybridFixedPointTangentMap(payload["model"], payload["data"]["y"], payload["data"]["u"]).theta0),
                "theta_reduced_dimension": config["M_x"] * config["na"] * config["M_x"],
                "selected_groups": len(edges), "baseline_fit_seconds": payload["baseline_fit_seconds"],
                "louis_seconds": payload["louis_seconds"], "completion_status": "validated"})
        update_manifest(output, manifest_rows)
        map_ok, jvp_ok = validation_passes(config, output, [m["run_id"] for m in stageA])
        if not map_ok or not jvp_ok:
            finalize(config, output, "map_equivalence" if not map_ok else "jvp_validation"); progress.complete("stopped: validation gate"); return
        first_meta = stageA[0]; first_payload = payloads[first_meta["run_id"]]
        first_engine = HybridFixedPointTangentMap(first_payload["model"], first_payload["data"]["y"], first_payload["data"]["u"])
        settings = tune_solver(config, output, first_engine, first_meta, edges_by_run[first_meta["run_id"]].iloc[0])
        for meta in stageA:
            run_core_network(config, output, payloads[meta["run_id"]], meta,
                             edges_by_run[meta["run_id"]], "full", settings, progress)
        fidelity, coverage, inflation = compute_fidelity(output, read_csv(output / "network_manifest.csv"))
        summary = method_summary(fidelity); full_ok = full_gate(summary, inflation, "I2_")
        if config["smoke_test"]:
            full_ok = True  # pipeline coverage only; scientific decision still uses the real gate
        if not full_ok:
            finalize(config, output, "full_implicit_legacy_fidelity"); progress.complete("stopped: I2 fidelity"); return
        for meta in stageA:
            run_core_network(config, output, payloads[meta["run_id"]], meta,
                             edges_by_run[meta["run_id"]], "reduced", settings, progress)
        fidelity, coverage, inflation = compute_fidelity(output, read_csv(output / "network_manifest.csv"))
        reduced_ok = full_gate(method_summary(fidelity), inflation, "I3_")
        for meta in stageA:
            for parameterization in ("full", "reduced"):
                run_neumann_network(config, output, payloads[meta["run_id"]], meta,
                                     edges_by_run[meta["run_id"]], parameterization, progress)
        if config["run_stageB_after_gate"] and (full_ok or reduced_ok):
            for meta in stageB:
                payload = get_baseline(config, output, meta); edges = add_truth_coefficients(get_edges(config, output, payload, meta), payload)
                append_rows(output / "validation_edge_manifest.csv", edges.to_dict("records"), ["run_id", "target", "source"])
                validate_map_and_jvp(config, output, payload, meta); progress.update(f"validate My={meta['M_y']} stageB")
                stage_map_ok, stage_jvp_ok = validation_passes(config, output, [meta["run_id"]])
                if not stage_map_ok or not stage_jvp_ok:
                    finalize(config, output, "stageB_map_or_jvp_validation"); progress.complete("stopped: Stage-B validation"); return
                update_manifest(output, [{**meta, "theta_full_dimension": len(HybridFixedPointTangentMap(payload["model"], payload["data"]["y"], payload["data"]["u"]).theta0),
                    "theta_reduced_dimension": config["M_x"] * config["na"] * config["M_x"],
                    "selected_groups": len(edges), "baseline_fit_seconds": payload["baseline_fit_seconds"],
                    "louis_seconds": payload["louis_seconds"], "completion_status": "validated"}])
                run_core_network(config, output, payload, meta, edges, "full", settings, progress)
                run_core_network(config, output, payload, meta, edges, "reduced", settings, progress)
                run_neumann_network(config, output, payload, meta, edges, "full", progress)
                run_neumann_network(config, output, payload, meta, edges, "reduced", progress)
        finalize(config, output)
        progress.complete("complete")
    except Exception as error:
        atomic_json({"completed": False, "error_type": type(error).__name__,
                     "error_message": str(error), "traceback": traceback.format_exc(),
                     "failed_at": time.time()}, output / "_FAILED.json")
        raise


if __name__ == "__main__":
    main()
