"""Experiment 34UI: mode-resolved diagnosis of ill-conditioned LRVB.

This experiment is diagnostic.  It reuses the completed 34UH Jacobians and
right-hand sides, audits finite-perturbation convergence, and only then tests
fixed spectral regularizations.  It does not define or select a deployable
LRVB estimator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
import warnings

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UI_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import spearmanr

import experiments.experiment_34ug_efficient_lrvb_stageb_comparison as ug
import experiments.experiment_34uh_explicit_factorized_lrvb as uh
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.exact_implicit_lrvb import HybridFixedPointTangentMap
from src.stats.factorized_implicit_lrvb import assemble_exact_jacobian
from src.stats.lrvb_stageB_smoother_feedback import (
    fit_perturbed_hybrid_vb, global_index,
)


OUTPUT_ROOT = Path("results/experiment_34ui_lrvb_mode_diagnosis")
SOURCE_UH = Path("results/experiment_34uh_explicit_factorized_lrvb")
SOURCE_UG = Path("results/experiment_34ug_efficient_lrvb_stageb_comparison")
EPS = 1e-15
CHI2_95_DF2 = 5.991464547107979
RELATIVE_THRESHOLDS = [10.0**power for power in range(-14, -5)]
BOTTOM_COUNTS = [1, 2, 5, 10, 20, 50]
HORIZONS = [1, 2, 5, 10, 20, 40]
ITERATION_CHECKPOINTS = [30, 50, 75, 100, 150]
REGULARIZATION_GRID = RELATIVE_THRESHOLDS
GAMMA_GRID = [.90, .95, .98, .99, .995, .999, 1.0]
REQUIRED = [
    "experiment_config.json", "source_34uh_manifest.csv",
    "C_rank_diagnostics.csv", "C_singular_values.csv",
    "dynamic_observability_summary.csv", "dynamic_observability_spectrum.csv",
    "lrvb_singular_values.csv", "lrvb_near_null_modes.csv",
    "singular_mode_block_energy.csv",
    "singular_mode_A_source_target_energy.csv",
    "singular_mode_observability_alignment.csv", "block_conditioning.csv",
    "schur_complement_diagnostics.csv", "nested_block_ablation.csv",
    "rhs_singular_mode_excitation.csv", "rhs_effective_conditioning.csv",
    "legacy_convergence_iteration_trace.csv",
    "legacy_convergence_covariance.csv", "legacy_convergence_summary.csv",
    "converged_epsilon_sensitivity.csv", "A_only_direct_results.csv",
    "regularized_tsvd_results.csv", "regularized_tikhonov_results.csv",
    "regularized_feedback_damping.csv", "runtime_summary.csv",
    "decision_summary.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a short convergence audit while retaining the real saved matrices.")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UI_SMOKE", "0") == "1")
    output = Path(args.results_dir or os.environ.get(
        "EXPERIMENT_34UI_RESULTS_DIR",
        str(OUTPUT_ROOT / "smoke_test" if smoke else OUTPUT_ROOT)))
    return {
        "experiment": "34UI_lrvb_mode_diagnosis",
        "smoke_test": smoke,
        "results_directory": str(output),
        "source_34UH": str(SOURCE_UH),
        "source_34UG": str(SOURCE_UG),
        "M_x": 20, "M_y_values": [40, 15], "T": 1000,
        "na": 2, "nb": 3, "C_family": "gaussian_isotropic",
        "stageA_network_ids": {"40": 347040, "15": 347015},
        "parameter_order": ["A_800", "alpha_380", "B_60", "Q_20"],
        "observability_horizons": HORIZONS,
        "relative_rank_thresholds": RELATIVE_THRESHOLDS,
        "bottom_mode_counts": BOTTOM_COUNTS,
        "legacy_epsilon_primary": 1e-4,
        "legacy_epsilon_grid": [3e-4, 1e-4, 3e-5, 1e-5],
        "legacy_max_iter": int(os.environ.get(
            "EXPERIMENT_34UI_PERTURB_MAX_ITER", "8" if smoke else "300")),
        "legacy_min_iter": 2,
        "legacy_convergence_tolerance": 1e-4,
        "legacy_iteration_checkpoints": ITERATION_CHECKPOINTS,
        "audit_edges": {"My15_worst": 1 if smoke else 3,
                        "My15_median": 0 if smoke else 2,
                        "My40_representative": 1 if smoke else 2},
        "tsvd_relative_cutoffs": REGULARIZATION_GRID,
        "tikhonov_relative_lambdas": REGULARIZATION_GRID,
        "feedback_gamma_grid": GAMMA_GRID,
        "inner_threads": max(1, int(os.environ.get("EXPERIMENT_34UI_INNER_THREADS", "1"))),
        "truth_used_for_method_design": False,
        "observability_used_as_edge_evidence": False,
        "full_90_or_380_edge_legacy_stageB_prohibited": True,
        "full_structured_VI_changed": False,
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def atomic_npz(path, **values):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def append_rows(path, rows, keys):
    if not rows:
        return
    new = pd.DataFrame(rows); old = read_csv(path)
    frame = pd.concat([old, new], ignore_index=True) if len(old) else new
    subset = [key for key in keys if key in frame]
    if subset:
        frame = frame.drop_duplicates(subset=subset, keep="last")
    atomic_csv(frame, str(path))


class Progress:
    """Single-line terminal progress; scientific results stay in CSV files."""
    def __init__(self):
        self.started = time.perf_counter(); self.phase_started = self.started
        self.label = ""; self.total = 1; self.done = 0

    def start(self, label, total, completed=0):
        self.label = label; self.total = max(int(total), 1)
        self.done = int(completed); self.phase_started = time.perf_counter()
        self.show()

    def update(self, done=None, detail=""):
        self.done = self.done + 1 if done is None else int(done)
        self.show(detail)

    def show(self, detail=""):
        fraction = min(self.done / self.total, 1.0); filled = round(30*fraction)
        elapsed = time.perf_counter()-self.phase_started
        eta = elapsed/max(self.done, 1)*max(self.total-self.done, 0)
        text = f"{self.label} {detail}"[:48]
        print(f"\r34UI [{'#'*filled}{'-'*(30-filled)}] {self.done}/{self.total} "
              f"{100*fraction:5.1f}% ETA {eta/60:6.1f}m {text}",
              end="\n" if self.done >= self.total else "", flush=True)


def initialize(config):
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    if (output / "_COMPLETED.json").exists():
        return output, True
    path = output / "experiment_config.json"
    protected = ["smoke_test", "M_x", "M_y_values", "T", "stageA_network_ids",
                 "legacy_max_iter", "legacy_min_iter", "legacy_convergence_tolerance",
                 "legacy_iteration_checkpoints", "legacy_epsilon_grid"]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"34UI checkpoint configuration mismatch: {changed}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Non-empty 34UI directory has no configuration: {output}")
    atomic_json(config, path)
    for name in ("plots", "checkpoints", "checkpoints/svd", "checkpoints/legacy"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output, False


def validate_sources():
    required_uh = ["_COMPLETED.json", "experiment_config.json", "network_manifest.csv",
                   "validation_edge_manifest.csv", "direct_full_results.csv",
                   "covariance_fidelity.csv", "solver_error_decomposition.csv"]
    missing = [name for name in required_uh if not (SOURCE_UH/name).exists()]
    if missing:
        raise FileNotFoundError(f"Completed 34UH artifacts missing: {missing}")
    for run_id in ("34UG_stageA_Mx20_My40_net347040",
                   "34UG_stageA_Mx20_My15_net347015"):
        directory = SOURCE_UH/"checkpoints"/run_id/"full"
        for name in ("J_full.npy", "I_minus_J_full.npy", "B_validation.npy",
                     "X_validation_direct.npy"):
            if not (directory/name).exists():
                raise FileNotFoundError(directory/name)


def source_manifest(output):
    names = ["experiment_config.json", "network_manifest.csv",
             "validation_edge_manifest.csv", "direct_full_results.csv",
             "direct_solve_residuals.csv", "covariance_fidelity.csv",
             "solver_error_decomposition.csv", "legacy_epsilon_sensitivity.csv",
             "_COMPLETED.json"]
    rows = []
    for name in names:
        path = SOURCE_UH/name
        rows.append({"source_experiment": "34UH", "artifact": name,
                     "path": str(path), "available": path.exists(),
                     "bytes": path.stat().st_size if path.exists() else 0,
                     "reused_read_only": True})
    for run_id in ("34UG_stageA_Mx20_My40_net347040",
                   "34UG_stageA_Mx20_My15_net347015"):
        for name in ("J_full.npy", "J_full_completed.npy", "I_minus_J_full.npy",
                     "B_validation.npy", "X_validation_direct.npy"):
            path = SOURCE_UH/"checkpoints"/run_id/"full"/name
            rows.append({"source_experiment": "34UH", "artifact": f"{run_id}/{name}",
                         "path": str(path), "available": path.exists(),
                         "bytes": path.stat().st_size if path.exists() else 0,
                         "reused_read_only": True})
    atomic_csv(pd.DataFrame(rows), str(output/"source_34uh_manifest.csv"))


def load_context(config, output, my):
    context = uh.context_from_34ug(config, output, my)
    context["meta"] = dict(context["meta"])
    context["meta"]["source_experiment_name"] = context["meta"].get("experiment_name")
    context["meta"]["experiment_name"] = config["experiment"]
    return context


def effective_rank(values):
    values = np.abs(np.asarray(values, float)); values = values[values > 0]
    if not len(values) or values.sum() <= 0:
        return 0.0
    probabilities = values/values.sum()
    return float(np.exp(-np.sum(probabilities*np.log(probabilities))))


def covariance_from_row(row):
    return np.asarray([[row.cov_00, row.cov_01],
                       [row.cov_10, row.cov_11]], float)


def stable_covariance(raw):
    return ug.ua.safe_covariance(np.asarray(raw, float))[0]


def edge_covariance_from_response(X, edge_position, target, source, M=20):
    selected = [global_index(target, lag, source, M, 2) for lag in range(2)]
    return stable_covariance(X[np.ix_(selected, [2*edge_position, 2*edge_position+1])])


def covariance_metrics(C, reference, edge, louis_trace):
    C = stable_covariance(C); reference = stable_covariance(reference)
    delta = edge[["center_lag1", "center_lag2"]].to_numpy(float) - edge[["truth_lag1", "truth_lag2"]].to_numpy(float)
    covered = bool(float(delta @ np.linalg.pinv(C) @ delta) <= CHI2_95_DF2)
    ref_covered = bool(float(delta @ np.linalg.pinv(reference) @ delta) <= CHI2_95_DF2)
    return {
        "relative_frobenius_error": float(np.linalg.norm(C-reference)/max(np.linalg.norm(reference), EPS)),
        "trace_relative_error": float(abs(np.trace(C)-np.trace(reference))/max(abs(np.trace(reference)), EPS)),
        "candidate_trace": float(np.trace(C)), "reference_trace": float(np.trace(reference)),
        "candidate_relative_inflation": float((np.trace(C)-louis_trace)/max(louis_trace, EPS)),
        "reference_relative_inflation": float((np.trace(reference)-louis_trace)/max(louis_trace, EPS)),
        "coverage_decision": covered, "reference_coverage_decision": ref_covered,
        "coverage_agreement": bool(covered == ref_covered),
        **ug.covariance_columns(C),
    }


def load_or_repair_matrices(output, context, progress):
    """Reuse valid 34UH matrices; repair only invalid/missing columns in 34UI."""
    run_id = context["meta"]["run_id"]
    source = SOURCE_UH/"checkpoints"/run_id/"full"
    engine = HybridFixedPointTangentMap(context["payload"]["model"],
        context["payload"]["data"]["y"], context["payload"]["data"]["u"])
    n = len(engine.theta0); source_j = source/"J_full.npy"
    valid_source = False
    if source_j.exists():
        candidate = np.load(source_j, mmap_mode="r")
        valid_source = candidate.shape == (n, n) and bool(np.all(np.isfinite(candidate)))
    if valid_source:
        J = np.asarray(candidate)
        reused = True
    else:
        repair = output/"checkpoints"/run_id/"full_repair"; repair.mkdir(parents=True, exist_ok=True)
        matrix_path = repair/"J_full.npy"; completion_path = repair/"J_full_completed.npy"
        if source_j.exists() and np.load(source_j, mmap_mode="r").shape == (n, n) and not matrix_path.exists():
            src = np.load(source_j, mmap_mode="r")
            dst = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float64, shape=(n, n))
            dst[:] = src; dst.flush()
            completed = np.all(np.isfinite(src), axis=0)
            with completion_path.open("wb") as handle: np.save(handle, completed)
        progress.start(f"repair saved J My={context['meta']['M_y']}", n, 0)
        J, _, _ = assemble_exact_jacobian(engine, repair, workers=1,
            inner_threads=1, checkpoint_every=20,
            progress_callback=lambda done, total, detail: progress.update(done, detail))
        reused = False
    source_l = source/"I_minus_J_full.npy"
    if reused and source_l.exists():
        L = np.asarray(np.load(source_l, mmap_mode="r"))
        valid_l = (L.shape == (n, n) and np.all(np.isfinite(L)) and
                   np.linalg.norm(L-(np.eye(n)-J))/max(np.linalg.norm(L), EPS) < 1e-12)
    else:
        valid_l = False
    if not valid_l:
        L = np.eye(n)-J
    B = np.asarray(np.load(source/"B_validation.npy"))
    X = np.asarray(np.load(source/"X_validation_direct.npy"))
    if B.shape != (n, 24) or X.shape != (n, 24):
        raise ValueError(f"Invalid saved 34UH RHS/response shape for {run_id}.")
    return engine, J, L, B, X, reused


def c_and_observability(config, output, context):
    meta = context["meta"]; model = context["payload"]["model"]
    C = np.asarray(model.C, float); R = np.asarray(model.R, float)
    singular = linalg.svdvals(C); maximum = singular[0]
    ranks = {threshold: int(np.sum(singular > threshold*maximum))
             for threshold in RELATIVE_THRESHOLDS}
    rank = int(np.linalg.matrix_rank(C)); gc = C.T @ linalg.solve(R, C, assume_a="pos")
    gc_eigen = np.linalg.eigvalsh((gc+gc.T)/2)
    row = {**meta, "C_rows": C.shape[0], "C_columns": C.shape[1],
           "rank_C": rank, "nullity_C": C.shape[1]-rank,
           "sigma_max_C": float(maximum),
           "sigma_min_nonzero_C": float(singular[rank-1]) if rank else 0.,
           "condition_C_nonzero_spectrum": float(maximum/singular[rank-1]) if rank else np.inf,
           "effective_rank_C": effective_rank(singular),
           "effective_rank_G_C": effective_rank(np.maximum(gc_eigen, 0)),
           "G_C_min_eigenvalue": float(gc_eigen.min()),
           "G_C_max_eigenvalue": float(gc_eigen.max()),
           "G_C_diagonal": json.dumps(np.diag(gc).tolist())}
    for threshold, value in ranks.items():
        row[f"numerical_rank_rel_{threshold:.0e}"] = value
    append_rows(output/"C_rank_diagnostics.csv", [row], ["run_id"])
    crows = [{**meta, "matrix": "C", "spectrum_index": index,
              "value": float(value)} for index, value in enumerate(singular)]
    crows += [{**meta, "matrix": "G_C", "spectrum_index": index,
               "value": float(value)} for index, value in enumerate(gc_eigen[::-1])]
    atomic_csv(pd.concat([read_csv(output/"C_singular_values.csv"), pd.DataFrame(crows)],
                         ignore_index=True).drop_duplicates(
        ["run_id", "matrix", "spectrum_index"], keep="last"),
        str(output/"C_singular_values.csv"))

    A = np.asarray(model.A_mean_matrices_, float); M = A.shape[1]
    companion = np.block([[A[0], A[1]], [np.eye(M), np.zeros((M, M))]])
    ctilde = np.hstack([C, np.zeros_like(C)])
    base = ctilde.T @ linalg.solve(R, ctilde, assume_a="pos")
    gramian = np.zeros_like(base); power = np.eye(2*M)
    summaries = []; spectra = []; dynamic_scores = {}
    for h in range(1, max(HORIZONS)+1):
        gramian += power.T @ base @ power
        power = power @ companion
        if h not in HORIZONS:
            continue
        symmetric = (gramian+gramian.T)/2
        values, vectors = np.linalg.eigh(symmetric); positive = values[values > max(values[-1], 1.)*1e-14]
        condition = float(values[-1]/positive[0]) if len(positive) else np.inf
        summaries.append({**meta, "horizon": h, "minimum_eigenvalue": float(values[0]),
            "maximum_eigenvalue": float(values[-1]), "condition_positive_spectrum": condition,
            "condition_number_2": (float(values[-1]/values[0])
                                   if values[0] > 0 else np.inf),
            "numerical_rank_rel_1e-12": int(np.sum(values > values[-1]*1e-12)),
            "effective_rank": effective_rank(np.maximum(values, 0)),
            "trace": float(np.trace(symmetric)),
            "logdet_positive_spectrum": float(np.sum(np.log(positive))) if len(positive) else -np.inf})
        for index, value in enumerate(values):
            spectra.append({**meta, "horizon": h, "eigen_index_ascending": index,
                "eigenvalue": float(value),
                "eigenvector": json.dumps(vectors[:, index].tolist()) if index < 5 else ""})
        dynamic_scores[h] = np.diag(symmetric)[:M].copy()
    append_rows(output/"dynamic_observability_summary.csv", summaries,
                ["run_id", "horizon"])
    append_rows(output/"dynamic_observability_spectrum.csv", spectra,
                ["run_id", "horizon", "eigen_index_ascending"])
    return {"C": C, "G_C": gc, "instantaneous": np.diag(gc),
            "dynamic": dynamic_scores, "rank": rank, "nullity": M-rank}


def svd_checkpoint(output, meta, L, progress):
    path = output/"checkpoints"/"svd"/f"{meta['run_id']}.npz"
    if path.exists():
        item = np.load(path); return item["U"], item["s"], item["Vt"]
    progress.start(f"full SVD My={meta['M_y']}", 1, 0)
    U, s, Vt = linalg.svd(L, full_matrices=False, check_finite=False,
                          lapack_driver="gesdd")
    atomic_npz(path, U=U, s=s, Vt=Vt); progress.update(1)
    return U, s, Vt


def svd_diagnostics(output, context, engine, J, L, U, s, Vt, obs):
    meta = context["meta"]; maximum = s[0]
    spectra = [{**meta, "singular_index_descending": index,
                "bottom_rank": len(s)-index,
                "singular_value": float(value),
                "relative_singular_value": float(value/maximum)}
               for index, value in enumerate(s)]
    append_rows(output/"lrvb_singular_values.csv", spectra,
                ["run_id", "singular_index_descending"])
    near = []
    for threshold in RELATIVE_THRESHOLDS:
        rank = int(np.sum(s > maximum*threshold))
        near.append({**meta, "relative_threshold": threshold,
            "record_type": "singular_rank_threshold",
            "numerical_rank": rank, "number_near_null_modes": len(s)-rank,
            "sigma_max": float(maximum), "sigma_min": float(s[-1]),
            "condition_number_2": float(maximum/s[-1])})
    append_rows(output/"lrvb_near_null_modes.csv", near,
                ["run_id", "record_type", "relative_threshold",
                 "eigen_rank_closest_to_one"])

    layout = engine.layout
    blocks = [("A", layout.A_slice), ("alpha", layout.alpha_slice),
              ("B", layout.B_slice), ("Q", layout.Q_slice)]
    energies = []; structures = []; alignments = []
    dynamic = obs["dynamic"][max(obs["dynamic"])]
    instant = obs["instantaneous"]
    for bottom_rank in range(1, min(50, len(s))+1):
        index = len(s)-bottom_rank
        for side, vector in (("right", Vt[index]), ("left", U[:, index])):
            total = float(vector@vector)
            for block, block_slice in blocks:
                energies.append({**meta, "bottom_rank": bottom_rank,
                    "singular_value": float(s[index]), "side": side,
                    "parameter_block": block,
                    "energy_fraction": float(vector[block_slice]@vector[block_slice]/total)})
        if bottom_rank not in BOTTOM_COUNTS:
            continue
        Avec = Vt[index, layout.A_slice].reshape(20, 2, 20)
        source_energy = np.sum(Avec**2, axis=(0, 1))
        target_energy = np.sum(Avec**2, axis=(1, 2))
        low = np.argsort(dynamic)[:5]
        low_source_fraction = float(source_energy[low].sum()/max(source_energy.sum(), EPS))
        low_target_fraction = float(target_energy[low].sum()/max(target_energy.sum(), EPS))
        for state in range(20):
            structures.append({**meta, "bottom_rank": bottom_rank,
                "latent_state": state, "source_energy": float(source_energy[state]),
                "target_energy": float(target_energy[state]),
                "instantaneous_observability": float(instant[state]),
                "dynamic_observability_H40": float(dynamic[state]),
                "lowest_dynamic_observability_quartile": bool(state in low)})
        alignments.append({**meta, "bottom_rank": bottom_rank,
            "source_vs_instantaneous_spearman": float(spearmanr(source_energy, instant).statistic),
            "target_vs_instantaneous_spearman": float(spearmanr(target_energy, instant).statistic),
            "source_vs_dynamic_spearman": float(spearmanr(source_energy, dynamic).statistic),
            "target_vs_dynamic_spearman": float(spearmanr(target_energy, dynamic).statistic),
            "low_observability_source_energy_fraction": low_source_fraction,
            "low_observability_target_energy_fraction": low_target_fraction,
            "low_observability_source_enrichment": low_source_fraction/.25,
            "low_observability_target_enrichment": low_target_fraction/.25})
    append_rows(output/"singular_mode_block_energy.csv", energies,
                ["run_id", "bottom_rank", "side", "parameter_block"])
    append_rows(output/"singular_mode_A_source_target_energy.csv", structures,
                ["run_id", "bottom_rank", "latent_state"])
    append_rows(output/"singular_mode_observability_alignment.csv", alignments,
                ["run_id", "bottom_rank"])

    eigen_path = output/"checkpoints"/"svd"/f"{meta['run_id']}_eigJ.npy"
    if eigen_path.exists():
        eigen_j = np.load(eigen_path)
    else:
        eigen_j = linalg.eigvals(J, check_finite=False)
        temporary = Path(str(eigen_path)+".tmp")
        with temporary.open("wb") as handle: np.save(handle, eigen_j)
        os.replace(temporary, eigen_path)
    closest = np.argsort(np.abs(1-eigen_j))[:50]
    eigen_rows = [{**meta, "relative_threshold": np.nan,
        "record_type": "eigenvalue_closest_to_one",
        "numerical_rank": np.nan, "number_near_null_modes": np.nan,
        "eigen_rank_closest_to_one": rank+1,
        "J_eigenvalue_real": float(eigen_j[index].real),
        "J_eigenvalue_imag": float(eigen_j[index].imag),
        "L_eigenvalue_real": float((1-eigen_j[index]).real),
        "L_eigenvalue_imag": float((1-eigen_j[index]).imag)}
        for rank, index in enumerate(closest)]
    append_rows(output/"lrvb_near_null_modes.csv", eigen_rows,
                ["run_id", "record_type", "relative_threshold",
                 "eigen_rank_closest_to_one"])


def cached_singular_values(path, matrix):
    if path.exists():
        return np.load(path)
    values = linalg.svdvals(matrix, check_finite=False)
    temporary = Path(str(path)+".tmp")
    with temporary.open("wb") as handle: np.save(handle, values)
    os.replace(temporary, path)
    return values


def response_from_svd(matrix, rhs, spectrum_path):
    U, s, Vt = linalg.svd(matrix, full_matrices=False, check_finite=False,
                          lapack_driver="gesdd")
    temporary = Path(str(spectrum_path)+".tmp")
    with temporary.open("wb") as handle: np.save(handle, s)
    os.replace(temporary, spectrum_path)
    response = Vt.T @ ((U.T @ rhs)/s[:, None])
    return response, s


def block_and_ablation(config, output, context, engine, L, B, s_full):
    meta = context["meta"]; layout = engine.layout; checkpoint = output/"checkpoints"/"svd"
    Aidx = np.arange(layout.A_slice.start, layout.A_slice.stop)
    alpha = np.arange(layout.alpha_slice.start, layout.alpha_slice.stop)
    bidx = np.arange(layout.B_slice.start, layout.B_slice.stop)
    qidx = np.arange(layout.Q_slice.start, layout.Q_slice.stop)
    eta = np.concatenate([alpha, bidx, qidx])
    LAA = L[np.ix_(Aidx, Aidx)]; Lee = L[np.ix_(eta, eta)]
    saa = cached_singular_values(checkpoint/f"{meta['run_id']}_LAA_s.npy", LAA)
    see = cached_singular_values(checkpoint/f"{meta['run_id']}_Letaeta_s.npy", Lee)
    block_rows = [
        {**meta, "system": "A", "dimension": len(Aidx), "sigma_min": saa[-1],
         "sigma_max": saa[0], "condition_number_2": saa[0]/saa[-1]},
        {**meta, "system": "eta", "dimension": len(eta), "sigma_min": see[-1],
         "sigma_max": see[0], "condition_number_2": see[0]/see[-1]},
        {**meta, "system": "full", "dimension": len(L), "sigma_min": s_full[-1],
         "sigma_max": s_full[0], "condition_number_2": s_full[0]/s_full[-1]},
    ]
    append_rows(output/"block_conditioning.csv", block_rows, ["run_id", "system"])

    schur_path = checkpoint/f"{meta['run_id']}_schur_A.npy"
    if schur_path.exists():
        schur = np.load(schur_path)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", linalg.LinAlgWarning)
            coupling = linalg.solve(Lee, L[np.ix_(eta, Aidx)],
                                    assume_a="gen", check_finite=False)
        schur = LAA-L[np.ix_(Aidx, eta)]@coupling
        temporary = Path(str(schur_path)+".tmp")
        with temporary.open("wb") as handle: np.save(handle, schur)
        os.replace(temporary, schur_path)
    ss = cached_singular_values(checkpoint/f"{meta['run_id']}_schur_A_s.npy", schur)
    schur_rows = [{**meta, "singular_index_descending": index,
        "singular_value": float(value), "sigma_min": float(ss[-1]),
        "sigma_max": float(ss[0]), "condition_number_2": float(ss[0]/ss[-1])}
        for index, value in enumerate(ss)]
    append_rows(output/"schur_complement_diagnostics.csv", schur_rows,
                ["run_id", "singular_index_descending"])

    systems = {
        "M0_A": Aidx, "M1_A_alpha": np.concatenate([Aidx, alpha]),
        "M2_A_B": np.concatenate([Aidx, bidx]),
        "M3_A_Q": np.concatenate([Aidx, qidx]),
        "M4_A_B_Q": np.concatenate([Aidx, bidx, qidx]),
        "M5_full": np.arange(len(L)),
    }
    legacy = context["legacy"]; rows = []; aonly = []
    for system, indices in systems.items():
        matrix = L[np.ix_(indices, indices)]; rhs = B[indices]
        solve_path = output/"checkpoints"/"svd"/f"{meta['run_id']}_{system}_response.npz"
        if solve_path.exists():
            saved = np.load(solve_path); response, singular = saved["X"], saved["s"]
        elif system == "M5_full":
            response = np.asarray(np.load(SOURCE_UH/"checkpoints"/meta["run_id"]/
                                          "full"/"X_validation_direct.npy"))
            singular = s_full
            atomic_npz(solve_path, X=response, s=singular)
        else:
            if system == "M0_A":
                singular = saa
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", linalg.LinAlgWarning)
                    response = linalg.solve(matrix, rhs, assume_a="gen",
                                            check_finite=False)
            else:
                response, singular = response_from_svd(matrix, rhs,
                    output/"checkpoints"/"svd"/f"{meta['run_id']}_{system}_s.npy")
            atomic_npz(solve_path, X=response, s=singular)
        for edge_position, edge in enumerate(context["edges"].itertuples(index=False)):
            target, source = int(edge.target), int(edge.source)
            global_selected = [global_index(target, lag, source, 20, 2) for lag in range(2)]
            local_selected = [int(np.flatnonzero(indices == value)[0]) for value in global_selected]
            C = stable_covariance(response[np.ix_(local_selected,
                [2*edge_position, 2*edge_position+1])])
            refrow = legacy[(legacy.target.eq(target))&legacy.source.eq(source)].iloc[-1]
            reference = covariance_from_row(refrow)
            edge_series = context["edges"][(context["edges"].target.eq(target))&
                                            context["edges"].source.eq(source)].iloc[-1]
            metrics = covariance_metrics(C, reference, edge_series,
                                         float(edge_series.louis_cov_trace))
            row = {**meta, "diagnostic_system": system, "dimension": len(indices),
                "condition_number_2": float(singular[0]/singular[-1]),
                "smallest_singular_value": float(singular[-1]),
                "target": target, "source": source,
                "response_norm_lag1": float(np.linalg.norm(response[:, 2*edge_position])),
                "response_norm_lag2": float(np.linalg.norm(response[:, 2*edge_position+1])),
                **metrics}
            rows.append(row)
            if system == "M0_A":
                aonly.append({**row, "method": "A_only_direct_diagnostic"})
    append_rows(output/"nested_block_ablation.csv", rows,
                ["run_id", "diagnostic_system", "target", "source"])
    append_rows(output/"A_only_direct_results.csv", aonly,
                ["run_id", "target", "source"])


def rhs_diagnostics(output, context, U, s, B, X, obs):
    meta = context["meta"]; projections = U.T @ B
    contributions = projections/np.maximum(s[:, None], np.finfo(float).tiny)
    energies = contributions**2; totals = np.sum(energies, axis=0)
    fidelity = read_csv(SOURCE_UH/"covariance_fidelity.csv")
    fidelity = fidelity[(fidelity.run_id.eq(meta["run_id"]))&
                        fidelity.method.eq("D0_full_dense_LU")]
    excitation = []; effective = []
    dynamic = obs["dynamic"][max(obs["dynamic"])]
    for position in range(B.shape[1]):
        edge_position, lag = divmod(position, 2)
        edge = context["edges"].iloc[edge_position]
        target, source = int(edge.target), int(edge.source)
        for count in BOTTOM_COUNTS:
            fraction = float(np.sum(energies[-count:, position])/max(totals[position], EPS))
            excitation.append({**meta, "target": target, "source": source,
                "lag": lag+1, "direction_index": global_index(target, lag, source, 20, 2),
                "bottom_mode_count": count,
                "near_null_projection_norm": float(np.linalg.norm(projections[-count:, position])),
                "amplified_contribution_norm": float(np.sqrt(np.sum(energies[-count:, position]))),
                "fraction_total_response_energy": fraction})
        fit = fidelity[(fidelity.target.eq(target))&fidelity.source.eq(source)]
        effective.append({**meta, "target": target, "source": source, "lag": lag+1,
            "direction_index": global_index(target, lag, source, 20, 2),
            "rhs_norm": float(np.linalg.norm(B[:, position])),
            "response_norm": float(np.linalg.norm(X[:, position])),
            "response_gain": float(np.linalg.norm(X[:, position])/max(np.linalg.norm(B[:, position]), EPS)),
            "normalized_effective_amplification": float(np.linalg.norm(X[:, position])/
                max(np.linalg.norm(B[:, position]), EPS)*s[-1]),
            "direct_vs_legacy_covariance_error": float(fit.iloc[-1].relative_frobenius_error) if len(fit) else np.nan,
            "source_instantaneous_observability": float(obs["instantaneous"][source]),
            "target_instantaneous_observability": float(obs["instantaneous"][target]),
            "source_dynamic_observability_H40": float(dynamic[source]),
            "target_dynamic_observability_H40": float(dynamic[target]),
            "louis_group_snr": float(edge.louis_group_snr),
            "edge_group_norm": float(np.hypot(edge.center_lag1, edge.center_lag2))})
    append_rows(output/"rhs_singular_mode_excitation.csv", excitation,
                ["run_id", "direction_index", "bottom_mode_count"])
    frame = pd.DataFrame(effective)
    correlations = []
    for variable in ("direct_vs_legacy_covariance_error",
                     "source_instantaneous_observability",
                     "source_dynamic_observability_H40",
                     "louis_group_snr", "edge_group_norm"):
        valid = frame[["response_gain", variable]].replace([np.inf,-np.inf],np.nan).dropna()
        correlations.append({**meta, "record_type":"descriptive_correlation",
            "direction_index":-1, "correlation_variable":variable,
            "response_gain_spearman":float(spearmanr(valid.response_gain,
                valid[variable]).statistic) if len(valid)>1 else np.nan})
    for row in effective:
        row["record_type"] = "RHS"
        row["correlation_variable"] = ""
    append_rows(output/"rhs_effective_conditioning.csv", effective+correlations,
                ["run_id", "record_type", "direction_index", "correlation_variable"])


def choose_audit_edges(config, contexts):
    fidelity = read_csv(SOURCE_UH/"covariance_fidelity.csv")
    rows = []
    for context in contexts:
        my = int(context["meta"]["M_y"]); frame = fidelity[
            (fidelity.run_id.eq(context["meta"]["run_id"]))&
            fidelity.method.eq("D0_full_dense_LU")].sort_values(
                "relative_frobenius_error", ascending=False)
        if my == 15:
            nworst = config["audit_edges"]["My15_worst"]
            for item in frame.head(nworst).itertuples(index=False):
                rows.append({"run_id": item.run_id, "M_y": my, "target": int(item.target),
                             "source": int(item.source), "audit_role": "My15_worst",
                             "direct_vs_legacy_error": item.relative_frobenius_error})
            nmedian = config["audit_edges"]["My15_median"]
            remaining = frame.iloc[nworst:].copy()
            if nmedian and len(remaining):
                ranks = np.argsort(np.abs(remaining.relative_frobenius_error.to_numpy()-
                                           remaining.relative_frobenius_error.median()))[:nmedian]
                for item in remaining.iloc[ranks].itertuples(index=False):
                    rows.append({"run_id": item.run_id, "M_y": my, "target": int(item.target),
                                 "source": int(item.source), "audit_role": "My15_median",
                                 "direct_vs_legacy_error": item.relative_frobenius_error})
        else:
            count = config["audit_edges"]["My40_representative"]
            ranks = np.argsort(np.abs(frame.relative_frobenius_error.to_numpy()-
                                       frame.relative_frobenius_error.median()))[:count]
            for item in frame.iloc[ranks].itertuples(index=False):
                rows.append({"run_id": item.run_id, "M_y": my, "target": int(item.target),
                             "source": int(item.source), "audit_role": "My40_representative",
                             "direct_vs_legacy_error": item.relative_frobenius_error})
    return pd.DataFrame(rows)


def fit_checkpoint_path(output, run_id, target, source, lag, sign, epsilon):
    token = f"eps{epsilon:.0e}".replace("+", "")
    return (output/"checkpoints"/"legacy"/run_id/
            f"t{target}_s{source}_lag{lag}_{sign}_{token}")


def run_or_load_fit(config, output, context, target, source, lag, sign,
                    epsilon, progress):
    directory = fit_checkpoint_path(output, context["meta"]["run_id"], target,
                                    source, lag, sign, epsilon)
    data_path = directory/"theta_checkpoints.npz"; meta_path = directory/"fit_summary.json"
    trace_path = directory/"iteration_trace.csv"
    if data_path.exists() and meta_path.exists() and trace_path.exists():
        item = np.load(data_path); summary = json.loads(meta_path.read_text(encoding="utf8"))
        return {key: item[key] for key in item.files}, summary, read_csv(trace_path)
    directory.mkdir(parents=True, exist_ok=True)
    model = context["payload"]["model"]; perturbation = np.zeros((20, 40))
    perturbation[target, (lag-1)*20+source] = epsilon if sign == "plus" else -epsilon
    label = f"My={context['meta']['M_y']} {target}<-{source} L{lag} {sign}"
    progress.start(label, config["legacy_max_iter"], 0)
    result = fit_perturbed_hybrid_vb(model, context["payload"]["data"]["y"],
        context["payload"]["data"]["u"], perturbation,
        max_iter=config["legacy_max_iter"],
        convergence_tol=config["legacy_convergence_tolerance"],
        min_iter=config["legacy_min_iter"],
        reestimate_B=True, reestimate_Q=True,
        checkpoint_iterations=ITERATION_CHECKPOINTS,
        require_loglike_convergence=True, record_trace=True,
        iteration_callback=lambda row: progress.update(row["iteration"]))
    arrays = {f"iter_{iteration}": value for iteration, value in
              result.theta_checkpoints.items()}
    arrays["final"] = result.theta.copy()
    atomic_npz(data_path, **arrays)
    trace = pd.DataFrame(result.iteration_trace)
    trace["run_id"] = context["meta"]["run_id"]; trace["M_y"] = context["meta"]["M_y"]
    trace["target"] = target; trace["source"] = source; trace["lag"] = lag
    trace["sign"] = sign; trace["epsilon"] = epsilon
    atomic_csv(trace, str(trace_path))
    summary = {"run_id": context["meta"]["run_id"], "M_y": context["meta"]["M_y"],
        "target": target, "source": source, "lag": lag, "sign": sign,
        "epsilon": epsilon, "converged": result.converged, "n_iter": result.n_iter,
        "hit_max_iter": bool(not result.converged and result.n_iter >= config["legacy_max_iter"]),
        "runtime_seconds": result.runtime_seconds,
        "final_relative_A_change": result.final_relative_A_change,
        "final_relative_B_change": result.final_relative_B_change,
        "final_relative_Q_change": result.final_relative_Q_change,
        "final_relative_alpha_change": result.final_relative_alpha_change,
        "final_loglikelihood": result.final_loglikelihood,
        "warning_flag": result.warning_flag}
    atomic_json(summary, meta_path)
    return arrays, summary, trace


def theta_at(arrays, summary, iteration):
    if iteration == "convergence":
        return arrays["final"]
    key = f"iter_{int(iteration)}"
    if key in arrays:
        return arrays[key]
    if bool(summary["converged"]) and int(summary["n_iter"]) < int(iteration):
        return arrays["final"]
    raise RuntimeError(f"Missing requested iteration {iteration} in perturbed-fit checkpoint.")


def legacy_convergence_audit(config, output, contexts, progress):
    audit = choose_audit_edges(config, contexts); all_trace = []; all_summaries = []
    fit_cache = {}; context_by_run = {c["meta"]["run_id"]: c for c in contexts}
    for edge in audit.itertuples(index=False):
        context = context_by_run[edge.run_id]
        for lag in (1, 2):
            for sign in ("plus", "minus"):
                key = (edge.run_id, edge.target, edge.source, lag, sign, 1e-4)
                arrays, summary, trace = run_or_load_fit(config, output, context,
                    edge.target, edge.source, lag, sign, 1e-4, progress)
                fit_cache[key] = (arrays, summary); all_trace.append(trace)
                all_summaries.append(summary)
    if all_trace:
        atomic_csv(pd.concat(all_trace, ignore_index=True),
                   str(output/"legacy_convergence_iteration_trace.csv"))

    covariance_rows = []
    stages = ITERATION_CHECKPOINTS + ["convergence"]
    for edge in audit.itertuples(index=False):
        context = context_by_run[edge.run_id]
        directrow = read_csv(SOURCE_UH/"direct_full_results.csv")
        directrow = directrow[(directrow.run_id.eq(edge.run_id))&
                              directrow.target.eq(edge.target)&
                              directrow.source.eq(edge.source)].iloc[-1]
        direct = covariance_from_row(directrow)
        legacyrow = context["legacy"][(context["legacy"].target.eq(edge.target))&
                                      context["legacy"].source.eq(edge.source)].iloc[-1]
        legacy = covariance_from_row(legacyrow)
        edge_series = context["edges"][(context["edges"].target.eq(edge.target))&
            context["edges"].source.eq(edge.source)].iloc[-1]
        for stage in stages:
            columns = []
            available = True
            for lag in (1, 2):
                plus, ps = fit_cache[(edge.run_id, edge.target, edge.source, lag, "plus", 1e-4)]
                minus, ms = fit_cache[(edge.run_id, edge.target, edge.source, lag, "minus", 1e-4)]
                try:
                    column = (theta_at(plus, ps, stage)-theta_at(minus, ms, stage))/(2e-4)
                except RuntimeError:
                    available = False; break
                selected = [global_index(edge.target, k, edge.source, 20, 2) for k in range(2)]
                columns.append(column[selected])
            if not available:
                continue
            C = stable_covariance(np.column_stack(columns))
            covariance_rows.append({"run_id": edge.run_id, "M_y": edge.M_y,
                "target": edge.target, "source": edge.source,
                "audit_role": edge.audit_role, "iteration_checkpoint": stage,
                "all_fits_converged_by_checkpoint": bool(all(
                    fit_cache[(edge.run_id, edge.target, edge.source, lag, sign, 1e-4)][1]["converged"] and
                    fit_cache[(edge.run_id, edge.target, edge.source, lag, sign, 1e-4)][1]["n_iter"] <=
                    (10**9 if stage == "convergence" else int(stage))
                    for lag in (1, 2) for sign in ("plus", "minus"))),
                "error_to_original_legacy": float(np.linalg.norm(C-legacy)/max(np.linalg.norm(legacy), EPS)),
                "error_to_direct_implicit": float(np.linalg.norm(C-direct)/max(np.linalg.norm(direct), EPS)),
                **ug.covariance_columns(C)})
    covariance_frame = pd.DataFrame(covariance_rows)
    atomic_csv(covariance_frame, str(output/"legacy_convergence_covariance.csv"))
    summaries = []
    fit_summary_frame = pd.DataFrame(all_summaries)
    for edge in audit.itertuples(index=False):
        fits = fit_summary_frame[(fit_summary_frame.run_id.eq(edge.run_id))&
            fit_summary_frame.target.eq(edge.target)&fit_summary_frame.source.eq(edge.source)]
        values = covariance_frame[(covariance_frame.run_id.eq(edge.run_id))&
            covariance_frame.target.eq(edge.target)&covariance_frame.source.eq(edge.source)]
        i30 = values[values.iteration_checkpoint.astype(str).eq("30")]
        final = values[values.iteration_checkpoint.astype(str).eq("convergence")]
        C30 = covariance_from_row(i30.iloc[-1]) if len(i30) else np.full((2, 2), np.nan)
        Cfinal = covariance_from_row(final.iloc[-1])
        summaries.append({"run_id": edge.run_id, "M_y": edge.M_y,
            "target": edge.target, "source": edge.source, "audit_role": edge.audit_role,
            "n_perturbed_fits": len(fits),
            "iter30_converged_fraction": float(np.mean(fits.converged & (fits.n_iter <= 30))),
            "final_converged_fraction": float(fits.converged.mean()),
            "maximum_iterations": int(fits.n_iter.max()),
            "all_final_converged": bool(fits.converged.all()),
            "covariance_change_after_iter30": float(np.linalg.norm(Cfinal-C30)/max(np.linalg.norm(C30), EPS)) if len(i30) else np.nan,
            "iter30_error_to_implicit": float(i30.iloc[-1].error_to_direct_implicit) if len(i30) else np.nan,
            "final_error_to_implicit": float(final.iloc[-1].error_to_direct_implicit),
            "converges_toward_implicit": bool(len(i30) and final.iloc[-1].error_to_direct_implicit < i30.iloc[-1].error_to_direct_implicit)})
    atomic_csv(pd.DataFrame(summaries), str(output/"legacy_convergence_summary.csv"))
    return audit, fit_cache, pd.DataFrame(summaries)


def add_converged_reference_errors(output):
    """Attach audited-final reference errors to A-only/full diagnostics."""
    convergence = read_csv(output/"legacy_convergence_covariance.csv")
    convergence = convergence[
        convergence.iteration_checkpoint.astype(str).eq("convergence")]
    if not len(convergence):
        return
    aonly = read_csv(output/"A_only_direct_results.csv")
    nested = read_csv(output/"nested_block_ablation.csv")
    direct = read_csv(SOURCE_UH/"direct_full_results.csv")
    for frame in (aonly, nested):
        if "relative_error_vs_audited_final" not in frame:
            frame["relative_error_vs_audited_final"] = np.nan
        if "audited_reference_all_fits_converged" not in frame:
            frame["audited_reference_all_fits_converged"] = pd.Series(
                pd.NA, index=frame.index, dtype="boolean")
        else:
            frame["audited_reference_all_fits_converged"] = frame[
                "audited_reference_all_fits_converged"].astype("boolean")
        for reference in convergence.itertuples(index=False):
            mask = (frame.run_id.eq(reference.run_id)&frame.target.eq(reference.target)&
                    frame.source.eq(reference.source))
            if not mask.any():
                continue
            R = covariance_from_row(reference)
            for index in frame.index[mask]:
                C = covariance_from_row(frame.loc[index])
                frame.loc[index, "relative_error_vs_audited_final"] = (
                    np.linalg.norm(C-R)/max(np.linalg.norm(R), EPS))
                frame.loc[index, "audited_reference_all_fits_converged"] = bool(
                    reference.all_fits_converged_by_checkpoint)
    atomic_csv(aonly, str(output/"A_only_direct_results.csv"))
    atomic_csv(nested, str(output/"nested_block_ablation.csv"))

    summary = read_csv(output/"legacy_convergence_summary.csv")
    if "full_direct_relative_error_vs_audited_final" not in summary:
        summary["full_direct_relative_error_vs_audited_final"] = np.nan
    if "audited_reference_all_fits_converged" not in summary:
        summary["audited_reference_all_fits_converged"] = pd.Series(
            pd.NA, index=summary.index, dtype="boolean")
    else:
        summary["audited_reference_all_fits_converged"] = summary[
            "audited_reference_all_fits_converged"].astype("boolean")
    for reference in convergence.itertuples(index=False):
        q = direct[(direct.run_id.eq(reference.run_id))&direct.target.eq(reference.target)&
                   direct.source.eq(reference.source)]
        if not len(q):
            continue
        R = covariance_from_row(reference); D = covariance_from_row(q.iloc[-1])
        mask = (summary.run_id.eq(reference.run_id)&summary.target.eq(reference.target)&
                summary.source.eq(reference.source))
        summary.loc[mask, "full_direct_relative_error_vs_audited_final"] = float(
            np.linalg.norm(D-R)/max(np.linalg.norm(R), EPS))
        summary.loc[mask, "audited_reference_all_fits_converged"] = bool(
            reference.all_fits_converged_by_checkpoint)
    atomic_csv(summary, str(output/"legacy_convergence_summary.csv"))


def converged_epsilon_audit(config, output, contexts, audit, fit_cache, progress):
    context_by_run = {c["meta"]["run_id"]: c for c in contexts}; rows = []
    worst = audit[audit.audit_role.eq("My15_worst")]
    direct_frame = read_csv(SOURCE_UH/"direct_full_results.csv")
    for edge in worst.itertuples(index=False):
        context = context_by_run[edge.run_id]
        directrow = direct_frame[(direct_frame.run_id.eq(edge.run_id))&
            direct_frame.target.eq(edge.target)&direct_frame.source.eq(edge.source)].iloc[-1]
        direct = covariance_from_row(directrow)
        for epsilon in config["legacy_epsilon_grid"]:
            columns = []; summaries = []
            for lag in (1, 2):
                values = []
                for sign in ("plus", "minus"):
                    key = (edge.run_id, edge.target, edge.source, lag, sign, epsilon)
                    if key not in fit_cache:
                        fit_cache[key] = run_or_load_fit(config, output, context,
                            edge.target, edge.source, lag, sign, epsilon, progress)[:2]
                    arrays, summary = fit_cache[key]; summaries.append(summary)
                    values.append(arrays["final"])
                selected = [global_index(edge.target, k, edge.source, 20, 2) for k in range(2)]
                columns.append(((values[0]-values[1])/(2*epsilon))[selected])
            C = stable_covariance(np.column_stack(columns))
            rows.append({"run_id": edge.run_id, "M_y": edge.M_y,
                "target": edge.target, "source": edge.source, "epsilon": epsilon,
                "all_fits_converged": bool(all(item["converged"] for item in summaries)),
                "maximum_iterations": int(max(item["n_iter"] for item in summaries)),
                "relative_error_vs_direct_implicit": float(np.linalg.norm(C-direct)/max(np.linalg.norm(direct), EPS)),
                **ug.covariance_columns(C)})
    atomic_csv(pd.DataFrame(rows), str(output/"converged_epsilon_sensitivity.csv"))


def regularization_metrics(context, responses, reference_frame, family, value,
                           condition, response_norm):
    rows = []; infl_candidate = []; infl_reference = []; cover = []
    for edge_position, edge in enumerate(context["edges"].itertuples(index=False)):
        reference_row = reference_frame[(reference_frame.run_id.eq(context["meta"]["run_id"]))&
            reference_frame.target.eq(edge.target)&reference_frame.source.eq(edge.source)]
        if not len(reference_row):
            continue
        reference = covariance_from_row(reference_row.iloc[-1]); C = edge_covariance_from_response(
            responses, edge_position, int(edge.target), int(edge.source))
        edge_series = context["edges"].iloc[edge_position]
        metrics = covariance_metrics(C, reference, edge_series,
                                     float(edge_series.louis_cov_trace))
        rows.append({**context["meta"], "regularization_family": family,
            "regularization_value": value, "target": edge.target, "source": edge.source,
            "condition_number": condition, "response_norm": response_norm, **metrics})
        rows[-1]["reference_all_fits_converged"] = bool(
            reference_row.iloc[-1].all_fits_converged_by_checkpoint)
        infl_candidate.append(metrics["candidate_relative_inflation"])
        infl_reference.append(metrics["reference_relative_inflation"])
        cover.append(metrics["coverage_agreement"])
    if rows:
        errors = [row["relative_frobenius_error"] for row in rows]
        trace_errors = [row["trace_relative_error"] for row in rows]
        aggregate = {**context["meta"], "regularization_family": family,
            "regularization_value": value, "target": -1, "source": -1,
            "condition_number": condition, "response_norm": response_norm,
            "relative_frobenius_error": float(np.mean(errors)),
            "trace_relative_error": float(np.mean(trace_errors)),
            "inflation_spearman": float(spearmanr(infl_candidate, infl_reference).statistic) if len(rows)>1 else np.nan,
            "coverage_agreement": float(np.mean(cover)), "aggregate_row": True}
        aggregate["reference_all_fits_converged"] = bool(all(
            row["reference_all_fits_converged"] for row in rows))
        for row in rows: row["aggregate_row"] = False
        rows.append(aggregate)
    return rows


def regularization_diagnostics(output, contexts, matrices):
    converged = read_csv(output/"legacy_convergence_covariance.csv")
    reference = converged[converged.iteration_checkpoint.astype(str).eq("convergence")]
    for context in contexts:
        run_id = context["meta"]["run_id"]
        U, s, Vt, J, L, B = (matrices[run_id][key] for key in
                             ("U", "s", "Vt", "J", "L", "B"))
        coefficients = U.T @ B; sigma_max = s[0]
        for tau in REGULARIZATION_GRID:
            keep = s/sigma_max >= tau
            filtered = np.zeros_like(coefficients); filtered[keep] = coefficients[keep]/s[keep, None]
            X = Vt.T @ filtered
            condition = float(sigma_max/s[keep][-1]) if np.any(keep) else np.inf
            rows = regularization_metrics(context, X, reference, "TSVD", tau,
                                          condition, float(np.linalg.norm(X)))
            append_rows(output/"regularized_tsvd_results.csv", rows,
                        ["run_id", "regularization_value", "target", "source"])
        for relative in REGULARIZATION_GRID:
            lam = relative*sigma_max
            factors = s/(s*s+lam*lam)
            X = Vt.T @ (factors[:, None]*coefficients)
            nonzero = factors[factors > factors.max()*np.finfo(float).eps]
            condition = float(nonzero.max()/nonzero.min()) if len(nonzero) else np.inf
            rows = regularization_metrics(context, X, reference, "Tikhonov", relative,
                                          condition, float(np.linalg.norm(X)))
            append_rows(output/"regularized_tikhonov_results.csv", rows,
                        ["run_id", "regularization_value", "target", "source"])
        for gamma in GAMMA_GRID:
            matrix = np.eye(len(J))-gamma*J
            started = time.perf_counter()
            lu, piv = linalg.lu_factor(matrix, check_finite=False)
            X = linalg.lu_solve((lu, piv), B, check_finite=False)
            try:
                reciprocal, info = linalg.lapack.dgecon(lu, np.linalg.norm(matrix, 1), norm="1")
                condition = float(1/reciprocal) if info == 0 and reciprocal > 0 else np.inf
            except Exception:
                condition = np.nan
            rows = regularization_metrics(context, X, reference, "feedback_damping", gamma,
                                          condition, float(np.linalg.norm(X)))
            for row in rows: row["solve_seconds"] = time.perf_counter()-started
            append_rows(output/"regularized_feedback_damping.csv", rows,
                        ["run_id", "regularization_value", "target", "source"])


def plot_results(output):
    plots = output/"plots"; plots.mkdir(exist_ok=True)
    def save(name, draw):
        fig, ax = plt.subplots(); draw(ax); fig.tight_layout(); fig.savefig(plots/name); plt.close(fig)
    c = read_csv(output/"C_singular_values.csv")
    save("01_C_singular_value_spectra.png", lambda ax: (
        [ax.semilogy(g.spectrum_index, g.value, label=f"My={my}") for my, g in c[c.matrix.eq("C")].groupby("M_y")],
        ax.set(xlabel="index", ylabel="singular value"), ax.legend()))
    o = read_csv(output/"dynamic_observability_spectrum.csv")
    save("02_dynamic_observability_spectra.png", lambda ax: (
        [ax.semilogy(g.eigen_index_ascending, np.maximum(g.eigenvalue, 1e-20), label=f"My={my}")
         for my, g in o[o.horizon.eq(40)].groupby("M_y")], ax.legend(),
        ax.set(xlabel="ascending index", ylabel="Gramian eigenvalue")))
    sv = read_csv(output/"lrvb_singular_values.csv")
    save("03_full_IminusJ_singular_spectrum.png", lambda ax: (
        [ax.semilogy(g.singular_index_descending, g.singular_value, label=f"My={my}") for my,g in sv.groupby("M_y")],
        ax.legend(), ax.set(xlabel="descending index", ylabel="singular value")))
    save("04_smallest_50_singular_values.png", lambda ax: (
        [ax.semilogy(g.bottom_rank, g.singular_value, marker=".", label=f"My={my}")
         for my,g in sv[sv.bottom_rank.le(50)].sort_values("bottom_rank").groupby("M_y")],
        ax.legend(), ax.set(xlabel="rank from smallest", ylabel="singular value")))
    be = read_csv(output/"singular_mode_block_energy.csv")
    save("05_singular_mode_block_energy.png", lambda ax: (
        be[(be.side.eq("right"))&be.bottom_rank.le(10)].groupby(
            ["M_y","parameter_block"]).energy_fraction.mean().unstack().plot.bar(ax=ax),
        ax.set_ylabel("mean bottom-10 energy fraction")))
    se = read_csv(output/"singular_mode_A_source_target_energy.csv")
    save("06_observability_vs_A_mode_energy.png", lambda ax: (
        [ax.scatter(g.dynamic_observability_H40, g.source_energy, label=f"My={my}", alpha=.7)
         for my,g in se.groupby("M_y")], ax.legend(),
        ax.set(xlabel="dynamic observability", ylabel="source-mode energy")))
    ex = read_csv(output/"rhs_singular_mode_excitation.csv")
    save("07_rhs_near_null_excitation.png", lambda ax: (
        ex[ex.bottom_mode_count.eq(10)].boxplot(column="fraction_total_response_energy", by="M_y", ax=ax),
        ax.set_ylabel("bottom-10 response-energy fraction")))
    ef = read_csv(output/"rhs_effective_conditioning.csv")
    save("08_response_gain_vs_covariance_discrepancy.png", lambda ax: (
        [ax.scatter(g.response_gain, g.direct_vs_legacy_covariance_error, label=f"My={my}") for my,g in ef.groupby("M_y")],
        ax.set_xscale("log"), ax.legend(), ax.set(xlabel="response gain", ylabel="direct/legacy error")))
    ab = read_csv(output/"nested_block_ablation.csv")
    save("09_condition_by_nested_system.png", lambda ax: (
        ab.drop_duplicates(["M_y","diagnostic_system"]).pivot(
            index="diagnostic_system", columns="M_y", values="condition_number_2").plot.bar(ax=ax),
        ax.set_yscale("log"), ax.set_ylabel("condition number")))
    sc = read_csv(output/"schur_complement_diagnostics.csv")
    def schur_plot(ax):
        for my, group in sc.groupby("M_y"):
            ax.semilogy(group.singular_index_descending, group.singular_value,
                        label=f"Schur My={my}")
            run_id = group.iloc[0].run_id
            path = output/"checkpoints"/"svd"/f"{run_id}_LAA_s.npy"
            if path.exists():
                values = np.load(path)
                ax.semilogy(np.arange(len(values)), values, linestyle="--",
                            label=f"A-only My={my}")
        ax.legend(); ax.set(xlabel="descending index", ylabel="singular value")
    save("10_schur_vs_Aonly_spectrum.png", schur_plot)
    lc = read_csv(output/"legacy_convergence_covariance.csv")
    numeric_lc = lc[lc.iteration_checkpoint.astype(str).str.fullmatch(r"\d+")].copy()
    numeric_lc["iteration_numeric"] = numeric_lc.iteration_checkpoint.astype(int)
    save("11_legacy_covariance_vs_iteration.png", lambda ax: (
        [ax.plot(g.iteration_numeric, g.cov_00, marker=".", alpha=.6) for _,g in numeric_lc.groupby(["run_id","target","source"])],
        ax.set(xlabel="iteration", ylabel="covariance[0,0]")))
    save("12_distance_to_implicit_vs_iteration.png", lambda ax: (
        [ax.plot(g.iteration_numeric, g.error_to_direct_implicit, marker=".", alpha=.6) for _,g in numeric_lc.groupby(["run_id","target","source"])],
        ax.set(xlabel="iteration", ylabel="error to implicit", yscale="log")))
    es = read_csv(output/"converged_epsilon_sensitivity.csv")
    save("13_converged_epsilon_sensitivity.png", lambda ax: (
        [ax.plot(g.epsilon, g.relative_error_vs_direct_implicit, marker="o") for _,g in es.groupby(["target","source"])],
        ax.set(xlabel="epsilon", ylabel="error to implicit", xscale="log", yscale="log")))
    for number, filename, source in (
        (14,"TSVD_cutoff_vs_covariance_error.png","regularized_tsvd_results.csv"),
        (15,"Tikhonov_lambda_vs_covariance_error.png","regularized_tikhonov_results.csv"),
        (16,"feedback_gamma_vs_covariance_error.png","regularized_feedback_damping.csv")):
        frame = read_csv(output/source); frame = frame[frame.target.eq(-1)]
        save(f"{number:02d}_{filename}", lambda ax, frame=frame: (
            [ax.plot(g.regularization_value, g.relative_frobenius_error, marker="o", label=f"My={my}") for my,g in frame.groupby("M_y")],
            ax.legend(), ax.set(xlabel="regularization value", ylabel="mean covariance error"),
            ax.set_xscale("log") if number < 16 else None))


def decision_summary(output):
    c = read_csv(output/"C_rank_diagnostics.csv"); obs = read_csv(output/"dynamic_observability_summary.csv")
    near = read_csv(output/"lrvb_near_null_modes.csv")
    near = near[near.relative_threshold.gt(0)]
    energy = read_csv(output/"singular_mode_block_energy.csv")
    rhs = read_csv(output/"rhs_singular_mode_excitation.csv")
    ab = read_csv(output/"nested_block_ablation.csv").drop_duplicates(["M_y","diagnostic_system"])
    schur = read_csv(output/"schur_complement_diagnostics.csv").drop_duplicates("M_y")
    legacy = read_csv(output/"legacy_convergence_summary.csv")
    tsvd = read_csv(output/"regularized_tsvd_results.csv"); tik = read_csv(output/"regularized_tikhonov_results.csv")
    damp = read_csv(output/"regularized_feedback_damping.csv")
    def one(frame, my, column):
        q = frame[frame.M_y.eq(my)]; return q.iloc[-1][column] if len(q) else np.nan
    def cond(my, system):
        q=ab[(ab.M_y.eq(my))&ab.diagnostic_system.eq(system)]; return q.iloc[-1].condition_number_2 if len(q) else np.nan
    bottom = energy[(energy.M_y.eq(15))&energy.side.eq("right")&energy.bottom_rank.le(10)]
    dominant = bottom.groupby("parameter_block").energy_fraction.mean().idxmax()
    aggregate = pd.concat([tsvd[tsvd.target.eq(-1)], tik[tik.target.eq(-1)], damp[damp.target.eq(-1)]], ignore_index=True)
    if len(aggregate):
        grouped = aggregate.groupby(["regularization_family","regularization_value"], as_index=False).relative_frobenius_error.mean()
        best = grouped.loc[grouped.relative_frobenius_error.idxmin()]
    else:
        best = pd.Series({"regularization_family":"none","regularization_value":np.nan,"relative_frobenius_error":np.nan})
    aonly = read_csv(output/"A_only_direct_results.csv")
    full = read_csv(SOURCE_UH/"covariance_fidelity.csv")
    convergence_comparison = read_csv(output/"legacy_convergence_summary.csv")
    audited_a = (aonly[np.isfinite(aonly["relative_error_vs_audited_final"])]
                 if "relative_error_vs_audited_final" in aonly else pd.DataFrame())
    a_error = (audited_a.groupby("M_y").relative_error_vs_audited_final.mean()
               if len(audited_a) else aonly.groupby("M_y").relative_frobenius_error.mean())
    if len(convergence_comparison) and "full_direct_relative_error_vs_audited_final" in convergence_comparison:
        f_error = convergence_comparison.groupby("M_y").full_direct_relative_error_vs_audited_final.mean()
    else:
        f_error = full[full.method.eq("D0_full_dense_LU")].groupby("M_y").relative_frobenius_error.mean()
    # The unresolved failure is the My=15 regime; My=40 is the control and
    # need not prefer the reduced diagnostic for this mechanism flag.
    closer = bool(a_error.get(15, np.inf) < f_error.get(15, np.inf))
    alpha_path = bool(cond(15,"M1_A_alpha") > 100*cond(15,"M0_A"))
    b_path = bool(cond(15,"M2_A_B") > 100*cond(15,"M0_A"))
    q_path = bool(cond(15,"M3_A_Q") > 100*cond(15,"M0_A"))
    a_path = bool(cond(15,"M0_A") > 1e10)
    final_converged = float(legacy.final_converged_fraction.mean()) if len(legacy) else np.nan
    toward = bool(legacy.converges_toward_implicit.mean() >= .5) if len(legacy) else False
    if final_converged < 1:
        mechanism = "legacy perturbed VB is not a reliable converged reference"
        recommendation = "define a numerically stable reference uncertainty calculation"
    elif toward:
        mechanism = "previous 30-iteration legacy Stage-B was under-converged"
        recommendation = "rebuild Stage-B with converged perturbations and reassess implicit LRVB"
    elif closer:
        mechanism = "nuisance feedback amplifies the full implicit A response"
        recommendation = "prospectively validate reduced-path LRVB"
    else:
        mechanism = "A-smoother feedback contains the near-singular response"
        recommendation = "test observability-sensitive regularization independently"
    row = {
        "My40_rank_C": one(c,40,"rank_C"), "My15_rank_C": one(c,15,"rank_C"),
        "My40_nullity_C": one(c,40,"nullity_C"), "My15_nullity_C": one(c,15,"nullity_C"),
        "My40_dynamic_observability_condition": one(obs[obs.horizon.eq(40)],40,"condition_positive_spectrum"),
        "My15_dynamic_observability_condition": one(obs[obs.horizon.eq(40)],15,"condition_positive_spectrum"),
        "My40_sigma_min_IminusJ": one(near[near.relative_threshold.eq(1e-10)],40,"sigma_min"),
        "My15_sigma_min_IminusJ": one(near[near.relative_threshold.eq(1e-10)],15,"sigma_min"),
        "My40_number_near_null_modes": one(near[near.relative_threshold.eq(1e-10)],40,"number_near_null_modes"),
        "My15_number_near_null_modes": one(near[near.relative_threshold.eq(1e-10)],15,"number_near_null_modes"),
        "My40_bottom10_rhs_energy_fraction": one(rhs[rhs.bottom_mode_count.eq(10)].groupby("M_y",as_index=False).fraction_total_response_energy.mean(),40,"fraction_total_response_energy"),
        "My15_bottom10_rhs_energy_fraction": one(rhs[rhs.bottom_mode_count.eq(10)].groupby("M_y",as_index=False).fraction_total_response_energy.mean(),15,"fraction_total_response_energy"),
        "dominant_near_null_parameter_block": dominant,
        "A_only_condition_My40": cond(40,"M0_A"), "A_only_condition_My15": cond(15,"M0_A"),
        "full_condition_My40": cond(40,"M5_full"), "full_condition_My15": cond(15,"M5_full"),
        "schur_A_condition_My40": one(schur,40,"condition_number_2"),
        "schur_A_condition_My15": one(schur,15,"condition_number_2"),
        "alpha_feedback_is_pathological": alpha_path, "B_feedback_is_pathological": b_path,
        "Q_feedback_is_pathological": q_path, "A_smoother_feedback_is_pathological": a_path,
        "legacy_iter30_converged_fraction": float(legacy.iter30_converged_fraction.mean()) if len(legacy) else np.nan,
        "legacy_final_converged_fraction": final_converged,
        "legacy_changes_after_iter30": float(legacy.covariance_change_after_iter30.mean()) if len(legacy) else np.nan,
        "legacy_converges_toward_implicit": toward,
        "A_only_closer_to_legacy_than_full": closer,
        "regularized_inverse_explains_legacy": bool(final_converged == 1 and
            np.isfinite(best.relative_frobenius_error) and best.relative_frobenius_error < .20),
        "best_exploratory_regularization_family": best.regularization_family,
        "best_exploratory_regularization_value": best.regularization_value,
        "mechanism_identified": mechanism,
        "recommended_next_experiment": recommendation,
    }
    atomic_csv(pd.DataFrame([row]), str(output/"decision_summary.csv"))


def main():
    args = parse_args(); config = configuration(args); output, completed = initialize(config)
    progress = Progress()
    if completed:
        progress.start("already complete", 1, 1); return
    try:
        validate_sources(); source_manifest(output)
        contexts = [load_context(config, output, my) for my in config["M_y_values"]]
        matrices = {}; runtime_rows = []
        for context in contexts:
            started = time.perf_counter(); meta = context["meta"]
            progress.start(f"diagnostics My={meta['M_y']}", 6, 0)
            engine, J, L, B, X, reused = load_or_repair_matrices(output, context, progress)
            progress.update(1, "saved matrices validated")
            obs = c_and_observability(config, output, context); progress.update(2, "observability")
            U, s, Vt = svd_checkpoint(output, meta, L, progress)
            progress.start(f"diagnostics My={meta['M_y']}", 6, 3)
            svd_diagnostics(output, context, engine, J, L, U, s, Vt, obs); progress.update(4, "singular modes")
            block_and_ablation(config, output, context, engine, L, B, s); progress.update(5, "block ablations")
            rhs_diagnostics(output, context, U, s, B, X, obs); progress.update(6, "RHS excitation")
            matrices[meta["run_id"]] = {"engine":engine,"J":J,"L":L,"B":B,"X":X,
                "U":U,"s":s,"Vt":Vt,"obs":obs}
            runtime_rows.append({**meta, "phase":"matrix_mode_diagnostics",
                "runtime_seconds":time.perf_counter()-started,
                "saved_34UH_J_reused":reused})
            append_rows(output/"runtime_summary.csv", [runtime_rows[-1]], ["run_id","phase"])
        started = time.perf_counter()
        audit, fit_cache, _ = legacy_convergence_audit(config, output, contexts, progress)
        add_converged_reference_errors(output)
        append_rows(output/"runtime_summary.csv", [{"run_id":"all","M_y":np.nan,
            "phase":"legacy_convergence_audit", "runtime_seconds":time.perf_counter()-started}],
            ["run_id","phase"])
        started = time.perf_counter()
        converged_epsilon_audit(config, output, contexts, audit, fit_cache, progress)
        append_rows(output/"runtime_summary.csv", [{"run_id":"all","M_y":np.nan,
            "phase":"converged_epsilon_sensitivity", "runtime_seconds":time.perf_counter()-started}],
            ["run_id","phase"])
        started = time.perf_counter()
        regularization_diagnostics(output, contexts, matrices)
        append_rows(output/"runtime_summary.csv", [{"run_id":"all","M_y":np.nan,
            "phase":"regularized_inverse_diagnostics", "runtime_seconds":time.perf_counter()-started}],
            ["run_id","phase"])
        decision_summary(output)
        for name in REQUIRED:
            if not (output/name).exists():
                atomic_csv(pd.DataFrame(), str(output/name))
        plot_results(output)
        atomic_json({"completed":True,"completed_at":time.time(),
                     "source_34UH_reused":True,
                     "legacy_fits_audited":int(len(audit)*4)}, output/"_COMPLETED.json")
        failed = output/"_FAILED.json"
        if failed.exists(): failed.unlink()
        progress.start("complete", 1, 1)
    except Exception as error:
        atomic_json({"completed":False,"error_type":type(error).__name__,
            "error_message":str(error),"traceback":traceback.format_exc(),
            "failed_at":time.time()}, output/"_FAILED.json")
        raise


if __name__ == "__main__":
    main()
