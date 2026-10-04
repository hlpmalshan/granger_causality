"""Explicit-Jacobian and reusable-factorization utilities for Hybrid LRVB.

Every Jacobian column is produced by the analytic tangent engine.  This module
contains no finite-difference Jacobian construction.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import dgecon
from scipy.sparse.linalg import LinearOperator, gmres

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.exact_implicit_lrvb import HybridFixedPointTangentMap


_WORKER_ENGINE = None


def _worker_initialize(model, y, u, reduced, inner_threads):
    global _WORKER_ENGINE
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(max(1, int(inner_threads)))
    _WORKER_ENGINE = HybridFixedPointTangentMap(model, y, u, reduced=bool(reduced))


def _worker_column(index):
    started = time.perf_counter()
    vector = np.zeros(len(_WORKER_ENGINE.theta0)); vector[int(index)] = 1.0
    column = _WORKER_ENGINE.jvp(vector)
    return int(index), column, time.perf_counter() - started, os.getpid()


def _atomic_bool_array(path, values):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, bool))
    os.replace(temporary, path)


def benchmark_column_workers(model, y, u, reduced, indices, worker_values,
                             inner_threads=1):
    """Benchmark independent exact-JVP columns for controlled worker counts."""
    rows = []
    for workers in sorted(set(max(1, int(value)) for value in worker_values)):
        started = time.perf_counter(); norms = []; finite = True
        if workers == 1:
            engine = HybridFixedPointTangentMap(model, y, u, reduced=reduced)
            for index in indices:
                vector = np.zeros(len(engine.theta0)); vector[int(index)] = 1.0
                column = engine.jvp(vector); norms.append(float(np.linalg.norm(column)))
                finite = finite and bool(np.all(np.isfinite(column)))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initialize,
                    initargs=(model, y, u, reduced, inner_threads)) as pool:
                futures = [pool.submit(_worker_column, int(index)) for index in indices]
                for future in as_completed(futures):
                    _, column, _, _ = future.result(); norms.append(float(np.linalg.norm(column)))
                    finite = finite and bool(np.all(np.isfinite(column)))
        elapsed = time.perf_counter() - started
        rows.append({"workers": workers, "n_benchmark_columns": len(indices),
            "runtime_seconds": elapsed, "seconds_per_column": elapsed / max(len(indices), 1),
            "columns_per_second": len(indices) / max(elapsed, 1e-15),
            "finite": finite, "median_column_norm": float(np.median(norms)) if norms else np.nan})
    return pd.DataFrame(rows)


def assemble_exact_jacobian(engine, checkpoint_directory, workers=1,
                            inner_threads=1, checkpoint_every=20,
                            progress_callback=None):
    """Assemble an exact dense J, resuming at individual-column boundaries."""
    directory = Path(checkpoint_directory); directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / ("J_reduced.npy" if engine.reduced else "J_full.npy")
    completed_path = directory / ("J_reduced_completed.npy" if engine.reduced else "J_full_completed.npy")
    runtime_path = directory / ("jacobian_reduced_column_runtime.csv" if engine.reduced else "jacobian_full_column_runtime.csv")
    segment_path = directory / ("jacobian_reduced_assembly_segments.csv" if engine.reduced else "jacobian_full_assembly_segments.csv")
    n = len(engine.theta0)
    if matrix_path.exists():
        matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
        if matrix.shape != (n, n):
            raise ValueError(f"Existing Jacobian has shape {matrix.shape}, expected {(n, n)}.")
    else:
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float64, shape=(n, n))
        matrix[:] = np.nan; matrix.flush()
    if completed_path.exists():
        completed = np.load(completed_path).astype(bool)
        if completed.shape != (n,):
            raise ValueError("Jacobian completion bitmap has the wrong shape.")
    else:
        completed = np.zeros(n, bool); _atomic_bool_array(completed_path, completed)
    # A completed bit is trusted only when its actual column is finite.
    completed &= np.all(np.isfinite(matrix), axis=0)
    pending = np.flatnonzero(~completed).tolist(); rows = []
    segment_started = time.perf_counter(); written = 0

    def accept(index, column, seconds, worker_id):
        nonlocal written
        column = np.asarray(column, np.float64)
        if column.shape != (n,) or not np.all(np.isfinite(column)):
            raise FloatingPointError(f"Invalid exact Jacobian column {index}.")
        matrix[:, index] = column; completed[index] = True; written += 1
        rows.append({"column_index": int(index), "runtime_seconds": float(seconds),
            "finite": True, "column_norm": float(np.linalg.norm(column)),
            "worker_id": int(worker_id)})
        if progress_callback is not None:
            progress_callback(int(completed.sum()), n, f"Jacobian column {index}")
        if written % max(1, int(checkpoint_every)) == 0:
            matrix.flush(); _atomic_bool_array(completed_path, completed)
            old = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
            combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("column_index", keep="last")
            atomic_csv(combined, str(runtime_path)); rows.clear()

    if pending and int(workers) == 1:
        for index in pending:
            started = time.perf_counter(); vector = np.zeros(n); vector[index] = 1.0
            accept(index, engine.jvp(vector), time.perf_counter() - started, os.getpid())
    elif pending:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_initialize,
                initargs=(engine.model, engine.y, engine.u, engine.reduced, inner_threads)) as pool:
            futures = {pool.submit(_worker_column, index): index for index in pending}
            for future in as_completed(futures):
                accept(*future.result())
    matrix.flush(); _atomic_bool_array(completed_path, completed)
    if rows:
        old = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
        combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("column_index", keep="last")
        atomic_csv(combined, str(runtime_path))
    segment = pd.DataFrame([{"started_at": time.time() - (time.perf_counter() - segment_started),
        "runtime_seconds": time.perf_counter() - segment_started, "columns_written": written,
        "workers": int(workers), "complete_after_segment": bool(completed.all())}])
    old_segments = pd.read_csv(segment_path) if segment_path.exists() else pd.DataFrame()
    atomic_csv(pd.concat([old_segments, segment], ignore_index=True), str(segment_path))
    if not completed.all() or not np.all(np.isfinite(matrix)):
        raise FloatingPointError("Explicit Jacobian assembly is incomplete or non-finite.")
    return np.asarray(matrix), runtime_path, segment_path


def factorize_response_matrix(J):
    J = np.asarray(J, np.float64); response = np.eye(len(J)) - J
    started = time.perf_counter(); lu, piv = lu_factor(response, check_finite=True)
    seconds = time.perf_counter() - started
    anorm = float(np.linalg.norm(response, 1))
    try:
        rcond, info = dgecon(lu, anorm, norm="1")
        rcond = float(rcond) if info == 0 else np.nan
    except Exception:
        rcond, info = np.nan, -1
    diagnostics = {"dimension": len(J), "factorization_seconds": seconds,
        "factorization_finite": bool(np.all(np.isfinite(lu))),
        "matrix_1_norm": anorm, "matrix_frobenius_norm": float(np.linalg.norm(response)),
        "asymmetry_relative_frobenius": float(np.linalg.norm(response-response.T) / max(np.linalg.norm(response), 1e-15)),
        "minimum_absolute_diagonal": float(np.min(np.abs(np.diag(response)))),
        "maximum_absolute_diagonal": float(np.max(np.abs(np.diag(response)))),
        "reciprocal_condition_estimate_1norm": rcond,
        "condition_estimate_1norm": float(1/rcond) if np.isfinite(rcond) and rcond > 0 else np.inf,
        "lapack_condition_info": int(info), "number_nontrivial_pivots": int(np.sum(piv != np.arange(len(piv)))),
        "maximum_pivot_displacement": int(np.max(np.abs(piv-np.arange(len(piv))))) if len(piv) else 0}
    return response, lu, piv, diagnostics


def solve_multiple_rhs(response, lu, piv, B):
    B = np.asarray(B, np.float64)
    started = time.perf_counter(); X_batch = lu_solve((lu, piv), B, check_finite=True)
    batch_seconds = time.perf_counter() - started
    started = time.perf_counter()
    X_loop = np.column_stack([lu_solve((lu, piv), B[:, index], check_finite=False)
                              for index in range(B.shape[1])])
    loop_seconds = time.perf_counter() - started
    residuals = np.linalg.norm(response @ X_batch - B, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-15)
    return X_batch, {"n_rhs": B.shape[1], "batched_seconds": batch_seconds,
        "looped_seconds": loop_seconds, "batched_faster": bool(batch_seconds <= loop_seconds),
        "batch_loop_relative_difference": float(np.linalg.norm(X_batch-X_loop) / max(np.linalg.norm(X_batch), 1e-15)),
        "maximum_relative_residual": float(np.max(residuals)),
        "median_relative_residual": float(np.median(residuals))}, residuals


def explicit_gmres(response, rhs, rtol=1e-5, restart=50, total_iterations=100,
                   preconditioner=None):
    response = np.asarray(response, float); rhs = np.asarray(rhs, float); residual_history = []
    operator = LinearOperator(response.shape, matvec=lambda value: response @ value, dtype=float)
    cycles = max(1, int(np.ceil(total_iterations / max(int(restart), 1))))
    started = time.perf_counter()
    solution, info = gmres(operator, rhs, M=preconditioner, rtol=rtol, atol=0.0,
        restart=int(restart), maxiter=cycles,
        callback=lambda value: residual_history.append(float(value)), callback_type="pr_norm")
    residual = float(np.linalg.norm(response @ solution-rhs) / max(np.linalg.norm(rhs), 1e-15))
    return solution, {"converged": bool(info == 0), "solver_info": int(info),
        "iterations": len(residual_history), "relative_residual": residual,
        "runtime_seconds": time.perf_counter()-started, "residual_history": residual_history}


def block_jacobi_preconditioner(response, block_slices):
    factors = []
    for name, block in block_slices:
        matrix = np.asarray(response[block, block], float)
        factors.append((name, block, lu_factor(matrix, check_finite=True)))
    def apply(value):
        result = np.empty_like(value)
        for _, block, factor in factors:
            result[block] = lu_solve(factor, value[block], check_finite=False)
        return result
    return LinearOperator(response.shape, matvec=apply, dtype=float), factors


__all__ = ["assemble_exact_jacobian", "benchmark_column_workers",
           "block_jacobi_preconditioner", "explicit_gmres",
           "factorize_response_matrix", "solve_multiple_rhs"]
