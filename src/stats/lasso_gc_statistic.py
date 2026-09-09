"""Eq. 7 residual-loss Granger score with a Numba-backed LASSO fit."""

import numpy as np
import pandas as pd

from src.ssm.em_varx_p_known_c_l1_mstep import (
    NUMBA_AVAILABLE,
    _coordinate_descent_core,
)
from src.stats.scalable_debiased_varx_network import (
    build_varx_design,
    source_lag_columns,
)


LGC_THRESHOLDS = (0.001, 0.003, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50)


def _threshold_label(value):
    return f"{value:g}".replace(".", "p")


def fit_standardized_lasso(X, y, lambda_fraction, max_iter=5000, tol=1e-8):
    """Fit no-intercept standardized LASSO; lambda is a fraction of lambda-max.

    The no-intercept convention matches ``build_varx_design`` and guarantees
    that lambda=0 has the stated Eq. 7 relationship to the existing raw GC.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if lambda_fraction < 0:
        raise ValueError("lambda_fraction must be nonnegative.")
    scale = np.sqrt(np.mean(X * X, axis=0))
    safe_scale = np.where(scale > 1e-12, scale, 1.0)
    X_std = np.ascontiguousarray(X / safe_scale, dtype=np.float64)
    n_samples, n_features = X_std.shape
    lambda_max = float(np.max(np.abs(X_std.T @ y)) / n_samples) if n_features else 0.0
    if lambda_fraction == 0.0:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted_residual = y - X @ beta
        return {
            "beta": beta, "intercept": 0.0,
            "loss": float(np.mean(fitted_residual ** 2)),
            "lambda_max": lambda_max, "lambda_effective": 0.0,
            "iterations": 1, "converged": True,
        }
    penalty = np.full(n_features, lambda_fraction * lambda_max, dtype=np.float64)
    ridge = np.zeros(n_features, dtype=np.float64)
    beta_std = np.zeros(n_features, dtype=np.float64)
    norms = np.mean(X_std * X_std, axis=0)
    beta_std, residual, iterations, converged = _coordinate_descent_core(
        X_std, y, penalty, ridge, beta_std, norms, max_iter, tol
    )
    beta = beta_std / safe_scale
    intercept = 0.0
    fitted_residual = y - X @ beta
    return {
        "beta": beta, "intercept": intercept,
        "loss": float(np.mean(fitted_residual ** 2)),
        "lambda_max": lambda_max,
        "lambda_effective": float(lambda_fraction * lambda_max),
        "iterations": int(iterations), "converged": bool(converged),
    }


def compute_lasso_gc_eq7_network(
        x, u, true_link_mask, na, nb, lambda_lasso,
        alpha=0.05, custom_thresholds=LGC_THRESHOLDS,
        max_iter=5000, tol=1e-8):
    """Compute Das--Babadi Eq. 7 score for every directed off-diagonal pair.

    LGC = (reduced residual MSE - full residual MSE) / full residual MSE.
    The raw value is retained; only ``lgc_clipped`` is clipped for scoring.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    true_link_mask = np.asarray(true_link_mask, dtype=bool)
    n_sources = x.shape[1]
    rows = []
    for target in range(n_sources):
        X_full, y, _ = build_varx_design(x, u, na, nb, target)
        full = fit_standardized_lasso(
            X_full, y, lambda_lasso, max_iter=max_iter, tol=tol
        )
        for source in range(n_sources):
            if source == target:
                continue
            removed = set(source_lag_columns(source, n_sources, na))
            keep = [column for column in range(X_full.shape[1]) if column not in removed]
            reduced = fit_standardized_lasso(
                X_full[:, keep], y, lambda_lasso, max_iter=max_iter, tol=tol
            )
            full_loss = max(full["loss"], 1e-15)
            lgc_raw = float((reduced["loss"] - full["loss"]) / full_loss)
            row = {
                "target": target, "source": source,
                "direction": f"{source}->{target}",
                "true_link": bool(true_link_mask[target, source]),
                "lgc_raw": lgc_raw, "lgc_clipped": max(lgc_raw, 0.0),
                "lgc_full_loss": full["loss"],
                "lgc_reduced_loss": reduced["loss"],
                "lgc_negative_flag": lgc_raw < 0.0,
                "lgc_lambda": float(lambda_lasso),
                "lgc_full_lambda_effective": full["lambda_effective"],
                "lgc_reduced_lambda_effective": reduced["lambda_effective"],
                "lgc_full_iterations": full["iterations"],
                "lgc_reduced_iterations": reduced["iterations"],
                "lgc_full_converged": full["converged"],
                "lgc_reduced_converged": reduced["converged"],
            }
            for threshold in custom_thresholds:
                row[f"lgc_detected_threshold_{_threshold_label(threshold)}"] = (
                    max(lgc_raw, 0.0) > threshold
                )
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "NUMBA_AVAILABLE", "LGC_THRESHOLDS", "fit_standardized_lasso",
    "compute_lasso_gc_eq7_network",
]
