"""Sparsity-aware covariance gates for selected-direction Stage-B LRVB."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GatedCovariance:
    covariance: np.ndarray
    raw_covariance: np.ndarray
    diagnostics: dict


def project_psd(matrix, eigen_floor_relative=1e-8, eigen_floor_absolute=1e-12):
    raw = .5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(raw); maximum = max(float(values.max()), eigen_floor_absolute)
    floor = max(eigen_floor_absolute, eigen_floor_relative * maximum)
    projected_values = np.maximum(values, floor)
    projected = vectors @ np.diag(projected_values) @ vectors.T
    diagnostics = {"number_negative_variances": int(np.sum(np.diag(raw) < 0)),
        "fraction_negative_variances": float(np.mean(np.diag(raw) < 0)),
        "min_eigenvalue_before_projection": float(values.min()),
        "number_negative_eigenvalues": int(np.sum(values < 0)),
        "negative_eigenvalue_mass": float(-np.sum(values[values < 0])),
        "psd_projection_used": bool(np.any(values < floor)), "eigen_floor": float(floor)}
    return .5 * (projected + projected.T), diagnostics


def gated_covariance(louis_covariance, stage_b_covariance, coefficient_weights,
                     conservative_diagonal=False):
    """Construct Sigma_L + W(Sigma_B-Sigma_L)W and stabilize it."""
    louis = np.asarray(louis_covariance, dtype=float); stage_b = np.asarray(stage_b_covariance, dtype=float)
    weights = np.asarray(coefficient_weights, dtype=float)
    if louis.shape != stage_b.shape or louis.shape[0] != len(weights):
        raise ValueError("Covariances and coefficient weights have incompatible shapes.")
    raw = louis + np.outer(weights, weights) * (stage_b - louis)
    if conservative_diagonal:
        diagonal = np.maximum(np.diag(raw), np.diag(louis)); raw = raw.copy(); np.fill_diagonal(raw, diagonal)
    covariance, diagnostics = project_psd(raw)
    return GatedCovariance(covariance, raw, diagnostics)


def globally_damped_covariance(louis_covariance, stage_b_covariance, damping):
    raw = np.asarray(louis_covariance) + float(damping) * (np.asarray(stage_b_covariance) - np.asarray(louis_covariance))
    covariance, diagnostics = project_psd(raw)
    return GatedCovariance(covariance, raw, diagnostics)


def hill_weight(score, threshold, power):
    score = max(float(score), 0.0); threshold = max(float(threshold), 1e-15)
    return score ** power / (score ** power + threshold ** power)


def logistic_weight(score, threshold, slope):
    argument = np.clip(float(slope) * (float(score) - float(threshold)), -60., 60.)
    return 1. / (1. + np.exp(-argument))


def group_activity_scores(mean_group, current_covariance, louis_covariance, alpha_mean):
    mean = np.asarray(mean_group, dtype=float)
    current = np.asarray(current_covariance, dtype=float); louis = np.asarray(louis_covariance, dtype=float)
    return {"group_norm": float(np.linalg.norm(mean)),
        "group_snr_current": float(np.sqrt(max(mean @ np.linalg.pinv(current) @ mean, 0.))),
        "group_snr_louis": float(np.sqrt(max(mean @ np.linalg.pinv(louis) @ mean, 0.))),
        "E_alpha": float(alpha_mean), "inverse_alpha_score": float(1. / alpha_mean),
        "negative_log_alpha": float(-np.log(alpha_mean))}


def covariance_diagnostics(covariance, ordinary_vb, stabilized_louis,
                           coefficient_types):
    covariance = np.asarray(covariance); ordinary = np.asarray(ordinary_vb); louis = np.asarray(stabilized_louis)
    diagonal = np.diag(covariance); types = np.asarray(coefficient_types); sd = np.sqrt(np.maximum(diagonal, 0)); louis_sd = np.sqrt(np.maximum(np.diag(louis), 0))
    row = {"mean_variance": float(np.mean(diagonal)), "median_variance": float(np.median(diagonal)),
        "min_variance": float(np.min(diagonal)), "max_variance": float(np.max(diagonal)),
        "trace_selected": float(np.trace(covariance)),
        "trace_over_vb_selected": float(np.trace(covariance) / max(np.trace(ordinary), 1e-15)),
        "trace_over_louis_selected": float(np.trace(covariance) / max(np.trace(louis), 1e-15))}
    for kind in ("diagonal", "offdiag_nonzero", "offdiag_zero"):
        mask = types == kind
        row[f"mean_sd_over_louis_sd_{kind}"] = float(np.mean(sd[mask] / np.maximum(louis_sd[mask], 1e-15))) if np.any(mask) else np.nan
    return row


__all__ = ["GatedCovariance", "covariance_diagnostics", "gated_covariance",
           "globally_damped_covariance", "group_activity_scores", "hill_weight",
           "logistic_weight", "project_psd"]
