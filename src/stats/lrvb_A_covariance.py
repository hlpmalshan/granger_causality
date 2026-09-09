"""Stage-A linear-response VB covariance for latent VARX A rows.

Stage A freezes the Kalman-smoother sufficient statistics and differentiates
only the fixed-point dependence between a row of A and its off-diagonal ARD
precision expectations.  Coefficients follow lag-major ordering.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LRVBRowResult:
    covariance_raw: np.ndarray
    covariance_psd: np.ndarray
    fixed_point_mean: np.ndarray
    fixed_point_alpha: np.ndarray
    jacobian: np.ndarray
    perturbation_jacobian: np.ndarray
    diagnostics: dict


def _source_columns(source, n_states, na):
    return np.asarray([lag * n_states + source for lag in range(na)], dtype=int)


def stage_a_row_map(eta, perturbation, S_zz, h_i, q_i, target, n_states,
                    na, a0=1e-3, b0=1e-3,
                    diagonal_prior_precision=1e-4,
                    posterior_jitter=1e-8):
    """Evaluate one simultaneous row-wise A/alpha variational update."""
    dimension = n_states * na
    sources = [source for source in range(n_states) if source != target]
    alpha = np.asarray(eta[dimension:], dtype=float)
    if len(alpha) != len(sources) or np.any(alpha <= 0):
        raise ValueError("eta has invalid ARD precision block.")
    precision_diagonal = np.empty(dimension)
    precision_diagonal[_source_columns(target, n_states, na)] = diagonal_prior_precision
    for index, source in enumerate(sources):
        precision_diagonal[_source_columns(source, n_states, na)] = alpha[index]
    precision = np.asarray(S_zz) / q_i + np.diag(precision_diagonal)
    precision += posterior_jitter * np.eye(dimension)
    covariance = np.linalg.pinv(.5 * (precision + precision.T))
    mean = covariance @ (np.asarray(h_i) / q_i + np.asarray(perturbation))
    shape = a0 + .5 * na
    alpha_new = np.empty(len(sources))
    for index, source in enumerate(sources):
        columns = _source_columns(source, n_states, na)
        second_moment = mean[columns] @ mean[columns] + np.trace(covariance[np.ix_(columns, columns)])
        alpha_new[index] = shape / (b0 + .5 * second_moment)
    return np.concatenate([mean, alpha_new]), covariance


def solve_stage_a_fixed_point(initial_mean, initial_alpha, S_zz, h_i, q_i,
                              target, n_states, na, max_iter=500, tol=1e-10,
                              **map_kwargs):
    """Converge the frozen-smoother Stage-A map before differentiation."""
    eta = np.concatenate([np.asarray(initial_mean), np.asarray(initial_alpha)])
    perturbation = np.zeros(n_states * na)
    residual = np.inf
    for iteration in range(int(max_iter)):
        updated, covariance = stage_a_row_map(eta, perturbation, S_zz, h_i, q_i,
                                               target, n_states, na, **map_kwargs)
        residual = np.linalg.norm(updated - eta) / max(np.linalg.norm(eta), 1e-12)
        eta = updated
        if residual < tol:
            break
    return eta, covariance, iteration + 1, float(residual)


def _central_jacobian(function, point, epsilon):
    point = np.asarray(point, dtype=float); baseline = np.asarray(function(point))
    result = np.empty((len(baseline), len(point)))
    for column in range(len(point)):
        shift = np.zeros_like(point); shift[column] = epsilon
        result[:, column] = (function(point + shift) - function(point - shift)) / (2.0 * epsilon)
    return result


def stage_a_lrvb_covariance(initial_mean, initial_alpha, S_zz, h_i, q_i,
                            target, n_states, na, epsilon=1e-4,
                            eigen_floor_relative=1e-8,
                            eigen_floor_absolute=1e-12, **map_kwargs):
    """Compute the row LRVB covariance using central finite differences."""
    dimension = n_states * na
    eta, _, fixed_iterations, fixed_residual = solve_stage_a_fixed_point(
        initial_mean, initial_alpha, S_zz, h_i, q_i, target, n_states, na,
        **map_kwargs)
    zero_t = np.zeros(dimension)

    def map_eta(value):
        return stage_a_row_map(value, zero_t, S_zz, h_i, q_i, target,
                               n_states, na, **map_kwargs)[0]

    def map_t(value):
        return stage_a_row_map(eta, value, S_zz, h_i, q_i, target,
                               n_states, na, **map_kwargs)[0]

    J = _central_jacobian(map_eta, eta, float(epsilon))
    B = _central_jacobian(map_t, zero_t, float(epsilon))
    system = np.eye(len(eta)) - J
    sensitivity = np.linalg.solve(system, B)
    raw = sensitivity[:dimension]
    symmetry_error = np.linalg.norm(raw - raw.T) / max(np.linalg.norm(raw), 1e-15)
    symmetric = .5 * (raw + raw.T)
    values, vectors = np.linalg.eigh(symmetric)
    max_eigenvalue = max(float(values.max()), eigen_floor_absolute)
    floor = max(eigen_floor_absolute, eigen_floor_relative * max_eigenvalue)
    projected_values = np.maximum(values, floor)
    projected = vectors @ np.diag(projected_values) @ vectors.T
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(J))))
    diagnostics = {
        "eps_used": float(epsilon), "row_dimension_D": dimension,
        "alpha_group_dimension_G": n_states - 1,
        "total_eta_dimension": len(eta), "spectral_radius_J": spectral_radius,
        "condition_number_I_minus_J": float(np.linalg.cond(system)),
        "trace_lrvb_raw": float(np.trace(symmetric)),
        "trace_lrvb_psd": float(np.trace(projected)),
        "min_eigenvalue_lrvb_raw": float(values.min()),
        "number_negative_eigenvalues_lrvb_raw": int(np.sum(values < 0)),
        "negative_eigenvalue_mass_lrvb_raw": float(-np.sum(values[values < 0])),
        "psd_projection_used": bool(np.any(values < floor)),
        "symmetry_error_lrvb_raw": float(symmetry_error),
        "condition_number_before_projection": float(np.linalg.cond(symmetric)),
        "eigen_floor": float(floor), "fixed_point_iterations": fixed_iterations,
        "fixed_point_relative_residual": fixed_residual,
    }
    return LRVBRowResult(raw, .5 * (projected + projected.T), eta[:dimension],
                         eta[dimension:], J, B, diagnostics)


def epsilon_stability(results_by_epsilon):
    """Summarize trace and diagonal-SD stability across epsilon choices."""
    epsilons = sorted(results_by_epsilon)
    traces = np.asarray([np.trace(results_by_epsilon[eps].covariance_psd) for eps in epsilons])
    sds = np.asarray([np.sqrt(np.maximum(np.diag(results_by_epsilon[eps].covariance_psd), 0)) for eps in epsilons])
    trace_cv = float(np.std(traces) / max(abs(np.mean(traces)), 1e-15))
    sd_cv = np.std(sds, axis=0) / np.maximum(np.mean(sds, axis=0), 1e-15)
    return {"trace_coefficient_of_variation_across_eps": trace_cv,
            "mean_diagonal_sd_cv_across_eps": float(np.mean(sd_cv)),
            "max_diagonal_sd_cv_across_eps": float(np.max(sd_cv))}


__all__ = ["LRVBRowResult", "epsilon_stability", "solve_stage_a_fixed_point",
           "stage_a_lrvb_covariance", "stage_a_row_map"]
