"""One-step observed-likelihood de-biasing for row-wise latent VARX A.

The score is the Fisher-identity observed score evaluated from exact Kalman
smoother moments.  It deliberately excludes the ARD prior.  Coefficients use
the project's lag-major ordering ``[A1[i, :], ..., Ap[i, :]]``.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.stats.louis_missing_information import (
    stabilize_observed_information,
    stabilized_louis_covariance,
)


@njit(cache=True, parallel=True)
def observed_score_rows_numba(S_zz, S_zx, beta_rows, q_diag):
    """Return unpenalized Fisher-identity scores for all target rows."""
    n_rows, dimension = beta_rows.shape
    scores = np.empty((n_rows, dimension))
    for target in prange(n_rows):
        for a in range(dimension):
            value = S_zx[a, target]
            for b in range(dimension):
                value -= S_zz[a, b] * beta_rows[target, b]
            scores[target, a] = value / q_diag[target]
    return scores


def observed_score_rows(sufficient_statistics, beta_rows, Q):
    """Compute row scores from the same sufficient statistics as q(A)."""
    G = np.asarray(sufficient_statistics["S_zz"], dtype=float)
    H = np.asarray(sufficient_statistics["S_zx"], dtype=float)
    beta = np.asarray(beta_rows, dtype=float)
    q_diag = np.diag(np.asarray(Q, dtype=float)).copy()
    if G.shape != (beta.shape[1], beta.shape[1]) or H.shape != (beta.shape[1], beta.shape[0]):
        raise ValueError("Sufficient statistics and beta_rows have incompatible shapes.")
    if np.any(q_diag <= 0):
        raise ValueError("Q diagonal must be positive.")
    return observed_score_rows_numba(G, H, beta, q_diag)


def build_information_variants(current_covariances, missing_information):
    """Build the three 34M information variants and their covariances."""
    information = {"currentInfo": [], "louis0p50": [], "stabilizedLouis": []}
    covariance = {key: [] for key in information}
    diagnostics = []
    for target, (current, missing) in enumerate(zip(current_covariances, missing_information)):
        complete = np.linalg.pinv(np.asarray(current, dtype=float))
        cov_050, info_050, diag_050 = stabilize_observed_information(complete, missing, .50)
        cov_stable, info_stable, diag_stable = stabilized_louis_covariance(
            complete, missing, eta=.70, tau=.90
        )
        information["currentInfo"].append(complete)
        covariance["currentInfo"].append(np.asarray(current, dtype=float))
        information["louis0p50"].append(info_050)
        covariance["louis0p50"].append(cov_050)
        information["stabilizedLouis"].append(info_stable)
        covariance["stabilizedLouis"].append(cov_stable)
        diagnostics.extend([
            {"target_row": target, "information_used_for_debias": "louis0p50", **diag_050},
            {"target_row": target, "information_used_for_debias": "stabilizedLouis", **diag_stable},
        ])
    return information, covariance, diagnostics


def solve_row_steps(scores, information_by_name, jitter=1e-10):
    """Solve I delta = score, returning per-information row corrections."""
    results = {}
    for name, matrices in information_by_name.items():
        rows = []
        for score, matrix in zip(scores, matrices):
            symmetric = .5 * (np.asarray(matrix) + np.asarray(matrix).T)
            try:
                rows.append(np.linalg.solve(symmetric + jitter * np.eye(len(score)), score))
            except np.linalg.LinAlgError:
                rows.append(np.linalg.pinv(symmetric) @ score)
        results[name] = np.asarray(rows)
    return results


def unpack_beta_rows(beta_rows, n_states, na):
    beta_rows = np.asarray(beta_rows)
    return np.asarray([
        beta_rows[:, lag * n_states:(lag + 1) * n_states].copy()
        for lag in range(na)
    ])


@dataclass
class DebiasedEstimate:
    raw_A: np.ndarray
    stable_A: np.ndarray
    raw_beta: np.ndarray
    stable_beta: np.ndarray
    spectral_radius_before: float
    spectral_radius_after: float
    stable_before: bool
    rescaling_factor: float


def apply_debias_step(beta_vb, delta_rows, step_size, n_states, na,
                      stability_threshold=.98, target_radius=.90):
    """Apply a damped correction and return raw plus stability-projected A."""
    raw_beta = np.asarray(beta_vb) + float(step_size) * np.asarray(delta_rows)
    raw_A = unpack_beta_rows(raw_beta, n_states, na)
    before = float(var_companion_spectral_radius(raw_A))
    stable_before = bool(np.isfinite(before) and before < stability_threshold)
    stable_A = raw_A.copy(); factor = 1.0; after = before
    if not stable_before:
        stable_list, after, factor = stabilize_A_matrices_if_needed(
            [x.copy() for x in raw_A], target_radius=target_radius
        )
        stable_A = np.asarray(stable_list)
    stable_beta = np.stack([
        np.concatenate([stable_A[lag, target] for lag in range(na)])
        for target in range(n_states)
    ])
    return DebiasedEstimate(raw_A, stable_A, raw_beta, stable_beta, before,
                            float(after), stable_before, float(factor))


def predicted_quadratic_gain(scores, delta_rows, information_rows, step_size):
    """Quadratic observed-likelihood gain for the applied damped step."""
    lam = float(step_size); values = []
    for score, delta, info in zip(scores, delta_rows, information_rows):
        values.append(lam * score @ delta - .5 * lam * lam * delta @ info @ delta)
    return np.asarray(values, dtype=float)


__all__ = [
    "DebiasedEstimate", "apply_debias_step", "build_information_variants",
    "observed_score_rows", "observed_score_rows_numba",
    "predicted_quadratic_gain", "solve_row_steps", "unpack_beta_rows",
]
