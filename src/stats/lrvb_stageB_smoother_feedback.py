"""Finite-difference Stage-B LRVB through the full Hybrid VB/smoother map."""
from dataclasses import dataclass
import time

import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.ssm.vb_ard_varx_ssm import HybridVBARDVARXSSMFixedBKnownC
from src.ssm.vb_ard_varx_ssm_b_controls import HybridVBARDVARXSSMKnownCWithBControls
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls


def global_index(target, lag, source, n_states, na=2):
    """Row-major index for beta rows, where each row is lag-major."""
    if not (0 <= lag < na): raise ValueError("lag is zero-based and outside range.")
    return target * (na * n_states) + lag * n_states + source


def decode_global_index(index, n_states, na=2):
    dimension = na * n_states; target, within = divmod(int(index), dimension)
    lag, source = divmod(within, n_states)
    return target, lag, source


def flatten_A(A_matrices):
    A = np.asarray(A_matrices); na, n_states, _ = A.shape
    return np.concatenate([np.concatenate([A[lag, target] for lag in range(na)]) for target in range(n_states)])


@dataclass
class PerturbedFitResult:
    theta: np.ndarray
    model: object
    converged: bool
    n_iter: int
    final_relative_A_change: float
    final_relative_alpha_change: float
    final_relative_B_change: float
    final_relative_Q_change: float
    final_loglikelihood: float
    runtime_seconds: float
    warning_flag: str


def fit_perturbed_hybrid_vb(baseline_model, y, u, perturbation, max_iter=50,
                            convergence_tol=1e-4, min_iter=5,
                            reestimate_B=None, reestimate_Q=None):
    """Warm-start and reconverge Hybrid VB with a linear A perturbation.

    ``perturbation`` has shape (M, na*M) and enters the Gaussian A-update
    natural parameter. C and R remain fixed; B and diagonal Q can optionally
    re-estimate so nuisance and smoother feedback enter the sensitivity.
    """
    perturbation = np.asarray(perturbation, dtype=float)
    M, dimension = perturbation.shape
    baseline_estimates_B = bool(getattr(baseline_model, "estimate_B", False))
    baseline_estimates_Q = bool(getattr(baseline_model, "estimate_Q", False))
    reestimate_B = baseline_estimates_B if reestimate_B is None else bool(reestimate_B)
    reestimate_Q = baseline_estimates_Q if reestimate_Q is None else bool(reestimate_Q)
    if reestimate_Q and not reestimate_B:
        # The BQ-controls class can hold B fixed through its fixed_true mode.
        B_update_mode = "fixed"
    else:
        B_update_mode = "free"
    common = dict(max_iter=max_iter, tol_objective=convergence_tol,
        tol_A_change=convergence_tol, a0=baseline_model.a0, b0=baseline_model.b0,
        diagonal_prior_precision=baseline_model.diagonal_prior_precision,
        posterior_jitter=baseline_model.posterior_jitter,
        init_A_matrices=baseline_model.A_mean_matrices_, verbose=False,
        random_state=baseline_model.random_state)
    if reestimate_Q:
        model = HybridVBARDVARXSSMKnownCWithBQControls(
            baseline_model.na, baseline_model.nb, baseline_model.C,
            baseline_model.Q, baseline_model.R, B_update_mode=B_update_mode,
            fixed_B_matrices=(baseline_model.B_matrices if not reestimate_B else None),
            initial_B_matrices=(baseline_model.B_matrices if reestimate_B else None),
            B_ridge_lambda=getattr(baseline_model, "effective_B_ridge", 0.),
            base_B_ridge=0., tol_B_change=convergence_tol,
            estimate_Q=True, Q_update_mode=getattr(baseline_model, "Q_update_mode", "diag_shrink_scalar"),
            Q_floor_mode=getattr(baseline_model, "Q_floor_mode", "none"),
            Q_floor_value=getattr(baseline_model, "Q_floor_value", 0.),
            Q_shrinkage_rho=getattr(baseline_model, "Q_shrinkage_rho", .25),
            Q_update_damping=getattr(baseline_model, "Q_update_damping", .5),
            include_A_posterior_uncertainty_in_Q=getattr(baseline_model, "include_A_posterior_uncertainty_in_Q", True),
            tol_Q_change=convergence_tol, initial_Q=baseline_model.Q,
            initial_alpha_mean=baseline_model.alpha_mean_, tol_alpha_change=convergence_tol,
            **common)
    elif reestimate_B:
        model = HybridVBARDVARXSSMKnownCWithBControls(
            baseline_model.na, baseline_model.nb, baseline_model.C,
            baseline_model.Q, baseline_model.R, B_update_mode="free",
            initial_B_matrices=baseline_model.B_matrices,
            B_ridge_lambda=getattr(baseline_model, "effective_B_ridge", 0.),
            base_B_ridge=0., tol_B_change=convergence_tol,
            tol_alpha_change=convergence_tol, **common)
    else:
        model = HybridVBARDVARXSSMFixedBKnownC(
            baseline_model.na, baseline_model.nb, baseline_model.C,
            baseline_model.B_matrices, baseline_model.Q, baseline_model.R,
            **common)
    if perturbation.shape != (M, baseline_model.na * M):
        raise ValueError("perturbation must have shape (M, na*M).")
    A = [np.asarray(value).copy() for value in baseline_model.A_mean_matrices_]
    model.B_matrices = [np.asarray(value).copy() for value in baseline_model.B_matrices]
    model.alpha_mean_ = np.asarray(baseline_model.alpha_mean_).copy()
    warning = ""; converged = False; started = time.perf_counter()
    A_change = alpha_change = B_change = Q_change = np.inf
    for iteration in range(int(max_iter)):
        smooth = model.smooth(y, u, A)
        stats = model._compute_posterior_sufficient_statistics(smooth, u)
        old_beta = model._pack_A(A); old_alpha = model.alpha_mean_.copy(); old_B = np.asarray(model.B_matrices).copy(); old_Q = model.Q.copy()
        perturbed_stats = dict(stats); perturbed_stats["S_zx"] = stats["S_zx"].copy()
        for target in range(M):
            perturbed_stats["S_zx"][:, target] += model.Q[target, target] * perturbation[target]
        model._update_q_A(perturbed_stats)
        radius = var_companion_spectral_radius(model.A_mean_matrices_)
        if not np.isfinite(radius): raise FloatingPointError("Non-finite perturbed companion radius.")
        if radius >= .98:
            model.A_mean_matrices_, _, _ = stabilize_A_matrices_if_needed(model.A_mean_matrices_, target_radius=.90)
            model._beta_means_ = model._pack_A(model.A_mean_matrices_); warning = "stability_rescaling"
        model._update_q_alpha()
        if reestimate_B: model._update_B(smooth, u)
        if reestimate_Q: model._update_Q(smooth, u)
        A = [value.copy() for value in model.A_mean_matrices_]
        A_change = np.linalg.norm(model._beta_means_ - old_beta) / max(np.linalg.norm(old_beta), 1e-12)
        mask = ~np.eye(M, dtype=bool)
        alpha_change = np.linalg.norm(model.alpha_mean_[mask] - old_alpha[mask]) / max(np.linalg.norm(old_alpha[mask]), 1e-12)
        B_change = np.linalg.norm(np.asarray(model.B_matrices)-old_B) / max(np.linalg.norm(old_B), 1e-12)
        Q_change = np.linalg.norm(model.Q-old_Q) / max(np.linalg.norm(old_Q), 1e-12)
        if iteration + 1 >= min_iter and A_change < convergence_tol and alpha_change < convergence_tol and (not reestimate_B or B_change < convergence_tol) and (not reestimate_Q or Q_change < convergence_tol):
            converged = True; break
    model.smooth_result_ = model.smooth(y, u, A); model.n_iter_ = iteration + 1; model.converged_ = converged
    model.filtered_state_mean_ = model.smooth_result_["filter"]["x_filt"][:, :M]
    model.smoothed_state_mean_ = model.smooth_result_["smoother"]["x_smooth"][:, :M]
    model._refresh_derived()
    return PerturbedFitResult(flatten_A(model.A_mean_matrices_), model, converged,
        iteration + 1, float(A_change), float(alpha_change), float(B_change), float(Q_change),
        float(model.smooth_result_["log_likelihood"]), time.perf_counter() - started,
        warning)


def sensitivity_column(baseline_model, y, u, direction_index, epsilon,
                       max_iter=50, convergence_tol=1e-4, min_iter=5,
                       reestimate_B=None, reestimate_Q=None):
    """Compute one central finite-difference Stage-B covariance column."""
    M, na = baseline_model.n_states, baseline_model.na
    target, lag, source = decode_global_index(direction_index, M, na)
    plus = np.zeros((M, na * M)); minus = np.zeros_like(plus)
    local = lag * M + source; plus[target, local] = epsilon; minus[target, local] = -epsilon
    fit_plus = fit_perturbed_hybrid_vb(baseline_model, y, u, plus, max_iter, convergence_tol, min_iter, reestimate_B, reestimate_Q)
    fit_minus = fit_perturbed_hybrid_vb(baseline_model, y, u, minus, max_iter, convergence_tol, min_iter, reestimate_B, reestimate_Q)
    column = (fit_plus.theta - fit_minus.theta) / (2. * epsilon)
    return column, fit_plus, fit_minus


def project_selected_covariance(columns, selected_indices,
                                eigen_floor_relative=1e-8,
                                eigen_floor_absolute=1e-12):
    """Symmetrize and PSD-project the selected square covariance block."""
    selected = np.asarray(selected_indices, dtype=int)
    raw_block = np.asarray(columns, dtype=float)[selected, :]
    symmetry_error = np.linalg.norm(raw_block - raw_block.T) / max(np.linalg.norm(raw_block), 1e-15)
    symmetric = .5 * (raw_block + raw_block.T)
    values, vectors = np.linalg.eigh(symmetric)
    maximum = max(float(values.max()), eigen_floor_absolute)
    floor = max(eigen_floor_absolute, eigen_floor_relative * maximum)
    projected_values = np.maximum(values, floor)
    projected = vectors @ np.diag(projected_values) @ vectors.T
    diagnostics = {"symmetry_error": float(symmetry_error),
        "min_eigenvalue_raw": float(values.min()),
        "number_negative_eigenvalues": int(np.sum(values < 0)),
        "negative_eigenvalue_mass": float(-np.sum(values[values < 0])),
        "psd_projection_used": bool(np.any(values < floor)),
        "eigen_floor": float(floor)}
    return raw_block, .5 * (projected + projected.T), diagnostics


__all__ = ["PerturbedFitResult", "decode_global_index", "fit_perturbed_hybrid_vb",
           "flatten_A", "global_index", "project_selected_covariance",
           "sensitivity_column"]
