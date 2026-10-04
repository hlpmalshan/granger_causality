"""Matrix-free implicit differentiation for the Hybrid Kalman + VB-ARD map.

This module linearizes one complete fixed-point update, not a converged
perturbed refit.  The update state contains A means, off-diagonal ARD means,
B, and diagonal Q.  Because the implementation updates these blocks
sequentially, its Jacobian is generally nonsymmetric and GMRES is the default.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.stats.lrvb_stageB_smoother_feedback import global_index


@dataclass
class HybridStateLayout:
    n_states: int
    na: int
    nb: int
    n_inputs: int
    offdiag: np.ndarray
    A_slice: slice
    alpha_slice: slice
    B_slice: slice
    Q_slice: slice
    size: int


def state_layout(model) -> HybridStateLayout:
    M, na, nb = model.n_states, model.na, model.nb
    n_inputs = np.asarray(model.B_matrices[0]).shape[1]
    n_A = M * na * M
    offdiag = ~np.eye(M, dtype=bool)
    n_alpha = int(offdiag.sum())
    n_B = nb * M * n_inputs
    A_slice = slice(0, n_A)
    alpha_slice = slice(A_slice.stop, A_slice.stop + n_alpha)
    B_slice = slice(alpha_slice.stop, alpha_slice.stop + n_B)
    Q_slice = slice(B_slice.stop, B_slice.stop + M)
    return HybridStateLayout(M, na, nb, n_inputs, offdiag, A_slice,
                             alpha_slice, B_slice, Q_slice, Q_slice.stop)


def pack_state(model, layout: HybridStateLayout | None = None) -> np.ndarray:
    layout = state_layout(model) if layout is None else layout
    beta = model._pack_A().reshape(-1)
    alpha = np.asarray(model.alpha_mean_)[layout.offdiag]
    B = np.asarray(model.B_matrices).reshape(-1)
    q = np.diag(model.Q)
    return np.concatenate([beta, alpha, B, q]).astype(float)


def set_state(model, theta: np.ndarray, layout: HybridStateLayout):
    theta = np.asarray(theta, float)
    if theta.shape != (layout.size,):
        raise ValueError("Hybrid fixed-point state has the wrong size.")
    beta = theta[layout.A_slice].reshape(layout.n_states, layout.na * layout.n_states)
    alpha = np.full((layout.n_states, layout.n_states), np.nan)
    alpha[layout.offdiag] = theta[layout.alpha_slice]
    if np.any(alpha[layout.offdiag] <= 0):
        raise FloatingPointError("Directional state produced nonpositive ARD precision.")
    B = theta[layout.B_slice].reshape(layout.nb, layout.n_states, layout.n_inputs)
    q = theta[layout.Q_slice]
    if np.any(q <= 0):
        raise FloatingPointError("Directional state produced nonpositive Q diagonal.")
    model._beta_means_ = beta.copy()
    model.A_mean_matrices_ = model._unpack_A(beta)
    if hasattr(model, "fixed_by_mask"):
        for matrix in model.A_mean_matrices_:
            matrix[model.fixed_by_mask] = 0.0
        model._beta_means_ = model._pack_A(model.A_mean_matrices_)
    model.alpha_mean_ = alpha
    model.B_matrices = [matrix.copy() for matrix in B]
    model.Q = np.diag(q)


def one_update_map(model, y, u, theta: np.ndarray,
                   perturbation: np.ndarray | None = None) -> np.ndarray:
    """Evaluate exactly one smoother/A/alpha/B/Q fixed-point update."""
    layout = state_layout(model)
    set_state(model, theta, layout)
    A = [value.copy() for value in model.A_mean_matrices_]
    smooth = model.smooth(y, u, A)
    stats = model._compute_posterior_sufficient_statistics(smooth, u)
    if perturbation is not None:
        perturbation = np.asarray(perturbation, float)
        if perturbation.shape != (layout.n_states, layout.na * layout.n_states):
            raise ValueError("Perturbation has the wrong shape.")
        stats = dict(stats)
        stats["S_zx"] = stats["S_zx"].copy()
        for target in range(layout.n_states):
            stats["S_zx"][:, target] += model.Q[target, target] * perturbation[target]
    model._update_q_A(stats)
    radius = var_companion_spectral_radius(model.A_mean_matrices_)
    if not np.isfinite(radius):
        raise FloatingPointError("Non-finite companion radius in one-update map.")
    if radius >= 0.98:
        model.A_mean_matrices_, _, _ = stabilize_A_matrices_if_needed(
            model.A_mean_matrices_, target_radius=0.90)
        model._beta_means_ = model._pack_A(model.A_mean_matrices_)
    model._update_q_alpha()
    if bool(getattr(model, "estimate_B", False)):
        model._update_B(smooth, u)
    if bool(getattr(model, "estimate_Q", False)):
        model._update_Q(smooth, u)
    return pack_state(model, layout)


class FixedPointLinearResponseOperator:
    """LinearOperator for (I - J_M) using directional finite differences."""

    def __init__(self, model, y, u, epsilon=1e-4):
        self.model = model
        self.y = np.asarray(y, float)
        self.u = np.asarray(u, float)
        self.epsilon = float(epsilon)
        self.layout = state_layout(model)
        self.theta0 = pack_state(model, self.layout)
        self.matvec_calls = 0
        self.map_calls = 0
        self.operator = LinearOperator((self.layout.size, self.layout.size),
                                       matvec=self.matvec, dtype=float)

    def map(self, theta, perturbation=None):
        self.map_calls += 1
        return one_update_map(self.model, self.y, self.u, theta, perturbation)

    def jvp(self, vector):
        vector = np.asarray(vector, float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros_like(vector)
        step = self.epsilon / max(norm, 1.0)
        plus = self.map(self.theta0 + step * vector)
        minus = self.map(self.theta0 - step * vector)
        return (plus - minus) / (2.0 * step)

    def matvec(self, vector):
        self.matvec_calls += 1
        vector = np.asarray(vector, float)
        return vector - self.jvp(vector)

    def perturbation_rhs(self, direction_index: int):
        target, within = divmod(int(direction_index), self.layout.na * self.layout.n_states)
        lag, source = divmod(within, self.layout.n_states)
        perturbation = np.zeros((self.layout.n_states, self.layout.na * self.layout.n_states))
        perturbation[target, lag * self.layout.n_states + source] = 1.0
        plus = self.map(self.theta0, self.epsilon * perturbation)
        minus = self.map(self.theta0, -self.epsilon * perturbation)
        return (plus - minus) / (2.0 * self.epsilon)

    def bilinear_asymmetry(self, n_trials=3, random_state=0):
        rng = np.random.default_rng(random_state)
        values = []
        for _ in range(n_trials):
            u = rng.normal(size=self.layout.size); u /= np.linalg.norm(u)
            v = rng.normal(size=self.layout.size); v /= np.linalg.norm(v)
            Mu, Mv = self.matvec(u), self.matvec(v)
            numerator = abs(u @ Mv - v @ Mu)
            denominator = max(abs(u @ Mv), abs(v @ Mu), 1e-12)
            values.append(numerator / denominator)
        return float(np.median(values)), float(np.max(values))

    def hutchinson_diagonal_preconditioner(self, n_probes=4, random_state=0):
        rng = np.random.default_rng(random_state)
        diagonal = np.zeros(self.layout.size)
        for _ in range(int(n_probes)):
            z = rng.choice([-1.0, 1.0], size=self.layout.size)
            diagonal += z * self.matvec(z)
        diagonal /= max(int(n_probes), 1)
        floor = max(1e-4 * np.median(np.abs(diagonal)), 1e-8)
        safe = np.where(np.abs(diagonal) >= floor, diagonal,
                        np.where(diagonal < 0, -floor, floor))
        return LinearOperator((self.layout.size, self.layout.size),
                              matvec=lambda value: np.asarray(value) / safe,
                              dtype=float), safe


def solve_response(operator: FixedPointLinearResponseOperator, rhs: np.ndarray,
                   rtol=1e-4, maxiter=40, preconditioner=None):
    residuals = []
    started = time.perf_counter()
    initial_calls = operator.matvec_calls
    # scipy counts ``maxiter`` in restart cycles for callback_type='pr_norm'.
    # Use one cycle whose Krylov dimension is the requested total iteration
    # budget so EXPERIMENT_34UB_CG_MAXITER has the documented meaning.
    total_iteration_budget = max(1, int(maxiter))
    solution, info = gmres(
        operator.operator, np.asarray(rhs, float), M=preconditioner,
        rtol=rtol, atol=0.0, restart=total_iteration_budget, maxiter=1,
        callback=lambda residual: residuals.append(float(residual)),
        callback_type="pr_norm",
    )
    absolute = float(np.linalg.norm(operator.matvec(solution) - rhs))
    relative = absolute / max(float(np.linalg.norm(rhs)), 1e-15)
    return solution, {
        "solver": "GMRES",
        "converged": bool(info == 0),
        "solver_info": int(info),
        "iterations": len(residuals),
        "final_residual_norm": absolute,
        "relative_residual": relative,
        "matvec_calls": operator.matvec_calls - initial_calls,
        "runtime_seconds": time.perf_counter() - started,
        "residual_history": residuals,
    }


def edge_response_block(operator: FixedPointLinearResponseOperator,
                        target: int, source: int, rtol=1e-4, maxiter=40,
                        preconditioner=None):
    indices = [global_index(target, lag, source, operator.layout.n_states,
                            operator.layout.na) for lag in range(operator.layout.na)]
    columns, diagnostics = [], []
    for lag, index in enumerate(indices):
        rhs = operator.perturbation_rhs(index)
        response, diagnostic = solve_response(operator, rhs, rtol, maxiter,
                                              preconditioner)
        columns.append(response[operator.layout.A_slice][indices])
        diagnostics.append({**diagnostic, "target": target, "source": source,
                            "lag": lag + 1, "direction_index": index})
    raw = np.column_stack(columns)
    symmetric = 0.5 * (raw + raw.T)
    values, vectors = np.linalg.eigh(symmetric)
    floor = max(1e-12, 1e-8 * max(float(values.max()), 1e-12))
    projected = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
    return raw, 0.5 * (projected + projected.T), diagnostics


__all__ = [
    "FixedPointLinearResponseOperator", "HybridStateLayout",
    "edge_response_block", "one_update_map", "pack_state", "set_state",
    "solve_response", "state_layout",
]
