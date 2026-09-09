"""Static frequency-domain diagnostics for stable VARX models.

The transfer score implemented here is total innovation-transfer influence for
diagonal Q; it may contain indirect paths.  The deletion score is a direct
edge-deletion contrast, not a refitted reduced-model Geweke statistic.
Exogenous transfer is returned separately and is never treated as endogenous GC.
"""

import numpy as np


def _A(A_mats):
    value = np.asarray(A_mats, dtype=float)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("A_mats must have shape (p, M, M).")
    return value


def _B(B_mats, n_states=None):
    value = np.asarray(B_mats, dtype=float)
    if value.ndim != 3 or (n_states is not None and value.shape[1] != n_states):
        raise ValueError("B_mats must have shape (q, M, n_inputs).")
    return value


def compute_A_frequency_matrix(A_mats, omega):
    A = _A(A_mats); result = np.eye(A.shape[1], dtype=complex)
    for lag, matrix in enumerate(A, start=1):
        result -= matrix * np.exp(-1j * float(omega) * lag)
    return result


def compute_A_transfer_function(A_mats, omega, jitter=0.0):
    matrix = compute_A_frequency_matrix(A_mats, omega)
    if jitter:
        matrix = matrix + float(jitter) * np.eye(matrix.shape[0])
    return np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=complex))


def compute_B_frequency_matrix(B_mats, omega):
    B = _B(B_mats); result = np.zeros(B.shape[1:], dtype=complex)
    for lag, matrix in enumerate(B):
        result += matrix * np.exp(-1j * float(omega) * lag)
    return result


def compute_innovation_spectrum(A_mats, Q, omega):
    H = compute_A_transfer_function(A_mats, omega)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (H.shape[0], H.shape[0]):
        raise ValueError("Q has incompatible dimensions.")
    return H @ Q @ H.conj().T


def compute_exogenous_transfer(A_mats, B_mats, omega):
    A = _A(A_mats); B = _B(B_mats, A.shape[1])
    return compute_A_transfer_function(A, omega) @ compute_B_frequency_matrix(B, omega)


def compute_exogenous_response_power(A_mats, B_mats, omega, S_u=None):
    response = compute_exogenous_transfer(A_mats, B_mats, omega)
    if S_u is None:
        S_u = np.eye(response.shape[1])
    S_u = np.asarray(S_u, dtype=complex)
    if S_u.shape != (response.shape[1], response.shape[1]):
        raise ValueError("S_u has incompatible dimensions.")
    return response @ S_u @ response.conj().T


def compute_pairwise_transfer_spectral_gc_diagonal_Q(A_mats, Q, omega, epsilon=1e-12,
                                                       return_diagnostics=False):
    """Return matrix[target, source] of total innovation-transfer influence."""
    A = _A(A_mats); Q = np.asarray(Q, dtype=float)
    if Q.shape != (A.shape[1], A.shape[1]) or not np.allclose(Q, np.diag(np.diag(Q))):
        raise ValueError("This score requires diagonal Q with compatible dimensions.")
    H = compute_A_transfer_function(A, omega); spectrum = H @ Q @ H.conj().T
    total = np.maximum(np.real(np.diag(spectrum)), epsilon)
    contribution = np.abs(H) ** 2 * np.diag(Q)[None, :]
    denominator_raw = total[:, None] - contribution
    denominator = np.maximum(denominator_raw, epsilon)
    score = np.maximum(np.real(np.log(total[:, None] / denominator)), 0.0)
    np.fill_diagonal(score, 0.0)
    diagnostics = {"denominator_violation_count": int(np.sum(denominator_raw < epsilon)),
                   "clipping_count": int(np.sum(denominator_raw < epsilon))}
    return (score, diagnostics) if return_diagnostics else score


def compute_edge_deletion_spectral_contrast(A_mats, Q, omega, source_j, target_i,
                                             epsilon=1e-12, return_diagnostics=False):
    """Direct edge-deletion diagnostic; this is not refitted conditional GC."""
    A = _A(A_mats); reduced = A.copy(); reduced[:, target_i, source_j] = 0.0
    full = max(float(np.real(compute_innovation_spectrum(A, Q, omega)[target_i, target_i])), epsilon)
    reduced_value = max(float(np.real(compute_innovation_spectrum(reduced, Q, omega)[target_i, target_i])), epsilon)
    raw = np.log(reduced_value / full); score = max(float(np.real(raw)), 0.0)
    diagnostics = {"clipping_count": int(raw < 0), "denominator_violation_count": 0}
    return (score, diagnostics) if return_diagnostics else score


def integrate_spectral_score(omega_grid, spectral_values, band=(0.0, np.pi)):
    omega, values = np.asarray(omega_grid, float), np.asarray(spectral_values, float)
    start, end = map(float, band); mask = (omega >= start - 1e-12) & (omega <= end + 1e-12)
    if mask.sum() < 2:
        raise ValueError("A frequency band must contain at least two grid points.")
    x, y = omega[mask], values[mask]; integral = float(np.trapezoid(y, x))
    peak_index = int(np.nanargmax(y))
    return {"integral": integral, "average": integral / max(end - start, 1e-12),
            "peak": float(y[peak_index]), "peak_frequency": float(x[peak_index])}


def validate_time_frequency_gc_2source(n_freqs=4097, n_samples=200000, seed=12345):
    """Reproducible bivariate sanity check against a reduced/full OLS GC."""
    A = np.array([[[.45, .28], [0.0, .35]]]); Q = np.diag([.7, .5])
    omega = np.linspace(0.0, np.pi, int(n_freqs))
    curve = np.array([compute_pairwise_transfer_spectral_gc_diagonal_Q(A, Q, w)[0, 1] for w in omega])
    spectral_gc = integrate_spectral_score(omega, curve, (0.0, np.pi))["average"]
    rng = np.random.default_rng(seed); burn = 1000; x = np.zeros((n_samples + burn, 2))
    noise = rng.multivariate_normal(np.zeros(2), Q, len(x))
    for t in range(1, len(x)): x[t] = A[0] @ x[t-1] + noise[t]
    x = x[burn:]; target, own, both = x[1:, 0], x[:-1, [0]], x[:-1]
    full_resid = target - both @ np.linalg.lstsq(both, target, rcond=None)[0]
    reduced_resid = target - own @ np.linalg.lstsq(own, target, rcond=None)[0]
    time_gc = float(np.log(np.mean(reduced_resid ** 2) / np.mean(full_resid ** 2)))
    difference = abs(time_gc - spectral_gc)
    return {"time_domain_gc": time_gc, "integrated_spectral_gc": spectral_gc,
            "absolute_difference": difference,
            "relative_difference": difference / max(abs(time_gc), 1e-12),
            "n_freqs": int(n_freqs), "n_samples": int(n_samples),
            "validation_passed": bool(difference < .02)}


__all__ = [name for name in globals() if name.startswith("compute_") or name.startswith("integrate_") or name.startswith("validate_")]
