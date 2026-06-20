import numpy as np

from scipy.linalg import cholesky

# Convert one-sided spectrum S[0...pi) to two-sided spectrumS[0...2pi)
def one_sided_to_two_sided(S_one_sided):
    S_one_sided = np.asarray(S_one_sided)

    if S_one_sided.ndim != 3:
        raise ValueError(
            "S_one_sided must have shape "
            "(n_freqs, n_channels, n_channels)"
        )
    
    # For real-valued processes:
    # S(-w) = S(w)^T = conj(S(w)) for Hermittian matrices
    mirrored = np.conj(np.swapaxes(S_one_sided[-2:0:-1], 1, 2))

    return np.concatenate([S_one_sided, mirrored], axis=0)

# Return the 0...pi part from a two-sided spectrum
def two_sided_to_one_sided(S_two_sided):
    n_total = S_two_sided.shape[0]
    n_one_sided = n_total // 2 + 1

    return S_two_sided[:n_one_sided]

# Enforce Hermittian symmetry and add epsilon*I
def _make_hermittian_positive_definite(S, regularization=1e-8):
    S = np.asarray(S).copy()
    n_freqs, n_channels, _ = S.shape
    I = np.eye(n_channels)

    for f in range(n_freqs):
        S[f] = 0.5 * (S[f] + np.conj(S[f].T))

        S[f] += regularization * I

    return S

# Wilson-style plus operator
# Keeps the causal Fourier coefficients and removes anti-casual ones
def _plus_operator(G):
    n_freqs, n_channels, _ = G.shape

    coeffs = np.fft.ifft(G, axis=0)

    coeffs_plus = np.zeros_like(coeffs)

    # Zero-lag coefficient
    # For matrix-valued spectra, use lower-triangular convention:
    # G0_plus + G0_plus^* = G0
    G0 = coeffs[0].copy()
    G0_plus = np.tril(G0)
    diag_idx = np.diag_indices(n_channels)
    G0_plus[diag_idx] = 0.5 * np.diag(G0)
    coeffs_plus[0] = G0_plus

    # Positive lags
    coeffs_plus[1:n_freqs // 2] = coeffs[1:n_freqs // 2]

    G_plus = np.fft.fft(coeffs_plus, axis=0)

    return G_plus

# Wilson-style spectral matrix factorization
def wilson_factorize(
        S_one_sided,
        max_iter=200,
        tol=1e-5,               # Stop when maximum relative reconstruction error is below tol
        regularization=1e-8,    # Diagonal loading added to each S(w)
        verbose=False
):
    S_two = one_sided_to_two_sided(S_one_sided)
    S_two = _make_hermittian_positive_definite(S_two, regularization=regularization)

    n_freqs, n_channels, _ = S_two.shape

    # Initial constant factor (a stable numerical starting point)
    S_mean = np.real(np.mean(S_two, axis=0))

    R =cholesky(S_mean + regularization * np.eye(n_channels), lower=True)

    spectral_factor = np.tile(R[None, :, :], (n_freqs, 1, 1)).astype(complex)

    I = np.eye(n_channels)
    errors = []
    converged = False
    for iteration in range(max_iter):
        G = np.zeros_like(S_two, dtype=complex)
        for f in range(n_freqs):
            inv_factor = np.linalg.inv(spectral_factor[f])
            G[f] = inv_factor @ S_two[f] @ np.conj(inv_factor.T) + I

        G_plus = _plus_operator(G)
        new_factor = np.zeros_like(spectral_factor)

        for f in range(n_freqs):
            new_factor[f] = spectral_factor[f] @ G_plus[f]

        spectral_factor = new_factor
        reconstructed_two = reconstruct_from_factor(spectral_factor)
        
        err = max_relative_spectral_error(S_two, reconstructed_two)
        errors.append(err)

        if verbose:
            print(f"Iteration {iteration:03d} | " f"max relative error = {err:.6e}")

        if err < tol:
            converged = True
            break
    
    # Convert to one-sided outputs
    spectral_factor_one = two_sided_to_one_sided(spectral_factor)

    # Zeroth Fourier coefficients of spectral factor
    R0 = np.real(np.fft.ifft(spectral_factor, axis=0)[0])

    factor_cov = R0 @ R0.T

    H = np.zeros_like(spectral_factor_one, dtype=complex)
    inv_R0 = np.linalg.inv(R0)
    for f in range(spectral_factor_one.shape[0]):
        H[f] = spectral_factor_one[f] @ inv_R0

    reconstruction_one = reconstruct_from_H(H, factor_cov)

    return {
        "H": H,
        "factor_cov": factor_cov,
        "spectral_factor": spectral_factor_one,
        "reconstruction": reconstruction_one,
        "errors": np.asarray(errors),
        "converged": converged,
        "n_iter": iteration + 1
    }

# Reconstruct S(w) = factor(w) factor(w)^*
def reconstruct_from_factor(spectral_factor):
    S_rec = np.zeros_like(spectral_factor, dtype=complex)
    for f in range(spectral_factor.shape[0]):
        S_rec[f] = spectral_factor[f] @ np.conj(spectral_factor[f].T)

    return S_rec

# Reconstruct S(w) = H(w) factor_cov H(w)^*
def reconstruct_from_H(H, factor_cov):
    S_rec = np.zeros_like(H, dtype=complex)
    for f in range(H.shape[0]):
        S_rec[f] = H[f] @ factor_cov @ np.conj(H[f].T)

    return S_rec

# Maximum relative Frobenius reconstruction error over frequencies
def max_relative_spectral_error(S_true, S_est):
    errors = []
    for f in range(S_true.shape[0]):
        numerator = np.linalg.norm(S_true[f] - S_est[f], ord="fro")
        denominator = np.linalg.norm(S_true[f], ord="fro") + 1e-12

        errors.append(numerator / denominator)

    return np.max(errors)
