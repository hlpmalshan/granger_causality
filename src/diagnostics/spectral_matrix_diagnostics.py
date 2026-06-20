import numpy as np 

# Compute maximum Hermitian symmetry violation
def hermitian_error(S):
    diff = S - np.conj(np.swapaxes(S, 1, 2))

    return np.max(np.abs(diff))

# Minimum eigenvalue of S(f) at each frequency
def min_eigenvalues(S):
    mins = []
    for Sf in S:
        eigvals = np.linalg.eigvals(Sf)
        mins.append(np.min(np.real(eigvals)))

    return np.asarray(mins)

# Condition number of S(f) at each frequency
def condition_numbers(S):
    conds = []
    for Sf in S:
        eigvals = np.linalg.eigvalsh(Sf)
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 1e-15)

        conds.append(np.max(eigvals) / np.min(eigvals))

    return np.asarray(conds)

# Add epsilon*I t oeach spectral matrix
# Useful for Wilson factorization
def regularize_spectral_matrix(S, epsilon=1e-8):
    S_reg = S.copy()
    n_channels = S.shape[1]
    I = np.eye(n_channels)

    for f_idx in range(S.shape[0]):
        S_reg[f_idx] += epsilon * I

    return S_reg

# Add frequency-dependent diagonal loading:
# S_reg(f) = S(f) + eps * trace(S(f))/n_channels * I
# This is better than fixed epsilon when spectral magnitudes vary strongly across frequencies.
def regularize_spectral_matrix_relative(S, relative_epsilon=1e-6):
    S = np.asarray(S)
    S_reg = S.copy()
    
    n_freqs, n_channels, _ = S.shape
    I = np.eye(n_channels)
    for f_idx in range(n_freqs):
        scale = np.real(np.trace(S[f_idx])) / n_channels
        loading = relative_epsilon * scale
        S_reg[f_idx] += loading + I

    return S_reg
