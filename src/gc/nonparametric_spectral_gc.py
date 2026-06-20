import numpy as np

from src.factorization.wilson_factorization import wilson_factorize

# Extract spectral submatrix S_indices, indices(f)
def extract_spectral_submatrix(S, indices):
    indices = list(indices)

    return S[:, indices, :][:, :, indices]

# Compute bivariate Geweke spectral GC from WIlson output
# Assumes S_pair is ordered as [target, source]
# Therefore, direction="second_to_first" computes source -> target
def bivariate_geweke_gc_from_factorization(
        S_pair,                         # (n_freqs, 2, 2)
        H,                              # (n_freqs, 2, 2)
        Sigma,                          # (2, 2)
        direction="second_to_first",
        clip_negative=True              #  If True, clips tiny negative numerical values to zero.
):
    if direction != "second_to_first":
        raise NotImplementedError(
            "Only direction='second_to_first' is implemented. "
            "Order S_pair as [target, source]."
        )
    
    sigma_x = np.real(Sigma[0, 0])
    upsilon = np.real(Sigma[0, 1])

    gc = []
    for f_idx in range(S_pair.shape[0]):
        Sxx = np.real(S_pair[f_idx, 0, 0])

        Hxx = H[f_idx, 0, 0]
        Hxy = H[f_idx, 0, 1]

        Hxx_tilde = Hxx + (upsilon / sigma_x) * Hxy

        intrinsic = sigma_x * np.abs(Hxx_tilde) ** 2

        value = np.log(Sxx / intrinsic)
        value = np.real(value)

        if clip_negative and value < 0:
            if value > -1e-8:
                value = 0.0

        gc.append(value)

    return np.asarray(gc)

# Nonparametric pairwise spectral GC
def nonparametric_pairwise_spectral_gc(
        S,
        target,
        source,
        wilson_max_iter=300,
        wilson_tol=5e-3,
        wilson_regularization=1e-10,
        verbose=False
):
    """
    Steps:
        1. Extract 2x2 spectral submatrix [target, source].
        2. Wilson factorize that spectral matrix.
        3. Compute Geweke spectral GC source -> target.
    """
    S_pair = extract_spectral_submatrix(S, indices=[target, source])

    wilson_result = wilson_factorize(
        S_pair,
        max_iter=wilson_max_iter,
        tol=wilson_tol,
        regularization=wilson_regularization,
        verbose=verbose
    )

    H = wilson_result["H"]
    Sigma = wilson_result["factor_cov"]

    gc = bivariate_geweke_gc_from_factorization(
        S_pair=S_pair,
        H=H,
        Sigma=Sigma,
        direction="second_to_first",
        clip_negative=True
    )

    return {
        "gc": gc,
        "S_pair": S_pair,
        "H": H,
        "Sigma": Sigma,
        "wilson_result": wilson_result
    }