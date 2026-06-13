import numpy as np

from src.models.var_model import VARModel
from src.utils.matrix_utils import transfer_function

# Helper function
def spectral_density(
        coef_matrices,
        Sigma,
        frequencies
):
    H_freq = transfer_function(coef_matrices, frequencies)

    spectra = []
    for H in H_freq:
        S = (H @ Sigma @ np.conj(H.T))
        spectra.append(S)

    return spectra

# Main function
def condtional_spectral_gc(
        data, 
        source,
        target,
        conditioning,
        order,
        n_freqs=512
):
    # Full model
    full_model = VARModel(order)
    full_model.fit(data)

    # Reduced data
    keep = [target, *conditioning]
    reduced_data = data[:, keep]

    # Reduced model
    reduced_model = VARModel(order)
    reduced_model.fit(reduced_data)

    frequencies = np.linspace(0, np.pi, n_freqs)
    
    full_spectra = spectral_density(full_model.coef_matrices, full_model.residual_covariance, frequencies)
    reduced_spectra = spectral_density(reduced_model.coef_matrices, reduced_model.residual_covariance, frequencies)

    # GC
    gc = []
    for Sf, Sr in zip(full_spectra, reduced_spectra):
        Sxx_full = np.real(Sf[target, target])
        Sxx_reduced = np.real(Sr[0, 0])

        gc.append(np.log(Sxx_full/Sxx_reduced))

    return (frequencies, np.asarray(gc))
