import numpy as np

from src.utils.matrix_utils import transfer_function

def dominant_frequency(frequencies, spectrum):
    idx = np.argmax(spectrum)
    return {
        "index": idx,
        "frequency": frequencies[idx],
        "value": spectrum[idx]
    }

def spectral_density_matrix(coef_matrices, Sigma, frequencies):
    H_freq = transfer_function(coef_matrices, frequencies)
    spectra = []

    for H in H_freq:
        S = (H @ Sigma @ np.conj(H.T))
        spectra.append(S)

    return spectra

def power_spectrum(spectra, variable_idx):
    psd = []
    for S in spectra:
        psd.append(np.real(S[variable_idx, variable_idx]))

    return np.asarray(psd)
