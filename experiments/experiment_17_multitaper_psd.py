import numpy as np

from src.simulation.var_generator import generate_var_data
from src.spectral.multitaper import multitaper_csd, extract_psd
from src.utils.spectral_analysis import dominant_frequency
from src.utils.plotting import plot_power_spectrum
from src.diagnostics.spectral_matrix_diagnostics import hermittian_error, min_eigenvalues, condition_numbers

# Known oscillatory VAR(2) system
r = 0.95
omega0 = 0.3 * np.pi

a1 = 2 * r * np.cos(omega0)
a2 = -r**2

A1 = np.array([
    [0.5, 0.3],
    [0.0, a1]
])

A2 = np.array([
    [0.0, 0.0],
    [0.0, a2]
])

data = generate_var_data(
    coeff_matrices=[A1, A2],
    covariance=np.eye(2),
    n_samples=30000
)

# Multitaper spectral matrix estimation
freqs, S = multitaper_csd(data, fs=1.0, time_bandwidth=3.0, n_tapers=5, n_fft=32768, demean=True)

psd_y = extract_psd(S, channel=1)
peak = dominant_frequency(freqs, psd_y)
print("Multitaper PSD peak:",peak["frequency"] / np.pi,"x pi")

# Diagnostics
print("Hermittian error:",hermittian_error(S))

mins = min_eigenvalues(S)
print("Minimum eigenvalue over all frequencies:", np.min(mins))

conds = condition_numbers(S)
print("Median condition number:", np.median(conds))
print("Maximum condition number:", np.max(conds))

# Plot
plot_power_spectrum(freqs, psd_y, title="Multitaper PSD of Y")