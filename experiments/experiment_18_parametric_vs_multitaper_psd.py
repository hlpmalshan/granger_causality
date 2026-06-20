import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.utils.spectral_analysis import spectral_density_matrix, power_spectrum, dominant_frequency
from src.spectral.multitaper import multitaper_csd, extract_psd
from src.utils.plotting import plot_spectrum_comparison

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

# Parametric VAR spectrum
model = VARModel(order=2)
model.fit(data)

freqs = np.linspace(0, np.pi, 1024)
S_parametric = spectral_density_matrix(model.coef_matrices, model.residual_covariance, freqs)
psd_y_parametric = power_spectrum(S_parametric, variable_idx=1)
peak_parametric = dominant_frequency(freqs, psd_y_parametric)

print("Parametric PSD peak:", peak_parametric["frequency"] / np.pi, "x pi")

# Multitaper spectrum
freqs_mt, S_mt = multitaper_csd(
    data,
    fs=1.0,
    time_bandwidth=10.0,
    n_tapers=9,
    n_fft=32768,
    demean=True
)

psd_y_mt = extract_psd(S_mt, channel=1)
peak_mt = dominant_frequency(freqs_mt, psd_y_mt)

print("Multitaper PSD peak:", peak_mt["frequency"] / np.pi, "x pi")

# Compare shapes
# Interpolating multitaper PSD onto the parametric frequency grid
psd_y_mt_interp = np.interp(freqs, freqs_mt, psd_y_mt)

plot_spectrum_comparison(
    freqs,
    psd_y_parametric,
    psd_y_mt_interp,
    label_1="Parametric VAR PSD",
    label_2="Multitaper PSD",
    title="Parametric vs Multitaper PSD of Y samples=30000, time_badwidth=10.0, n_tapers=9"
)