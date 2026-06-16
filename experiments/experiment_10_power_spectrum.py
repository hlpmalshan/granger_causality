import numpy as np

from src.models.var_model import VARModel
from src.simulation.var_generator import generate_var_data
from src.utils.spectral_analysis import spectral_density_matrix, power_spectrum, dominant_frequency
from src.utils.plotting import plot_power_spectrum

r = 0.95
omega0 = 0.3*np.pi

a1 = 2*r*np.cos(omega0)
a2 = -r**2

A1 = np.array([
    [0.5, 0.3],
    [0.0, a1]
])

A2 = np.array([
    [0.0, 0.0],
    [0.0, a2]
])

data = generate_var_data([A1, A2], np.eye(2), 30000)

model = VARModel(order=2)
model.fit(data)

freqs = np.linspace(0, np.pi, 1024)

spectra = spectral_density_matrix(model.coef_matrices, model.residual_covariance, freqs)

psd_y = power_spectrum(spectra, variable_idx=1)

plot_power_spectrum(freqs, psd_y, title="Power Spectrum of Y")

peak = dominant_frequency(freqs, psd_y)
print("PSD peak:", peak["frequency"]/np.pi, "x pi")