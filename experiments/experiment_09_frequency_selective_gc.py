import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.gc.geweke_spectral_gc import geweke_spectral_gc
from src.utils.plotting import plot_spectral_gc
from src.utils.spectral_analysis import dominant_frequency

r = 0.95
omega0 = 0.3 * np.pi

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

Sigma = np.eye(2)
data = generate_var_data([A1, A2], Sigma, 30000)

model = VARModel(order=2)
model.fit(data)

freqs, gc = geweke_spectral_gc(model.coef_matrices, model.residual_covariance, n_freqs=1024)

plot_spectral_gc(freqs, gc, title="Frequency Selective Y->X GC")

peak = dominant_frequency(freqs, gc)
print("Peak frequency", peak["frequency"] / np.pi, "x pi")

