import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.gc.geweke_spectral_gc import geweke_spectral_gc
from src.utils.plotting import plot_spectral_gc

A1 = np.array([
    [0.5, 0.3],
    [0.0, 1.6]
])

A2 = np.array([
    [0.0, 0.0],
    [0.0, -0.81]
])

Sigma = np.eye(2)

data = generate_var_data([A1, A2], Sigma, 20000)

model = VARModel(order=2)
model.fit(data)

freqs, gc = geweke_spectral_gc(model.coef_matrices, model.residual_covariance)

plot_spectral_gc(freqs, gc, title="Oscillatory Y->X Spectral GC")