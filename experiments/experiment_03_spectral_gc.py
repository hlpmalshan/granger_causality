import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.gc.spectral_gc import spectral_gc_bivariate
from src.utils.plotting import plot_spectral_gc

A1 = np.array([
    [0.5, 0.4],
    [0.0, 0.8]
])

data = generate_var_data(
    [A1],
    np.eye(2),
    10000
)

model = VARModel(order=1)
model.fit(data)

freqs, gc = spectral_gc_bivariate(
    model.coef_matrices,
    model.residual_covariance
)

plot_spectral_gc(freqs, gc, title="Y->X Spectral GC")
