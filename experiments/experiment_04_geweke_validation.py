import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.gc.time_domain_gc import time_domain_gc
from src.gc.geweke_spectral_gc import geweke_spectral_gc
from src.utils.integration import integrate_spectral_gc

A1 = np.array([
    [0.5, 0.4],
    [0.0, 0.8]
])

Sigma = np.array([
    [1.0, 0.3],
    [0.3, 1.0]
])

data = generate_var_data([A1], Sigma, 15000)

model = VARModel(order=1)
model.fit(data)

F_time = time_domain_gc(
    data,
    source=1,
    target=0,
    order=1
)

freqs, F_spec = ( geweke_spectral_gc(model.coef_matrices, model.residual_covariance))

F_integrated = ( integrate_spectral_gc(freqs, F_spec))

print("Time-domain GC:", F_time)
print("Integrated Spectral GC:", F_integrated)