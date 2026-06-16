import numpy as np

from src.simulation.tvvar2_generator import generate_time_varying_var2
from src.models.adaptive_var_rls_p import AdaptiveVARRLSP
from src.utils.adaptive_utils import extract_var_coefficients
from src.gc.adaptive_spectral_gc import spectral_gc_single_time
from src.utils.plotting import plot_gc_surface

data, true_coupling = generate_time_varying_var2(n_samples=5000, noise_cov=np.eye(2))

rls = AdaptiveVARRLSP(n_vars=2, order=2, forgetting_factor=0.995)
history = rls.fit(data)

freqs = np.linspace(0, np.pi, 128)

gc_surface = []
Sigma = np.eye(2)

for theta in history[::20]:
    coef_matrices = extract_var_coefficients(theta, n_vars=2, order=2)
    gc = spectral_gc_single_time(coef_matrices, Sigma, freqs)

    gc_surface.append(gc)

gc_surface = np.asarray(gc_surface)

plot_gc_surface(gc_surface, freqs)