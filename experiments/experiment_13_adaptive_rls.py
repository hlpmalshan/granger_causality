import numpy as np

from src.simulation.tvvar_generator import generate_smoothly_varying_var
from src.models.adaptive_var_rls import AdaptiveVARRLS
from src.utils.plotting import plot_adaptive_coupling

data, true_coupling = generate_smoothly_varying_var(n_samples=10000, noise_cov=np.eye(2))

rls = AdaptiveVARRLS(n_vars=2, forgetting_factor=0.995)
history = rls.fit(data)

estimated = history[:, 0, 1]

plot_adaptive_coupling(true_coupling[:len(estimated)], estimated, order=1)