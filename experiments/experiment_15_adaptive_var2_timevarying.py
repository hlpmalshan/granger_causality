import numpy as np

from src.simulation.tvvar2_generator import generate_time_varying_var2
from src. models.adaptive_var_rls_p import AdaptiveVARRLSP
from src.utils.plotting import plot_adaptive_coupling

data, true_coupling = generate_time_varying_var2(n_samples=10000, noise_cov=np.eye(2))

rls = AdaptiveVARRLSP(n_vars=2, order=2, forgetting_factor=0.995)
history = rls.fit(data)

estimated = []
for theta in history:
    A1 = theta[:, :2]
    estimated.append(A1[0, 1])

estimated = np.asarray(estimated)

plot_adaptive_coupling(true_coupling[:len(estimated)], estimated, order=2)