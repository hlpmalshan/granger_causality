import numpy as np

from src.simulation.tvvar_generator import generate_smoothly_varying_var
from src.gc.pairwise_gc import pairwise_gc
from src.gc.sliding_window_gc import sliding_window_gc
from src.utils.plotting import plot_true_vs_estimated_gc

N = 10000
data, true_coupling = generate_smoothly_varying_var(n_samples=N, noise_cov=np.eye(2))

global_gc = pairwise_gc(data, source=1, target=0, order=1)
print("Global GC:", global_gc)

gc_times, gc_values = (
    sliding_window_gc(
        data,
        source=1,
        target=0,
        order=1,
        window_size=1000,
        step_size=100
    )
)

plot_true_vs_estimated_gc(
    np.arange(len(true_coupling)),
    true_coupling,
    gc_times,
    gc_values
)