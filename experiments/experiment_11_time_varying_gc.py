import numpy as np

from src.simulation.tvvar_generator import generate_piecewise_var
from src.gc.sliding_window_gc import sliding_window_gc
from src.gc.pairwise_gc import pairwise_gc
from src.utils.plotting import plot_gc_evolution

N = 10000
A1_strong = np.array([
    [0.5, 0.8],
    [0.0, 0.8]
])
A1_weak = np.array([
    [0.5, 0.1],
    [0.0, 0.8]
])

data = generate_piecewise_var(coef_sets=[[A1_strong], [A1_weak]], change_points=[5000], noise_cov=np.eye(2), n_samples=N)

global_gc = pairwise_gc(data, source=1, target=0, order=1)
print("Global GC:", global_gc)

times, gc_values = (
    sliding_window_gc(
        data,
        source=1,
        target=0,
        order=1,
        window_size=1000,
        step_size=100
    )
)

plot_gc_evolution(times, gc_values, change_point=5000, title="Time-Varying GC")