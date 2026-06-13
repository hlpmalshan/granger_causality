import numpy as np

from src.simulation.var_generator import generate_var_data
from src.gc.pairwise_gc import pairwise_gc
from src.gc.conditional_gc import conditional_gc
from src.gc.conditional_spectral_gc import condtional_spectral_gc
from src.utils.plotting import plot_spectral_gc

# Interpretation of A1
# Y --> Z
# Z --> X
# No direct Y --> X
A1 = np.array([
    [0.6,0.0,0.5],
    [0.0,0.8,0.0],
    [0.0,0.6,0.5]
])

data = generate_var_data([A1], np.eye(3), 20000)
print("Pairwise GC Y->X:",
    pairwise_gc(
        data,
        source=1,
        target=0,
        order=1
    )
)

print("Conditional GC Y->X|Z:",
    conditional_gc(
        data,
        source=1,
        target=0,
        conditioning=[2],
        order=1
    )
)

freqs, gc = ( 
    condtional_spectral_gc(
        data,
        source=1,
        target=0,
        conditioning=[2],
        order=1
    )
)

plot_spectral_gc(freqs, gc, title="Conditional Spectral GC Y->X given Z")