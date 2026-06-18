import numpy as np

from src.simulation.var_generator import generate_var_data
from src.gc.pairwise_gc import pairwise_gc

A1 = np.array([
    [0.5, 0.8],
    [0.0, 0.8]
])

Sigma = np.eye(2)

data = generate_var_data(
    coeff_matrices=[A1],
    covariance=Sigma,
    n_samples=5000
)

print("Pairwise Y->X",
    pairwise_gc(
        data,
        source=1,
        target=0,
        order=1
    )
)

print("Pairwise X->Y",
    pairwise_gc(
        data,
        source=0,
        target=1,
        order=1
    )
)