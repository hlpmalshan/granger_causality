import numpy as np

from src.simulation.var_generator import generate_var_data

A1 = np.array([
    [0.5, 0.4],
    [0.5, 0.8]
])

Sigma = np.eye(2)

data = generate_var_data(
    coeff_matrices=[A1],
    covariance=Sigma,
    n_samples=5000
)

print(data.shape)