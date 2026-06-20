import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.utils.spectral_analysis import spectral_density_matrix, power_spectrum
from src.factorization.wilson_factorization import wilson_factorize, max_relative_spectral_error
from src.utils.plotting import plot_spectrum_comparison

# Known oscillatory VAR(2) system
r = 0.95
omega0 = 0.3 * np.pi

a1 = 2 * r * np.cos(omega0)
a2 = -r**2

A1 = np.array([
    [0.5, 0.3],
    [0.0, a1]
])

A2 = np.array([
    [0.0, 0.0],
    [0.0, a2]
])

data = generate_var_data(
    coeff_matrices=[A1, A2],
    covariance=np.eye(2),
    n_samples=30000
)

# Parametric VAR spectrum
model = VARModel(order=2)
model.fit(data)

freqs = np.linspace(0, np.pi, 513)
S_parametric = np.asarray(spectral_density_matrix(model.coef_matrices, model.residual_covariance, freqs))

# Wilson factorization
result = wilson_factorize(S_parametric, max_iter=200, tol=1e-5, regularization=1e-8, verbose=True)
S_reconstructed= result["reconstruction"]

err = max_relative_spectral_error(S_parametric, S_reconstructed)

print("\nWilson factorization results")
print("Converged:", result["converged"])
print("Iterations:", result["n_iter"])
print("Final iteration error:", result["errors"][-1])
print("Max reconstruction error:", err)

print("\nFactor covariance matrix:")
print(result["factor_cov"])

# Compare PSDs
psd_original_y = power_spectrum(S_parametric, variable_idx=1)
psd_reconstructed_y = power_spectrum(S_reconstructed, variable_idx=1)

plot_spectrum_comparison(
    freqs, 
    psd_original_y, 
    psd_reconstructed_y,
    label_1="Original parametric PSD",
    label_2="Wilson reconstructed PSD",
    title="Wilson Factorization Reconstruction Check"
)