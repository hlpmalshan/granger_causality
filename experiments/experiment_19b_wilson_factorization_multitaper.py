import numpy as np

from src.simulation.var_generator import generate_var_data
from src.spectral.multitaper import multitaper_csd, extract_psd
from src.factorization.wilson_factorization import wilson_factorize, max_relative_spectral_error
from src.diagnostics.spectral_matrix_diagnostics import hermitian_error, min_eigenvalues, condition_numbers, regularize_spectral_matrix_relative
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

# Multitaper spectral matrix
freqs_mt, S_mt = multitaper_csd(
    data,
    fs=1.0,
    time_bandwidth=3.0,
    n_tapers=5,
    n_fft=32768,
    demean=True
)

print("\nBefore regularization")
print("Hermitian error:", hermitian_error(S_mt))
print("Minimum eigenvalue:", np.min(min_eigenvalues(S_mt)))
print("Median condition number:", np.median(condition_numbers(S_mt)))
print("Maximum condition number:", np.max(condition_numbers(S_mt)))

# Regularize multitaper spectral matrix
S_mt_reg = regularize_spectral_matrix_relative(S_mt, relative_epsilon=1e-6)

print("\nAfter regularization")
print("Hermitian error:", hermitian_error(S_mt_reg))
print("Minimum eigenvalue:", np.min(min_eigenvalues(S_mt_reg)))
print("Median condition number:", np.median(condition_numbers(S_mt_reg)))
print("Maximum condition number:", np.max(condition_numbers(S_mt_reg)))

# Wilson factorization
result = wilson_factorize(
    S_mt_reg,
    max_iter=300,
    tol=1e-5,
    regularization=1e-10,
    verbose=True
)

S_reconstructed = result["reconstruction"]

err = max_relative_spectral_error(S_mt_reg, S_reconstructed)

print("\nWilson factorization on multitaper spectrum")
print("Converged:", result["converged"])
print("Iterations:", result["n_iter"])
print("Final iteration error:", result["errors"][-1])
print("Max reconstruction error:", err)

print("\nFactor covariance matrix:")
print(result["factor_cov"])

# Compare PSD reconstruction of Y
psd_original_y = extract_psd(S_mt_reg, channel=1)
psd_reconstructed_y = extract_psd(S_reconstructed, channel=1)

plot_spectrum_comparison(
    freqs_mt,
    psd_original_y,
    psd_reconstructed_y,
    label_1="Multitaper PSD",
    label_2="Wilson reconstructed PSD",
    title="Wilson Reconstruction of Multitaper PSD"
)