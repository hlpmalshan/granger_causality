import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.gc.geweke_spectral_gc import geweke_spectral_gc
from src.spectral.multitaper import multitaper_csd
from src.gc.nonparametric_spectral_gc import nonparametric_pairwise_spectral_gc
from src.utils.spectral_analysis import dominant_frequency
from src.utils.plotting import plot_gc_comparison
from src.diagnostics.spectral_matrix_diagnostics import hermitian_error, min_eigenvalues, condition_numbers, regularize_spectral_matrix_relative

# Configuration
USE_EXTERNAL_REGULARIZATION = False
N_SAMPLES = 8192
N_FFT = 8192
TIME_BANDWIDTH = 3.0
N_TAPERS = 5

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
    n_samples=N_SAMPLES
)

# Parametric spectral GC
model = VARModel(order=2)
model.fit(data)

freqs_param, gc_param = geweke_spectral_gc(model.coef_matrices, model.residual_covariance, n_freqs=N_FFT // 2 + 1)

peak_param = dominant_frequency(freqs_param, gc_param)

print("\nParametric spectral GC")
print("Peak frequency:", peak_param["frequency"] / np.pi, "x pi")
print("Peak GC:", peak_param["value"])

# Multitaper spectral matrix
freqs_mt, S_mt = multitaper_csd(
    data,
    fs=1.0,
    time_bandwidth=TIME_BANDWIDTH,
    n_tapers=N_TAPERS,
    n_fft=N_FFT,
    demean=True
)

print("\nMultitaper spectral matrix diagnostics")
print("Hermitian error:", hermitian_error(S_mt))
print("Minimum eigenvalue:", np.min(min_eigenvalues(S_mt)))
print("Median condition number:", np.median(condition_numbers(S_mt)))
print("Maximum condition number:", np.max(condition_numbers(S_mt)))

# Optional external regularization
if USE_EXTERNAL_REGULARIZATION:
    print("\nUsing relative spectral regularization")
    S_input = regularize_spectral_matrix_relative(
        S_mt,
        relative_epsilon=1e-6
    )

else:
    print("\nUsing raw multitaper spectral matrix")
    S_input = S_mt

# Nonparametric spectral GC
nonparam_result = nonparametric_pairwise_spectral_gc(
    S=S_input,
    target=0,
    source=1,
    wilson_max_iter=300,
    wilson_tol=5e-3,
    wilson_regularization=1e-10,
    verbose=True
)

gc_nonparam = nonparam_result["gc"]
wilson_result = nonparam_result["wilson_result"]
peak_nonparam = dominant_frequency(freqs_mt, gc_nonparam)

print("\nNonparametric spectral GC")
print("Wilson converged:", wilson_result["converged"])
print("Wilson iterations:", wilson_result["n_iter"])
print("Wilson final error:", wilson_result["errors"][-1])

print("Peak frequency:", peak_nonparam["frequency"] / np.pi, "x pi")
print("Peak GC:", peak_nonparam["value"])

print("\nWilson factor covariance:")
print(nonparam_result["Sigma"])

# Compare curves
plot_gc_comparison(
    freqs_param,
    gc_nonparam,
    gc_param,
    label_1="Multitaper + Wilson spectral GC",
    label_2="Parametric VAR spectral GC",
    title="Parametrice vs Nonparametric (Multitaper with Wilson) Spectral GC Y to X"
)