import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_simulator import generate_linear_ssm_varx
from src.ssm.kalman import kalman_filter_varx, rts_smoother
from src.varx.varx_gc import varx_gc
from src.utils.plotting import plot_ssm_smoothing

# Configuration
N_SAMPLES = 3000
# x[:, 0] = X
# x[:, 1] = Y

A = np.array([
    [0.60, 0.30],
    [0.00, 0.55]
])

B = np.array([
    [0.50],
    [0.40]
])

Q = 0.15 * np.eye(2)
R = 0.60 * np.eye(2)

C = np.eye(2)

# Exogenous input
u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

# Simulate latent state and noisy observations
x_true, y_obs = generate_linear_ssm_varx(A=A, B=B, u=u, Q=Q, R=R, C=C, random_seed=2)

# Kalman filter + RTS smoother with known parameters
filter_result = kalman_filter_varx(y=y_obs, u=u, A=A, B=B, Q=Q, R=R, C=C)
smooth_result = rts_smoother(filter_result, A=A)

x_smooth = smooth_result["x_smooth"]

# MSE comparison
mse_observation = np.mean((y_obs - x_true) ** 2)
mse_smooth = np.mean((x_smooth - x_true) ** 2)

print("Observation MSE vs true state:", mse_observation)
print("Smoothed-state MSE vs true state:", mse_smooth)
print("Log-likelihood:", filter_result["log_likelihood"])

# GC Comparison
true_gc = varx_gc(y=x_true, u=u, source=1, target=0, na=1, nb=1, ridge=0.0)["gc"]
observed_gc = varx_gc(y=y_obs, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]
smoothed_gc = varx_gc(y=x_smooth, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

print("\nVARX GC Y -> X")
print("True latent state:", true_gc)
print("Noisy observation:", observed_gc)
print("Kalman smoothed state:", smoothed_gc)

# Plot
plot_ssm_smoothing(
    true_state=x_true,
    observations=y_obs,
    smoothed_state=x_smooth,
    channel=0,
    n_points=500,
    title="State-space VARX Kalman Smoothing of X"
)