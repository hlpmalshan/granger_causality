import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_simulator import generate_linear_ssm_varx
from src.ssm.em_varx_ssm import EMVARXSSM
from src.varx.varx_gc import varx_gc
from src.utils.plotting import plot_ssm_smoothing, plot_em_log_likelihood

# Configuration
N_SAMPLES = 3000

A_true = np.array([
    [0.60, 0.30],
    [0.00, 0.55]
])

B_true = np.array([
    [0.50],
    [0.40]
])

Q_true = 0.15 * np.eye(2)
R_true = 0.60 * np.eye(2)

C = np.eye(2)

# Exogenous input
u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

# Simulate latent state and noisy observations
x_true, y_obs = generate_linear_ssm_varx(A=A_true, B=B_true, u=u, Q=Q_true, R=R_true, C=C, random_seed=2)

# Fit EM state-space VARX
model = EMVARXSSM(n_states=2, n_inputs=1, ridge=1e-6)
model.fit(y=y_obs, u=u, max_iter=50, tol=1e-3, verbose=True)

x_smooth = model.smooth_result["x_smooth"]

# Parameter comparison
print("\nTrue A")
print(A_true)

print("\nEstimated A")
print(model.A)

print("\nTrue B")
print(B_true)

print("\nEstimated B")
print(model.B)

print("\nTrue Q")
print(Q_true)

print("\nEstimated Q")
print(model.Q)

print("\nTrue R")
print(R_true)

print("\nEstimated R")
print(model.R)

# MSE comparison
mse_observation = np.mean((y_obs - x_true) ** 2)
mse_smooth = np.mean((x_smooth - x_true) ** 2)

print("\nMSE comparison")
print("Observation MSE:", mse_observation)
print("EM smoothed-state MSE:", mse_smooth)

# GC Comparison
true_gc = varx_gc(y=x_true, u=u, source=1, target=0, na=1, nb=1, ridge=0.0)["gc"]
observed_gc = varx_gc(y=y_obs, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]
smoothed_gc = varx_gc(y=x_smooth, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

print("\nVARX GC Y -> X")
print("True latent state:", true_gc)
print("Noisy observation:", observed_gc)
print("EM smoothed state:", smoothed_gc)

print("\nParameter-level interpretation")
print("True A[0,1] Y->X:", A_true[0, 1])
print("Estimated A[0,1] Y->X:", model.A[0, 1])
print("True A[1,0] X->Y:", A_true[1, 0])
print("Estimated A[1,0] X->Y:", model.A[1, 0])

# Plots
plot_ssm_smoothing(true_state=x_true, observations=y_obs, smoothed_state=x_smooth, channel=0, n_points=500, title="EM State-space VARX Smoothed X")
plot_em_log_likelihood(model.log_likelihoods, title="EM Log-likelihood")