import numpy as np
from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_simulator import generate_linear_ssm_varx
from src.ssm.em_varx_ssm_known_c import EMVARXSSMKnownC
from src.varx.varx_gc import varx_gc
from src.utils.plotting import plot_ssm_mixed_observation, plot_em_log_likelihood

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

# Nontrivial observation matrix.
# y is a mixed version of x.
C_true = np.array([
    [1.00, 0.35],
    [0.25, 1.00]
])

R_true = 0.60 * np.eye(2)

# Exogenous input
u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

# Simulate latent state and mixed noisy observations
x_true, y_obs = generate_linear_ssm_varx(A=A_true, B=B_true, u=u, Q=Q_true, R=R_true, C=C_true ,random_seed=2)

# Fit EM state-space VARX with known C
model = EMVARXSSMKnownC(n_states=2, n_inputs=1, C=C_true, ridge=1e-6)
model.fit(y=y_obs, u=u, max_iter=75, tol=1e-3, verbose=True)
x_smooth = model.smooth_result["x_smooth"]

# Pseudo-inverse observation proxy
C_pinv = np.linalg.pinv(C_true)
x_proxy = (C_pinv @ y_obs.T).T

# Parameter comparison
print("\nTrue C")
print(C_true)

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
mse_proxy = np.mean((x_proxy - x_true)**2)
mse_smooth = np.mean((x_smooth - x_true)**2)
print("\nMSE comparison")
print("Pseudo-inverse observation proxy MSE:", mse_proxy)
print("EM smoothed-state MSE:", mse_smooth)

# GC comparison
true_gc = varx_gc(y=x_true, u=u, source=1, target=0, na=1, nb=1, ridge=0.0)["gc"]
proxy_gc = varx_gc(y=x_proxy, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]
smoothed_gc = varx_gc(y=x_smooth, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

print("\nVARX GC Y -> X")
print("True latent state:", true_gc)
print("Pseudo-inverse proxy:", proxy_gc)
print("EM smoothed state:", smoothed_gc)

print("\nParameter-level interpretation")
print("True A[0,1] Y->X:", A_true[0, 1])
print("Estimated A[0,1] Y->X:", model.A[0, 1])
print("True A[1,0] X->Y:", A_true[1, 0])
print("Estimated A[1,0] X->Y:", model.A[1, 0])

# Plots
plot_ssm_mixed_observation(
    true_state=x_true, 
    proxy_observation=x_proxy, 
    smoothed_state=x_smooth, 
    channel=0, 
    n_points=500, 
    title="State-space VARX with known C Latent X Recovery"
)

plot_em_log_likelihood(model.log_likelihoods, title="EM Log-likelihood with known C")
