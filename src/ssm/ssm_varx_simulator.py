import numpy as np

# Generate a linear Gaussian state-space VARX system
# x_t = A x_{t-1} + B u_t + w_t
# y_t = C x_t + D u_t + v_t
def generate_linear_ssm_varx(
        A,  # State transition matrix 
        B,  # Exogenous input matrix 
        u,  # Exogenous input 
        Q,  # Process noise covariance
        R,  # Observation noise covariance
        C=None, # Observation matrix. If None, identity is used.     
        D=None, # Direct exogenous-to-observation matrix If None, zero is used.
        x0=None, random_seed=42
):
    rng = np.random.default_rng(random_seed)

    u = np.asarray(u)
    n_samples, n_inputs = u.shape
    n_states = A.shape[0]
    if C is None: C = np.eye(n_states)
    n_obs = C.shape[0]
    if D is None: D = np.zeros((n_obs, n_inputs))
    if x0 is None: x0 = np.zeros(n_states)
    
    x  = np.zeros((n_samples, n_states))
    y = np.zeros((n_samples, n_obs))

    process_noise = rng.multivariate_normal(mean=np.zeros(n_states), cov=Q, size=n_samples)
    obs_noise = rng.multivariate_normal(mean=np.zeros(n_obs), cov=R, size=n_samples)

    x[0] = x0 + process_noise[0]
    y[0] = C @ x[0] + D @ u[0] + obs_noise[0]

    for t in range(1, n_samples):
        x[t] = A @ x[t-1] + B @ u[t] + process_noise[t]
        y[t] = C @ x[t] + D @ u[t] + obs_noise[t]

    return x, y     # Latent states, Observations