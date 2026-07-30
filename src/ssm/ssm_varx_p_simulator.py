import numpy as np

# Original model: x_t = A1 x_{t-1} + A2 x_{t-2} + ... + Ap x_{t-p} + w_t
# Companion state: s_t = [x_t, x_{t-1}, ..., x_{t-p+1}]^T
# Companion update: s_t = F s_{t-1} + eta_t
def build_var_companion_matrix(A_matrices):
    if len(A_matrices) < 1:
        raise ValueError("A_matrices must contain at least one matrix.")
    
    A_matrices = [np.asarray(A, dtype=float) for A in A_matrices]
    n_states = A_matrices[0].shape[0]
    p = len(A_matrices)
    for A in A_matrices:
        if A.shape != (n_states, n_states):
            raise ValueError("All A matrices must have shape (n_states, n_states).")
        
    F = np.zeros((n_states * p, n_states * p))
    # Top block: [A1 A2 ... Ap]
    F[:n_states, :] = np.hstack(A_matrices)

    # Lower shift blocks:
    # x_{t-1} <- x_t
    # x_{t-2} <- x_{t-1}
    # ...
    for block in range(1, p):
        row_start = block * n_states
        row_stop = (block + 1) * n_states

        col_start = (block - 1) * n_states
        col_stop = block * n_states

        F[row_start:row_stop, col_start:col_stop] = np.eye(n_states)

    return F

# Original model: x_t = ... + B0 u_t + B1 u_{t-1} + ... + Bq u_{t-q} + w_t
# Define: phi_t = [u_t, u_{t-1}, ..., u_{t-q}]^T
# Then: s_t = F s_{t-1} + G phi_t + eta_t
# Only the first block of the companion state receives direct input.
def build_varx_companion_input_matrix(B_matrices, n_states, p):
    if len(B_matrices) < 1:
        raise ValueError("B_matrices must contain at least one matrix.")
    B_matrices = [np.asarray(B, dtype=float) for B in B_matrices] 
    n_inputs = B_matrices[0].shape[1]
    nb = len(B_matrices)
    for B in B_matrices:
        if B.shape != (n_states, n_inputs):
            raise ValueError("All B matrices must have shape (n_states, n_inputs).")
        
    G = np.zeros((n_states * p, n_inputs * nb))
    G[:n_states, :] = np.hstack(B_matrices)

    return G

# Only the current latent state x_t receives process noise.
# The lag-storage blocks are deterministic shifts.
def build_companion_process_covariance(Q, p):
    Q = np.asarray(Q, dtype=float)
    n_states = Q.shape[0]
    Q_aug = np.zeros((n_states * p, n_states * p))
    Q_aug[:n_states, :n_states] = Q

    return Q_aug

# Original observation:
# y_t = C x_t + v_t
# Companion observation:
# y_t = C_aug s_t + v_t
# where:
# C_aug = [C, 0, ..., 0]
def build_companion_observation_matrix(C, p):
    C = np.asarray(C, dtype=float)
    n_obs, n_states = C.shape
    C_aug = np.zeros((n_obs, n_states * p))
    C_aug[:, :n_states] = C

    return C_aug

# Build phi_t = [u_t, u_{t-1}, ..., u_{t-nb+1}].
# If an early lag is unavailable, zeros are used.
def make_exogenous_regressor(u, t, nb):
    u = np.asarray(u, dtype=float)
    T, n_inputs = u.shape
    values = []
    for lag in range(nb):
        idx = t - lag
        if idx >= 0:
            values.append(u[idx])

        else:
            values.append(np.zeros(n_inputs))

    return np.concatenate(values)

# Compute the spectral radius of the VAR(p) companion matrix.
# Stability condition: spectral_radius(F) < 1
def var_companion_spectral_radius(A_matrices):
    F = build_var_companion_matrix(A_matrices)
    eigvals = np.linalg.eigvals(F)
    spectral_radius = float(np.max(np.abs(eigvals)))

    return spectral_radius

def is_stable_var(A_matrices, threshold=1.0):
    rho = var_companion_spectral_radius(A_matrices)
    return rho < threshold

# Simulate latent state-space VARX(p):
# x_t = sum_k A_k x_{t-k} + sum_l B_l u_{t-l} + w_t
# y_t = C x_t + D u_t + v_t
def generate_ssm_varx_p_data(
        A_matrices, # VAR coefficient matrices [A1, ..., Ap]
        B_matrices, # Exogenous matrices [B0, ..., Bq]
        u, # Exogenous input
        Q, # Latent process-noise covariance
        R, # Observation-noise covariance
        C=None, # Observation matrix
        D=None, # Direct input-to-observation matrix. For MEG-like simulations, use D = 0
        burn_in=0, # Number of initial samples to discard
        random_seed=42,
        x0=None, 
        return_augmented=False # Whether to return companion states as well
    ):
    rng = np.random.default_rng(random_seed)
    
    A_matrices = [np.asarray(A, dtype=float) for A in A_matrices]
    B_matrices = [np.asarray(B, dtype=float) for B in B_matrices]
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    T_total, n_inputs = u.shape
    
    p = len(A_matrices)
    nb = len(B_matrices)
    
    n_states = A_matrices[0].shape[0]
    
    Q = np.asarray(Q, dtype=float)
    
    if C is None:
        C = np.eye(n_states)
    else:
        C = np.asarray(C, dtype=float)
    n_obs = C.shape[0]

    R = np.asarray(R, dtype=float)

    if D is None:
        D = np.zeros((n_obs, n_inputs))
    else:
        D = np.asarray(D, dtype=float)

    F = build_var_companion_matrix(A_matrices)
    G = build_varx_companion_input_matrix(B_matrices=B_matrices, n_states=n_states, p=p)
    Q_aug = build_companion_process_covariance(Q=Q, p=p)
    C_aug = build_companion_observation_matrix(C=C, p=p)
    n_aug = n_states * p
    s = np.zeros((T_total, n_aug))
    x = np.zeros((T_total, n_states))
    y = np.zeros((T_total, n_obs))

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)

        if x0.shape != (n_states,):
            raise ValueError("x0 must have shape (n_states,).")
        
        s[0, :n_states] = x0

    process_noise = rng.multivariate_normal(mean=np.zeros(n_aug), cov=Q_aug + 1e-12 * np.eye(n_aug), size=T_total)
    observation_noise = rng.multivariate_normal(mean=np.zeros(n_obs), cov=R, size=T_total)

    for t in range(T_total):
        phi_t = make_exogenous_regressor(u=u, t=t, nb=nb)

        if t == 0:
            s[t] = s[t-1] + G @ phi_t + process_noise[t]
        else:
            s[t] = F @ s[t-1] + G @ phi_t + process_noise[t]

        x[t] = s[t, :n_states]
        y[t] = C_aug @ s[t] + D @ u[t] + observation_noise[t]

    if burn_in < 0 or burn_in >= T_total:
        raise ValueError("burn_in must be between 0 and T_total - 1")
    
    result = {
        "x": x[burn_in:],
        "y": y[burn_in:],
        "u": u[burn_in:],
        "F": F,
        "G": G,
        "Q_aug": Q_aug,
        "C_aug": C_aug,
        "A_matrices": A_matrices,
        "B_matrices": B_matrices,
        "Q": Q,
        "R": R,
        "C": C,
        "D": D,
        "spectral_radius": var_companion_spectral_radius(
            A_matrices
        ),
        "stable": is_stable_var(
            A_matrices
        )
    }

    if return_augmented:
        result["s"] = s[burn_in:]

    return result