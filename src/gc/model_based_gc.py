import numpy as np

from scipy.linalg import solve_discrete_lyapunov

# Solve for stationary covariance P
# P = A P A.T + Q
# This requires stable A
def stationary_covariance(A, Q):
    return solve_discrete_lyapunov(A, Q)

# Compute innovations covariance of a subprocess
# Full latent model:
# x_t = A x_{t-1} + w_t where w_t ~ N(0, Q)
# Observed process:
# z_t = C x_t where C selcts observed_indices
# The return innovation covariance is the steady-state covariance of the one-step prediction error z_t using only the past of z_t
def reduced_innovation_covariance(A, Q, observed_indices, max_iter=5000, tol=1e-10):
    A = np.asarray(A)
    Q = np.asarray(Q)

    n_states = A.shape[0]
    observed_indices = list(observed_indices)
    C = np.zeros((len(observed_indices), n_states))
    for row, idx in enumerate(observed_indices):
        C[row, idx] = 1.0

    # Start from unconditional stationary covariance
    try:
        P_filt = stationary_covariance(A, Q)
    except Exception:
        P_filt = np.eye(n_states)

    S_previous = None
    for _ in range(max_iter):
        # Prediction covariance
        P_pred = A @ P_filt @ A.T + Q

        # Innovation covariance
        S = C @ P_pred @ C.T
        S = 0.5 * (S + S.T)

        # Kalman update with zero measurement noise
        K = P_pred @ C.T @ np.linalg.pinv(S)
        P_new = P_pred - K @ S @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        if S_previous is not None:
            diff = np.linalg.norm(S - S_previous, ord="fro")
            scale = np.linalg.norm(S_previous, ord="fro") + 1e-12

            if diff / scale < tol:
                return S
            
        S_previous = S
        P_filt = P_new

    return S

# Compute model-based time-domain GC from source to target using full-model parameters A and Q
# For pairwise GC: pairwise_gc=None
# For condional GC: conditioning=[...]
def model_based_pairwise_gc(A, Q, source, target, conditioning=None, clip_negative=True):
    if conditioning is None:
        conditioning = []

    observed_indices = [target, *conditioning]
    S_reduced = reduced_innovation_covariance(A=A, Q=Q, observed_indices=observed_indices)
    sigma_reduced = np.real(S_reduced[0, 0])
    sigma_full = np.real(Q[target, target])
    gc = np.log(sigma_reduced / sigma_full)
    gc = np.real(gc)

    if clip_negative and gc < 0:
        if gc > -1e-8:
            gc = 0.0

    return {
        "gc": gc,
        "sigma_full": sigma_full,
        "sigma_reduced": sigma_reduced,
        "S_reduced": S_reduced
    }

# Compute pairwise model-based GC matrix
# Entry[i, j] = GC j -> i
def model_based_gc_matrix(A, Q):
    n_states = A.shape[0]
    G = np.zeros((n_states, n_states))
    for target in range(n_states):
        for source in range(n_states):
            if source == target:
                continue

            result = model_based_pairwise_gc(A=A, Q=Q, source=source, target=target)
            G[target, source] = result["gc"]

    return G
