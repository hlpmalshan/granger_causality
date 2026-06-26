import numpy as np

# Generate scalar colored exogenous input
# u_t = ar_coeff * u_{t-1} + noise_t
def generate_colored_input(
        n_samples,
        ar_coeff=0.9,
        noise_std=1.0,
        burn_in=200,
        random_seed=42
):
    rng = np.random.default_rng(random_seed)

    total = n_samples + burn_in
    u = np.zeros(total)
    noise = rng.normal(loc=0.0, scale=noise_std, size=total)

    for t in range(1, total):
        u[t] = ar_coeff * u[t-1] + noise[t]

    u = u[burn_in:]

    return u.reshape(-1, 1)

# Generate VARX data
# y_t = sum_k A_k y_{t-k} + sum_l B_l u_{t-l} + e_t
def generate_varx_data(
        A_matrices,
        B_matrices,
        u,
        noise_cov,
        burn_in=200,
        random_seed=123
):
    rng = np.random.default_rng(random_seed)

    u = np.asarray(u)
    n_samples, n_u = u.shape
    n_y = A_matrices[0].shape[0]

    na = len(A_matrices)
    nb = len(B_matrices)

    max_lag = max(na, nb - 1)
    total = n_samples + burn_in

    y = np.zeros((total, n_y))

    # Extend ipnut through burn-in by prepending zeros
    u_full = np.zeros((total, n_u))
    u_full[burn_in:] = u

    noise = rng.multivariate_normal(mean=np.zeros(n_y), cov=noise_cov, size=total)

    for t in range(max_lag, total):
        value = np.zeros(n_y)
        for k, A in enumerate(A_matrices, start=1):
            value += A @ y[t-k]

        for lag, B in enumerate(B_matrices):
            value += B @ u_full[t-lag]

        y[t] = value + noise[t]

    return y[burn_in:]