import numpy as np

def generate_var_data(
    coeff_matrices,
    covariance,
    n_samples,
    burn_in=200,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)
    p = len(coeff_matrices)
    n_vars = coeff_matrices[0].shape[0]
    total_length = n_samples + burn_in
    data = np.zeros((total_length, n_vars))
    noise = rng.multivariate_normal(
        mean=np.zeros(n_vars),
        cov=covariance,
        size=total_length
    )

    for t in range(p, total_length):
        x_t = np.zeros(n_vars)

        for lag in range(p):
            x_t += coeff_matrices[lag] @ data[t-lag-1]

        data[t] = x_t + noise[t]

    return data[burn_in:]