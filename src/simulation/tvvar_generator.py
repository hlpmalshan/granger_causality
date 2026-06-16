import numpy as np

def generate_piecewise_var(
        coef_sets, # List of VAR coefficient lists (Example: [[A1_regime1], [A1_regime2]])
        change_points, # change_points=[5000] means: regime1:t < 5000 regime2:t >= 5000
        noise_cov,
        n_samples,
        burn_in=500
):
    n_vars = coef_sets[0][0].shape[0]
    max_order = max(len(c) for c in coef_sets)
    total = n_samples + burn_in

    data = np.zeros((total, n_vars))

    noise = np.random.multivariate_normal(np.zeros(n_vars), noise_cov, size=total)

    for t in range(max_order, total):
        actual_t = t - burn_in
        regime = 0
        for cp in change_points:
            if actual_t >= cp:
                regime += 1
            
        coeffs = coef_sets[regime]
        value = np.zeros(n_vars)

        for lag, A in enumerate (coeffs, start=1):
            value += (A @ data[t-lag])

        value += noise[t]
        data[t] = value

    return data[burn_in:]

def generate_smoothly_varying_var(n_samples, noise_cov, burn_in=500):
    total = n_samples + burn_in
    n_vars = 2
    data = np.zeros((total, n_vars))
    noise = np.random.multivariate_normal(np.zeros(n_vars), noise_cov, size=total)

    coupling_history = []
    for t in range(1, total):
        normalized_time = (t-burn_in) / n_samples
        normalized_time = np.clip(normalized_time, 0, 1)
        coupling = (0.8 - 0.7 * normalized_time)
        coupling_history.append(coupling)

        A = np.array([
            [0.5, coupling],
            [0.0, 0.8]
        ])

        data[t] = ( A @ data[t-1] + noise[t])

    return (data[burn_in:], np.asarray(coupling_history))