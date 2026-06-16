import numpy as np 

def generate_time_varying_var2(n_samples, noise_cov, burn_in=500):
    total = n_samples + burn_in
    data = np.zeros((total, 2))
    noise = np.random.multivariate_normal(np.zeros(2), noise_cov, size=total)

    true_coupling = []

    r = 0.95
    omega0 = 0.3 * np.pi
    a1_y = 2 * r * np.cos(omega0)
    a2_y = -r**2

    for t in range(2, total):
        normalized_time = (t-burn_in) / n_samples
        normalized_time = np.clip(normalized_time, 0, 1)
        coupling = 0.8 - 0.7 * normalized_time
        true_coupling.append(coupling)

        A1 = np.array([
            [0.5, coupling],
            [0.0, a1_y]
        ])

        A2 = np.array([
            [0.0, 0.0],
            [0.0, a2_y]
        ])

        data[t] = A1 @ data[t-1] + A2 @ data[t-2] + noise[t]

    return data[burn_in:], np.asarray(true_coupling)