import numpy as np

def transfer_function_timevarying(coef_matrices, frequencies):
    n_vars = coef_matrices[0].shape[0]
    H_all = []

    for w in frequencies:
        A = np.eye(n_vars, dtype=complex)
        for lag, Ak in enumerate(coef_matrices, start=1):
            A -= Ak * np.exp(-1j * w * lag)

        H_all.append(np.linalg.inv(A))

    return H_all

def spectral_gc_single_time(coef_matrices, residual_covariance, frequencies):
    H_freq = transfer_function_timevarying(coef_matrices, frequencies)

    gc = []
    sigma_x = residual_covariance[0, 0]
    sigma_y = residual_covariance[1, 1]

    for H in H_freq:
        Hxx = H[0, 0]
        Hxy = H[0, 1]

        numerator = np.abs(Hxx)**2 * sigma_x + np.abs(Hxy)**2 * sigma_y
        denominator = np.abs(Hxx)**2 * sigma_x

        gc.append(np.log(numerator / denominator))

    return np.asarray(gc)