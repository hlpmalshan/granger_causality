import numpy as np

def var_to_frequency_domain(coef_matrices, frequencies):
    """
    Construct A(w) = I - sum(Ak exp(-iwk))
    """
    n_vars = coef_matrices[0].shape[0]
    A_freq = []

    for w in frequencies:
        A = np.eye(n_vars, dtype=complex)
        
        for k, Ak in enumerate(coef_matrices, start=1):
            A -= Ak * np.exp(-1j * w * k)

        A_freq.append(A)

    return A_freq

def transfer_function(coef_matrices, frequencies):
    A_freq = var_to_frequency_domain(coef_matrices, frequencies)

    H_freq = []

    for A in A_freq:
        H_freq.append(np.linalg.inv(A))

    return H_freq