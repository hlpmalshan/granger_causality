import numpy as np

def extract_var_coefficients(theta, n_vars, order):
    matrices = []
    for lag in range(order):
        start = lag * n_vars
        stop = (lag+1) * n_vars
        matrices.append(theta[:, start:stop])

    return matrices