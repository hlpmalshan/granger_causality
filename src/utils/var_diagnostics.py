import numpy as np

def companion_matrix(coef_matrices):
    p = len(coef_matrices)
    n = coef_matrices[0].shape[0]
    top_row = np.hstack(coef_matrices)
    
    if p == 1:
        return top_row
    
    lower = np.zeros((p-1)*n, p*n)

    for i in range(p-1):
        start_col = i*n
        lower[i*n:(i+1)*n, start_col:start_col+n] = np.eye(n)

    companion = np.vstack([top_row, lower])

    return companion

def check_stability(coef_matrices, verbose=True):
    C = companion_matrix(coef_matrices)
    eigvals = np.linalg.eigvals(C)
    spectral_radius = np.max(np.abs(eigvals))

    stable = spectral_radius < 1

    if verbose:
        print("Spectral Radius:", spectral_radius)
        print("Stable:", stable)

    return {
        "stable": stable,
        "spectral_radius": spectral_radius,
        "eigenvalues": eigvals
    }

def compute_aic(residual_covariance, n_params, n_obs):
    return (np.log(np.linalg.det(residual_covariance)) + 2*n_params/n_obs)

def compute_bic(residual_covariance, n_params, n_obs):
    return (np.log(np.linalg.det(residual_covariance)) + np.log(n_obs)*n_params/n_obs)

def residual_autocorrelation(residuals, lag=20):
    n_vars = residuals.shape[1]
    results = {}

    for v in range(n_vars):
        x = residuals[:, v]
        acf = []
        for k in range(1, lag+1):
            corr = np.corrcoef(x[:-k], x[k:])[0, 1]
            acf.append(corr)

        results[v] = np.array(acf)

    return results

def count_var_parameters(n_vars, order):
    return (n_vars*n_vars*order)
