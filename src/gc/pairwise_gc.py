import numpy as np

def pairwise_gc(
        data,
        source,
        target,
        order
):
    T = len(data)
    full_y = data[order:, target]
    full_X = []

    for t in range(order, T):
        row = []

        for lag in range(1, order+1):
            row.append(data[t-lag, target])
            row.append(data[t-lag, source])

        full_X.append(row)

    full_X = np.asarray(full_X)
    beta = np.linalg.lstsq(full_X, full_y, rcond=None)[0]
    residuals = full_y - full_X @ beta
    sigma_full = np.var(residuals, ddof = 0)

    reduced_X = []
    for t in range(order, T):
        
        row = []
        for lag in range(1, order+1):
            row.append(data[t-lag, target])

        reduced_X.append(row)

    reduced_X = np.asarray(reduced_X)
    beta = np.linalg.lstsq(reduced_X, full_y, rcond=None)[0]
    residuals = full_y - reduced_X @ beta
    sigma_reduced = np.var(residuals, ddof=0)

    gc = np.log(sigma_reduced / sigma_full)

    return gc
