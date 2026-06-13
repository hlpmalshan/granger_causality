import numpy as np

def conditional_gc(
        data, 
        source,
        target,
        conditioning,
        order
):
    T = len(data)
    y = data[order:, target]
    full_X = []

    for t in range(order, T):

        row = []
        for lag in range(1, order+1):
            row.append(data[t-lag, target])
            row.append(data[t-lag, source])

            for z in conditioning:
                row.append(data[t-lag, z])

        full_X.append(row)

    full_X = np.asarray(full_X)
    beta = np.linalg.lstsq(full_X, y, rcond=None)[0]
    sigma_full = np.var(y - full_X @ beta, ddof=0)

    reduced_X = []
    for t in range(order, T):
        
        row = []
        for lag in range(1, order+1):
            row.append(data[t-lag, target])

            for z in conditioning:
                row.append(data[t-lag, z])

        reduced_X.append(row)

    reduced_X = np.asarray(reduced_X)
    beta = np.linalg.lstsq(reduced_X, y, rcond=None)[0]
    sigma_reduced = np.var(y - reduced_X @ beta, ddof=0)

    return np.log(sigma_reduced / sigma_full)
