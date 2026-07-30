import numpy as np

def _unique_preserve_order(values):
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)

    return output

# Build a scalar-target VARX design matrix
# y_target[t]
#     =
# sum_{k=1}^{na} endogenous_predictors[t-k] * coefficients
#     +
# sum_{l=0}^{nb-1} u[t-l] * coefficients
#     +
# error[t]
def build_varx_target_design(y, u, target, endogenous_predictors, na, nb, include_intercept=False):
    '''
    Parameters
    ----------
    y : ndarray
        Endogenous data, shape (T, n_y).

    u : ndarray or None
        Exogenous input, shape (T, n_u).
        If None, only endogenous history is used.

    target : int
        Target endogenous variable index.

    endogenous_predictors : list[int]
        Endogenous variables whose histories are included.

    na : int
        Endogenous autoregressive order.

    nb : int
        Number of exogenous lags including current input.
        nb=0 means no exogenous input.
        nb=3 means u[t], u[t-1], u[t-2].

    include_intercept : bool
        Whether to include an intercept column.
    '''
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y must have shape (T, n_y)")
    
    T, n_y = y.shape

    if target < 0 or target >= n_y:
        raise ValueError("target index is out of range")
    
    endogenous_predictors = _unique_preserve_order(list(endogenous_predictors))
    for idx in endogenous_predictors:
        if idx < 0 or idx >= n_y:
            raise ValueError(f"endogenous predictor index {idx} is out of range")
        
    if u is None:
        u = np.zeros((T, 0))
        nb = 0
    else:
        u = np.asarray(u)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        if u.shape[0] != T:
            raise ValueError("y and u must have the same number of samples")
        
    n_u = u.shape[1]

    if na < 1:
        raise ValueError("na must be at least 1.")

    if nb < 0:
        raise ValueError("nb must be nonnegative.")

    max_lag = max(na, nb - 1 if nb > 0 else 0 )

    X_rows = []
    Y_values = []
    feature_names = []

    if include_intercept:
        feature_names.append("intercept")

    for lag in range(1, na+1):
        for idx in endogenous_predictors:
            feature_names.append(f"y{idx}_lag{lag}")

    for lag in range(nb):
        for j in range(n_u):
            feature_names.append(f"u{j}_lag{lag}")

    for t in range(max_lag, T):
        row = []

        if include_intercept:
            row.append(1.0)

        for lag in range(1, na+1):
            for idx in endogenous_predictors:
                row.append(y[t-lag, idx])

        for lag in range(nb):
            for j in range(n_u):
                row.append(u[t-lag, j])

        X_rows.append(row)
        Y_values.append(y[t, target])

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_values, dtype=float)

    return X, Y, feature_names
        