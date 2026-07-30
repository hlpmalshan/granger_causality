import numpy as np

def build_varx_multitarget_design(y, u, na, nb, include_intercept=False, start_lag=None):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must have shape (T, n_y).")
    
    T, n_y = y.shape

    if u is None:
        u = np.zeros((T, 0))
        nb = 0
    else:
        u = np.asarray(u, dtype=float)

        if u.ndim == 1:
            u = u.reshape(-1, 1)

        if u.shape[0] != T:
            raise ValueError("y and u must have the same number of samples.")

    n_u = u.shape[1]

    if na < 1:
        raise ValueError("na must be at least 1.")

    if nb < 0:
        raise ValueError("nb must be nonnegative.")
    
    minimum_lag = max(
        na,
        nb - 1 if nb > 0 else 0
    )

    if start_lag is None:
        start_lag = minimum_lag

    if start_lag < minimum_lag:
        raise ValueError(
            "start_lag is too small for the requested na and nb."
        )
    
    X_rows = []
    Y_rows = []

    feature_names = []

    if include_intercept:
        feature_names.append("intercept")

    for lag in range(1, na + 1):
        for i in range(n_y):
            feature_names.append(f"y{i}_lag{lag}")

    for lag in range(nb):
        for j in range(n_u):
            feature_names.append(f"u{j}_lag{lag}")

    for t in range(start_lag, T):
        row = []
        if include_intercept:
            row.append(1.0)

        for lag in range(1, na + 1):
            for i in range(n_y):
                row.append(y[t - lag, i])

        for lag in range(nb):
            for j in range(n_u):
                row.append(u[t - lag, j])

        X_rows.append(row)
        Y_rows.append(y[t])

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)

    return X, Y, feature_names

def fit_varx_multitarget(y, u, na, nb, gamma=0.0, penalty="diag", include_intercept=False, start_lag=None):
    X, Y, feature_names = build_varx_multitarget_design(y=y, u=u, na=na, nb=nb, include_intercept=include_intercept, start_lag=start_lag)
    T_eff, n_features = X.shape
    n_targets = Y.shape[1]

    Rxx = X.T @ X
    Rxy = X.T @ Y

    if penalty == "diag":
        Gamma = np.diag(np.diag(Rxx))
    elif penalty == "identity":
        Gamma = np.eye(n_features)
    elif penalty is None:
        Gamma = np.zeros_like(Rxx)
    else:
        raise ValueError("penalty must be 'diag', 'identity', or None.")
    
    beta = np.linalg.pinv(Rxx + gamma * Gamma) @ Rxy
    residuals = Y - X @ beta
    residual_covariance = (residuals.T @ residuals) / T_eff
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)

    return {
        "X": X,
        "Y": Y,
        "beta": beta,
        "residuals": residuals,
        "residual_covariance": residual_covariance,
        "T_eff": T_eff,
        "n_features": n_features,
        "n_targets": n_targets,
        "n_parameters": n_features * n_targets,
        "feature_names": feature_names,
        "na": na,
        "nb": nb,
        "gamma": gamma
    }

# Numerically stable log determinant for residual covariance.
def safe_logdet(matrix, jitter=1e-10, max_attempts=8):
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    
    n = matrix.shape[0]
    scale = np.trace(matrix) / max(n, 1)
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    for attempt in range(max_attempts):
        added = (jitter * (10 ** attempt) * scale )
        candidate = matrix + added * np.eye(n)
        sign, logdet = np.linalg.slogdet(candidate)

        if sign > 0 and np.isfinite(logdet):
            return float(logdet)

    raise np.linalg.LinAlgError("Could not compute a positive log determinant.")

# Compute Gaussian VARX AIC/BIC, ignoring constants shared across candidate orders.
def compute_varx_aic_bic(fit_result):
    T_eff = fit_result["T_eff"]
    K = fit_result["n_parameters"]

    Sigma = fit_result["residual_covariance"]

    logdet = safe_logdet(Sigma)

    # AIC = T log|Sigma| + 2K
    aic = (T_eff * logdet + 2 * K)
    # BIC = T log|Sigma| + K log(T)
    bic = (T_eff * logdet + K * np.log(T_eff))

    return {
        "logdet_sigma": logdet,
        "aic": float(aic),
        "bic": float(bic),
        "T_eff": T_eff,
        "n_parameters": K
    }

# If common_start_lag=True, all candidate orders are fitted on the same time indices. This makes AIC/BIC comparison fair.
def evaluate_varx_orders(y, u, na_candidates, nb, gamma=0.0, penalty="diag", include_intercept=False, common_start_lag=True ):
    na_candidates = list(na_candidates)

    if common_start_lag:
        start_lag = max(max(na_candidates), nb - 1 if nb > 0 else 0)
    else:
        start_lag = None

    rows = []

    for na in na_candidates:

        fit = fit_varx_multitarget(y=y, u=u, na=na, nb=nb, gamma=gamma, penalty=penalty, include_intercept=include_intercept, start_lag=start_lag)

        ic = compute_varx_aic_bic(fit)

        rows.append({
            "na": na,
            "nb": nb,
            "gamma": gamma,
            "T_eff": ic["T_eff"],
            "n_parameters": ic["n_parameters"],
            "logdet_sigma": ic["logdet_sigma"],
            "aic": ic["aic"],
            "bic": ic["bic"]
        })

    best_aic = min(rows, key=lambda row: row["aic"])["na"]

    best_bic = min(rows, key=lambda row: row["bic"])["na"]

    return {
        "rows": rows,
        "best_aic": best_aic,
        "best_bic": best_bic,
        "start_lag": start_lag
    }