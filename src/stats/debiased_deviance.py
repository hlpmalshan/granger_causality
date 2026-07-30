import numpy as np

from scipy.stats import chi2

# Fir scalar-output ridge/Tikhonov regression
# Objective: ||y - X h||^2 + gamma * h.T Gamma h
# The VARX paper uses: Gamma = diag(X.T X)
# so that predictors are regularized in a scale-normalized way
def fit_tikhonov_regression(X, y, gamma=0.0, penalty="diag"):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    if len(y) != X.shape[0]:
        raise ValueError("X and y have incompatible sample sizes.")
    
    T, n_features = X.shape

    Rxx = X.T @ X
    Rxy = X.T @ y

    if penalty == "diag":
        Gamma = np.diag(np.diag(Rxx))
    elif penalty == "identity":
        Gamma = np.eye(n_features)
    elif penalty is None:
        Gamma = np.zeros_like(Rxx)
    else:
        raise ValueError("penalty must be 'diag', 'identity', or None.")
    
    lhs = Rxx + gamma * Gamma
    h = np.linalg.pinv(lhs) @ Rxy
    residuals = y - X @ h
    rss = float(residuals.T @ residuals)
    sigma2 = rss / T
    Rxe = Rxy - Rxx @ h

    return{
        "h": h,
        "residuals": residuals,
        "rss": rss,
        "sigma2": sigma2,
        "Rxx": Rxx,
        "Rxy": Rxy,
        "Rxe": Rxe,
        "Gamma": Gamma,
        "T": T,
        "n_features": n_features,
        "gamma": gamma
    }

# Compute the L2 deviance bias term
# Scalar-output version
# b = 1/(2 * r_ee) * r_xe.T R_xx^{-1} r_xe
# where: r_ee = ||y - X h_hat||^2 and r_xe = X.T y - X.T X h_hat
def l2_bias_term(fit_result, eps=1e-12):
    Rxx = fit_result["Rxx"]
    Rxe = fit_result["Rxe"]
    rss = fit_result["rss"]

    if rss <= eps:
        return 0.0
    
    value = Rxe.T @ np.linalg.pinv(Rxx) @ Rxe

    b = 0.5 * float(value) / float(rss + eps)

    return b

# Compute raw GC, raw debiance, de-biased deviance and p-value
# Raw GC :              F = log(sigma_r^2 / sigma_f^2)
# Raw deviance :        D = T * F
# De-debiased deviance: D_db = T_eff * F - b_r + b_f
# p-value:              p = 1 - chi2_cdf(D_db, df_removed)
def debiased_deviance_from_fits(
        full_fit, 
        reduced_fit, 
        df_removed, 
        effective_t_mode="full_minus_features",
        clip_negative_debiased=True
):
    sigma_f = float(full_fit["sigma2"])
    sigma_r = float(reduced_fit["sigma2"])
    if sigma_f <= 0 or sigma_r <= 0:
        raise ValueError("Residual variances must be positive.")
    
    T = int(full_fit["T"])
    if reduced_fit["T"] != T:
        raise ValueError("Full and reduced fits must use the same T.")
    
    n_full_features = int(full_fit["n_features"])
    if effective_t_mode == "n_obs":
        T_eff = T
    elif effective_t_mode == "full_minus_features":
        T_eff = max(T - n_full_features, 1)
    else:
        raise ValueError("effective_t_mode must be 'n_obs' or 'full_minus_features'.")
    
    gc_raw = float(np.log(sigma_r / sigma_f))

    deviance_raw = T * gc_raw
    
    b_f = l2_bias_term(full_fit) 
    b_r = l2_bias_term(reduced_fit)
    deviance_debiased = T * gc_raw - b_r + b_f

    if clip_negative_debiased and deviance_debiased < 0:
         deviance_for_p = 0.0
    else:
        deviance_for_p = deviance_debiased

    p_value = float(chi2.sf(deviance_for_p, df=df_removed))

    gc_debiased = float(deviance_debiased / T_eff)

    return {
        "gc_raw": gc_raw,
        "gc_debiased": gc_debiased,
        "deviance_raw": deviance_raw,
        "deviance_debiased": deviance_debiased,
        "deviance_for_p": deviance_for_p,
        "p_value": p_value,
        "df": df_removed,
        "T": T,
        "T_eff": T_eff,
        "bias_full": b_f,
        "bias_reduced": b_r,
        "sigma_full": sigma_f,
        "sigma_reduced": sigma_r
    }