import numpy as np

from scipy.stats import chi2
from src.ssm.ssm_varx_p_simulator import make_exogenous_regressor

def _unique_preserve_order(values):
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)

    return output

def _regularize_scalar(value, eps=1e-12):
    return max(float(value), eps)

# Build posterior expected sufficient statistics for one scalar latent VARX target regression.
# Dynamics regression: x_target,t = h^T z_t + e_t
# where: z_t = [selected components of s_{t-1}, phi_t]
# and:
# s_{t-1} = [x_{t-1}, x_{t-2}, ..., x_{t-na}]
# phi_t = [u_t, u_{t-1}, ..., u_{t-nb+1}]
def build_latent_varx_target_sufficient_statistics(
        smooth_mean, # Smoothed companion-state mean
        smooth_cov, # Smoothed companion-state covariance
        smooth_lag_cov, # Approximate lag-one smoothed covariance, smooth_lag_cov[t] ≈ Cov(s_t, s_{t-1} | y)
        u,
        target,
        endogenous_predictors,  # Latent variables whose histories are included
        na,
        nb,
        n_latent,
        start_time=1
):
    smooth_mean = np.asarray(smooth_mean, dtype=float)
    smooth_cov = np.asarray(smooth_cov, dtype=float)
    smooth_lag_cov = np.asarray(smooth_lag_cov, dtype=float)
    
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    T = smooth_mean.shape[0]
    n_inputs = u.shape[1]
    
    endogenous_predictors = _unique_preserve_order(endogenous_predictors)

    # Indices inside s_{t-1}.
    # s_{t-1} = [x_{t-1}, x_{t-2}, ..., x_{t-na}]
    latent_feature_indices = []
    feature_names = []

    for lag in range(1, na + 1):
        block_start = (lag - 1) * n_latent
        for idx in endogenous_predictors:
            latent_feature_indices.append(block_start + idx)
            feature_names.append(f"x{idx}_lag{lag}")

    for lag in range(nb):
        for j in range(n_inputs):
            feature_names.append(f"u{j}_lag{lag}")

    n_latent_features = len(latent_feature_indices)

    n_phi = n_inputs * nb
    z_dim = n_latent_features + n_phi

    S_zz = np.zeros((z_dim, z_dim))
    S_zx = np.zeros(z_dim)
    S_xx = 0.0
    n_used = 0

    for t in range(start_time, T):

        mu_t = smooth_mean[t]
        P_t = smooth_cov[t]

        mu_prev = smooth_mean[t - 1]
        P_prev = smooth_cov[t - 1]

        P_t_prev = smooth_lag_cov[t]

        mu_x = mu_t[target]
        var_x = P_t[target, target]

        phi_t = make_exogenous_regressor(u=u, t=t, nb=nb)
        mu_prev_selected = mu_prev[latent_feature_indices]
        P_prev_selected = P_prev[np.ix_(latent_feature_indices, latent_feature_indices)]
        cov_x_prev_selected = P_t_prev[target, latent_feature_indices]

        # E[z]
        z_mean = np.concatenate([mu_prev_selected, phi_t])

        # E[z z^T]
        E_zz = np.zeros((z_dim, z_dim))
        E_prev_prev = (P_prev_selected + np.outer(mu_prev_selected, mu_prev_selected))

        E_zz[:n_latent_features, :n_latent_features] = E_prev_prev
        E_zz[:n_latent_features, n_latent_features:] = np.outer(mu_prev_selected, phi_t)
        E_zz[n_latent_features:, :n_latent_features] = np.outer(phi_t, mu_prev_selected)
        E_zz[n_latent_features:, n_latent_features:] = np.outer(phi_t, phi_t)

        # E[z x]
        E_zx = np.zeros(z_dim)
        E_zx[:n_latent_features] = (cov_x_prev_selected + mu_prev_selected * mu_x)
        E_zx[n_latent_features:] = (phi_t * mu_x)

        # E[x^2]
        E_xx = (var_x + mu_x ** 2)

        S_zz += E_zz
        S_zx += E_zx
        S_xx += E_xx

        n_used += 1

    return {
        "S_zz": S_zz,
        "S_zx": S_zx,
        "S_xx": float(S_xx),
        "n_used": n_used,
        "feature_names": feature_names,
        "latent_feature_indices": latent_feature_indices,
        "target": target,
        "endogenous_predictors": endogenous_predictors,
        "na": na,
        "nb": nb
    }

# Fit ridge/Tikhonov regression using posterior expected sufficient statistics.
def fit_expected_latent_regression(stats, gamma=0.0, penalty="diag", eps=1e-12):
    S_zz = stats["S_zz"]
    S_zx = stats["S_zx"]
    S_xx = stats["S_xx"]
    n_used = stats["n_used"]

    n_features = S_zz.shape[0]

    if penalty == "diag":
        Gamma = np.diag(np.diag(S_zz))
    elif penalty == "identity":
        Gamma = np.eye(n_features)
    elif penalty is None:
        Gamma = np.zeros_like(S_zz)
    else:
        raise ValueError("penalty must be 'diag', 'identity', or None.")

    h = (np.linalg.pinv(S_zz + gamma * Gamma) @ S_zx)

    expected_rss = (S_xx - 2.0 * float(h.T @ S_zx) + float(h.T @ S_zz @ h))
    expected_rss = _regularize_scalar(expected_rss, eps=eps)

    sigma2 = (expected_rss / max(n_used, 1))

    R_ze = (S_zx - S_zz @ h)

    return {
        "h": h,
        "expected_rss": expected_rss,
        "sigma2": sigma2,
        "S_zz": S_zz,
        "S_zx": S_zx,
        "S_xx": S_xx,
        "R_ze": R_ze,
        "Gamma": Gamma,
        "n_used": n_used,
        "n_features": n_features,
        "gamma": gamma,
        "feature_names": stats["feature_names"]
    }

# Latent analogue of the L2 bias term.
# Uses expected sufficient statistics instead of sample sums.
def latent_l2_bias_term(fit_result, eps=1e-12):
    S_zz = fit_result["S_zz"]
    R_ze = fit_result["R_ze"]
    rss = fit_result["expected_rss"]

    if rss <= eps:
        return 0.0

    value = (R_ze.T @ np.linalg.pinv(S_zz) @ R_ze)

    return 0.5 * float(value) / float(rss + eps)

# Compute latent complete-data de-biased deviance from expected full/reduced regression fits.
def debiased_deviance_from_latent_fits(
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

    T = int(full_fit["n_used"])
    if int(reduced_fit["n_used"]) != T:
        raise ValueError("Full and reduced fits must use the same n_used.")

    n_full_features = int(full_fit["n_features"])

    if effective_t_mode == "n_obs":
        T_eff = T
    elif effective_t_mode == "full_minus_features":
        T_eff = max(T - n_full_features, 1)
    else:
        raise ValueError("effective_t_mode must be 'n_obs' or 'full_minus_features'.")

    gc_raw = float(np.log(sigma_r / sigma_f))

    deviance_raw = T * gc_raw

    b_f = latent_l2_bias_term(full_fit)
    b_r = latent_l2_bias_term(reduced_fit)

    deviance_debiased = (T_eff * gc_raw - b_r + b_f)

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

# Latent complete-data de-biased VARX GC using posterior sufficient statistics from a smoother.
# This is the latent analogue of debiased_varx_gc().
def debiased_latent_varx_gc_from_smoother(
        smooth_mean,
        smooth_cov,
        smooth_lag_cov,
        u,
        source,
        target,
        na,
        nb,
        n_latent,
        conditioning=None,
        gamma=0.0,
        penalty="diag",
        effective_t_mode="full_minus_features"
):
    if conditioning is None:
        conditioning = []

    if source == target:
        raise ValueError("source and target must be different.")

    full_predictors = _unique_preserve_order([target, source, *conditioning])
    reduced_predictors = _unique_preserve_order([target, *conditioning])

    full_stats = build_latent_varx_target_sufficient_statistics(
        smooth_mean=smooth_mean,
        smooth_cov=smooth_cov,
        smooth_lag_cov=smooth_lag_cov,
        u=u,
        target=target,
        endogenous_predictors=full_predictors,
        na=na,
        nb=nb,
        n_latent=n_latent
    )

    reduced_stats = build_latent_varx_target_sufficient_statistics(
        smooth_mean=smooth_mean,
        smooth_cov=smooth_cov,
        smooth_lag_cov=smooth_lag_cov,
        u=u,
        target=target,
        endogenous_predictors=reduced_predictors,
        na=na,
        nb=nb,
        n_latent=n_latent
    )

    full_fit = fit_expected_latent_regression(
        stats=full_stats,
        gamma=gamma,
        penalty=penalty
    )

    reduced_fit = fit_expected_latent_regression(
        stats=reduced_stats,
        gamma=gamma,
        penalty=penalty
    )

    df_removed = na

    stats = debiased_deviance_from_latent_fits(
        full_fit=full_fit,
        reduced_fit=reduced_fit,
        df_removed=df_removed,
        effective_t_mode=effective_t_mode
    )

    return {
        **stats,
        "source": source,
        "target": target,
        "na": na,
        "nb": nb,
        "conditioning": conditioning,
        "gamma": gamma,
        "full_fit": full_fit,
        "reduced_fit": reduced_fit,
        "full_feature_names": full_fit["feature_names"],
        "reduced_feature_names": reduced_fit["feature_names"]
    }