import numpy as np
from scipy.stats import chi2

# Get the final observed-data log-likelihood from a fitted EM model.
def final_observed_log_likelihood(model):
    if model.smooth_result is None:
        raise RuntimeError("Model has no smoothing result. Fit the model first.")

    if "log_likelihood" in model.smooth_result:
        return float(model.smooth_result["log_likelihood"])

    raise RuntimeError("Could not find final log_likelihood in model.smooth_result.")

# Observed-data likelihood-ratio deviance:
# D_obs = 2 [log p(y | theta_full) - log p(y | theta_reduced)]
# The p-value reported here is the asymptotic chi-square p-value. 
# As with the latent complete-data statistic, we should empirically validate this reference distribution later.
def observed_likelihood_deviance(full_model, reduced_model, df_removed, clip_negative=True):
    ll_full = final_observed_log_likelihood(full_model)
    ll_reduced = final_observed_log_likelihood(reduced_model)

    deviance = 2.0 * (ll_full - ll_reduced)

    if clip_negative and deviance < 0:
        deviance_for_p = 0.0
    else:
        deviance_for_p = deviance

    p_value = float(chi2.sf(deviance_for_p, df=df_removed))

    return {
        "ll_full": ll_full,
        "ll_reduced": ll_reduced,
        "observed_likelihood_deviance": float(deviance),
        "deviance_for_p": float(deviance_for_p),
        "p_value": p_value,
        "df": df_removed
    }