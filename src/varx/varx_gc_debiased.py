from src.varx.design_matrix import build_varx_target_design, _unique_preserve_order
from src.stats.debiased_deviance import fit_tikhonov_regression, debiased_deviance_from_fits

# Bias-corrected sample-based VARX Granger test
# Tests: source -> target
# Full model includes: target history, source history, conditioning history, exogenous input history
# Reduced model removes: source history
def debiased_varx_gc(y, u, source, target, na, nb, conditioning=None, gamma=0.0, penalty="diag", include_intercept=False, effective_t_mode="full_minus_features"):
    if conditioning is None:
        conditioning = []

    if source == target:
        raise ValueError("source and target must be different.")
    
    full_predictors = _unique_preserve_order([target, source, *conditioning])
    reduced_predictors = _unique_preserve_order([target, *conditioning])

    X_full, Y_full, full_feature_names = build_varx_target_design(
        y=y, u=u, target=target, endogenous_predictors=full_predictors, na=na, nb=nb, include_intercept=include_intercept
    )
    
    X_reduced, Y_reduced, reduced_feature_names = build_varx_target_design(
        y=y, u=u, target=target, endogenous_predictors=reduced_predictors, na=na, nb=nb, include_intercept=include_intercept
    )

    if len(Y_full) != len(Y_reduced):
        raise RuntimeError(
            "Full and reduced models have different target lengths."
        )
    
    full_fit = fit_tikhonov_regression(X=X_full, y=Y_full, gamma=gamma, penalty=penalty)
    reduced_fit = fit_tikhonov_regression(X=X_reduced, y=Y_reduced, gamma=gamma, penalty=penalty)

    # For one scalar source and one scalar target:
    # removing source history removes na parameters
    df_removed = na

    stats = debiased_deviance_from_fits(full_fit=full_fit, reduced_fit=reduced_fit, df_removed=df_removed, effective_t_mode=effective_t_mode)

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
        "full_feature_names": full_feature_names,
        "reduced_feature_names": reduced_feature_names
    }