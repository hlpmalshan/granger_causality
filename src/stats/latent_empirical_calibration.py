import numpy as np

from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c import EMVARXPSSMKnownC
from src.stats.latent_debiased_deviance import debiased_latent_varx_gc_from_smoother
from src.stats.empirical_pvalue import empirical_upper_tail_p_value, clipped_deviance_statistic

def fit_em_known_c_fixed_r(
        y_obs,
        u,
        C,
        R_fixed,
        na,
        nb,
        Q_init=None,
        max_iter=50,
        tol=1e-5,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_floor=0.05,
        verbose=False
):
    
    C = np.asarray(C, dtype=float)

    n_latent = C.shape[1]

    if Q_init is None:
        Q_init = 0.50 * np.eye(n_latent)

    model = EMVARXPSSMKnownC(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=max_iter,
        tol=tol,
        ridge_m_step=ridge_m_step,
        covariance_floor=covariance_floor,
        R_init=R_fixed,
        Q_init=Q_init,
        estimate_R=False,
        R_floor=R_floor,
        verbose=verbose
    )

    model.fit(y=y_obs, u=u)
    return model

# Compute latent complete-data posterior-statistic GC from a fitted EM model.
def latent_complete_data_gc_from_model(
        model,
        u,
        source,
        target,
        na,
        nb,
        gamma,
        penalty="diag",
        effective_t_mode="full_minus_features"
):
    smoother = model.smooth_result["smoother"]

    n_latent = model.C.shape[1]

    result = debiased_latent_varx_gc_from_smoother(
        smooth_mean=smoother["x_smooth"],
        smooth_cov=smoother["P_smooth"],
        smooth_lag_cov=smoother["P_lag_one"],
        u=u,
        source=source,
        target=target,
        na=na,
        nb=nb,
        n_latent=n_latent,
        conditioning=None,
        gamma=gamma,
        penalty=penalty,
        effective_t_mode=effective_t_mode
    )

    return result

# Construct directional null dynamics.
# For source -> target, remove: A_k[target, source] for every lag k.
def make_directional_null_A_matrices(
        A_matrices,
        source,
        target
):
    A_null = [np.asarray(A, dtype=float).copy() for A in A_matrices]
    for A in A_null:
        A[target, source] = 0.0

    return A_null

# Simulate a bootstrap dataset from the fitted directional null model.
# The exogenous input u is held fixed.
def simulate_from_directional_null_model(
        observed_model,
        observed_u,
        source,
        target,
        random_seed
):
    A_null = make_directional_null_A_matrices(
        A_matrices=observed_model.A_matrices,
        source=source,
        target=target
    )

    sim = generate_ssm_varx_p_data(
        A_matrices=A_null,
        B_matrices=observed_model.B_matrices,
        u=observed_u,
        Q=observed_model.Q,
        R=observed_model.R,
        C=observed_model.C,
        D=observed_model.D,
        burn_in=0,
        random_seed=random_seed,
        return_augmented=False
    )

    return sim["y"]

# Empirical null calibration for one tested direction.
def empirical_calibration_for_direction(
        observed_model,
        observed_u,
        source,
        target,
        na,
        nb,
        gamma,
        n_bootstrap,
        base_seed,
        em_max_iter=50,
        em_tol=1e-5,
        verbose_every=25
):
    observed_result = latent_complete_data_gc_from_model(
        model=observed_model,
        u=observed_u,
        source=source,
        target=target,
        na=na,
        nb=nb,
        gamma=gamma
    )

    observed_statistic = clipped_deviance_statistic(observed_result)

    null_statistics = []
    bootstrap_rows = []

    for b in range(n_bootstrap):
        y_boot = simulate_from_directional_null_model(
            observed_model=observed_model,
            observed_u=observed_u,
            source=source,
            target=target,
            random_seed=base_seed + b
        )

        boot_model = fit_em_known_c_fixed_r(
            y_obs=y_boot,
            u=observed_u,
            C=observed_model.C,
            R_fixed=observed_model.R,
            na=na,
            nb=nb,
            Q_init=observed_model.Q,
            max_iter=em_max_iter,
            tol=em_tol,
            verbose=False
        )

        boot_result = latent_complete_data_gc_from_model(
            model=boot_model,
            u=observed_u,
            source=source,
            target=target,
            na=na,
            nb=nb,
            gamma=gamma
        )

        boot_statistic = clipped_deviance_statistic(boot_result)
        null_statistics.append(boot_statistic)

        bootstrap_rows.append({
            "bootstrap_index": b,
            "bootstrap_statistic": boot_statistic,
            "bootstrap_debiased_deviance": boot_result["deviance_debiased"],
            "bootstrap_chi_square_p_value": boot_result["p_value"],
            "bootstrap_raw_gc": boot_result["gc_raw"],
            "bootstrap_spectral_radius": boot_model.spectral_radius(),
            "bootstrap_em_iterations": len(boot_model.log_likelihoods)
        })

        if verbose_every is not None and verbose_every > 0:
            if (b + 1) % verbose_every == 0:
                print(f"    bootstrap completed {b + 1}/{n_bootstrap}")

    null_statistics = np.asarray(null_statistics, dtype=float)

    empirical_p_value = empirical_upper_tail_p_value(
        observed_statistic=observed_statistic,
        null_statistics=null_statistics
    )

    summary = {
        "observed_raw_gc": observed_result["gc_raw"],
        "observed_debiased_deviance": observed_result["deviance_debiased"],
        "observed_statistic_clipped": observed_statistic,
        "chi_square_p_value": observed_result["p_value"],
        "empirical_p_value": empirical_p_value,
        "null_mean_statistic": float(np.mean(null_statistics)),
        "null_std_statistic": float(np.std(null_statistics)),
        "null_median_statistic": float(np.median(null_statistics)),
        "null_90_statistic": float(np.percentile(null_statistics, 90)),
        "null_95_statistic": float(np.percentile(null_statistics, 95)),
        "null_99_statistic": float(np.percentile(null_statistics, 99)),
        "n_bootstrap": n_bootstrap,
        "min_possible_empirical_p": 1.0 / (n_bootstrap + 1)
    }

    return summary, bootstrap_rows

def fit_em_known_c_estimated_r(
        y_obs,
        u,
        C,
        na,
        nb,
        R_init=None,
        Q_init=None,
        R_floor=0.30,
        max_iter=50,
        tol=1e-5,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        verbose=False
):
    C = np.asarray(C, dtype=float)

    n_latent = C.shape[1]
    n_obs = C.shape[0]

    if Q_init is None:
        Q_init = 0.50 * np.eye(n_latent)

    if R_init is None:
        R_init = R_floor * np.eye(n_obs)

    model = EMVARXPSSMKnownC(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=max_iter,
        tol=tol,
        ridge_m_step=ridge_m_step,
        covariance_floor=covariance_floor,
        R_init=R_init,
        Q_init=Q_init,
        estimate_R=True,
        R_floor=R_floor,
        verbose=verbose
    )

    model.fit(y=y_obs, u=u)
    return model

# Empirical null calibration for one tested direction when R is estimated.
# Bootstrap procedure:
#     1. Build fitted directional null model.
#     2. Simulate y* from the fitted null.
#     3. Refit EM with estimated R and the same R floor.
#     4. Recompute latent complete-data deviance.
#     5. Compute empirical upper-tail p-value.
def empirical_calibration_for_direction_estimated_r(
        observed_model,
        observed_u,
        source,
        target,
        na,
        nb,
        gamma,
        n_bootstrap,
        base_seed,
        R_floor=0.30,
        em_max_iter=50,
        em_tol=1e-5,
        verbose_every=25
):
    observed_result = latent_complete_data_gc_from_model(
        model=observed_model,
        u=observed_u,
        source=source,
        target=target,
        na=na,
        nb=nb,
        gamma=gamma
    )

    observed_statistic = clipped_deviance_statistic(observed_result)

    null_statistics = []
    bootstrap_rows = []

    for b in range(n_bootstrap):

        y_boot = simulate_from_directional_null_model(
            observed_model=observed_model,
            observed_u=observed_u,
            source=source,
            target=target,
            random_seed=base_seed + b
        )

        boot_model = fit_em_known_c_estimated_r(
            y_obs=y_boot,
            u=observed_u,
            C=observed_model.C,
            na=na,
            nb=nb,
            R_init=observed_model.R,
            Q_init=observed_model.Q,
            R_floor=R_floor,
            max_iter=em_max_iter,
            tol=em_tol,
            verbose=False
        )

        boot_result = latent_complete_data_gc_from_model(
            model=boot_model,
            u=observed_u,
            source=source,
            target=target,
            na=na,
            nb=nb,
            gamma=gamma
        )

        boot_statistic = clipped_deviance_statistic(boot_result)
        null_statistics.append(boot_statistic)
        bootstrap_rows.append({
            "bootstrap_index": b,
            "bootstrap_statistic": boot_statistic,
            "bootstrap_debiased_deviance": boot_result["deviance_debiased"],
            "bootstrap_chi_square_p_value": boot_result["p_value"],
            "bootstrap_raw_gc": boot_result["gc_raw"],
            "bootstrap_spectral_radius": boot_model.spectral_radius(),
            "bootstrap_em_iterations": len(boot_model.log_likelihoods),
            "bootstrap_R00": boot_model.R[0, 0],
            "bootstrap_R11": boot_model.R[1, 1],
            "bootstrap_R01": boot_model.R[0, 1]
        })

        if verbose_every is not None and verbose_every > 0:
            if (b + 1) % verbose_every == 0:
                print(f"    bootstrap completed {b + 1}/{n_bootstrap}")

    null_statistics = np.asarray(null_statistics, dtype=float)

    empirical_p_value = empirical_upper_tail_p_value(
        observed_statistic=observed_statistic,
        null_statistics=null_statistics
    )

    summary = {
        "observed_raw_gc": observed_result["gc_raw"],
        "observed_debiased_deviance": observed_result["deviance_debiased"],
        "observed_statistic_clipped": observed_statistic,
        "chi_square_p_value": observed_result["p_value"],
        "empirical_p_value": empirical_p_value,

        "null_mean_statistic": float(np.mean(null_statistics)),
        "null_std_statistic": float(np.std(null_statistics)),
        "null_median_statistic": float(np.median(null_statistics)),
        "null_90_statistic": float(np.percentile(null_statistics, 90)),
        "null_95_statistic": float(np.percentile(null_statistics, 95)),
        "null_99_statistic": float(np.percentile(null_statistics, 99)),

        "n_bootstrap": n_bootstrap,
        "min_possible_empirical_p": 1.0 / (n_bootstrap + 1)
    }

    return summary, bootstrap_rows