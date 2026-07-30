import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import (
    generate_colored_input
)

from src.ssm.ssm_varx_p_simulator import (
    generate_ssm_varx_p_data
)

from src.ssm.em_varx_p_known_c_warmstart import (
    EMVARXPSSMKnownCConstrainedWarmStart,
    extract_model_parameters
)

from src.stats.observed_likelihood_deviance import (
    observed_likelihood_deviance,
    final_observed_log_likelihood
)

from src.stats.empirical_pvalue import (
    empirical_upper_tail_p_value
)


# ------------------------------------------------------
# Experiment 31E revised pilot
# Empirical calibration of observed-likelihood deviance
# on problematic null false-positive runs only.
# ------------------------------------------------------

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

N_BOOTSTRAP = 100

# Must match the no_link base seed used in your 31D/31E diagnostic runs.
NO_LINK_BASE_SEED = 1800000

PROBLEMATIC_TESTS = [
    {"outer_run": 0, "direction": "Y->X", "source": 1, "target": 0},
    {"outer_run": 1, "direction": "Y->X", "source": 1, "target": 0},
    {"outer_run": 6, "direction": "Y->X", "source": 1, "target": 0},
    {"outer_run": 10, "direction": "Y->X", "source": 1, "target": 0},
    {"outer_run": 11, "direction": "Y->X", "source": 1, "target": 0},
    {"outer_run": 7, "direction": "X->Y", "source": 0, "target": 1},
]

print("\nExperiment 31E revised pilot")
print("----------------------------")
print("Observed-likelihood empirical calibration on problematic null runs")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("N bootstrap:", N_BOOTSTRAP)
print("Minimum possible empirical p:", 1.0 / (N_BOOTSTRAP + 1))


def simulate_no_link_case(
        outer_run
):
    """
    Recreate the exact no-link dataset for a given outer_run.

    This assumes the original no_link base seed was NO_LINK_BASE_SEED.
    """

    random_seed = NO_LINK_BASE_SEED + outer_run

    u_total = generate_colored_input(
        n_samples=N_SAMPLES + BURN_IN,
        ar_coeff=0.95,
        noise_std=1.0,
        random_seed=random_seed
    )

    A1 = np.array([
        [0.55, 0.00],
        [0.00, 0.50]
    ])

    A2 = np.array([
        [-0.10, 0.00],
        [0.00, -0.08]
    ])

    B0 = np.array([
        [0.00],
        [0.90]
    ])

    B1 = np.array([
        [0.00],
        [0.00]
    ])

    B2 = np.array([
        [0.90],
        [0.00]
    ])

    Q = 0.50 * np.eye(2)
    R = 0.60 * np.eye(2)

    C = np.array([
        [1.00, 0.35],
        [0.25, 1.00]
    ])

    sim = generate_ssm_varx_p_data(
        A_matrices=[A1, A2],
        B_matrices=[B0, B1, B2],
        u=u_total,
        Q=Q,
        R=R,
        C=C,
        D=None,
        burn_in=BURN_IN,
        random_seed=random_seed + 10000,
        return_augmented=True
    )

    return {
        "x_true": sim["x"],
        "y_obs": sim["y"],
        "u": sim["u"],
        "C": C,
        "Q_true": Q,
        "R_true": R
    }


def make_directional_zero_constraint(
        source,
        target
):
    return [
        {
            "source": source,
            "target": target,
            "lags": "all"
        }
    ]


def fit_em_model(
        y_obs,
        u,
        C,
        zero_constraints=None,
        initial_parameters=None,
        label="model",
        verbose=False
):
    if zero_constraints is None:
        zero_constraints = []

    model = EMVARXPSSMKnownCConstrainedWarmStart(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=MAX_ITER,
        tol=TOL,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_init=0.60 * np.eye(2),
        Q_init=0.50 * np.eye(2),
        estimate_R=True,
        R_floor=R_FLOOR,
        zero_constraints=zero_constraints,
        initial_parameters=initial_parameters,
        jitter_scale=0.0,
        random_seed=None,
        verbose=verbose
    )

    model.fit(
        y=y_obs,
        u=u
    )

    model.fit_label = label

    return model


def model_ll(
        model
):
    return final_observed_log_likelihood(
        model
    )


def choose_best_model(
        models
):
    if len(models) < 1:
        raise ValueError(
            "models must contain at least one fitted model."
        )

    return max(
        models,
        key=model_ll
    )


def fit_warmstarted_case_models(
        y_obs,
        u,
        C
):
    """
    Same warm-start strategy as 31C.

    This fits both reduced directions so that the best full model is selected
    in the same way as in the previous observed-likelihood experiments.
    """

    full_candidates = []

    full_proxy = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        zero_constraints=[],
        initial_parameters=None,
        label="full_proxy",
        verbose=False
    )

    full_candidates.append(
        full_proxy
    )

    direction_specs = [
        {
            "direction": "Y->X",
            "source": 1,
            "target": 0
        },
        {
            "direction": "X->Y",
            "source": 0,
            "target": 1
        }
    ]

    reduced_results = {}

    for spec in direction_specs:

        direction = spec["direction"]
        source = spec["source"]
        target = spec["target"]

        zero_constraints = make_directional_zero_constraint(
            source=source,
            target=target
        )

        reduced_proxy = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=zero_constraints,
            initial_parameters=None,
            label=f"reduced_{direction}_proxy",
            verbose=False
        )

        reduced_from_full = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=zero_constraints,
            initial_parameters=extract_model_parameters(
                full_proxy
            ),
            label=f"reduced_{direction}_from_full_proxy",
            verbose=False
        )

        best_reduced = choose_best_model([
            reduced_proxy,
            reduced_from_full
        ])

        full_from_reduced = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=[],
            initial_parameters=extract_model_parameters(
                best_reduced
            ),
            label=f"full_from_best_reduced_{direction}",
            verbose=False
        )

        full_candidates.append(
            full_from_reduced
        )

        reduced_results[direction] = {
            "source": source,
            "target": target,
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
            "best_reduced": best_reduced,
            "full_from_reduced": full_from_reduced
        }

    best_full = choose_best_model(
        full_candidates
    )

    return {
        "best_full": best_full,
        "full_candidates": full_candidates,
        "reduced_results": reduced_results
    }


def observed_likelihood_statistic_for_direction(
        fitted,
        direction
):
    """
    Compute raw observed-likelihood deviance for one direction.
    """

    best_full = fitted["best_full"]
    best_reduced = fitted["reduced_results"][direction]["best_reduced"]

    result = observed_likelihood_deviance(
        full_model=best_full,
        reduced_model=best_reduced,
        df_removed=na,
        clip_negative=True
    )

    statistic = float(
        result["deviance_for_p"]
    )

    return result, statistic


def simulate_from_reduced_null_model(
        reduced_model,
        u,
        random_seed
):
    """
    Simulate bootstrap data from the fitted reduced model.

    The exogenous input u is held fixed.
    """

    sim = generate_ssm_varx_p_data(
        A_matrices=reduced_model.A_matrices,
        B_matrices=reduced_model.B_matrices,
        u=u,
        Q=reduced_model.Q,
        R=reduced_model.R,
        C=reduced_model.C,
        D=reduced_model.D,
        burn_in=0,
        random_seed=random_seed,
        return_augmented=False
    )

    return sim["y"]


def empirical_calibration_for_problematic_test(
        outer_run,
        direction,
        source,
        target,
        bootstrap_base_seed
):
    """
    Recreate one problematic null dataset and empirically calibrate
    the observed-likelihood deviance for the requested direction.
    """

    data = simulate_no_link_case(
        outer_run=outer_run
    )

    y_obs = data["y_obs"]
    u = data["u"]
    C = data["C"]
    x_true = data["x_true"]

    print("\n" + "=" * 100)
    print(f"Problematic no_link test | outer_run={outer_run} | direction={direction}")
    print("=" * 100)

    observed_fitted = fit_warmstarted_case_models(
        y_obs=y_obs,
        u=u,
        C=C
    )

    observed_result, observed_statistic = observed_likelihood_statistic_for_direction(
        fitted=observed_fitted,
        direction=direction
    )

    observed_full = observed_fitted["best_full"]
    observed_reduced = observed_fitted["reduced_results"][direction]["best_reduced"]

    x_smooth = observed_full.smoothed_latent_state()

    observed_mse = float(
        np.mean(
            (x_smooth - x_true) ** 2
        )
    )

    print("Observed best full label:", observed_full.fit_label)
    print("Observed reduced label:", observed_reduced.fit_label)
    print("Observed LL full:", observed_result["ll_full"])
    print("Observed LL reduced:", observed_result["ll_reduced"])
    print("Observed raw deviance:", observed_result["observed_likelihood_deviance"])
    print("Observed statistic clipped:", observed_statistic)
    print("Observed chi-square p:", observed_result["p_value"])
    print("Observed chi-square detected:", observed_result["p_value"] < ALPHA)
    print("Observed full MSE:", observed_mse)
    print("Observed full R diag:", observed_full.R[0, 0], observed_full.R[1, 1])

    null_statistics = []
    bootstrap_rows = []

    for b in range(N_BOOTSTRAP):

        y_boot = simulate_from_reduced_null_model(
            reduced_model=observed_reduced,
            u=u,
            random_seed=bootstrap_base_seed + b
        )

        boot_fitted = fit_warmstarted_case_models(
            y_obs=y_boot,
            u=u,
            C=C
        )

        boot_result, boot_statistic = observed_likelihood_statistic_for_direction(
            fitted=boot_fitted,
            direction=direction
        )

        null_statistics.append(
            boot_statistic
        )

        boot_full = boot_fitted["best_full"]
        boot_reduced = boot_fitted["reduced_results"][direction]["best_reduced"]

        bootstrap_rows.append({
            "outer_run": outer_run,
            "direction": direction,
            "source": source,
            "target": target,
            "bootstrap_index": b,
            "bootstrap_statistic": boot_statistic,
            "bootstrap_raw_deviance": boot_result["observed_likelihood_deviance"],
            "bootstrap_chi_square_p": boot_result["p_value"],
            "bootstrap_chi_square_detected": boot_result["p_value"] < ALPHA,
            "bootstrap_ll_full": boot_result["ll_full"],
            "bootstrap_ll_reduced": boot_result["ll_reduced"],
            "bootstrap_ll_gap": boot_result["ll_full"] - boot_result["ll_reduced"],
            "bootstrap_full_label": boot_full.fit_label,
            "bootstrap_reduced_label": boot_reduced.fit_label,
            "bootstrap_full_spectral_radius": boot_full.spectral_radius(),
            "bootstrap_full_R00": boot_full.R[0, 0],
            "bootstrap_full_R11": boot_full.R[1, 1],
            "bootstrap_full_R01": boot_full.R[0, 1]
        })

        if (b + 1) % 10 == 0:
            print(
                f"  bootstrap completed {b + 1}/{N_BOOTSTRAP}"
            )

    null_statistics = np.asarray(
        null_statistics,
        dtype=float
    )

    empirical_p = empirical_upper_tail_p_value(
        observed_statistic=observed_statistic,
        null_statistics=null_statistics
    )

    summary = {
        "case": "no_link",
        "outer_run": outer_run,
        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": False,

        "observed_ll_full": observed_result["ll_full"],
        "observed_ll_reduced": observed_result["ll_reduced"],
        "observed_ll_gap": observed_result["ll_full"] - observed_result["ll_reduced"],

        "observed_raw_deviance": observed_result["observed_likelihood_deviance"],
        "observed_statistic_clipped": observed_statistic,

        "chi_square_p_value": observed_result["p_value"],
        "chi_square_detected": observed_result["p_value"] < ALPHA,

        "empirical_p_value": empirical_p,
        "empirical_detected": empirical_p < ALPHA,

        "null_mean_statistic": float(np.mean(null_statistics)),
        "null_std_statistic": float(np.std(null_statistics)),
        "null_median_statistic": float(np.median(null_statistics)),
        "null_90_statistic": float(np.percentile(null_statistics, 90)),
        "null_95_statistic": float(np.percentile(null_statistics, 95)),
        "null_99_statistic": float(np.percentile(null_statistics, 99)),

        "n_bootstrap": N_BOOTSTRAP,
        "min_possible_empirical_p": 1.0 / (N_BOOTSTRAP + 1),

        "observed_full_model_label": observed_full.fit_label,
        "observed_reduced_model_label": observed_reduced.fit_label,

        "observed_full_mse": observed_mse,
        "observed_full_spectral_radius": observed_full.spectral_radius(),
        "observed_full_em_iterations": len(observed_full.log_likelihoods),
        "observed_reduced_spectral_radius": observed_reduced.spectral_radius(),
        "observed_reduced_em_iterations": len(observed_reduced.log_likelihoods),

        "observed_full_R00": observed_full.R[0, 0],
        "observed_full_R11": observed_full.R[1, 1],
        "observed_full_R01": observed_full.R[0, 1]
    }

    print("\nResult")
    print("------")
    print("outer_run:", outer_run)
    print("direction:", direction)
    print("observed statistic:", observed_statistic)
    print("chi-square p:", summary["chi_square_p_value"])
    print("empirical p:", summary["empirical_p_value"])
    print("chi-square detected:", summary["chi_square_detected"])
    print("empirical detected:", summary["empirical_detected"])
    print("null 95% statistic:", summary["null_95_statistic"])

    return summary, bootstrap_rows


# ------------------------------------------------------
# Main run
# ------------------------------------------------------

summary_rows = []
bootstrap_rows_all = []

for test_index, test in enumerate(PROBLEMATIC_TESTS):

    summary, bootstrap_rows = empirical_calibration_for_problematic_test(
        outer_run=test["outer_run"],
        direction=test["direction"],
        source=test["source"],
        target=test["target"],
        bootstrap_base_seed=2000000 + 100000 * test_index
    )

    summary_rows.append(
        summary
    )

    bootstrap_rows_all.extend(
        bootstrap_rows
    )

    # Save after each problematic test so progress is not lost.
    os.makedirs(
        "data",
        exist_ok=True
    )

    pd.DataFrame(summary_rows).to_csv(
        "data/experiment_31e_revised_empirical_pilot_summary_partial.csv",
        index=False
    )

    pd.DataFrame(bootstrap_rows_all).to_csv(
        "data/experiment_31e_revised_empirical_pilot_bootstrap_partial.csv",
        index=False
    )


summary_df = pd.DataFrame(
    summary_rows
)

bootstrap_df = pd.DataFrame(
    bootstrap_rows_all
)


# ------------------------------------------------------
# Save final results
# ------------------------------------------------------

summary_path = "data/experiment_31e_revised_empirical_pilot_summary.csv"
bootstrap_path = "data/experiment_31e_revised_empirical_pilot_bootstrap.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)


# ------------------------------------------------------
# Print compact summary
# ------------------------------------------------------

print("\n" + "=" * 120)
print("Experiment 31E revised empirical pilot summary")
print("=" * 120)

columns_to_show = [
    "case",
    "outer_run",
    "direction",
    "observed_raw_deviance",
    "chi_square_p_value",
    "empirical_p_value",
    "chi_square_detected",
    "empirical_detected",
    "null_mean_statistic",
    "null_95_statistic",
    "null_99_statistic",
    "observed_full_mse",
    "observed_full_spectral_radius",
    "observed_full_R00",
    "observed_full_R11"
]

print(
    summary_df[columns_to_show].to_string(
        index=False
    )
)

print("\nSaved summary to:")
print(summary_path)

print("\nSaved bootstrap details to:")
print(bootstrap_path)


# ------------------------------------------------------
# Interpretation guide
# ------------------------------------------------------

print("\n" + "=" * 120)
print("Interpretation guide")
print("=" * 120)

print(
    "\nThese are all null directions that were problematic under the chi-square reference."
)

print(
    "\nSuccess criterion:"
    "\n    chi_square_detected may be True,"
    "\n    but empirical_detected should become False."
)

print(
    "\nIf empirical p-values remove most or all of these false positives, "
    "then observed-likelihood deviance also needs empirical calibration."
)

print(
    "\nIf some empirical false positives remain, inspect whether observed_raw_deviance "
    "is above the bootstrap null_95_statistic or whether EM produced unstable fits."
)