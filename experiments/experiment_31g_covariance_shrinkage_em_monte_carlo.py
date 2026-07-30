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
    extract_model_parameters
)

from src.ssm.em_varx_p_known_c_shrinkage import (
    EMVARXPSSMKnownCConstrainedWarmStartShrinkage
)

from src.stats.observed_likelihood_deviance import (
    observed_likelihood_deviance,
    final_observed_log_likelihood
)


# ------------------------------------------------------
# Experiment 31G
# Covariance shrinkage inside EM for observed-likelihood GC.
# ------------------------------------------------------

# Stage 1:
# Keep 1000 first to isolate the effect of covariance shrinkage.
N_SAMPLES = 50

# After Stage 1, set:
# N_SAMPLES = 50
# and increase N_OUTER_RUNS.
N_OUTER_RUNS = 50                                                                                      

BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

NO_LINK_BASE_SEED = 1800000
TRUE_LINK_BASE_SEED = 1900000

SHRINKAGE_SPECS = [
    {
        "name": "none",
        "alpha_Q": 0.00,
        "alpha_R": 0.00,
        "target_Q": "spherical",
        "target_R": "spherical"
    },
    {
        "name": "R_0p10",
        "alpha_Q": 0.00,
        "alpha_R": 0.10,
        "target_Q": "spherical",
        "target_R": "spherical"
    },
    {
        "name": "QR_0p10",
        "alpha_Q": 0.10,
        "alpha_R": 0.10,
        "target_Q": "spherical",
        "target_R": "spherical"
    },
    {
        "name": "QR_0p20",
        "alpha_Q": 0.20,
        "alpha_R": 0.20,
        "target_Q": "spherical",
        "target_R": "spherical"
    }
]

print("\nExperiment 31G: covariance shrinkage inside EM")
print("------------------------------------------------")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("shrinkage specs:", [s["name"] for s in SHRINKAGE_SPECS])


def simulate_case(
        n_samples,
        burn_in,
        y_to_x_lag1,
        y_to_x_lag2,
        random_seed
):
    u_total = generate_colored_input(
        n_samples=n_samples + burn_in,
        ar_coeff=0.95,
        noise_std=1.0,
        random_seed=random_seed
    )

    A1 = np.array([
        [0.55, y_to_x_lag1],
        [0.00, 0.50]
    ])

    A2 = np.array([
        [-0.10, y_to_x_lag2],
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
        burn_in=burn_in,
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
        shrinkage_spec,
        zero_constraints=None,
        initial_parameters=None,
        label="model",
        verbose=False
):
    if zero_constraints is None:
        zero_constraints = []

    model = EMVARXPSSMKnownCConstrainedWarmStartShrinkage(
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
        verbose=verbose,
        alpha_Q=shrinkage_spec["alpha_Q"],
        alpha_R=shrinkage_spec["alpha_R"],
        shrinkage_target_Q=shrinkage_spec["target_Q"],
        shrinkage_target_R=shrinkage_spec["target_R"]
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
    return max(
        models,
        key=model_ll
    )


def fit_warmstarted_case_models(
        y_obs,
        u,
        C,
        shrinkage_spec
):
    full_candidates = []

    full_proxy = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        shrinkage_spec=shrinkage_spec,
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
            shrinkage_spec=shrinkage_spec,
            zero_constraints=zero_constraints,
            initial_parameters=None,
            label=f"reduced_{direction}_proxy",
            verbose=False
        )

        reduced_from_full = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            shrinkage_spec=shrinkage_spec,
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
            shrinkage_spec=shrinkage_spec,
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
            "best_reduced": best_reduced,
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
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


def collect_direction_result(
        case_label,
        outer_run,
        shrinkage_spec,
        direction,
        source,
        target,
        expected_significant,
        best_full,
        best_reduced,
        full_mse
):
    obs_result = observed_likelihood_deviance(
        full_model=best_full,
        reduced_model=best_reduced,
        df_removed=na,
        clip_negative=True
    )

    ll_gap = (
        obs_result["ll_full"]
        - obs_result["ll_reduced"]
    )

    return {
        "case": case_label,
        "outer_run": outer_run,
        "n_samples": N_SAMPLES,

        "shrinkage_name": shrinkage_spec["name"],
        "alpha_Q": shrinkage_spec["alpha_Q"],
        "alpha_R": shrinkage_spec["alpha_R"],
        "target_Q": shrinkage_spec["target_Q"],
        "target_R": shrinkage_spec["target_R"],

        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": expected_significant,

        "ll_full": obs_result["ll_full"],
        "ll_reduced": obs_result["ll_reduced"],
        "ll_gap_full_minus_reduced": ll_gap,
        "negative_ll_gap": ll_gap < -1e-8,

        "observed_deviance": obs_result["observed_likelihood_deviance"],
        "observed_deviance_for_p": obs_result["deviance_for_p"],
        "p_value": obs_result["p_value"],
        "detected": obs_result["p_value"] < ALPHA,

        "full_model_label": best_full.fit_label,
        "reduced_model_label": best_reduced.fit_label,

        "full_mse": full_mse,
        "full_spectral_radius": best_full.spectral_radius(),
        "reduced_spectral_radius": best_reduced.spectral_radius(),

        "full_em_iterations": len(best_full.log_likelihoods),
        "reduced_em_iterations": len(best_reduced.log_likelihoods),

        "full_Q00": best_full.Q[0, 0],
        "full_Q11": best_full.Q[1, 1],
        "full_Q01": best_full.Q[0, 1],

        "full_R00": best_full.R[0, 0],
        "full_R11": best_full.R[1, 1],
        "full_R01": best_full.R[0, 1],

        "full_A1_01": best_full.A_matrices[0][0, 1],
        "full_A2_01": best_full.A_matrices[1][0, 1],
        "full_A1_10": best_full.A_matrices[0][1, 0],
        "full_A2_10": best_full.A_matrices[1][1, 0]
    }


def run_outer_case(
        case_label,
        y_to_x_lag1,
        y_to_x_lag2,
        expected_y_to_x_significant,
        base_seed
):
    rows = []

    for outer_run in range(N_OUTER_RUNS):

        print("\n" + "=" * 120)
        print(f"Case={case_label} | outer run {outer_run + 1}/{N_OUTER_RUNS}")
        print("=" * 120)

        seed = base_seed + outer_run

        data = simulate_case(
            n_samples=N_SAMPLES,
            burn_in=BURN_IN,
            y_to_x_lag1=y_to_x_lag1,
            y_to_x_lag2=y_to_x_lag2,
            random_seed=seed
        )

        x_true = data["x_true"]
        y_obs = data["y_obs"]
        u = data["u"]
        C = data["C"]

        for shrinkage_spec in SHRINKAGE_SPECS:

            print("\nShrinkage:", shrinkage_spec["name"])

            fitted = fit_warmstarted_case_models(
                y_obs=y_obs,
                u=u,
                C=C,
                shrinkage_spec=shrinkage_spec
            )

            best_full = fitted["best_full"]

            x_full_smooth = best_full.smoothed_latent_state()

            full_mse = float(
                np.mean(
                    (x_full_smooth - x_true) ** 2
                )
            )

            print("Best full label:", best_full.fit_label)
            print("Best full LL:", model_ll(best_full))
            print("Full MSE:", full_mse)
            print("Full spectral radius:", best_full.spectral_radius())
            print("Full Q diag:", best_full.Q[0, 0], best_full.Q[1, 1])
            print("Full R diag:", best_full.R[0, 0], best_full.R[1, 1])

            direction_specs = [
                {
                    "direction": "Y->X",
                    "source": 1,
                    "target": 0,
                    "expected_significant": expected_y_to_x_significant
                },
                {
                    "direction": "X->Y",
                    "source": 0,
                    "target": 1,
                    "expected_significant": False
                }
            ]

            for spec in direction_specs:

                direction = spec["direction"]

                best_reduced = fitted["reduced_results"][direction]["best_reduced"]

                row = collect_direction_result(
                    case_label=case_label,
                    outer_run=outer_run,
                    shrinkage_spec=shrinkage_spec,
                    direction=direction,
                    source=spec["source"],
                    target=spec["target"],
                    expected_significant=spec["expected_significant"],
                    best_full=best_full,
                    best_reduced=best_reduced,
                    full_mse=full_mse
                )

                rows.append(
                    row
                )

                print(
                    f"{direction}: "
                    f"D={row['observed_deviance']:.4f}, "
                    f"p={row['p_value']:.4g}, "
                    f"detected={row['detected']}, "
                    f"LL gap={row['ll_gap_full_minus_reduced']:.4f}"
                )

        os.makedirs(
            "results",
            exist_ok=True
        )

        pd.DataFrame(rows).to_csv(
            f"results/experiment_31g_{case_label}_partial_{N_OUTER_RUNS}runs.csv",
            index=False
        )

    return rows


def summarize_results(
        results_df
):
    summary_rows = []

    grouped = results_df.groupby([
        "n_samples",
        "shrinkage_name",
        "case",
        "direction",
        "expected_significant"
    ])

    for keys, group in grouped:

        n_samples, shrinkage_name, case_label, direction, expected_significant = keys

        summary_rows.append({
            "n_samples": n_samples,
            "shrinkage_name": shrinkage_name,
            "case": case_label,
            "direction": direction,
            "expected_significant": expected_significant,

            "mean_deviance": group["observed_deviance"].mean(),
            "std_deviance": group["observed_deviance"].std(),
            "median_p_value": group["p_value"].median(),
            "detection_rate": group["detected"].mean(),

            "negative_ll_gap_fraction": group["negative_ll_gap"].mean(),
            "min_ll_gap": group["ll_gap_full_minus_reduced"].min(),
            "mean_ll_gap": group["ll_gap_full_minus_reduced"].mean(),

            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_reduced_spectral_radius": group["reduced_spectral_radius"].mean(),

            "mean_full_em_iterations": group["full_em_iterations"].mean(),
            "mean_reduced_em_iterations": group["reduced_em_iterations"].mean(),

            "mean_full_Q00": group["full_Q00"].mean(),
            "mean_full_Q11": group["full_Q11"].mean(),
            "mean_full_Q01": group["full_Q01"].mean(),

            "mean_full_R00": group["full_R00"].mean(),
            "mean_full_R11": group["full_R11"].mean(),
            "mean_full_R01": group["full_R01"].mean(),

            "mean_full_A1_01": group["full_A1_01"].mean(),
            "mean_full_A2_01": group["full_A2_01"].mean(),
            "mean_full_A1_10": group["full_A1_10"].mean(),
            "mean_full_A2_10": group["full_A2_10"].mean()
        })

    return pd.DataFrame(
        summary_rows
    )


# ------------------------------------------------------
# Main run
# ------------------------------------------------------

all_rows = []

no_link_rows = run_outer_case(
    case_label="no_link",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=NO_LINK_BASE_SEED
)

all_rows.extend(
    no_link_rows
)

true_link_rows = run_outer_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=TRUE_LINK_BASE_SEED
)

all_rows.extend(
    true_link_rows
)

results_df = pd.DataFrame(
    all_rows
)

summary_df = summarize_results(
    results_df
)


# ------------------------------------------------------
# Print summary
# ------------------------------------------------------

print("\n" + "=" * 150)
print("Experiment 31G covariance-shrinkage EM summary")
print("=" * 150)

summary_columns = [
    "n_samples",
    "shrinkage_name",
    "case",
    "direction",
    "expected_significant",
    "mean_deviance",
    "median_p_value",
    "detection_rate",
    "negative_ll_gap_fraction",
    "min_ll_gap",
    "mean_full_mse",
    "mean_full_spectral_radius",
    "mean_full_Q00",
    "mean_full_Q11",
    "mean_full_Q01",
    "mean_full_R00",
    "mean_full_R11",
    "mean_full_R01",
    "mean_full_A1_01",
    "mean_full_A2_01",
    "mean_full_A1_10",
    "mean_full_A2_10"
]

print(
    summary_df[summary_columns].to_string(
        index=False
    )
)


# ------------------------------------------------------
# Save results
# ------------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

results_path = f"results/experiment_31g_covariance_shrinkage_em_results_T{N_SAMPLES}_{N_OUTER_RUNS}runs.csv"
summary_path = f"results/experiment_31g_covariance_shrinkage_em_summary_T{N_SAMPLES}_{N_OUTER_RUNS}runs.csv"

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved detailed results to:")
print(results_path)

print("\nSaved summary to:")
print(summary_path)


print("\nInterpretation guide")
print("--------------------")
print("Compare shrinkage_name='none' against R_0p10, QR_0p10, and QR_0p20.")
print("A good shrinkage setting should reduce null false positives without hurting true Y->X power.")
print("Also check Q/R off-diagonal shrinkage, MSE, spectral radius, and negative_ll_gap_fraction.")