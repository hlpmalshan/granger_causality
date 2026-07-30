import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

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
# Experiment 32A
# Sample-size sensitivity for observed-likelihood GC.
# ------------------------------------------------------

# Since T=50 and T=1000 were already run, start with missing values.
# To rerun the full curve, use:
# T_VALUES = [20, 30, 50, 100, 250, 500, 1000]
T_VALUES = [20, 30, 100, 250, 500]

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

# Keep only the most useful settings.
SHRINKAGE_SPECS = [
    {
        "name": "none",
        "alpha_Q": 0.00,
        "alpha_R": 0.00,
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

CUSTOM_THRESHOLDS = [
    8.0,
    10.0,
    12.0,
    15.0
]


print("\nExperiment 32A: sample-size sensitivity")
print("----------------------------------------")
print("T values:", T_VALUES)
print("N outer runs:", N_OUTER_RUNS)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("shrinkage specs:", [s["name"] for s in SHRINKAGE_SPECS])
print("custom thresholds:", CUSTOM_THRESHOLDS)


def simulate_case(
        n_samples,
        burn_in,
        y_to_x_lag1,
        y_to_x_lag2,
        random_seed
):
    """
    Simulate latent VARX(2) state-space data.

    Direction convention:
        Y->X means source=1, target=0.
        X->Y means source=0, target=1.
    """

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
    """
    Same warm-start strategy as 31B/31C/31G.
    """

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


def compute_decision_rule_flags(
        deviance,
        n_samples,
        df
):
    """
    Compute several threshold-based decisions from the same deviance.
    """

    deviance_for_p = max(
        float(deviance),
        0.0
    )

    chi_square_p = float(
        chi2.sf(
            deviance_for_p,
            df=df
        )
    )

    bic_threshold = float(
        df * np.log(
            n_samples
        )
    )

    aic_threshold = float(
        2.0 * df
    )

    flags = {
        "chi_square_threshold": float(
            chi2.ppf(
                1.0 - ALPHA,
                df=df
            )
        ),
        "chi_square_p_value": chi_square_p,
        "chi_square_detected": chi_square_p < ALPHA,

        "aic_threshold": aic_threshold,
        "aic_detected": deviance_for_p > aic_threshold,

        "bic_threshold": bic_threshold,
        "bic_detected": deviance_for_p > bic_threshold
    }

    for threshold in CUSTOM_THRESHOLDS:

        key = f"D_gt_{str(threshold).replace('.', 'p')}"

        flags[f"{key}_threshold"] = threshold
        flags[f"{key}_detected"] = deviance_for_p > threshold

    return flags


def collect_direction_result(
        n_samples,
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

    deviance = float(
        obs_result["observed_likelihood_deviance"]
    )

    decision_flags = compute_decision_rule_flags(
        deviance=deviance,
        n_samples=n_samples,
        df=na
    )

    ll_gap = (
        obs_result["ll_full"]
        - obs_result["ll_reduced"]
    )

    row = {
        "n_samples": n_samples,

        "case": case_label,
        "outer_run": outer_run,

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

        "observed_deviance": deviance,
        "observed_deviance_for_p": obs_result["deviance_for_p"],

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

    row.update(
        decision_flags
    )

    return row


def run_one_sample_size(
        n_samples
):
    rows = []

    case_specs = [
        {
            "case_label": "no_link",
            "y_to_x_lag1": 0.00,
            "y_to_x_lag2": 0.00,
            "expected_y_to_x_significant": False,
            "base_seed": NO_LINK_BASE_SEED
        },
        {
            "case_label": "true_link",
            "y_to_x_lag1": 0.25,
            "y_to_x_lag2": 0.12,
            "expected_y_to_x_significant": True,
            "base_seed": TRUE_LINK_BASE_SEED
        }
    ]

    for case_spec in case_specs:

        case_label = case_spec["case_label"]

        for outer_run in range(N_OUTER_RUNS):

            print("\n" + "=" * 120)
            print(
                f"T={n_samples} | "
                f"case={case_label} | "
                f"outer run {outer_run + 1}/{N_OUTER_RUNS}"
            )
            print("=" * 120)

            seed = case_spec["base_seed"] + outer_run

            data = simulate_case(
                n_samples=n_samples,
                burn_in=BURN_IN,
                y_to_x_lag1=case_spec["y_to_x_lag1"],
                y_to_x_lag2=case_spec["y_to_x_lag2"],
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

                direction_specs = [
                    {
                        "direction": "Y->X",
                        "source": 1,
                        "target": 0,
                        "expected_significant": case_spec["expected_y_to_x_significant"]
                    },
                    {
                        "direction": "X->Y",
                        "source": 0,
                        "target": 1,
                        "expected_significant": False
                    }
                ]

                for direction_spec in direction_specs:

                    direction = direction_spec["direction"]

                    best_reduced = fitted["reduced_results"][direction]["best_reduced"]

                    row = collect_direction_result(
                        n_samples=n_samples,
                        case_label=case_label,
                        outer_run=outer_run,
                        shrinkage_spec=shrinkage_spec,
                        direction=direction,
                        source=direction_spec["source"],
                        target=direction_spec["target"],
                        expected_significant=direction_spec["expected_significant"],
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
                        f"chi_p={row['chi_square_p_value']:.4g}, "
                        f"chi_det={row['chi_square_detected']}, "
                        f"BIC_det={row['bic_detected']}, "
                        f"D>10={row['D_gt_10p0_detected']}"
                    )

            # Save partial progress.
            os.makedirs(
                "results",
                exist_ok=True
            )

            partial_path = (
                f"results/experiment_32a_sample_size_sensitivity_partial_T{n_samples}.csv"
            )

            pd.DataFrame(rows).to_csv(
                partial_path,
                index=False
            )

    return rows


def summarize_by_direction(
        results_df
):
    grouped = results_df.groupby([
        "n_samples",
        "shrinkage_name",
        "case",
        "direction",
        "expected_significant"
    ])

    summary_rows = []

    for keys, group in grouped:

        n_samples, shrinkage_name, case_label, direction, expected_significant = keys

        row = {
            "n_samples": n_samples,
            "shrinkage_name": shrinkage_name,
            "case": case_label,
            "direction": direction,
            "expected_significant": expected_significant,

            "n_tests": len(group),

            "mean_deviance": group["observed_deviance"].mean(),
            "std_deviance": group["observed_deviance"].std(),
            "median_deviance": group["observed_deviance"].median(),

            "median_chi_square_p_value": group["chi_square_p_value"].median(),

            "chi_square_detection_rate": group["chi_square_detected"].mean(),
            "aic_detection_rate": group["aic_detected"].mean(),
            "bic_detection_rate": group["bic_detected"].mean(),

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
        }

        for threshold in CUSTOM_THRESHOLDS:

            key = f"D_gt_{str(threshold).replace('.', 'p')}"
            row[f"{key}_detection_rate"] = group[f"{key}_detected"].mean()

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def compute_confusion_metrics(
        y_true,
        y_pred
):
    y_true = np.asarray(
        y_true,
        dtype=bool
    )

    y_pred = np.asarray(
        y_pred,
        dtype=bool
    )

    tp = int(
        np.sum(
            y_true & y_pred
        )
    )

    fp = int(
        np.sum(
            ~y_true & y_pred
        )
    )

    tn = int(
        np.sum(
            ~y_true & ~y_pred
        )
    )

    fn = int(
        np.sum(
            y_true & ~y_pred
        )
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if np.isfinite(precision) and np.isfinite(tpr) and (precision + tpr) > 0
        else np.nan
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "f1": f1
    }


def summarize_by_decision_rule(
        results_df
):
    """
    Aggregate all null directions together and all true directions together.

    Null directions are:
        no_link X->Y
        no_link Y->X
        true_link X->Y

    True direction is:
        true_link Y->X
    """

    decision_columns = [
        "chi_square_detected",
        "aic_detected",
        "bic_detected"
    ]

    for threshold in CUSTOM_THRESHOLDS:
        key = f"D_gt_{str(threshold).replace('.', 'p')}_detected"
        decision_columns.append(
            key
        )

    grouped = results_df.groupby([
        "n_samples",
        "shrinkage_name"
    ])

    summary_rows = []

    for keys, group in grouped:

        n_samples, shrinkage_name = keys

        for decision_column in decision_columns:

            metrics = compute_confusion_metrics(
                y_true=group["expected_significant"].values,
                y_pred=group[decision_column].values
            )

            row = {
                "n_samples": n_samples,
                "shrinkage_name": shrinkage_name,
                "decision_rule": decision_column,
                "n_tests": len(group)
            }

            row.update(
                metrics
            )

            summary_rows.append(
                row
            )

    return pd.DataFrame(
        summary_rows
    )


# ------------------------------------------------------
# Main run
# ------------------------------------------------------

all_rows = []

for n_samples in T_VALUES:

    rows_this_T = run_one_sample_size(
        n_samples=n_samples
    )

    all_rows.extend(
        rows_this_T
    )

    current_df = pd.DataFrame(
        all_rows
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    current_df.to_csv(
        "results/experiment_32a_sample_size_sensitivity_results_partial.csv",
        index=False
    )


results_df = pd.DataFrame(
    all_rows
)

direction_summary_df = summarize_by_direction(
    results_df
)

decision_summary_df = summarize_by_decision_rule(
    results_df
)


# ------------------------------------------------------
# Print summaries
# ------------------------------------------------------

print("\n" + "=" * 160)
print("Experiment 32A direction-wise summary")
print("=" * 160)

direction_columns = [
    "n_samples",
    "shrinkage_name",
    "case",
    "direction",
    "expected_significant",
    "mean_deviance",
    "median_chi_square_p_value",
    "chi_square_detection_rate",
    "bic_detection_rate",
    "D_gt_10p0_detection_rate",
    "negative_ll_gap_fraction",
    "mean_full_mse",
    "mean_full_spectral_radius",
    "mean_full_Q01",
    "mean_full_R01",
    "mean_full_A1_01",
    "mean_full_A2_01",
    "mean_full_A1_10",
    "mean_full_A2_10"
]

print(
    direction_summary_df[direction_columns].to_string(
        index=False
    )
)


print("\n" + "=" * 160)
print("Experiment 32A decision-rule summary")
print("=" * 160)

decision_columns = [
    "n_samples",
    "shrinkage_name",
    "decision_rule",
    "fpr",
    "tpr",
    "precision",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn"
]

print(
    decision_summary_df[decision_columns].to_string(
        index=False
    )
)


# ------------------------------------------------------
# Save final outputs
# ------------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

results_path = "results/experiment_32a_sample_size_sensitivity_results.csv"
direction_summary_path = "results/experiment_32a_sample_size_sensitivity_direction_summary.csv"
decision_summary_path = "results/experiment_32a_sample_size_sensitivity_decision_summary.csv"

results_df.to_csv(
    results_path,
    index=False
)

direction_summary_df.to_csv(
    direction_summary_path,
    index=False
)

decision_summary_df.to_csv(
    decision_summary_path,
    index=False
)

print("\nSaved detailed results to:")
print(results_path)

print("\nSaved direction-wise summary to:")
print(direction_summary_path)

print("\nSaved decision-rule summary to:")
print(decision_summary_path)


print("\nInterpretation guide")
print("--------------------")
print("This experiment estimates the minimum useful sample size T.")
print("The direction-wise summary shows each case/direction separately.")
print("The decision-rule summary aggregates all null directions and the true Y->X direction.")
print("A usable regime should have low FPR and nontrivial TPR.")
print("Compare none vs QR_0p10 vs QR_0p20.")
print("Compare chi-square vs BIC vs D>10.")