import os
import numpy as np
import pandas as pd

from scipy.stats import chi2
from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c_warmstart import EMVARXPSSMKnownCConstrainedWarmStart, extract_model_parameters
from src.stats.observed_likelihood_deviance import observed_likelihood_deviance, final_observed_log_likelihood
from src.stats.observed_debiased_deviance import finite_difference_score_information, observed_debiased_deviance_from_precomputed_information

N_SAMPLES = 1000
BURN_IN = 300

N_OUTER_RUNS = 20

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

PARAMETER_BLOCKS = ("A", "B", "Q", "R")

FD_EPSILON = 3e-4
RELATIVE_INFO_FLOOR = 1e-8

# Same seeds as the failed observed de-biased deviance run.
NO_LINK_BASE_SEED = 1800000
TRUE_LINK_BASE_SEED = 1900000

INVERSE_METHOD_SPECS = [
    {
        "name": "solve",
        "method": "solve",
        "kwargs": {}
    },
    {
        "name": "pinv_1e-6",
        "method": "pinv",
        "kwargs": {"pinv_rcond": 1e-6}
    },
    {
        "name": "pinv_1e-4",
        "method": "pinv",
        "kwargs": {"pinv_rcond": 1e-4}
    },
    {
        "name": "pinv_psd_1e-6",
        "method": "pinv_psd",
        "kwargs": {"pinv_rcond": 1e-6}
    },
    {
        "name": "pinv_psd_1e-4",
        "method": "pinv_psd",
        "kwargs": {"pinv_rcond": 1e-4}
    },
    {
        "name": "eig_clip_1e-6",
        "method": "eig_clip",
        "kwargs": {"eigen_floor_rel": 1e-6}
    },
    {
        "name": "eig_clip_1e-4",
        "method": "eig_clip",
        "kwargs": {"eigen_floor_rel": 1e-4}
    },
    {
        "name": "ridge_1e-4",
        "method": "ridge",
        "kwargs": {"ridge_rel": 1e-4, "eigen_floor_rel": 1e-8}
    },
    {
        "name": "ridge_1e-3",
        "method": "ridge",
        "kwargs": {"ridge_rel": 1e-3, "eigen_floor_rel": 1e-8}
    },
    {
        "name": "shrink_0p05",
        "method": "shrinkage",
        "kwargs": {"shrinkage_alpha": 0.05, "eigen_floor_rel": 1e-8}
    },
    {
        "name": "shrink_0p10",
        "method": "shrinkage",
        "kwargs": {"shrinkage_alpha": 0.10, "eigen_floor_rel": 1e-8}
    },
    {
        "name": "shrink_0p20",
        "method": "shrinkage",
        "kwargs": {"shrinkage_alpha": 0.20, "eigen_floor_rel": 1e-8}
    }
]


print("\nExperiment 31F: Robust information inverse diagnostic")
print("----------------------------------------------------")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("parameter blocks:", PARAMETER_BLOCKS)
print("finite-difference epsilon:", FD_EPSILON)
print("relative information floor:", RELATIVE_INFO_FLOOR)
print("inverse methods:", [spec["name"] for spec in INVERSE_METHOD_SPECS])

# Simulate latent VARX(2) state-space data.
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


def make_directional_zero_constraint(source, target):
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

    model.fit(y=y_obs, u=u)
    model.fit_label = label
    return model


def model_ll(model):
    return final_observed_log_likelihood(model)

def choose_best_model(models):
    return max(models, key=model_ll)

def fit_warmstarted_case_models(y_obs, u, C):
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

    full_candidates.append(full_proxy)

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

        zero_constraints = make_directional_zero_constraint(source=source, target=target)

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
            initial_parameters=extract_model_parameters(full_proxy),
            label=f"reduced_{direction}_from_full_proxy",
            verbose=False
        )

        best_reduced = choose_best_model([reduced_proxy, reduced_from_full])

        full_from_reduced = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=[],
            initial_parameters=extract_model_parameters(best_reduced),
            label=f"full_from_best_reduced_{direction}",
            verbose=False
        )

        full_candidates.append(full_from_reduced)

        reduced_results[direction] = {
            "source": source,
            "target": target,
            "best_reduced": best_reduced,
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
            "full_from_reduced": full_from_reduced
        }

    best_full = choose_best_model(full_candidates)

    return {
        "best_full": best_full,
        "full_candidates": full_candidates,
        "reduced_results": reduced_results
    }

def compute_information_for_model(
        model,
        y_obs,
        u,
        label
):
    print(f"    finite-difference information: {label}")

    info = finite_difference_score_information(
        model=model,
        y=y_obs,
        u=u,
        parameter_blocks=PARAMETER_BLOCKS,
        fd_epsilon=FD_EPSILON,
        relative_info_floor=RELATIVE_INFO_FLOOR,
        verbose=False
    )

    return info

# Diagnostic guard only. 31F does not use this as the final rule yet.
# It helps us see which inverse methods would pass a practical safety test.
def is_correction_numerically_safe(
        result,
        full_model,
        reduced_model,
        raw_deviance,
        condition_threshold=1e7,
        spectral_radius_threshold=0.98,
        correction_multiplier=5.0
):
    bias_correction = result["bias_correction"]

    if not np.isfinite(bias_correction):
        return False

    if not np.isfinite(result["bias_full"]):
        return False

    if not np.isfinite(result["bias_reduced"]):
        return False

    if result["full_used_condition_number"] > condition_threshold:
        return False

    if result["reduced_used_condition_number"] > condition_threshold:
        return False

    if full_model.spectral_radius() >= spectral_radius_threshold:
        return False

    if reduced_model.spectral_radius() >= spectral_radius_threshold:
        return False

    allowed_size = correction_multiplier * max(abs(raw_deviance),result["df"], 1.0)

    if abs(bias_correction) > allowed_size:
        return False

    return True

def collect_method_result_row(
        case_label,
        outer_run,
        direction,
        source,
        target,
        expected_significant,
        full_model,
        reduced_model,
        raw_result,
        full_info,
        reduced_info,
        inverse_spec,
        full_mse
):
    robust_result = observed_debiased_deviance_from_precomputed_information(
        raw_result=raw_result,
        full_info=full_info,
        reduced_info=reduced_info,
        df_removed=na,
        inverse_method_spec=inverse_spec,
        clip_negative=True
    )

    raw_deviance = robust_result["observed_raw_deviance"]

    correction_safe = is_correction_numerically_safe(
        result=robust_result,
        full_model=full_model,
        reduced_model=reduced_model,
        raw_deviance=raw_deviance
    )

    if correction_safe:
        guarded_deviance = robust_result["observed_debiased_deviance"]
        guarded_deviance_for_p = robust_result["observed_debiased_deviance_for_p"]
        guarded_used_debiased = True
    else:
        guarded_deviance = raw_deviance
        guarded_deviance_for_p = max(raw_deviance, 0.0)
        guarded_used_debiased = False

    guarded_p_value = float(chi2.sf(guarded_deviance_for_p, df=na))

    return {
        "case": case_label,
        "outer_run": outer_run,
        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": expected_significant,

        "inverse_method_name": inverse_spec["name"],
        "inverse_method": inverse_spec["method"],

        "ll_full": raw_result["ll_full"],
        "ll_reduced": raw_result["ll_reduced"],
        "ll_gap_full_minus_reduced": raw_result["ll_full"] - raw_result["ll_reduced"],

        "raw_deviance": robust_result["observed_raw_deviance"],
        "raw_p_value": robust_result["observed_raw_p_value"],
        "raw_detected": robust_result["observed_raw_p_value"] < ALPHA,

        "bias_full": robust_result["bias_full"],
        "bias_reduced": robust_result["bias_reduced"],
        "bias_correction": robust_result["bias_correction"],

        "debiased_deviance": robust_result["observed_debiased_deviance"],
        "debiased_deviance_for_p": robust_result["observed_debiased_deviance_for_p"],
        "debiased_p_value": robust_result["observed_debiased_p_value"],
        "debiased_detected": robust_result["observed_debiased_p_value"] < ALPHA,

        "correction_safe": correction_safe,

        "guarded_deviance": guarded_deviance,
        "guarded_p_value": guarded_p_value,
        "guarded_detected": guarded_p_value < ALPHA,
        "guarded_used_debiased": guarded_used_debiased,

        "full_score_norm": robust_result["full_score_norm"],
        "reduced_score_norm": robust_result["reduced_score_norm"],

        "full_effective_rank": robust_result["full_effective_rank"],
        "reduced_effective_rank": robust_result["reduced_effective_rank"],

        "full_min_info_eigenvalue": robust_result["full_min_info_eigenvalue"],
        "reduced_min_info_eigenvalue": robust_result["reduced_min_info_eigenvalue"],

        "full_original_condition_number": robust_result["full_original_condition_number"],
        "reduced_original_condition_number": robust_result["reduced_original_condition_number"],

        "full_used_condition_number": robust_result["full_used_condition_number"],
        "reduced_used_condition_number": robust_result["reduced_used_condition_number"],

        "full_bias_is_finite": robust_result["full_bias_is_finite"],
        "reduced_bias_is_finite": robust_result["reduced_bias_is_finite"],

        "full_model_label": full_model.fit_label,
        "reduced_model_label": reduced_model.fit_label,

        "full_mse": full_mse,
        "full_spectral_radius": full_model.spectral_radius(),
        "reduced_spectral_radius": reduced_model.spectral_radius(),

        "full_em_iterations": len(full_model.log_likelihoods),
        "reduced_em_iterations": len(reduced_model.log_likelihoods),

        "full_R00": full_model.R[0, 0],
        "full_R11": full_model.R[1, 1],
        "full_R01": full_model.R[0, 1],

        "full_A1_01": full_model.A_matrices[0][0, 1],
        "full_A2_01": full_model.A_matrices[1][0, 1],
        "full_A1_10": full_model.A_matrices[0][1, 0],
        "full_A2_10": full_model.A_matrices[1][1, 0]
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

        fitted = fit_warmstarted_case_models(y_obs=y_obs, u=u, C=C)

        best_full = fitted["best_full"]

        x_full_smooth = best_full.smoothed_latent_state()

        full_mse = float(np.mean((x_full_smooth - x_true) ** 2))

        print("Best full label:", best_full.fit_label)
        print("Best full LL:", model_ll(best_full))
        print("Full MSE:", full_mse)
        print("Full spectral radius:", best_full.spectral_radius())
        print("Full R diag:", best_full.R[0, 0], best_full.R[1, 1])

        full_info = compute_information_for_model(
            model=best_full,
            y_obs=y_obs,
            u=u,
            label="full"
        )

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

            print("\nDirection:", direction)
            print("Reduced label:", best_reduced.fit_label)

            raw_result = observed_likelihood_deviance(
                full_model=best_full,
                reduced_model=best_reduced,
                df_removed=na,
                clip_negative=True
            )

            print("Raw deviance:", raw_result["observed_likelihood_deviance"])
            print("Raw p:", raw_result["p_value"])

            reduced_info = compute_information_for_model(
                model=best_reduced,
                y_obs=y_obs,
                u=u,
                label=f"reduced {direction}"
            )

            for inverse_spec in INVERSE_METHOD_SPECS:

                row = collect_method_result_row(
                    case_label=case_label,
                    outer_run=outer_run,
                    direction=direction,
                    source=spec["source"],
                    target=spec["target"],
                    expected_significant=spec["expected_significant"],
                    full_model=best_full,
                    reduced_model=best_reduced,
                    raw_result=raw_result,
                    full_info=full_info,
                    reduced_info=reduced_info,
                    inverse_spec=inverse_spec,
                    full_mse=full_mse
                )

                rows.append(row)

            print("Completed inverse-method comparisons for:", direction)

        # Save partial progress after every outer run.
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv(f"results/experiment_31f_{case_label}_partial.csv", index=False)

    return rows

def summarize_results(results_df):
    summary_rows = []

    grouped = results_df.groupby([
        "inverse_method_name",
        "case",
        "direction",
        "expected_significant"
    ])

    for keys, group in grouped:
        inverse_method_name, case_label, direction, expected_significant = keys

        summary_rows.append({
            "inverse_method_name": inverse_method_name,
            "case": case_label,
            "direction": direction,
            "expected_significant": expected_significant,

            "mean_raw_deviance": group["raw_deviance"].mean(),
            "median_raw_p_value": group["raw_p_value"].median(),
            "raw_detection_rate": group["raw_detected"].mean(),

            "mean_debiased_deviance": group["debiased_deviance"].mean(),
            "std_debiased_deviance": group["debiased_deviance"].std(),
            "median_debiased_p_value": group["debiased_p_value"].median(),
            "debiased_detection_rate": group["debiased_detected"].mean(),

            "mean_bias_full": group["bias_full"].mean(),
            "mean_bias_reduced": group["bias_reduced"].mean(),
            "mean_bias_correction": group["bias_correction"].mean(),

            "median_bias_full": group["bias_full"].median(),
            "median_bias_reduced": group["bias_reduced"].median(),
            "median_bias_correction": group["bias_correction"].median(),

            "max_abs_bias_correction": np.max(np.abs(group["bias_correction"])),

            "correction_safe_fraction": group["correction_safe"].mean(),

            "guarded_detection_rate": group["guarded_detected"].mean(),
            "guarded_used_debiased_fraction": group["guarded_used_debiased"].mean(),
            "median_guarded_p_value": group["guarded_p_value"].median(),

            "mean_full_score_norm": group["full_score_norm"].mean(),
            "mean_reduced_score_norm": group["reduced_score_norm"].mean(),

            "mean_full_original_condition_number": group["full_original_condition_number"].mean(),
            "mean_reduced_original_condition_number": group["reduced_original_condition_number"].mean(),

            "mean_full_used_condition_number": group["full_used_condition_number"].mean(),
            "mean_reduced_used_condition_number": group["reduced_used_condition_number"].mean(),

            "mean_full_effective_rank": group["full_effective_rank"].mean(),
            "mean_reduced_effective_rank": group["reduced_effective_rank"].mean(),

            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_reduced_spectral_radius": group["reduced_spectral_radius"].mean(),

            "mean_full_R00": group["full_R00"].mean(),
            "mean_full_R11": group["full_R11"].mean()
        })

    return pd.DataFrame(summary_rows)

# Main run
all_rows = []

no_link_rows = run_outer_case(
    case_label="no_link",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=NO_LINK_BASE_SEED
)

all_rows.extend(no_link_rows)

true_link_rows = run_outer_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=TRUE_LINK_BASE_SEED
)

all_rows.extend(true_link_rows)
results_df = pd.DataFrame(all_rows)
summary_df = summarize_results(results_df)

# Print summaries
print("\n" + "=" * 160)
print("Experiment 31F robust inverse aggregate summary")
print("=" * 160)

summary_columns = [
    "inverse_method_name",
    "case",
    "direction",
    "expected_significant",

    "mean_raw_deviance",
    "median_raw_p_value",
    "raw_detection_rate",

    "mean_debiased_deviance",
    "median_debiased_p_value",
    "debiased_detection_rate",

    "mean_bias_full",
    "mean_bias_reduced",
    "mean_bias_correction",
    "max_abs_bias_correction",

    "correction_safe_fraction",
    "guarded_detection_rate",
    "guarded_used_debiased_fraction",

    "mean_full_used_condition_number",
    "mean_reduced_used_condition_number",

    "mean_full_effective_rank",
    "mean_reduced_effective_rank",

    "mean_full_mse",
    "mean_full_spectral_radius",
    "mean_full_R00",
    "mean_full_R11"
]

print(summary_df[summary_columns].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

results_path = "results/experiment_31f_robust_information_inverse_results.csv"
summary_path = "results/experiment_31f_robust_information_inverse_summary.csv"

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\nSaved detailed results to:")
print(results_path)

print("\nSaved aggregate summary to:")
print(summary_path)

# Interpretation guide
print("\n" + "=" * 160)
print("Interpretation guide")
print("=" * 160)

print(
    "\nThis experiment tests robust alternatives for I^{-1} in the observed de-biased correction:"
    "\n    B = s^T I^{-1} s"
)

print(
    "\nGood inverse methods should satisfy:"
    "\n    1. no catastrophic bias corrections,"
    "\n    2. lower used condition numbers,"
    "\n    3. null false-positive rate not worse than raw,"
    "\n    4. true Y->X power close to raw,"
    "\n    5. correction_safe_fraction reasonably high."
)

print(
    "\nThe guarded columns are diagnostic for the next experiment."
    "\nThey show what would happen if we skipped the de-biased correction whenever diagnostics fail."
)

print(
    "\nIf pinv/eig_clip/shrinkage still perform worse than raw deviance,"
    "\nthen the conclusion is that the score/Hessian observed de-biasing is not useful in this setting,"
    "\nand we should keep raw observed deviance with diagnostic guards."
)

print(
    "\nIf one robust method improves false positives without hurting true-link power,"
    "\nthat method becomes the candidate for Experiment 31G."
)