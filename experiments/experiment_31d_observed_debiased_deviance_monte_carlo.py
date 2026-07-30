import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c_warmstart import EMVARXPSSMKnownCConstrainedWarmStart, extract_model_parameters
from src.stats.observed_likelihood_deviance import final_observed_log_likelihood
from src.stats.observed_debiased_deviance import observed_debiased_likelihood_deviance

N_SAMPLES = 1000
BURN_IN = 300

N_OUTER_RUNS = 20

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

# Full observed-information correction.
# This includes A, B, Q, R. C is known and D is fixed.
# PARAMETER_BLOCKS = ("A", "B", "Q", "R")
PARAMETER_BLOCKS = ("A", "B", "Q")

FD_EPSILON = 3e-4
RELATIVE_INFO_FLOOR = 1e-8

print("\nExperiment 31D: Observed-data de-biased deviance Monte Carlo")
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

# Simulate latent VARX(2) state-space data.
# True endogenous order: na = 2
# True exogenous order:  nb = 3
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
        "A1_true": A1,
        "A2_true": A2,
        "Q_true": Q,
        "R_true": R,
        "C": C
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
    if len(models) < 1:
        raise ValueError("models must contain at least one fitted model.")

    return max(models, key=model_ll)

# For each dataset:
# 1. Fit full model from proxy.
# 2. Fit each reduced model from proxy and from full initialization.
# 3. Fit full models warm-started from the best reduced models.
# 4. Choose the best full model by observed-data likelihood.
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
            initial_parameters=extract_model_parameters(
                full_proxy
            ),
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
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
            "best_reduced": best_reduced,
            "full_from_reduced": full_from_reduced
        }

    best_full = choose_best_model(full_candidates)

    return {
        "best_full": best_full,
        "full_candidates": full_candidates,
        "reduced_results": reduced_results
    }

def collect_direction_result(
        case_label,
        outer_run,
        direction,
        source,
        target,
        expected_significant,
        best_full,
        best_reduced,
        full_mse,
        y_obs,
        u
):
    result = observed_debiased_likelihood_deviance(
        full_model=best_full,
        reduced_model=best_reduced,
        y=y_obs,
        u=u,
        df_removed=na,
        parameter_blocks=PARAMETER_BLOCKS,
        fd_epsilon=FD_EPSILON,
        relative_info_floor=RELATIVE_INFO_FLOOR,
        clip_negative=True,
        verbose=False
    )

    ll_gap = (result["ll_full"] - result["ll_reduced"])

    return {
        "case": case_label,
        "outer_run": outer_run,
        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": expected_significant,

        "full_model_label": best_full.fit_label,
        "reduced_model_label": best_reduced.fit_label,

        "ll_full": result["ll_full"],
        "ll_reduced": result["ll_reduced"],
        "ll_gap_full_minus_reduced": ll_gap,
        "negative_ll_gap": ll_gap < -1e-8,

        "observed_raw_deviance": result["observed_raw_deviance"],
        "observed_raw_p_value": result["observed_raw_p_value"],
        "observed_raw_detected": result["observed_raw_p_value"] < ALPHA,

        "bias_full": result["bias_full"],
        "bias_reduced": result["bias_reduced"],
        "bias_correction": result["bias_correction"],

        "observed_debiased_deviance": result["observed_debiased_deviance"],
        "observed_debiased_deviance_for_p": result["observed_debiased_deviance_for_p"],
        "observed_debiased_p_value": result["observed_debiased_p_value"],
        "observed_debiased_detected": result["observed_debiased_p_value"] < ALPHA,

        "full_score_norm": result["full_score_norm"],
        "reduced_score_norm": result["reduced_score_norm"],

        "full_n_parameters": result["full_n_parameters"],
        "reduced_n_parameters": result["reduced_n_parameters"],

        "full_min_info_eigenvalue": result["full_min_info_eigenvalue"],
        "reduced_min_info_eigenvalue": result["reduced_min_info_eigenvalue"],

        "full_info_condition_number": result["full_info_condition_number"],
        "reduced_info_condition_number": result["reduced_info_condition_number"],

        "df": result["df"],

        "full_mse": full_mse,
        "full_spectral_radius": best_full.spectral_radius(),
        "full_em_iterations": len(best_full.log_likelihoods),
        "reduced_spectral_radius": best_reduced.spectral_radius(),
        "reduced_em_iterations": len(best_reduced.log_likelihoods),

        "full_A1_01": best_full.A_matrices[0][0, 1],
        "full_A2_01": best_full.A_matrices[1][0, 1],
        "full_A1_10": best_full.A_matrices[0][1, 0],
        "full_A2_10": best_full.A_matrices[1][1, 0],

        "full_R00": best_full.R[0, 0],
        "full_R11": best_full.R[1, 1],
        "full_R01": best_full.R[0, 1],

        "parameter_blocks": result["parameter_blocks"],
        "fd_epsilon": result["fd_epsilon"],
        "relative_info_floor": result["relative_info_floor"]
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
        print(f"\nCase={case_label} | outer run {outer_run + 1}/{N_OUTER_RUNS}")
        
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
        print("Full EM iterations:", len(best_full.log_likelihoods))
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
            print("\nComputing observed de-biased deviance for:", direction)

            row = collect_direction_result(
                case_label=case_label,
                outer_run=outer_run,
                direction=direction,
                source=spec["source"],
                target=spec["target"],
                expected_significant=spec["expected_significant"],
                best_full=best_full,
                best_reduced=best_reduced,
                full_mse=full_mse,
                y_obs=y_obs,
                u=u
            )

            rows.append(row)

            print("Reduced label:", best_reduced.fit_label)
            print("LL reduced:", row["ll_reduced"])
            print("LL gap:", row["ll_gap_full_minus_reduced"])
            print("Raw deviance:", row["observed_raw_deviance"])
            print("Raw p:", row["observed_raw_p_value"])
            print("Raw detected:", row["observed_raw_detected"])
            print("Bias full:", row["bias_full"])
            print("Bias reduced:", row["bias_reduced"])
            print("Bias correction -Br+Bf:", row["bias_correction"])
            print("Debiased deviance:", row["observed_debiased_deviance"])
            print("Debiased p:", row["observed_debiased_p_value"])
            print("Debiased detected:", row["observed_debiased_detected"])
            print("Full score norm:", row["full_score_norm"])
            print("Reduced score norm:", row["reduced_score_norm"])

    return rows

def summarize_results(outer_df):
    summary_rows = []

    grouped = outer_df.groupby([
        "case",
        "direction",
        "expected_significant"
    ])

    for keys, group in grouped:
        case_label, direction, expected_significant = keys

        summary_rows.append({
            "case": case_label,
            "direction": direction,
            "expected_significant": expected_significant,

            "mean_raw_deviance": group["observed_raw_deviance"].mean(),
            "std_raw_deviance": group["observed_raw_deviance"].std(),
            "median_raw_p_value": group["observed_raw_p_value"].median(),
            "raw_detection_rate": group["observed_raw_detected"].mean(),

            "mean_debiased_deviance": group["observed_debiased_deviance"].mean(),
            "std_debiased_deviance": group["observed_debiased_deviance"].std(),
            "median_debiased_p_value": group["observed_debiased_p_value"].median(),
            "debiased_detection_rate": group["observed_debiased_detected"].mean(),

            "mean_bias_full": group["bias_full"].mean(),
            "mean_bias_reduced": group["bias_reduced"].mean(),
            "mean_bias_correction": group["bias_correction"].mean(),

            "median_bias_full": group["bias_full"].median(),
            "median_bias_reduced": group["bias_reduced"].median(),
            "median_bias_correction": group["bias_correction"].median(),

            "mean_full_score_norm": group["full_score_norm"].mean(),
            "mean_reduced_score_norm": group["reduced_score_norm"].mean(),

            "mean_full_info_condition_number": group["full_info_condition_number"].mean(),
            "mean_reduced_info_condition_number": group["reduced_info_condition_number"].mean(),

            "negative_ll_gap_fraction": group["negative_ll_gap"].mean(),
            "min_ll_gap": group["ll_gap_full_minus_reduced"].min(),
            "mean_ll_gap": group["ll_gap_full_minus_reduced"].mean(),

            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_full_em_iterations": group["full_em_iterations"].mean(),
            "mean_reduced_em_iterations": group["reduced_em_iterations"].mean(),

            "mean_full_A1_01": group["full_A1_01"].mean(),
            "mean_full_A2_01": group["full_A2_01"].mean(),
            "mean_full_A1_10": group["full_A1_10"].mean(),
            "mean_full_A2_10": group["full_A2_10"].mean(),

            "mean_full_R00": group["full_R00"].mean(),
            "mean_full_R11": group["full_R11"].mean(),
            "mean_full_R01": group["full_R01"].mean()
        })

    return pd.DataFrame(summary_rows)

# Main run
outer_rows_all = []

no_link_rows = run_outer_case(
    case_label="no_link",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=1800000
)
outer_rows_all.extend(no_link_rows)

true_link_rows = run_outer_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=1900000
)
outer_rows_all.extend(true_link_rows)

outer_df = pd.DataFrame(outer_rows_all)
summary_df = summarize_results(outer_df)

# Print compact tables
print("\nExperiment 31D outer-run results")
outer_columns = [
    "case",
    "outer_run",
    "direction",
    "expected_significant",
    "observed_raw_deviance",
    "observed_raw_p_value",
    "observed_raw_detected",
    "bias_full",
    "bias_reduced",
    "bias_correction",
    "observed_debiased_deviance",
    "observed_debiased_p_value",
    "observed_debiased_detected",
    "full_score_norm",
    "reduced_score_norm",
    "ll_gap_full_minus_reduced",
    "full_mse",
    "full_spectral_radius",
    "full_R00",
    "full_R11"
]

print(outer_df[outer_columns].to_string(index=False))
print("Experiment 31D aggregate summary")

summary_columns = [
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

    "mean_full_score_norm",
    "mean_reduced_score_norm",

    "negative_ll_gap_fraction",
    "min_ll_gap",

    "mean_full_mse",
    "mean_full_spectral_radius",
    "mean_full_R00",
    "mean_full_R11"
]

print(summary_df[summary_columns].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

outer_path = "results/experiment_31d_observed_debiased_deviance_outer_results.csv"
summary_path = "results/experiment_31d_observed_debiased_deviance_summary.csv"

outer_df.to_csv(outer_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\nSaved outer-run results to:")
print(outer_path)

print("\nSaved aggregate summary to:")
print(summary_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nThis experiment compares raw observed-likelihood deviance against "
    "observed-data de-biased deviance."
)

print(
    "\nThe de-biased statistic is:"
    "\n    D_db = D_raw - B_reduced + B_full"
    "\nwhere:"
    "\n    B_m = s_m^T I_m^{-1} s_m"
)

print(
    "\nThe current finite-difference information calculation includes A, B, Q, and R. "
    "C is known and D is fixed."
)

print(
    "\nFor null directions, debiased_detection_rate should be close to alpha = 0.05."
)

print(
    "\nFor true_link Y->X, debiased_detection_rate should remain high."
)

print(
    "\nIf the bias correction terms are very small, then raw and de-biased observed "
    "deviance will be nearly identical. That would mean EM has approximately reached "
    "a stationary point of the observed likelihood."
)

print(
    "\nIf the bias correction changes decisions substantially, inspect score norms, "
    "information condition numbers, and whether R is near the floor."
)