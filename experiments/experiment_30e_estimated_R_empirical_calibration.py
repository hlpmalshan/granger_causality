import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.latent_empirical_calibration import fit_em_known_c_estimated_r, empirical_calibration_for_direction_estimated_r

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

N_BOOTSTRAP = 100

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

# R_floor prevents the EM collapse R -> 0.
R_FLOOR = 0.30

print("\nExperiment 30E: Estimated R with empirical calibration")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("N bootstrap:", N_BOOTSTRAP)
print("gamma_gc:", gamma_gc)
print("R floor:", R_FLOOR)
print("Minimum possible empirical p:", 1.0 / (N_BOOTSTRAP + 1))

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
        "B_matrices_true": [B0, B1, B2],
        "Q_true": Q,
        "R_true": R,
        "C": C
    }

def add_summary_row(
        rows,
        case_label,
        direction,
        source,
        target,
        expected_significant,
        calibration_summary,
        observed_model,
        em_mse
):
    rows.append({
        "case": case_label,
        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": expected_significant,

        "observed_raw_gc": calibration_summary["observed_raw_gc"],
        "observed_debiased_deviance": calibration_summary["observed_debiased_deviance"],
        "observed_statistic_clipped": calibration_summary["observed_statistic_clipped"],

        "chi_square_p_value": calibration_summary["chi_square_p_value"],
        "empirical_p_value": calibration_summary["empirical_p_value"],

        "chi_square_detected": calibration_summary["chi_square_p_value"] < ALPHA,
        "empirical_detected": calibration_summary["empirical_p_value"] < ALPHA,

        "null_mean_statistic": calibration_summary["null_mean_statistic"],
        "null_std_statistic": calibration_summary["null_std_statistic"],
        "null_median_statistic": calibration_summary["null_median_statistic"],
        "null_90_statistic": calibration_summary["null_90_statistic"],
        "null_95_statistic": calibration_summary["null_95_statistic"],
        "null_99_statistic": calibration_summary["null_99_statistic"],

        "n_bootstrap": calibration_summary["n_bootstrap"],
        "min_possible_empirical_p": calibration_summary["min_possible_empirical_p"],

        "em_mse": em_mse,
        "spectral_radius": observed_model.spectral_radius(),
        "em_iterations": len(observed_model.log_likelihoods),

        "estimated_R00": observed_model.R[0, 0],
        "estimated_R11": observed_model.R[1, 1],
        "estimated_R01": observed_model.R[0, 1],

        "estimated_Q00": observed_model.Q[0, 0],
        "estimated_Q11": observed_model.Q[1, 1],
        "estimated_Q01": observed_model.Q[0, 1]
    })

def run_case(
        case_label,
        y_to_x_lag1,
        y_to_x_lag2,
        expected_y_to_x_significant,
        random_seed,
        base_bootstrap_seed
):
    print("\n"+case_label)
    
    data = simulate_case(
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        y_to_x_lag1=y_to_x_lag1,
        y_to_x_lag2=y_to_x_lag2,
        random_seed=random_seed
    )

    x_true = data["x_true"]
    y_obs = data["y_obs"]
    u = data["u"]
    C = data["C"]

    # Controlled initialization:
    # R is initialized near the true scale but is estimated afterward.
    R_init = 0.60 * np.eye(2)
    Q_init = 0.50 * np.eye(2)

    observed_model = fit_em_known_c_estimated_r(
        y_obs=y_obs,
        u=u,
        C=C,
        na=na,
        nb=nb,
        R_init=R_init,
        Q_init=Q_init,
        R_floor=R_FLOOR,
        max_iter=50,
        tol=1e-5,
        verbose=True
    )

    x_em = observed_model.smoothed_latent_state()

    em_mse = float(np.mean((x_em - x_true) ** 2))

    print("\nEstimated A1")
    print(observed_model.A_matrices[0])

    print("\nEstimated A2")
    print(observed_model.A_matrices[1])

    print("\nEstimated Q")
    print(observed_model.Q)

    print("\nEstimated R")
    print(observed_model.R)

    print("\nEM smoothed MSE:", em_mse)
    print("Spectral radius:", observed_model.spectral_radius())
    print("EM iterations:", len(observed_model.log_likelihoods))

    summaries = []
    bootstrap_rows_all = []

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

    for direction_index, spec in enumerate(direction_specs):
        print("\nEmpirical calibration:", case_label, spec["direction"])
       
        calibration_summary, bootstrap_rows = empirical_calibration_for_direction_estimated_r(
            observed_model=observed_model,
            observed_u=u,
            source=spec["source"],
            target=spec["target"],
            na=na,
            nb=nb,
            gamma=gamma_gc,
            n_bootstrap=N_BOOTSTRAP,
            base_seed=base_bootstrap_seed + 10000 * direction_index,
            R_floor=R_FLOOR,
            em_max_iter=50,
            em_tol=1e-5,
            verbose_every=25
        )

        add_summary_row(
            rows=summaries,
            case_label=case_label,
            direction=spec["direction"],
            source=spec["source"],
            target=spec["target"],
            expected_significant=spec["expected_significant"],
            calibration_summary=calibration_summary,
            observed_model=observed_model,
            em_mse=em_mse
        )

        for row in bootstrap_rows:
            row["case"] = case_label
            row["direction"] = spec["direction"]
            row["source"] = spec["source"]
            row["target"] = spec["target"]
            row["expected_significant"] = spec["expected_significant"]

        bootstrap_rows_all.extend(bootstrap_rows)

        print("Observed raw GC:", calibration_summary["observed_raw_gc"])
        print("Observed deviance:", calibration_summary["observed_debiased_deviance"])
        print("Observed clipped statistic:", calibration_summary["observed_statistic_clipped"])
        print("Chi-square p:", calibration_summary["chi_square_p_value"])
        print("Empirical p:", calibration_summary["empirical_p_value"])
        print("Null 95% statistic:", calibration_summary["null_95_statistic"])

    return summaries, bootstrap_rows_all

# Main run
summary_rows = []
bootstrap_rows_all = []

null_summary, null_bootstrap = run_case(
    case_label="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    random_seed=1,
    base_bootstrap_seed=1200000
)

summary_rows.extend(null_summary)
bootstrap_rows_all.extend(null_bootstrap)

true_summary, true_bootstrap = run_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    random_seed=2,
    base_bootstrap_seed=1300000
)

summary_rows.extend(true_summary)
bootstrap_rows_all.extend(true_bootstrap)

summary_df = pd.DataFrame(summary_rows)
bootstrap_df = pd.DataFrame(bootstrap_rows_all)

# Print summary table
print("\nExperiment 30E summary table")

columns_to_show = [
    "case",
    "direction",
    "expected_significant",
    "observed_raw_gc",
    "observed_debiased_deviance",
    "chi_square_p_value",
    "empirical_p_value",
    "chi_square_detected",
    "empirical_detected",
    "null_95_statistic",
    "em_mse",
    "estimated_R00",
    "estimated_R11",
    "estimated_R01",
    "spectral_radius",
    "em_iterations"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

summary_path = "results/experiment_30e_estimated_R_empirical_summary.csv"
bootstrap_path = "results/experiment_30e_estimated_R_bootstrap_results.csv"

summary_df.to_csv(summary_path, index=False)
bootstrap_df.to_csv(bootstrap_path, index=False)

print("\nSaved summary to:")
print(summary_path)

print("\nSaved bootstrap results to:")
print(bootstrap_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nThe main question is whether estimating R with a floor still preserves "
    "the good empirical-calibrated behavior from 30C/30D."
)

print(
    "\nCheck whether estimated R collapsed to the floor. If R00 and R11 are "
    "exactly near R_floor, then the model is still trying to overtrust "
    "observations and the floor is acting as a regularizer."
)

print(
    "\nFor null directions, empirical_detected should be False."
)

print(
    "\nFor true-link Y->X, empirical_detected should be True."
)

print(
    "\nIf this single-dataset pilot works, the next step is 30F: Monte Carlo "
    "validation with estimated R."
)