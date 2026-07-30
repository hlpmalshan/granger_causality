import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.latent_empirical_calibration import fit_em_known_c_estimated_r, empirical_calibration_for_direction_estimated_r

N_SAMPLES = 1000
BURN_IN = 300

# Start here. Increase after confirming runtime.
N_OUTER_RUNS = 20
N_BOOTSTRAP = 100

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 30F: Monte Carlo validation of empirical p-values with estimated R")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("N bootstrap:", N_BOOTSTRAP)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("gamma_gc:", gamma_gc)
print("Minimum possible empirical p-value:", 1.0 / (N_BOOTSTRAP + 1))


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

def add_outer_result_row(
        rows,
        case_label,
        outer_run,
        direction,
        source,
        target,
        expected_significant,
        calibration_summary,
        em_mse,
        spectral_radius,
        em_iterations
):
    rows.append({
        "case": case_label,
        "outer_run": outer_run,
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
        "spectral_radius": spectral_radius,
        "em_iterations": em_iterations
    })

def summarize_outer_results(results_df):
    summary_rows = []

    grouped = results_df.groupby([
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

            "mean_raw_gc": group["observed_raw_gc"].mean(),
            "std_raw_gc": group["observed_raw_gc"].std(),
            "negative_raw_gc_fraction": np.mean(group["observed_raw_gc"] < 0),

            "mean_debiased_deviance": group["observed_debiased_deviance"].mean(),
            "std_debiased_deviance": group["observed_debiased_deviance"].std(),

            "median_chi_square_p": group["chi_square_p_value"].median(),
            "median_empirical_p": group["empirical_p_value"].median(),

            "chi_square_detection_rate": group["chi_square_detected"].mean(),
            "empirical_detection_rate": group["empirical_detected"].mean(),

            "mean_null_95_statistic": group["null_95_statistic"].mean(),
            "mean_observed_statistic": group["observed_statistic_clipped"].mean(),

            "mean_em_mse": group["em_mse"].mean(),
            "mean_spectral_radius": group["spectral_radius"].mean(),
            "mean_em_iterations": group["em_iterations"].mean()
        })

    return pd.DataFrame(summary_rows)

# Run one case over multiple outer Monte Carlo datasets.
def run_outer_case(
        case_label,
        y_to_x_lag1,
        y_to_x_lag2,
        expected_y_to_x_significant,
        base_seed,
        base_bootstrap_seed
):
    outer_rows = []
    bootstrap_rows_all = []

    for outer_run in range(N_OUTER_RUNS):
        print(f"Case={case_label} | outer run {outer_run + 1}/{N_OUTER_RUNS}")
        
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
        Q_init = data["Q_true"]

        observed_model = fit_em_known_c_estimated_r(
            y_obs=y_obs,
            u=u,
            C=C,
            R_floor=R_FLOOR,
            na=na,
            nb=nb,
            Q_init=Q_init,
            max_iter=50,
            tol=1e-5,
            verbose=False
        )

        x_em = observed_model.smoothed_latent_state()

        em_mse = float(np.mean((x_em - x_true) ** 2))

        spectral_radius = float(observed_model.spectral_radius())

        em_iterations = len(observed_model.log_likelihoods)

        print("Observed EM MSE:", em_mse)
        print("Observed spectral radius:", spectral_radius)
        print("Observed EM iterations:", em_iterations)

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

            direction = spec["direction"]
            source = spec["source"]
            target = spec["target"]
            expected_significant = spec["expected_significant"]

            print(f"\nCalibrating direction {direction}")

            calibration_summary, bootstrap_rows = empirical_calibration_for_direction_estimated_r(
                observed_model=observed_model,
                observed_u=u,
                source=source,
                target=target,
                na=na,
                nb=nb,
                gamma=gamma_gc,
                n_bootstrap=N_BOOTSTRAP,
                base_seed=( base_bootstrap_seed + 100000 * outer_run + 10000 * direction_index),
                R_floor=R_FLOOR,
                em_max_iter=50,
                em_tol=1e-5,
                verbose_every=25
            )

            add_outer_result_row(
                rows=outer_rows,
                case_label=case_label,
                outer_run=outer_run,
                direction=direction,
                source=source,
                target=target,
                expected_significant=expected_significant,
                calibration_summary=calibration_summary,
                em_mse=em_mse,
                spectral_radius=spectral_radius,
                em_iterations=em_iterations
            )

            for row in bootstrap_rows:
                row["case"] = case_label
                row["outer_run"] = outer_run
                row["direction"] = direction
                row["source"] = source
                row["target"] = target
                row["expected_significant"] = expected_significant

            bootstrap_rows_all.extend(bootstrap_rows)

            print("Observed deviance:", calibration_summary["observed_debiased_deviance"])
            print("Chi-square p:", calibration_summary["chi_square_p_value"])
            print("Empirical p:", calibration_summary["empirical_p_value"])
            print("Null 95% statistic:", calibration_summary["null_95_statistic"])

    return outer_rows, bootstrap_rows_all

# Main run
outer_rows_all = []
bootstrap_rows_all = []

print("\nRunning null case...")
null_outer_rows, null_bootstrap_rows = run_outer_case(
    case_label="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=800000,
    base_bootstrap_seed=900000
)

outer_rows_all.extend(null_outer_rows)
bootstrap_rows_all.extend(null_bootstrap_rows)

print("\nRunning true-link case...")
true_outer_rows, true_bootstrap_rows = run_outer_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=1000000,
    base_bootstrap_seed=1100000
)

outer_rows_all.extend(true_outer_rows)
bootstrap_rows_all.extend(true_bootstrap_rows)

outer_df = pd.DataFrame(outer_rows_all)
bootstrap_df = pd.DataFrame(bootstrap_rows_all)

summary_df = summarize_outer_results(outer_df)

# Print results
print("\nExperiment 30F outer-run summary")
outer_columns = [
    "case",
    "outer_run",
    "direction",
    "expected_significant",
    "observed_raw_gc",
    "observed_debiased_deviance",
    "chi_square_p_value",
    "empirical_p_value",
    "chi_square_detected",
    "empirical_detected",
    "null_95_statistic",
    "em_mse"
]

print(outer_df[outer_columns].to_string(index=False))

print("Experiment 30F aggregate summary")

summary_columns = [
    "case",
    "direction",
    "expected_significant",
    "mean_raw_gc",
    "negative_raw_gc_fraction",
    "mean_debiased_deviance",
    "median_chi_square_p",
    "median_empirical_p",
    "chi_square_detection_rate",
    "empirical_detection_rate",
    "mean_null_95_statistic",
    "mean_observed_statistic",
    "mean_em_mse",
    "mean_em_iterations"
]

print(summary_df[summary_columns].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

outer_path = "results/experiment_30f_outer_results.csv"
bootstrap_path = "results/experiment_30f_bootstrap_results.csv"
summary_path = "results/experiment_30f_summary.csv"

outer_df.to_csv(outer_path, index=False)
bootstrap_df.to_csv(bootstrap_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\nSaved outer-run results to:")
print(outer_path)

print("\nSaved bootstrap results to:")
print(bootstrap_path)

print("\nSaved aggregate summary to:")
print(summary_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nFor null directions, empirical_detection_rate should be much closer "
    "to alpha than chi_square_detection_rate."
)

print(
    "\nFor true-link Y->X, empirical_detection_rate should remain high."
)

print(
    "\nIf chi-square detects many null directions but empirical calibration does not, "
    "that confirms that the empirical null is correcting the liberal chi-square reference."
)

print(
    "\nBecause N_BOOTSTRAP is finite, empirical p-values cannot be smaller than "
    "1 / (N_BOOTSTRAP + 1)."
)

print(
    "\nIf results are promising with N_OUTER_RUNS=20 and N_BOOTSTRAP=100, rerun with "
    "N_OUTER_RUNS=20 or 50 for stronger reporting."
)