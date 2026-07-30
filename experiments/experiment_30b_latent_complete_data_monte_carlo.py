import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c import EMVARXPSSMKnownC
from src.varx.varx_gc_debiased import debiased_varx_gc
from src.stats.latent_debiased_deviance import debiased_latent_varx_gc_from_smoother

N_SAMPLES = 1000
BURN_IN = 300

# Start with 50 for speed. Increase to 100 for final reporting.
N_RUNS = 100

na = 2
nb = 3

ALPHA = 0.05

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 30B: Monte Carlo latent complete-data GC")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N runs:", N_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("gamma_gc:", gamma_gc)


def simulate_case(n_samples, burn_in, y_to_x_lag1, y_to_x_lag2, random_seed):
    u_total = generate_colored_input(n_samples=n_samples + burn_in, ar_coeff=0.95, noise_std=1.0, random_seed=random_seed)

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
        "A1": A1,
        "A2": A2,
        "B_matrices": [B0, B1, B2],
        "Q": Q,
        "R": R,
        "C": C
    }


def fit_em_known_c_fixed_r(y_obs, u, C):
    model = EMVARXPSSMKnownC(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=50,
        tol=1e-5,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_init=0.60 * np.eye(2),
        Q_init=0.50 * np.eye(2),
        estimate_R=False,
        R_floor=0.05,
        verbose=False
    )

    model.fit(y=y_obs, u=u)

    return model


def sample_gc_pair(y_like, u):
    y_to_x = debiased_varx_gc(
        y=y_like,
        u=u,
        source=1,
        target=0,
        na=na,
        nb=nb,
        gamma=gamma_gc,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    x_to_y = debiased_varx_gc(
        y=y_like,
        u=u,
        source=0,
        target=1,
        na=na,
        nb=nb,
        gamma=gamma_gc,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    return y_to_x, x_to_y


def latent_complete_data_gc_pair(model, u):
    smoother = model.smooth_result["smoother"]

    smooth_mean = smoother["x_smooth"]
    smooth_cov = smoother["P_smooth"]
    smooth_lag_cov = smoother["P_lag_one"]

    y_to_x = debiased_latent_varx_gc_from_smoother(
        smooth_mean=smooth_mean,
        smooth_cov=smooth_cov,
        smooth_lag_cov=smooth_lag_cov,
        u=u,
        source=1,
        target=0,
        na=na,
        nb=nb,
        n_latent=2,
        conditioning=None,
        gamma=gamma_gc,
        penalty="diag",
        effective_t_mode="full_minus_features"
    )

    x_to_y = debiased_latent_varx_gc_from_smoother(
        smooth_mean=smooth_mean,
        smooth_cov=smooth_cov,
        smooth_lag_cov=smooth_lag_cov,
        u=u,
        source=0,
        target=1,
        na=na,
        nb=nb,
        n_latent=2,
        conditioning=None,
        gamma=gamma_gc,
        penalty="diag",
        effective_t_mode="full_minus_features"
    )

    return y_to_x, x_to_y

def add_result_row(
        rows,
        case_label,
        run,
        method,
        direction,
        expected_significant,
        result,
        mse=None,
        spectral_radius=None,
        n_em_iterations=None
):
    rows.append({
        "case": case_label,
        "run": run,
        "method": method,
        "direction": direction,
        "expected_significant": expected_significant,
        "gc_raw": result["gc_raw"],
        "deviance_debiased": result["deviance_debiased"],
        "p_value": result["p_value"],
        "detected": result["p_value"] < ALPHA,
        "sigma_full": result["sigma_full"],
        "sigma_reduced": result["sigma_reduced"],
        "bias_full": result["bias_full"],
        "bias_reduced": result["bias_reduced"],
        "mse": mse,
        "spectral_radius": spectral_radius,
        "n_em_iterations": n_em_iterations
    })

def run_monte_carlo_case(case_label, y_to_x_lag1, y_to_x_lag2, expected_y_to_x_significant, base_seed):
    rows = []
    for run in range(N_RUNS):
        seed = base_seed + run
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

        model = fit_em_known_c_fixed_r(y_obs=y_obs, u=u, C=C)
        x_em_smooth = model.smoothed_latent_state()

        em_mse = float(np.mean((x_em_smooth - x_true) ** 2))

        spectral_radius = float(model.spectral_radius())

        n_em_iterations = len(model.log_likelihoods)

        # Method 1: oracle latent sample GC.
        oracle_y_to_x, oracle_x_to_y = sample_gc_pair(y_like=x_true, u=u)

        # Method 2: sample GC on EM posterior mean.
        em_mean_y_to_x, em_mean_x_to_y = sample_gc_pair(y_like=x_em_smooth, u=u)

        # Method 3: latent complete-data posterior-stat GC.
        latent_cd_y_to_x, latent_cd_x_to_y = latent_complete_data_gc_pair(model=model, u=u)

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="oracle_latent_sample",
            direction="Y->X",
            expected_significant=expected_y_to_x_significant,
            result=oracle_y_to_x,
            mse=0.0,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="oracle_latent_sample",
            direction="X->Y",
            expected_significant=False,
            result=oracle_x_to_y,
            mse=0.0,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="em_posterior_mean_sample",
            direction="Y->X",
            expected_significant=expected_y_to_x_significant,
            result=em_mean_y_to_x,
            mse=em_mse,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="em_posterior_mean_sample",
            direction="X->Y",
            expected_significant=False,
            result=em_mean_x_to_y,
            mse=em_mse,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="latent_complete_data",
            direction="Y->X",
            expected_significant=expected_y_to_x_significant,
            result=latent_cd_y_to_x,
            mse=em_mse,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        add_result_row(
            rows=rows,
            case_label=case_label,
            run=run,
            method="latent_complete_data",
            direction="X->Y",
            expected_significant=False,
            result=latent_cd_x_to_y,
            mse=em_mse,
            spectral_radius=spectral_radius,
            n_em_iterations=n_em_iterations
        )

        if (run + 1) % 5 == 0:
            print(f"{case_label}: completed {run + 1}/{N_RUNS}")

    return rows


def summarize_results(df):
    summary_rows = []

    grouped = df.groupby(
        [
            "case",
            "method",
            "direction",
            "expected_significant"
        ]
    )

    for keys, group in grouped:
        case_label, method, direction, expected_significant = keys

        summary_rows.append({
            "case": case_label,
            "method": method,
            "direction": direction,
            "expected_significant": expected_significant,
            "mean_raw_gc": group["gc_raw"].mean(),
            "std_raw_gc": group["gc_raw"].std(),
            "negative_raw_gc_fraction": np.mean(group["gc_raw"] < 0),
            "mean_debiased_deviance": group["deviance_debiased"].mean(),
            "std_debiased_deviance": group["deviance_debiased"].std(),
            "median_p_value": group["p_value"].median(),
            "p05_value": group["p_value"].quantile(0.05),
            "p95_value": group["p_value"].quantile(0.95),
            "detection_rate": group["detected"].mean(),
            "mean_mse": group["mse"].dropna().mean(),
            "mean_spectral_radius": group["spectral_radius"].dropna().mean(),
            "mean_em_iterations": group["n_em_iterations"].dropna().mean()
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df

# Main run
print("\nRunning null case...")
null_rows = run_monte_carlo_case(
    case_label="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=400000
)

print("\nRunning true-link case...")
true_rows = run_monte_carlo_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=500000
)

results_df = pd.DataFrame(null_rows + true_rows)

summary_df = summarize_results(results_df)
# Print summary
print("\nExperiment 30B summary")
columns_to_show = [
    "case",
    "method",
    "direction",
    "expected_significant",
    "mean_raw_gc",
    "negative_raw_gc_fraction",
    "mean_debiased_deviance",
    "median_p_value",
    "detection_rate",
    "mean_mse",
    "mean_em_iterations"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

results_path = "results/experiment_30b_all_results.csv"
summary_path = "results/experiment_30b_summary.csv"

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\nSaved all run-level results to:")
print(results_path)

print("\nSaved summary to:")
print(summary_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nFor null directions, detection_rate should be close to or below alpha =",
    ALPHA
)

print(
    "\nFor true-link Y->X, detection_rate should be high."
)

print(
    "\nCompare em_posterior_mean_sample vs latent_complete_data. "
    "If latent_complete_data has lower false positives and raw GC closer "
    "to oracle_latent_sample, then posterior sufficient statistics are helping."
)

print(
    "\nIf latent_complete_data still has elevated false positives, then the next "
    "step should be empirical null calibration or observed-likelihood deviance."
)