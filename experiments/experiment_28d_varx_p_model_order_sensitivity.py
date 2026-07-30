import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
N_RUNS = 100

TRUE_NA = 2
TRUE_NB = 3

FIT_NA_LIST = [1, 2, 3, 5]
FIT_NB = 3

ALPHA = 0.05

LAMBDA_REGULARIZATION = 1.0

GAMMA_MODES = [
    "ols",
    "ridge"
]

print("\nExperiment 28D: VARX(p) model-order sensitivity")
print("N samples:", N_SAMPLES)
print("N runs:", N_RUNS)
print("True na:", TRUE_NA)
print("True nb:", TRUE_NB)
print("Fit na values:", FIT_NA_LIST)
print("Fit nb:", FIT_NB)
print("alpha:", ALPHA)
print("lambda regularization:", LAMBDA_REGULARIZATION)
print("gamma modes:", GAMMA_MODES)


def gamma_from_mode(mode, n_samples):
    if mode == "ols":
        return 0.0

    if mode == "ridge":
        return LAMBDA_REGULARIZATION / np.sqrt(n_samples)

    raise ValueError(f"Unknown gamma mode: {mode}")

def simulate_true_varx2_case(n_samples, y_to_x_lag1, y_to_x_lag2, random_seed):
    u = generate_colored_input(n_samples=n_samples, ar_coeff=0.95, noise_std=1.0, random_seed=random_seed)

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

    noise_cov = 0.5 * np.eye(2)

    y = generate_varx_data(A_matrices=[A1, A2], B_matrices=[B0, B1, B2], u=u, noise_cov=noise_cov, random_seed=random_seed + 10000)

    return y, u, A1, A2

def raw_p_value_from_deviance(deviance_raw, df):
    deviance_for_p = max(float(deviance_raw), 0.0)

    return float(chi2.sf(deviance_for_p, df=df))

def run_single_gc_pair(y, u, source, target, fit_na, gamma):
    result = debiased_varx_gc(
        y=y,
        u=u,
        source=source,
        target=target,
        na=fit_na,
        nb=FIT_NB,
        gamma=gamma,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    raw_p = raw_p_value_from_deviance(deviance_raw=result["deviance_raw"], df=result["df"])

    result["p_value_raw_chi2"] = raw_p
    result["raw_detected"] = raw_p < ALPHA
    result["debiased_detected"] = result["p_value"] < ALPHA

    return result

def summarize_result_list(results):
    raw_gc = np.array([r["gc_raw"] for r in results])
    dev_raw = np.array([r["deviance_raw"] for r in results])
    dev_db = np.array([r["deviance_debiased"] for r in results])
    p_raw = np.array([r["p_value_raw_chi2"] for r in results])
    p_db = np.array([r["p_value"] for r in results])
    bias_full = np.array([r["bias_full"] for r in results])
    bias_reduced = np.array([r["bias_reduced"] for r in results])

    return {
        "mean_raw_gc": np.mean(raw_gc),
        "std_raw_gc": np.std(raw_gc),
        "negative_raw_gc_fraction": np.mean(raw_gc < 0),

        "mean_raw_deviance": np.mean(dev_raw),
        "std_raw_deviance": np.std(dev_raw),

        "mean_debiased_deviance": np.mean(dev_db),
        "std_debiased_deviance": np.std(dev_db),

        "mean_bias_full": np.mean(bias_full),
        "mean_bias_reduced": np.mean(bias_reduced),
        "mean_bias_correction_minus_br_plus_bf": np.mean(-bias_reduced + bias_full),

        "median_raw_p": np.median(p_raw),
        "median_debiased_p": np.median(p_db),

        "raw_detection_rate": np.mean(p_raw < ALPHA),
        "debiased_detection_rate": np.mean(p_db < ALPHA),

        "p05_debiased": np.percentile(p_db, 5),
        "p95_debiased": np.percentile(p_db, 95)
    }

def collect_results(fit_na, gamma_mode, case_label, y_to_x_lag1, y_to_x_lag2):
    gamma = gamma_from_mode(mode=gamma_mode, n_samples=N_SAMPLES)

    y_to_x_results = []
    x_to_y_results = []

    for run in range(N_RUNS):

        seed = (
            200000
            + 10000 * FIT_NA_LIST.index(fit_na)
            + 1000 * GAMMA_MODES.index(gamma_mode)
            + run
        )

        y, u, A1, A2 = simulate_true_varx2_case(n_samples=N_SAMPLES, y_to_x_lag1=y_to_x_lag1, y_to_x_lag2=y_to_x_lag2, random_seed=seed)

        result_y_to_x = run_single_gc_pair(y=y, u=u, source=1, target=0, fit_na=fit_na, gamma=gamma)
        result_x_to_y = run_single_gc_pair(y=y, u=u, source=0, target=1, fit_na=fit_na, gamma=gamma)

        y_to_x_results.append(result_y_to_x)
        x_to_y_results.append(result_x_to_y)

    return y_to_x_results, x_to_y_results

summary_rows = []

# Main loop
for fit_na in FIT_NA_LIST:

    for gamma_mode in GAMMA_MODES:
        gamma = gamma_from_mode(mode=gamma_mode, n_samples=N_SAMPLES)
        print(f"fit_na = {fit_na} | " f"gamma_mode = {gamma_mode} | " f"gamma = {gamma}")
        
        # Case 1: no true Y -> X
        null_y_to_x, null_x_to_y = collect_results(fit_na=fit_na, gamma_mode=gamma_mode, case_label="null", y_to_x_lag1=0.00, y_to_x_lag2=0.00)

        # Case 2: true Y -> X
        true_y_to_x, true_x_to_y = collect_results(fit_na=fit_na, gamma_mode=gamma_mode, case_label="true_link", y_to_x_lag1=0.25, y_to_x_lag2=0.12)

        groups = [
            ("null", "Y->X", False, null_y_to_x),
            ("null", "X->Y", False, null_x_to_y),
            ("true_link", "Y->X", True, true_y_to_x),
            ("true_link", "X->Y", False, true_x_to_y)
        ]

        for case_label, direction, expected_significant, results in groups:
            summary = summarize_result_list(results)

            row = {
                "true_na": TRUE_NA,
                "fit_na": fit_na,
                "fit_nb": FIT_NB,
                "gamma_mode": gamma_mode,
                "gamma": gamma,
                "case": case_label,
                "direction": direction,
                "expected_significant": expected_significant,
                "df": fit_na,
                **summary
            }

            summary_rows.append(row)

            print(
                f"\n{case_label:10s} | {direction:4s} | "
                f"expected={expected_significant}"
            )

            print(
                "debiased detect:",
                summary["debiased_detection_rate"],
                "| raw detect:",
                summary["raw_detection_rate"],
                "| mean raw GC:",
                summary["mean_raw_gc"],
                "| median debiased p:",
                summary["median_debiased_p"],
                "| negative raw GC:",
                summary["negative_raw_gc_fraction"]
            )

# Summary table
summary_df = pd.DataFrame(summary_rows)

print("Experiment 28D summary table")

columns_to_show = [
    "true_na",
    "fit_na",
    "gamma_mode",
    "case",
    "direction",
    "expected_significant",
    "mean_raw_gc",
    "negative_raw_gc_fraction",
    "debiased_detection_rate",
    "median_debiased_p",
    "mean_debiased_deviance",
    "mean_bias_correction_minus_br_plus_bf"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)
output_path = "results/experiment_28d_summary.csv"
summary_df.to_csv(output_path, index=False)

print("\nSaved summary to:")
print(output_path)

# Interpretation guide
print("Interpretation guide")
print(
    "\nThe true system is VARX(2). Therefore fit_na=2 is correctly specified."
)

print(
    "\nfit_na=1 is under-specified and may miss lag-2 effects."
)

print(
    "\nfit_na=3 or fit_na=5 are over-specified. They may preserve detection "
    "of true links but can increase false positives, especially with short T."
)

print(
    "\nFor null directions, debiased_detection_rate should be near or below alpha =",
    ALPHA
)

print(
    "\nFor true-link Y->X, debiased_detection_rate should remain high."
)

print(
    "\nThe degrees of freedom for a scalar source-to-target test equal fit_na, "
    "because fit_na source-history coefficients are removed in the reduced model."
)