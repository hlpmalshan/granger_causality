import numpy as np
import pandas as pd

from scipy.stats import chi2
from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_gc_debiased import debiased_varx_gc

T_LIST = [500, 1000, 3000]
N_RUNS = 100
na = 2
nb = 3
ALPHA = 0.05
LAMBDA_REGULARIZATION = 1.0
GAMMA_MODES = ["ols", "ridge"]

print("\nExperiment 28C: VARX(p) sample-size and regularization sensitivity")
print("T values:", T_LIST)
print("N runs:", N_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("lambda regularization:", LAMBDA_REGULARIZATION)
print("gamma modes:", GAMMA_MODES)

def gamma_from_mode(mode, n_samples):
    if mode == "ols":
        return 0.0
    if mode == "ridge":
        return LAMBDA_REGULARIZATION / np.sqrt(n_samples)
    raise ValueError (f"Unknown gamma mode: {mode}")

def simulate_varx_case(n_samples, y_to_x_lag1, y_to_x_lag2, random_seed):
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

    y = generate_varx_data(A_matrices=[A1, A2], B_matrices=[B0, B1, B2], u=u, noise_cov=noise_cov, random_seed=random_seed+10000)
    
    return y, u

def raw_p_value_from_deviance(deviance_raw, df):
    deviance_for_p = max(float(deviance_raw), 0.0)

    return float(chi2.sf(deviance_for_p, df=df))

def run_single_gc_pair(y, u, source, target, gamma):
    result = debiased_varx_gc(y=y, u=u, source=source, target=target, na=na, nb=nb, gamma=gamma, penalty="diag", include_intercept=False, effective_t_mode="full_minus_features")

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
        "mean_debiased_deviance": np.mean(dev_db),

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

def collect_results(n_samples, gamma_mode, case_name, y_to_x_lag1, y_to_x_lag2):
    gamma = gamma_from_mode(mode=gamma_mode, n_samples=n_samples)

    y_to_x_results = []
    x_to_y_results = []
    for run in range(N_RUNS):
        seed = 100000 + 1000 * T_LIST.index(n_samples) + 100 * GAMMA_MODES.index(gamma_mode) + run
        y, u = simulate_varx_case(n_samples=n_samples, y_to_x_lag1=y_to_x_lag1, y_to_x_lag2=y_to_x_lag2, random_seed=seed)
        
        result_y_to_x = run_single_gc_pair(y=y, u=u, source=1, target=0, gamma=gamma)
        result_x_to_y = run_single_gc_pair(y=y, u=u, source=0, target=1, gamma=gamma)

        y_to_x_results.append(result_y_to_x)
        x_to_y_results.append(result_x_to_y)

    return y_to_x_results, x_to_y_results

summary_rows = []

# Main loop
for n_samples in T_LIST:
    for gamma_mode in GAMMA_MODES:
        gamma = gamma_from_mode(mode=gamma_mode, n_samples=n_samples)
        print("\n"f"T = {n_samples} | " f"gamma_mode = {gamma_mode} | " f"gamma = {gamma}")

        # Case 1: null Y -> X
        null_y_to_x, null_x_to_y = collect_results(n_samples=n_samples, gamma_mode=gamma_mode, case_name="null", y_to_x_lag1=0.00, y_to_x_lag2=0.00)

        # Case 2: true Y -> X
        true_y_to_x, true_x_to_y = collect_results(n_samples=n_samples, gamma_mode=gamma_mode, case_name="true_link", y_to_x_lag1=0.25, y_to_x_lag2=0.12)

        groups = [("null", "Y->X", False, null_y_to_x),
                  ("null", "X->Y", False, null_x_to_y),
                  ("true_link", "Y->X", True, true_y_to_x),
                  ("true_link", "X->Y", False, true_x_to_y)
        ]

        for case_label, direction, expected_significant, results in groups:
            summary = summarize_result_list(results)

            row = {
                "T": n_samples,
                "gamma_mode": gamma_mode,
                "gamma": gamma,
                "case": case_label,
                "direction": direction,
                "expected_significant": expected_significant,
                **summary
            }

            summary_rows.append(row)

            print(
                f"\n{case_label:10s} | {direction:4s} | "
                f"expected={expected_significant}"
            )

            print(
                "raw detect:",
                summary["raw_detection_rate"],
                "| debiased detect:",
                summary["debiased_detection_rate"],
                "| mean raw GC:",
                summary["mean_raw_gc"],
                "| negative raw GC:",
                summary["negative_raw_gc_fraction"]
            )

# Summary table
summary_df = pd.DataFrame(summary_rows)
print("\nExperiment 28C summary table")

columns_to_show = [
    "T",
    "gamma_mode",
    "case",
    "direction",
    "expected_significant",
    "mean_raw_gc",
    "negative_raw_gc_fraction",
    "raw_detection_rate",
    "debiased_detection_rate",
    "median_debiased_p",
    "mean_debiased_deviance",
    "mean_bias_correction_minus_br_plus_bf"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save results
output_path = "results/experiment_28c_summary.csv"
summary_df.to_csv(output_path, index=False)
print("\nSaved summary to:")
print(output_path)

# Interpretation guide
print("\nInterpretation guide")
print(
    "\nFor null directions, debiased_detection_rate should be "
    "near or below alpha =",
    ALPHA
)

print(
    "\nFor true-link Y->X, debiased_detection_rate should be high."
)

print(
    "\nOLS should usually have nonnegative raw GC because the full model "
    "contains the reduced model."
)

print(
    "\nRidge may produce negative raw GC because the full and reduced "
    "models are penalized differently."
)

print(
    "\nThe trusted p-value under ridge is the de-biased p-value, "
    "not the raw chi-square p-value."
)