import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_order_selection import evaluate_varx_orders
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
N_RUNS = 100

TRUE_NA = 2
TRUE_NB = 3

NA_CANDIDATES = [1, 2, 3, 5]
FIT_NB = 3

ALPHA = 0.05

LAMBDA_REGULARIZATION = 1.0

print("\nExperiment 28E: VARX(p) AIC/BIC order selection")
print("N samples:", N_SAMPLES)
print("N runs:", N_RUNS)
print("True na:", TRUE_NA)
print("True nb:", TRUE_NB)
print("Candidate na:", NA_CANDIDATES)
print("Fit nb:", FIT_NB)
print("alpha:", ALPHA)


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

def run_gc_pair_with_selected_order(y, u, selected_na, gamma):
    y_to_x = debiased_varx_gc(
        y=y,
        u=u,
        source=1,
        target=0,
        na=selected_na,
        nb=FIT_NB,
        gamma=gamma,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    x_to_y = debiased_varx_gc(
        y=y,
        u=u,
        source=0,
        target=1,
        na=selected_na,
        nb=FIT_NB,
        gamma=gamma,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    y_to_x["p_value_raw_chi2"] = raw_p_value_from_deviance(y_to_x["deviance_raw"], y_to_x["df"])
    x_to_y["p_value_raw_chi2"] = raw_p_value_from_deviance(x_to_y["deviance_raw"], x_to_y["df"])

    return y_to_x, x_to_y

def summarize_gc_results(results):
    raw_gc = np.array([r["gc_raw"] for r in results])
    dev_db = np.array([r["deviance_debiased"] for r in results])
    p_db = np.array([r["p_value"] for r in results])
    selected_na = np.array([r["selected_na"] for r in results])

    return {
        "mean_selected_na": np.mean(selected_na),
        "fraction_selected_true_na": np.mean(selected_na == TRUE_NA),
        "mean_raw_gc": np.mean(raw_gc),
        "std_raw_gc": np.std(raw_gc),
        "negative_raw_gc_fraction": np.mean(raw_gc < 0),
        "mean_debiased_deviance": np.mean(dev_db),
        "std_debiased_deviance": np.std(dev_db),
        "median_debiased_p": np.median(p_db),
        "debiased_detection_rate": np.mean(p_db < ALPHA),
        "p05_debiased": np.percentile(p_db, 5),
        "p95_debiased": np.percentile(p_db, 95)
    }

def run_case(case_label, y_to_x_lag1, y_to_x_lag2, expected_y_to_x_significant):
    """
    For each run:

    1. Simulate data.
    2. Select order by AIC and BIC.
    3. Compute de-biased GC using selected order.
    """

    all_ic_rows = []

    aic_y_to_x_results = []
    aic_x_to_y_results = []

    bic_y_to_x_results = []
    bic_x_to_y_results = []

    for run in range(N_RUNS):

        seed = (300000 + 10000 * (case_label == "true_link") + run)

        y, u, A1, A2 = simulate_true_varx2_case(
            n_samples=N_SAMPLES,
            y_to_x_lag1=y_to_x_lag1,
            y_to_x_lag2=y_to_x_lag2,
            random_seed=seed
        )

        # AIC/BIC selection is done using OLS likelihood.
        # This is the standard interpretation of AIC/BIC.
        
        selection = evaluate_varx_orders(
            y=y,
            u=u,
            na_candidates=NA_CANDIDATES,
            nb=FIT_NB,
            gamma=0.0,
            penalty="diag",
            include_intercept=False,
            common_start_lag=True
        )

        best_aic = selection["best_aic"]
        best_bic = selection["best_bic"]

        for row in selection["rows"]:
            all_ic_rows.append({
                "case": case_label,
                "run": run,
                "na": row["na"],
                "aic": row["aic"],
                "bic": row["bic"],
                "logdet_sigma": row["logdet_sigma"],
                "n_parameters": row["n_parameters"],
                "T_eff": row["T_eff"],
                "best_aic": best_aic,
                "best_bic": best_bic
            })

        # Compute GC with selected orders.
        # For the GC test itself, we use ridge with the same lambda/sqrt(T) rule used in 28B/28C/28D.
        gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

        aic_y_to_x, aic_x_to_y = run_gc_pair_with_selected_order(
            y=y,
            u=u,
            selected_na=best_aic,
            gamma=gamma_gc
        )

        bic_y_to_x, bic_x_to_y = run_gc_pair_with_selected_order(
            y=y,
            u=u,
            selected_na=best_bic,
            gamma=gamma_gc
        )

        aic_y_to_x["selected_na"] = best_aic
        aic_x_to_y["selected_na"] = best_aic

        bic_y_to_x["selected_na"] = best_bic
        bic_x_to_y["selected_na"] = best_bic

        aic_y_to_x_results.append(aic_y_to_x)
        aic_x_to_y_results.append(aic_x_to_y)

        bic_y_to_x_results.append(bic_y_to_x)
        bic_x_to_y_results.append(bic_x_to_y)

        if (run + 1) % 10 == 0:
            print(
                f"{case_label}: completed {run + 1}/{N_RUNS}"
            )

    return {
        "ic_rows": all_ic_rows,
        "aic_y_to_x_results": aic_y_to_x_results,
        "aic_x_to_y_results": aic_x_to_y_results,
        "bic_y_to_x_results": bic_y_to_x_results,
        "bic_x_to_y_results": bic_x_to_y_results,
        "expected_y_to_x_significant": expected_y_to_x_significant
    }


# Run null and true-link cases
print("\nRunning null case...")
null_case = run_case(
    case_label="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False
)

print("\nRunning true-link case...")
true_case = run_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True
)

# Summaries
summary_rows = []

summary_specs = [
    (
        "null",
        "AIC",
        "Y->X",
        False,
        null_case["aic_y_to_x_results"]
    ),
    (
        "null",
        "AIC",
        "X->Y",
        False,
        null_case["aic_x_to_y_results"]
    ),
    (
        "null",
        "BIC",
        "Y->X",
        False,
        null_case["bic_y_to_x_results"]
    ),
    (
        "null",
        "BIC",
        "X->Y",
        False,
        null_case["bic_x_to_y_results"]
    ),
    (
        "true_link",
        "AIC",
        "Y->X",
        True,
        true_case["aic_y_to_x_results"]
    ),
    (
        "true_link",
        "AIC",
        "X->Y",
        False,
        true_case["aic_x_to_y_results"]
    ),
    (
        "true_link",
        "BIC",
        "Y->X",
        True,
        true_case["bic_y_to_x_results"]
    ),
    (
        "true_link",
        "BIC",
        "X->Y",
        False,
        true_case["bic_x_to_y_results"]
    )
]

for case_label, criterion, direction, expected_significant, results in summary_specs:

    summary = summarize_gc_results(
        results
    )

    row = {
        "case": case_label,
        "criterion": criterion,
        "direction": direction,
        "expected_significant": expected_significant,
        **summary
    }

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)

ic_df = pd.DataFrame(
    null_case["ic_rows"]
    + true_case["ic_rows"]
)

# Selection frequencies
selection_rows = []

for case_label in ["null", "true_link"]:

    case_ic = ic_df[
        ic_df["case"] == case_label
    ]

    # one row per run is enough because best_aic/best_bic
    # are repeated across candidate na values
    run_level = case_ic.drop_duplicates(
        subset=["case", "run"]
    )

    for criterion_col, criterion_name in [
        ("best_aic", "AIC"),
        ("best_bic", "BIC")
    ]:

        counts = (
            run_level[criterion_col]
            .value_counts()
            .sort_index()
        )

        for na in NA_CANDIDATES:

            selection_rows.append({
                "case": case_label,
                "criterion": criterion_name,
                "selected_na": na,
                "count": int(counts.get(na, 0)),
                "fraction": float(counts.get(na, 0) / N_RUNS)
            })

selection_df = pd.DataFrame(selection_rows)

# Print outputs

print("\nOrder selection frequencies")
print(selection_df.to_string(index=False))

print("\nGC performance using selected order")
columns_to_show = [
    "case",
    "criterion",
    "direction",
    "expected_significant",
    "mean_selected_na",
    "fraction_selected_true_na",
    "mean_raw_gc",
    "negative_raw_gc_fraction",
    "debiased_detection_rate",
    "median_debiased_p",
    "mean_debiased_deviance"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save files
os.makedirs("results", exist_ok=True)

summary_path = "results/experiment_28e_summary.csv"
selection_path = "results/experiment_28e_selection_frequencies.csv"
ic_path = "results/experiment_28e_ic_by_run.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

selection_df.to_csv(
    selection_path,
    index=False
)

ic_df.to_csv(
    ic_path,
    index=False
)

print("\nSaved summary to:")
print(summary_path)

print("\nSaved selection frequencies to:")
print(selection_path)

print("\nSaved IC-by-run table to:")
print(ic_path)


# Interpretation guide
print("Interpretation guide")
print(
    "\nThe true order is na = 2. Good order selection should often choose 2."
)

print(
    "\nAIC may select na > 2 because it penalizes extra parameters less strongly."
)

print(
    "\nBIC may be more conservative and should often prefer simpler models."
)

print(
    "\nIf selected-order GC keeps true Y->X power high and reverse/null false "
    "positive rates near alpha, then this is enough for our direct VARX baseline."
)

print(
    "\nAfter this, we should move to latent state-space VARX(p), not spend too "
    "much more time fine-tuning direct-observation VARX."
)