import numpy as np

from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
N_RUNS = 100

na = 2
nb = 3

ALPHA = 0.05

lambda_regularization = 1.0
gamma_ridge = lambda_regularization / np.sqrt(N_SAMPLES)

print("\nExperiment 28B: Monte Carlo short-signal VARX(p)")
print("N samples:", N_SAMPLES)
print("N runs:", N_RUNS)
print("na:", na)
print("nb:", nb)
print("lambda:", lambda_regularization)
print("gamma ridge:", gamma_ridge)
print("alpha:", ALPHA)

# Simulate directly observed VARX(2)
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

def run_gc_tests(y, u, gamma):
    y_to_x = debiased_varx_gc(
        y=y,
        u=u,
        source=1,
        target=0,
        na=na,
        nb=nb,
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
        na=na,
        nb=nb,
        gamma=gamma,
        penalty="diag",
        include_intercept=False,
        effective_t_mode="full_minus_features"
    )

    return y_to_x, x_to_y

def collect_results_for_case(case_name, y_to_x_lag1, y_to_x_lag2, gamma):
    print("\n"+case_name)

    results_y_to_x = []
    results_x_to_y = []

    for run in range(N_RUNS):
        seed = 1000 + run
        y, u = simulate_varx_case(n_samples=N_SAMPLES, y_to_x_lag1=y_to_x_lag1, y_to_x_lag2=y_to_x_lag2, random_seed=seed)

        result_y_to_x, result_x_to_y = run_gc_tests(y=y, u=u, gamma=gamma)

        results_y_to_x.append(result_y_to_x)
        results_x_to_y.append(result_x_to_y)

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{N_RUNS}")

    return results_y_to_x, results_x_to_y

def summarize_results(label, results, expected_significant):
    raw_gc = np.array([r["gc_raw"] for r in results])
    debiased_dev = np.array([r["deviance_debiased"] for r in results])
    dev_for_p = np.array([r["deviance_for_p"] for r in results])
    p_values = np.array([r["p_value"] for r in results])

    detections = p_values < ALPHA
    negative_raw_gc_fraction = np.mean(raw_gc<0)
    detection_rate = np.mean(detections)

    print("\n" + label)
    print("-" * len(label))

    print("Expected significant:", expected_significant)

    print("Mean raw GC:", np.mean(raw_gc))
    print("Std raw GC:", np.std(raw_gc))

    print("Mean de-biased deviance:", np.mean(debiased_dev))
    print("Std de-biased deviance:", np.std(debiased_dev))

    print("Mean deviance used for p:", np.mean(dev_for_p))

    print("Median p-value:", np.median(p_values))
    print("5th percentile p-value:", np.percentile(p_values, 5))
    print("95th percentile p-value:", np.percentile(p_values, 95))

    print("Detection rate p < alpha:", detection_rate)
    print("Negative raw GC fraction:", negative_raw_gc_fraction)

    return {
        "raw_gc": raw_gc,
        "debiased_dev": debiased_dev,
        "dev_for_p": dev_for_p,
        "p_values": p_values,
        "detections": detections,
        "detection_rate": detection_rate,
        "negative_raw_gc_fraction": negative_raw_gc_fraction
    }

# Case 1: null Y -> X
null_y_to_x_results, null_x_to_y_results = collect_results_for_case(
    case_name="Case 1: No true endogenous Y -> X",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    gamma=gamma_ridge
)    

# Case 2: true Y -> X
true_y_to_x_results, true_x_to_y_results = collect_results_for_case(
    case_name="Case 2: True endogenous Y -> X",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    gamma=gamma_ridge
)

# Summaries
print("\n"+"Monte Carlo summary")

summary_null_y_to_x = summarize_results(
    label="Null case: Y -> X",
    results=null_y_to_x_results,
    expected_significant=False
)

summary_null_x_to_y = summarize_results(
    label="Null case: X -> Y",
    results=null_x_to_y_results,
    expected_significant=False
)

summary_true_y_to_x = summarize_results(
    label="True-link case: Y -> X",
    results=true_y_to_x_results,
    expected_significant=True
)

summary_true_x_to_y = summarize_results(
    label="True-link case: X -> Y",
    results=true_x_to_y_results,
    expected_significant=False
)

print("\nInterpretation guide")
print("\nFalse positive rate for null Y->X should be near or below alpha:")
print(summary_null_y_to_x["detection_rate"])

print("\nPower for true Y->X should be high:")
print(summary_true_y_to_x["detection_rate"])

print("\nReverse direction X->Y should usually not be significant:")
print(summary_true_x_to_y["detection_rate"])

print("\nNegative raw GC under ridge is allowed, especially under null.")
print("The p-value is computed from max(D_debiased, 0).")