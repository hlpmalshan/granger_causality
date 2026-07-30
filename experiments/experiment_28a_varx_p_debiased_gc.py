import numpy as np

from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_gc_debiased import debiased_varx_gc

# Experiment 28A: Directly observed VARX(p) with Eq. 18-style de-biased deviance.

N_SAMPLES = 1000
na = 2
nb = 3

# The VARX supplement suggests scaling regularization down as sample size increases.
# Here we start with lambda = 1.
lambda_regularization = 1.0
gamma = lambda_regularization / np.sqrt(N_SAMPLES)

print("N samples:", N_SAMPLES)
print("na:", na)
print("nb:", nb)
print("lambda:", lambda_regularization)
print("gamma:", gamma)

def print_gc_result(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print("Raw GC:", result["gc_raw"])
    print("Raw deviance:", result["deviance_raw"])
    print("De-biased deviance:", result["deviance_debiased"])
    print("De-biased devianced used for p-value:", result["deviance_for_p"])
    print("p-value:", result["p_value"])
    print("df:", result["df"])
    print("T:", result["T"])
    print("T_eff:", result["T_eff"])
    print("sigma_full:", result["sigma_full"])
    print("sigma_reduced:", result["sigma_reduced"])
    print("bias_full:", result["bias_full"])
    print("bias_reduced:", result["bias_reduced"])

def run_case(case_name, y_to_x_lag1, y_to_x_lag2, random_seed):
    print(case_name)

    # Exogenous input
    # u is colored. This make it a relaistic confound: past u contains predictive temporal structure
    u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=random_seed)

    # Directly observed VARX(2)
    # y[:, 0] = X
    # y[:, 1] = Y
    # The possible endogenous link being tested is: Y -> X
    # That link appears in: A1[0, 1], A2[0, 1]
    A1 = np.array([
        [0.55, y_to_x_lag1],
        [0.00, 0.50]
    ])

    A2 = np.array([
        [-0.10, y_to_x_lag2],
        [0.00, -0.08]
    ])

    # Exogenous input drives both channels with different delays
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
    y = generate_varx_data(A_matrices=[A1, A2], B_matrices=[B0, B1, B2], u=u, noise_cov=noise_cov, random_seed=random_seed+100)

    print("\nTrue A1")
    print(A1)

    print("\nTrue A2")
    print(A2)

    # Bias-corrected VARX GC tests
    result_y_to_x = debiased_varx_gc(
        y=y, u=u, source=1, target=0, na=na, nb=nb, gamma=gamma, penalty="diag", include_intercept=False, effective_t_mode="full_minus_features"
    )
    result_x_to_y = debiased_varx_gc(
        y=y, u=u, source=0, target=1, na=na, nb=nb, gamma=gamma, penalty="diag", include_intercept=False, effective_t_mode="full_minus_features"
    )
    print_gc_result("Y -> X", result_y_to_x)
    print_gc_result("X -> Y", result_x_to_y)

    return {
        "y": y,
        "u": u,
        "A1": A1,
        "A2": A2,
        "result_y_to_x": result_y_to_x,
        "result_x_to_y": result_x_to_y
    }

# Case 1: no true endogenous Y -> X 
case_1 = run_case(case_name="Case 1: No true endogenous Y -> X", y_to_x_lag1=0.00, y_to_x_lag2=0.00, random_seed=1)

# Case 2: true endogenous Y -> X
case_2 = run_case(case_name="Case 2: True endogenous Y -> X", y_to_x_lag1=0.25, y_to_x_lag2=0.12, random_seed=2)