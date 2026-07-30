import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data, var_companion_spectral_radius
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

LAMBDA_REGULARIZATION = 1.0
gamma = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 29A: Latent state-space VARX(p) simulator")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("gamma:", gamma)


def print_gc_result(name, result):
    print(f"\n{name}")
    print("Raw GC:", result["gc_raw"])
    print("De-biased deviance:", result["deviance_debiased"])
    print("p-value:", result["p_value"])
    print("df:", result["df"])
    print("sigma_full:", result["sigma_full"])
    print("sigma_reduced:", result["sigma_reduced"])
    print("bias_full:", result["bias_full"])
    print("bias_reduced:", result["bias_reduced"])


def run_case(case_name, y_to_x_lag1, y_to_x_lag2, random_seed):
    print("\n" + case_name)
    # Generate exogenous input with burn-in included.
    u_total = generate_colored_input(
        n_samples=N_SAMPLES + BURN_IN,
        ar_coeff=0.95,
        noise_std=1.0,
        random_seed=random_seed
    )

    # Possible true endogenous link: Y -> X appears through:
    # A1[0, 1] and A2[0, 1]
    A1 = np.array([
        [0.55, y_to_x_lag1],
        [0.00, 0.50]
    ])

    A2 = np.array([
        [-0.10, y_to_x_lag2],
        [0.00, -0.08]
    ])

    # Exogenous input design
    # B0: u_t drives Y immediately.
    # B1: no lag-1 direct exogenous effect.
    # B2: u_{t-2} drives X after two samples.
    # This creates a common-input temporal confound.
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

    # Latent process noise
    Q = 0.50 * np.eye(2)

    # Observation noise
    R = 0.60 * np.eye(2)

    # Observation/mixing matrix.
    # This mimics MEG-like spatial mixing in a small toy system.
    C = np.array([
        [1.00, 0.35],
        [0.25, 1.00]
    ])

    result = generate_ssm_varx_p_data(
        A_matrices=[A1, A2],
        B_matrices=[B0, B1, B2],
        u=u_total,
        Q=Q,
        R=R,
        C=C,
        D=None,
        burn_in=BURN_IN,
        random_seed=random_seed + 10000,
        return_augmented=True
    )

    x_true = result["x"]
    y_obs = result["y"]
    u = result["u"]

    # Crude proxy source estimate using known C.
    # This is a two-stage baseline, not the final method.
    C_pinv = np.linalg.pinv(C)
    x_proxy = y_obs @ C_pinv.T

    proxy_mse = np.mean((x_proxy - x_true) ** 2)

    print("\nTrue A1")
    print(A1)

    print("\nTrue A2")
    print(A2)

    print("\nObservation matrix C")
    print(C)

    print("\nVAR companion spectral radius:")
    print(result["spectral_radius"])

    print("Stable:")
    print(result["stable"])

    print("\nShapes")
    print("x_true:", x_true.shape)
    print("y_obs:", y_obs.shape)
    print("u:", u.shape)

    print("\nProxy MSE after C pseudo-inverse:")
    print(proxy_mse)

    # GC tests
    # 1. Latent x_true:
    #    Oracle benchmark. Available only in simulation.
    # 2. Observed y_obs:
    #    Naive sensor-space VARX.
    # 3. Proxy x_proxy:
    
    latent_y_to_x = debiased_varx_gc(
        y=x_true,
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

    latent_x_to_y = debiased_varx_gc(
        y=x_true,
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

    observed_y_to_x = debiased_varx_gc(
        y=y_obs,
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

    observed_x_to_y = debiased_varx_gc(
        y=y_obs,
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

    proxy_y_to_x = debiased_varx_gc(
        y=x_proxy,
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

    proxy_x_to_y = debiased_varx_gc(
        y=x_proxy,
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

    print_gc_result("Oracle latent x: Y -> X", latent_y_to_x)
    print_gc_result("Oracle latent x: X -> Y", latent_x_to_y)
    print_gc_result("Observed mixed y: Y -> X", observed_y_to_x)
    print_gc_result("Observed mixed y: X -> Y", observed_x_to_y)
    print_gc_result("Pseudo-inverse proxy x: Y -> X", proxy_y_to_x)
    print_gc_result("Pseudo-inverse proxy x: X -> Y", proxy_x_to_y)

    return {
        "A1": A1,
        "A2": A2,
        "B0": B0,
        "B1": B1,
        "B2": B2,
        "Q": Q,
        "R": R,
        "C": C,
        "x_true": x_true,
        "y_obs": y_obs,
        "u": u,
        "x_proxy": x_proxy,
        "proxy_mse": proxy_mse,
        "latent_y_to_x": latent_y_to_x,
        "latent_x_to_y": latent_x_to_y,
        "observed_y_to_x": observed_y_to_x,
        "observed_x_to_y": observed_x_to_y,
        "proxy_y_to_x": proxy_y_to_x,
        "proxy_x_to_y": proxy_x_to_y
    }


# Case 1: no true endogenous Y -> X
case_1 = run_case(
    case_name="Case 1: No true endogenous Y -> X",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    random_seed=1
)
# Case 2: true endogenous Y -> X
case_2 = run_case(
    case_name="Case 2: True endogenous Y -> X",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    random_seed=2
)


print("Experiment 29A interpretation guide")
print(
    "\nThe oracle latent x result is the simulation benchmark."
)

print(
    "\nThe observed mixed y result shows what happens if we naively run VARX "
    "on sensor-space observations."
)

print(
    "\nThe pseudo-inverse proxy result is a simple two-stage baseline."
)

print(
    "\nIf observed/proxy GC differs from latent GC, that motivates the next "
    "step: Kalman smoothing and EM-based latent VARX(p) estimation."
)