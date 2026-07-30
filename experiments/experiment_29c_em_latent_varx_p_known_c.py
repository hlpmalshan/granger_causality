import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c import EMVARXPSSMKnownC
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 29C: EM estimation for latent VARX(p), known C")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("gamma_gc:", gamma_gc)


def print_gc_result(name, result):
    print(f"\n{name}")
    print("Raw GC:", result["gc_raw"])
    print("De-biased deviance:", result["deviance_debiased"])
    print("p-value:", result["p_value"])
    print("df:", result["df"])
    print("sigma_full:", result["sigma_full"])
    print("sigma_reduced:", result["sigma_reduced"])


def compute_gc_pair(label, y_like, u):
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

    print_gc_result(f"{label}: Y -> X", y_to_x)
    print_gc_result(f"{label}: X -> Y", x_to_y)

    return y_to_x, x_to_y


def print_parameter_comparison(true_A, est_A, name):
    print(f"\n{name}")
    print("True:")
    print(true_A)
    print("Estimated:")
    print(est_A)
    print("Error:")
    print(est_A - true_A)


def run_case(case_name, y_to_x_lag1, y_to_x_lag2, random_seed):
    print(case_name)
    
    u_total = generate_colored_input(
        n_samples=N_SAMPLES + BURN_IN,
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
        burn_in=BURN_IN,
        random_seed=random_seed + 10000,
        return_augmented=True
    )

    x_true = sim["x"]
    y_obs = sim["y"]
    u = sim["u"]

    C_pinv = np.linalg.pinv(C)
    x_proxy = y_obs @ C_pinv.T

    model = EMVARXPSSMKnownC(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=50,
        tol=1e-4,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_init=0.60 * np.eye(2),
        Q_init=0.50 * np.eye(2),
        estimate_R=False,
        R_floor=0.05,
        verbose=True
    )

    model.fit(y=y_obs, u=u)
    x_em_smooth = model.smoothed_latent_state()

    obs_mse = np.mean((y_obs - x_true) ** 2)
    proxy_mse = np.mean((x_proxy - x_true) ** 2)
    em_smooth_mse = np.mean((x_em_smooth - x_true) ** 2)

    print("\nTrue A1")
    print(A1)

    print("\nTrue A2")
    print(A2)

    print("\nEstimated A1")
    print(model.A_matrices[0])

    print("\nEstimated A2")
    print(model.A_matrices[1])

    print("\nEstimated B0")
    print(model.B_matrices[0])

    print("\nEstimated B1")
    print(model.B_matrices[1])

    print("\nEstimated B2")
    print(model.B_matrices[2])

    print("\nTrue Q")
    print(Q)

    print("\nEstimated Q")
    print(model.Q)

    print("\nTrue R")
    print(R)

    print("\nEstimated R")
    print(model.R)

    print("\nFinal spectral radius")
    print(model.spectral_radius())

    print("\nLog-likelihoods")
    print(model.log_likelihoods)

    print("\nMSE comparisons")
    print("Observed y vs true x MSE:", obs_mse)
    print("Pseudo-inverse proxy vs true x MSE:", proxy_mse)
    print("EM smoothed x vs true x MSE:", em_smooth_mse)

    print_parameter_comparison(true_A=A1, est_A=model.A_matrices[0], name="A1 comparison")
    print_parameter_comparison(true_A=A2, est_A=model.A_matrices[1], name="A2 comparison")

    latent_y_to_x, latent_x_to_y = compute_gc_pair(label="Oracle latent x", y_like=x_true, u=u)
    observed_y_to_x, observed_x_to_y = compute_gc_pair(label="Observed mixed y", y_like=y_obs, u=u)
    proxy_y_to_x, proxy_x_to_y = compute_gc_pair(label="Pseudo-inverse proxy x", y_like=x_proxy, u=u)
    em_y_to_x, em_x_to_y = compute_gc_pair(label="EM smoothed x", y_like=x_em_smooth, u=u)

    return {
        "A1_true": A1,
        "A2_true": A2,
        "B0_true": B0,
        "B1_true": B1,
        "B2_true": B2,
        "Q_true": Q,
        "R_true": R,
        "C": C,
        "x_true": x_true,
        "y_obs": y_obs,
        "u": u,
        "x_proxy": x_proxy,
        "x_em_smooth": x_em_smooth,
        "model": model,
        "obs_mse": obs_mse,
        "proxy_mse": proxy_mse,
        "em_smooth_mse": em_smooth_mse,
        "latent_y_to_x": latent_y_to_x,
        "latent_x_to_y": latent_x_to_y,
        "observed_y_to_x": observed_y_to_x,
        "observed_x_to_y": observed_x_to_y,
        "proxy_y_to_x": proxy_y_to_x,
        "proxy_x_to_y": proxy_x_to_y,
        "em_y_to_x": em_y_to_x,
        "em_x_to_y": em_x_to_y
    }


case_1 = run_case(case_name="Case 1: No true endogenous Y -> X", y_to_x_lag1=0.00, y_to_x_lag2=0.00, random_seed=1)
case_2 = run_case(case_name="Case 2: True endogenous Y -> X", y_to_x_lag1=0.25, y_to_x_lag2=0.12, random_seed=2)

print("Experiment 29C interpretation guide")

print(
    "\nThe main question is whether EM-smoothed MSE improves over the "
    "pseudo-inverse proxy MSE."
)

print(
    "\nParameter recovery should be judged mainly by A1[0,1], A2[0,1], "
    "A1[1,0], and A2[1,0]."
)

print(
    "\nIn the true-link case, estimated A1[0,1] and A2[0,1] should be positive."
)

print(
    "\nIn the null case, estimated cross-lag coefficients should ideally be close "
    "to zero, but leakage may remain because this is short, noisy latent data."
)

print(
    "\nGC on EM smoothed posterior means is still diagnostic only. The final "
    "method should use posterior sufficient statistics for the latent "
    "complete-data de-biased deviance."
)