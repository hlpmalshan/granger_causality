import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.kalman_varx_p import kalman_smooth_varx_p_companion, extract_current_latent_state
from src.varx.varx_gc_debiased import debiased_varx_gc

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

LAMBDA_REGULARIZATION = 1.0
gamma = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 29B: Known-parameter Kalman smoothing for latent VARX(p)")
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


def compute_gc_pair(label, y_like, u):
    y_to_x = debiased_varx_gc(
        y=y_like,
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
        y=y_like,
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

    print_gc_result(f"{label}: Y -> X", y_to_x)
    print_gc_result(f"{label}: X -> Y", x_to_y)

    return y_to_x, x_to_y


def run_case(case_name, y_to_x_lag1, y_to_x_lag2, random_seed):
    print("\n"+case_name)
    
    u_total = generate_colored_input(n_samples=N_SAMPLES + BURN_IN, ar_coeff=0.95, noise_std=1.0, random_seed=random_seed)

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

    # Known-parameter Kalman smoothing.
    # We use the true F, G, Q_aug, C_aug, R.
    # This tests whether the state-space representation can recover latent x when parameters are known.
    
    smooth_result = kalman_smooth_varx_p_companion(
        y=y_obs,
        u=u,
        F=sim["F"],
        G=sim["G"],
        Q_aug=sim["Q_aug"],
        R=sim["R"],
        C_aug=sim["C_aug"],
        nb=nb,
        D=sim["D"],
        s0_mean=None,
        s0_cov=None
    )

    s_smooth = smooth_result["smoother"]["x_smooth"]

    x_smooth = extract_current_latent_state(companion_states=s_smooth, n_latent=2)

    proxy_mse = np.mean((x_proxy - x_true) ** 2)
    smooth_mse = np.mean((x_smooth - x_true) ** 2)
    obs_mse = np.mean((y_obs - x_true) ** 2)

    print("\nTrue A1")
    print(A1)

    print("\nTrue A2")
    print(A2)

    print("\nObservation matrix C")
    print(C)

    print("\nStable:", sim["stable"])
    print("Spectral radius:", sim["spectral_radius"])

    print("\nLog-likelihood from Kalman filter:")
    print(smooth_result["log_likelihood"])

    print("\nMSE comparisons")
    print("Observed y vs true x MSE:", obs_mse)
    print("Pseudo-inverse proxy vs true x MSE:", proxy_mse)
    print("Kalman smoothed x vs true x MSE:", smooth_mse)

    # GC comparisons
    latent_y_to_x, latent_x_to_y = compute_gc_pair(label="Oracle latent x", y_like=x_true,u=u)
    observed_y_to_x, observed_x_to_y = compute_gc_pair(label="Observed mixed y", y_like=y_obs, u=u)
    proxy_y_to_x, proxy_x_to_y = compute_gc_pair(label="Pseudo-inverse proxy x", y_like=x_proxy, u=u)
    smooth_y_to_x, smooth_x_to_y = compute_gc_pair(label="Kalman smoothed x", y_like=x_smooth, u=u)

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
        "x_smooth": x_smooth,
        "proxy_mse": proxy_mse,
        "smooth_mse": smooth_mse,
        "obs_mse": obs_mse,
        "log_likelihood": smooth_result["log_likelihood"],
        "latent_y_to_x": latent_y_to_x,
        "latent_x_to_y": latent_x_to_y,
        "observed_y_to_x": observed_y_to_x,
        "observed_x_to_y": observed_x_to_y,
        "proxy_y_to_x": proxy_y_to_x,
        "proxy_x_to_y": proxy_x_to_y,
        "smooth_y_to_x": smooth_y_to_x,
        "smooth_x_to_y": smooth_x_to_y
    }


case_1 = run_case(
    case_name="Case 1: No true endogenous Y -> X",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    random_seed=1
)

case_2 = run_case(
    case_name="Case 2: True endogenous Y -> X",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    random_seed=2
)


print("\n" + "=" * 80)
print("Experiment 29B interpretation guide")
print("=" * 80)

print(
    "\nThe key reconstruction metric is whether Kalman smoothed MSE is lower "
    "than pseudo-inverse proxy MSE."
)

print(
    "\nGC on Kalman smoothed x is diagnostic only. It is still a sample-based "
    "regression on posterior means, not the final latent complete-data "
    "de-biased deviance."
)

print(
    "\nIf smoothing improves MSE but GC is still distorted, that is expected "
    "and motivates using posterior sufficient statistics in Experiment 30A."
)