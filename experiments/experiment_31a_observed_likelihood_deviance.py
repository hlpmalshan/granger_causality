import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c_constrained import EMVARXPSSMKnownCConstrained
from src.stats.observed_likelihood_deviance import observed_likelihood_deviance
from src.stats.latent_debiased_deviance import debiased_latent_varx_gc_from_smoother

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

R_FLOOR = 0.30

print("\nExperiment 31A: Observed-data likelihood deviance")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("gamma_gc:", gamma_gc)
print("R floor:", R_FLOOR)

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
        "Q_true": Q,
        "R_true": R,
        "C": C
    }

def make_directional_zero_constraint(source, target):
    return [
        {
            "source": source,
            "target": target,
            "lags": "all"
        }
    ]

def fit_em_model(y_obs, u, C, zero_constraints=None, verbose=False):
    model = EMVARXPSSMKnownCConstrained(
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
        estimate_R=True,
        R_floor=R_FLOOR,
        zero_constraints=zero_constraints,
        verbose=verbose
    )

    model.fit(y=y_obs, u=u)
    return model

def latent_complete_data_gc_from_model(model, u, source, target):
    smoother = model.smooth_result["smoother"]

    return debiased_latent_varx_gc_from_smoother(
        smooth_mean=smoother["x_smooth"],
        smooth_cov=smoother["P_smooth"],
        smooth_lag_cov=smoother["P_lag_one"],
        u=u,
        source=source,
        target=target,
        na=na,
        nb=nb,
        n_latent=2,
        conditioning=None,
        gamma=gamma_gc,
        penalty="diag",
        effective_t_mode="full_minus_features"
    )

def print_model_summary(label, model):
    print(f"\n{label}")

    print("Final log-likelihood:")
    print(model.smooth_result["log_likelihood"])

    print("Spectral radius:")
    print(model.spectral_radius())

    print("EM iterations:")
    print(len(model.log_likelihoods))

    print("Estimated A1:")
    print(model.A_matrices[0])

    print("Estimated A2:")
    print(model.A_matrices[1])

    print("Estimated Q:")
    print(model.Q)

    print("Estimated R:")
    print(model.R)

def print_observed_likelihood_result(label, result):
    print(f"\n{label}")
    print("LL full:", result["ll_full"])
    print("LL reduced:", result["ll_reduced"])
    print("Observed-likelihood deviance:", result["observed_likelihood_deviance"])
    print("Deviance used for p:", result["deviance_for_p"])
    print("Chi-square p-value:", result["p_value"])
    print("df:", result["df"])

def print_latent_cd_result(label, result):
    print(f"\n{label}")
    print("Latent CD raw GC:", result["gc_raw"])
    print("Latent CD deviance:", result["deviance_debiased"])
    print("Latent CD chi-square p:", result["p_value"])
    print("df:", result["df"])

def run_case(
        case_name,
        y_to_x_lag1,
        y_to_x_lag2,
        random_seed
):
    print("\n"+case_name)
    
    data = simulate_case(
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        y_to_x_lag1=y_to_x_lag1,
        y_to_x_lag2=y_to_x_lag2,
        random_seed=random_seed
    )

    x_true = data["x_true"]
    y_obs = data["y_obs"]
    u = data["u"]
    C = data["C"]

    # Full model
    full_model = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        zero_constraints=[],
        verbose=True
    )

    x_full_smooth = full_model.smoothed_latent_state()

    full_mse = float(np.mean((x_full_smooth - x_true) ** 2))

    print_model_summary("Full observed-likelihood model", full_model)

    print("\nFull model smoothed-state MSE:")
    print(full_mse)

    # Direction Y -> X
    # source=1, target=0
    reduced_y_to_x = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        zero_constraints=make_directional_zero_constraint(source=1, target=0),
        verbose=False
    )

    obs_ll_y_to_x = observed_likelihood_deviance(
        full_model=full_model,
        reduced_model=reduced_y_to_x,
        df_removed=na,
        clip_negative=True
    )

    latent_cd_y_to_x = latent_complete_data_gc_from_model(
        model=full_model,
        u=u,
        source=1,
        target=0
    )

    print_model_summary("Reduced model for Y -> X null", reduced_y_to_x)
    print_observed_likelihood_result("Observed-likelihood GC: Y -> X", obs_ll_y_to_x)
    print_latent_cd_result("Latent complete-data GC from full model: Y -> X", latent_cd_y_to_x)

    # Direction X -> Y
    # source=0, target=1
    reduced_x_to_y = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        zero_constraints=make_directional_zero_constraint(source=0, target=1),
        verbose=False
    )

    obs_ll_x_to_y = observed_likelihood_deviance(
        full_model=full_model,
        reduced_model=reduced_x_to_y,
        df_removed=na,
        clip_negative=True
    )

    latent_cd_x_to_y = latent_complete_data_gc_from_model(
        model=full_model,
        u=u,
        source=0,
        target=1
    )

    print_model_summary("Reduced model for X -> Y null", reduced_x_to_y)
    print_observed_likelihood_result("Observed-likelihood GC: X -> Y", obs_ll_x_to_y)
    print_latent_cd_result("Latent complete-data GC from full model: X -> Y", latent_cd_x_to_y)

    return {
        "case_name": case_name,
        "data": data,
        "full_model": full_model,
        "reduced_y_to_x": reduced_y_to_x,
        "reduced_x_to_y": reduced_x_to_y,
        "full_mse": full_mse,
        "obs_ll_y_to_x": obs_ll_y_to_x,
        "obs_ll_x_to_y": obs_ll_x_to_y,
        "latent_cd_y_to_x": latent_cd_y_to_x,
        "latent_cd_x_to_y": latent_cd_x_to_y
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

print("\nExperiment 31A interpretation guide")

print(
    "\nObserved-likelihood deviance compares log p(y | full model) and "
    "log p(y | reduced model), rather than applying GC to posterior mean states."
)

print(
    "\nFor the true-link case, observed-likelihood Y->X deviance should be large."
)

print(
    "\nFor null directions, observed-likelihood deviance should ideally be small, "
    "but chi-square p-values should still be treated as provisional."
)

print(
    "\nIf the full model log-likelihood is lower than the reduced model, the "
    "observed-likelihood deviance can be negative because EM may converge to "
    "different local optima. In that case, it is clipped only for p-value computation."
)

print(
    "\nThe next step after this pilot is empirical calibration of observed-likelihood "
    "deviance, analogous to Experiments 30C-30F."
)