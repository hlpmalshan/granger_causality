import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c_warmstart import EMVARXPSSMKnownCConstrainedWarmStart, extract_model_parameters
from src.stats.observed_likelihood_deviance import observed_likelihood_deviance, final_observed_log_likelihood
from src.stats.latent_debiased_deviance import debiased_latent_varx_gc_from_smoother

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 80
TOL = 1e-6

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 31B: Warm-started observed-data likelihood deviance")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("gamma_gc:", gamma_gc)

# Simulate latent VARX(2) state-space data.
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

def fit_em_model(
        y_obs,
        u,
        C,
        zero_constraints=None,
        initial_parameters=None,
        label="model",
        verbose=False
):
    if zero_constraints is None:
        zero_constraints = []

    model = EMVARXPSSMKnownCConstrainedWarmStart(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=MAX_ITER,
        tol=TOL,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_init=0.60 * np.eye(2),
        Q_init=0.50 * np.eye(2),
        estimate_R=True,
        R_floor=R_FLOOR,
        zero_constraints=zero_constraints,
        initial_parameters=initial_parameters,
        jitter_scale=0.0,
        random_seed=None,
        verbose=verbose
    )

    model.fit(y=y_obs, u=u)
    model.fit_label = label

    return model

def model_ll(model):
    return final_observed_log_likelihood(model)

# Choose the fitted model with the highest observed-data log-likelihood.
def choose_best_model(models):
    if len(models) < 1:
        raise ValueError("models must contain at least one fitted model.")

    best = max(models, key=model_ll)
    return best

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

# Fit full/reduced models with warm starts.
# Strategy:
#     1. Fit full model from proxy.
#     2. For each direction:
#         a. Fit reduced from proxy.
#         b. Fit reduced from full-proxy initialization.
#         c. Pick best reduced.
#         d. Fit full from best reduced initialization.
#     3. Pick one global best full model among:
#         a. full from proxy
#         b. full from reduced Y->X
#         c. full from reduced X->Y
def fit_warmstarted_case_models(y_obs, u, C):
    direction_specs = [
        {
            "direction": "Y->X",
            "source": 1,
            "target": 0,
            "expected_significant": None
        },
        {
            "direction": "X->Y",
            "source": 0,
            "target": 1,
            "expected_significant": False
        }
    ]

    full_candidates = []
    print("\nFitting full model from proxy initialization...")

    full_proxy = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        zero_constraints=[],
        initial_parameters=None,
        label="full_proxy",
        verbose=True
    )

    full_candidates.append(full_proxy)
    reduced_results = {}
    for spec in direction_specs:
        direction = spec["direction"]
        source = spec["source"]
        target = spec["target"]

        print(f"Fitting reduced model for {direction}")

        zero_constraints = make_directional_zero_constraint(source=source, target=target)

        reduced_proxy = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=zero_constraints,
            initial_parameters=None,
            label=f"reduced_{direction}_proxy",
            verbose=False
        )

        reduced_from_full = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=zero_constraints,
            initial_parameters=extract_model_parameters(full_proxy),
            label=f"reduced_{direction}_from_full_proxy",
            verbose=False
        )

        best_reduced = choose_best_model([reduced_proxy, reduced_from_full])

        print(f"Reduced {direction} candidates:")
        print("  proxy LL:", model_ll(reduced_proxy))
        print("  from full LL:", model_ll(reduced_from_full))
        print("  best:", best_reduced.fit_label, model_ll(best_reduced))

        full_from_reduced = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            zero_constraints=[],
            initial_parameters=extract_model_parameters(best_reduced),
            label=f"full_from_best_reduced_{direction}",
            verbose=False
        )

        print("Full warm-started from best reduced:")
        print("  label:", full_from_reduced.fit_label)
        print("  LL:", model_ll(full_from_reduced))

        full_candidates.append(full_from_reduced)

        reduced_results[direction] = {
            "spec": spec,
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
            "best_reduced": best_reduced,
            "full_from_reduced": full_from_reduced
        }

    best_full = choose_best_model(full_candidates)

    print("\nFull model candidates")

    for candidate in full_candidates:
        print(candidate.fit_label, "LL =", model_ll(candidate))

    print("Best full model:", best_full.fit_label)
    print("Best full LL:", model_ll(best_full))

    return {
        "best_full": best_full,
        "full_candidates": full_candidates,
        "reduced_results": reduced_results
    }

def collect_direction_result(
        case_name,
        direction,
        source,
        target,
        expected_significant,
        best_full,
        best_reduced,
        full_mse,
        data_u
):
    obs_result = observed_likelihood_deviance(
        full_model=best_full,
        reduced_model=best_reduced,
        df_removed=na,
        clip_negative=True
    )

    latent_cd_result = latent_complete_data_gc_from_model(
        model=best_full,
        u=data_u,
        source=source,
        target=target
    )

    ll_gap = (obs_result["ll_full"] - obs_result["ll_reduced"])

    return {
        "case": case_name,
        "direction": direction,
        "source": source,
        "target": target,
        "expected_significant": expected_significant,

        "full_model_label": best_full.fit_label,
        "reduced_model_label": best_reduced.fit_label,

        "ll_full": obs_result["ll_full"],
        "ll_reduced": obs_result["ll_reduced"],
        "ll_gap_full_minus_reduced": ll_gap,

        "observed_likelihood_deviance": obs_result["observed_likelihood_deviance"],
        "observed_deviance_for_p": obs_result["deviance_for_p"],
        "observed_likelihood_p_value": obs_result["p_value"],
        "observed_likelihood_detected": obs_result["p_value"] < ALPHA,

        "latent_cd_raw_gc": latent_cd_result["gc_raw"],
        "latent_cd_debiased_deviance": latent_cd_result["deviance_debiased"],
        "latent_cd_p_value": latent_cd_result["p_value"],
        "latent_cd_detected": latent_cd_result["p_value"] < ALPHA,

        "df": na,

        "full_mse": full_mse,
        "full_spectral_radius": best_full.spectral_radius(),
        "full_em_iterations": len(best_full.log_likelihoods),
        "reduced_spectral_radius": best_reduced.spectral_radius(),
        "reduced_em_iterations": len(best_reduced.log_likelihoods),

        "full_A1_01": best_full.A_matrices[0][0, 1],
        "full_A2_01": best_full.A_matrices[1][0, 1],
        "full_A1_10": best_full.A_matrices[0][1, 0],
        "full_A2_10": best_full.A_matrices[1][1, 0],

        "full_R00": best_full.R[0, 0],
        "full_R11": best_full.R[1, 1],
        "full_R01": best_full.R[0, 1]
    }

def print_case_model_summary(label, model):
    print("\n" + label)
    print("Label:", model.fit_label)
    print("LL:", model_ll(model))
    print("Spectral radius:", model.spectral_radius())
    print("EM iterations:", len(model.log_likelihoods))

    print("A1:")
    print(model.A_matrices[0])

    print("A2:")
    print(model.A_matrices[1])

    print("Q:")
    print(model.Q)

    print("R:")
    print(model.R)


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

    fitted = fit_warmstarted_case_models(y_obs=y_obs, u=u, C=C)
    best_full = fitted["best_full"]
    x_full_smooth = best_full.smoothed_latent_state()

    full_mse = float(np.mean((x_full_smooth - x_true) ** 2))

    print_case_model_summary("Best full model", best_full)

    print("Best full smoothed-state MSE:", full_mse)

    rows = []

    direction_info = {
        "Y->X": {
            "source": 1,
            "target": 0,
            "expected_significant": y_to_x_lag1 != 0.0 or y_to_x_lag2 != 0.0
        },
        "X->Y": {
            "source": 0,
            "target": 1,
            "expected_significant": False
        }
    }

    for direction, info in direction_info.items():
        best_reduced = fitted["reduced_results"][direction]["best_reduced"]

        print_case_model_summary(f"Best reduced model for {direction}", best_reduced)

        row = collect_direction_result(
            case_name=case_name,
            direction=direction,
            source=info["source"],
            target=info["target"],
            expected_significant=info["expected_significant"],
            best_full=best_full,
            best_reduced=best_reduced,
            full_mse=full_mse,
            data_u=u
        )

        rows.append(row)

        print("\nObserved-likelihood result:", case_name, direction)
        print("LL full:", row["ll_full"])
        print("LL reduced:", row["ll_reduced"])
        print("LL gap:", row["ll_gap_full_minus_reduced"])
        print("Observed deviance:", row["observed_likelihood_deviance"])
        print("Observed p:", row["observed_likelihood_p_value"])
        print("Observed detected:", row["observed_likelihood_detected"])

        print("\nLatent complete-data comparison:", case_name, direction)
        print("Latent CD raw GC:", row["latent_cd_raw_gc"])
        print("Latent CD deviance:", row["latent_cd_debiased_deviance"])
        print("Latent CD p:", row["latent_cd_p_value"])
        print("Latent CD detected:", row["latent_cd_detected"])

    return rows

# Main run
all_rows = []

case_1_rows = run_case(
    case_name="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    random_seed=1
)

all_rows.extend(case_1_rows)

case_2_rows = run_case(
    case_name="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    random_seed=2
)

all_rows.extend(case_2_rows)

summary_df = pd.DataFrame(all_rows)

# Print compact summary
print("\nExperiment 31B compact summary")

columns_to_show = [
    "case",
    "direction",
    "expected_significant",
    "full_model_label",
    "reduced_model_label",
    "ll_full",
    "ll_reduced",
    "ll_gap_full_minus_reduced",
    "observed_likelihood_deviance",
    "observed_likelihood_p_value",
    "observed_likelihood_detected",
    "latent_cd_debiased_deviance",
    "latent_cd_p_value",
    "latent_cd_detected",
    "full_mse",
    "full_spectral_radius",
    "full_em_iterations",
    "full_R00",
    "full_R11"
]

print(summary_df[columns_to_show].to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)
summary_path = "results/experiment_31b_warmstarted_observed_likelihood_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary to:")
print(summary_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nThe key diagnostic is ll_gap_full_minus_reduced. "
    "It should be nonnegative or very close to zero."
)

print(
    "\nIf 31B removes the negative deviance from 31A while preserving the same "
    "correct decisions, then observed-likelihood deviance is numerically stable enough "
    "for Monte Carlo validation."
)

print(
    "\nExpected result:"
    "\n  null Y->X: nonsignificant"
    "\n  null X->Y: nonsignificant"
    "\n  true_link Y->X: significant"
    "\n  true_link X->Y: nonsignificant"
)

print(
    "\nThe latent complete-data columns are included only for comparison. "
    "The main statistic in 31B is observed_likelihood_deviance."
)