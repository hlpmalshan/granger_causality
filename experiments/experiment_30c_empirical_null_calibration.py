import os
import copy
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c import EMVARXPSSMKnownC
from src.stats.latent_debiased_deviance import debiased_latent_varx_gc_from_smoother
from src.stats.empirical_pvalue import empirical_upper_tail_p_value, clipped_deviance_statistic

N_SAMPLES = 1000
BURN_IN = 300

na = 2
nb = 3

ALPHA = 0.05

# Start with 50 because each bootstrap refits EM.
# Increase to 200 or 500 for final reporting.
N_BOOTSTRAP = 500

LAMBDA_REGULARIZATION = 1.0
gamma_gc = LAMBDA_REGULARIZATION / np.sqrt(N_SAMPLES)

print("\nExperiment 30C: Empirical null calibration")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("N bootstrap:", N_BOOTSTRAP)
print("gamma_gc:", gamma_gc)


def simulate_case(n_samples, burn_in, y_to_x_lag1, y_to_x_lag2, random_seed):
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
        "A_matrices_true": [A1, A2],
        "B_matrices_true": [B0, B1, B2],
        "Q_true": Q,
        "R_true": R,
        "C": C
    }


def fit_em_known_c_fixed_r(y_obs, u, C, R_fixed, Q_init=None, verbose=False):
    if Q_init is None:
        Q_init = 0.50 * np.eye(2)

    model = EMVARXPSSMKnownC(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=50,
        tol=1e-5,
        ridge_m_step=1e-6,
        covariance_floor=1e-6,
        R_init=R_fixed,
        Q_init=Q_init,
        estimate_R=False,
        R_floor=0.05,
        verbose=verbose
    )

    model.fit(y=y_obs, u=u)

    return model


def latent_cd_gc_from_model(model, u, source, target):
    smoother = model.smooth_result["smoother"]

    result = debiased_latent_varx_gc_from_smoother(
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

    return result

# Construct null dynamics for the tested direction.
# Example: source = 1, target = 0 means Y -> X.
# Null: A_k[target, source] = 0 for every lag k.
def make_directional_null_A_matrices(A_matrices, source, target):
    A_null = [np.array(A, dtype=float).copy() for A in A_matrices]
    for A in A_null:
        A[target, source] = 0.0

    return A_null

# Simulate y* from the fitted directional null model.
# We condition on the same exogenous input u used in the observed dataset.
def simulate_bootstrap_from_null_model(
        A_null,
        B_matrices,
        Q,
        R,
        C,
        u,
        random_seed
):
    sim = generate_ssm_varx_p_data(
        A_matrices=A_null,
        B_matrices=B_matrices,
        u=u,
        Q=Q,
        R=R,
        C=C,
        D=None,
        burn_in=0,
        random_seed=random_seed,
        return_augmented=False
    )

    return sim["y"]

# Calibrate one tested direction using a fitted directional null model.
def empirical_calibration_for_direction(
        observed_model,
        observed_u,
        source,
        target,
        case_label,
        direction_label,
        base_seed
):
    observed_result = latent_cd_gc_from_model(
        model=observed_model,
        u=observed_u,
        source=source,
        target=target
    )

    observed_stat = clipped_deviance_statistic(observed_result)

    A_null = make_directional_null_A_matrices(
        A_matrices=observed_model.A_matrices,
        source=source,
        target=target
    )

    null_stats = []
    null_rows = []

    print(f"\nBootstrap calibration: {case_label}, {direction_label}")

    for b in range(N_BOOTSTRAP):
        seed = base_seed + b

        y_boot = simulate_bootstrap_from_null_model(
            A_null=A_null,
            B_matrices=observed_model.B_matrices,
            Q=observed_model.Q,
            R=observed_model.R,
            C=observed_model.C,
            u=observed_u,
            random_seed=seed
        )

        boot_model = fit_em_known_c_fixed_r(
            y_obs=y_boot,
            u=observed_u,
            C=observed_model.C,
            R_fixed=observed_model.R,
            Q_init=observed_model.Q,
            verbose=False
        )

        boot_result = latent_cd_gc_from_model(
            model=boot_model,
            u=observed_u,
            source=source,
            target=target
        )

        boot_stat = clipped_deviance_statistic(boot_result)

        null_stats.append(boot_stat)

        null_rows.append({
            "case": case_label,
            "direction": direction_label,
            "bootstrap_index": b,
            "bootstrap_statistic": boot_stat,
            "bootstrap_debiased_deviance": boot_result["deviance_debiased"],
            "bootstrap_chi_square_p": boot_result["p_value"],
            "bootstrap_raw_gc": boot_result["gc_raw"],
            "bootstrap_spectral_radius": boot_model.spectral_radius(),
            "bootstrap_em_iterations": len(boot_model.log_likelihoods)
        })

        if (b + 1) % 10 == 0:
            print(f"  completed {b + 1}/{N_BOOTSTRAP}")

    null_stats = np.asarray(null_stats, dtype=float)

    empirical_p = empirical_upper_tail_p_value(
        observed_statistic=observed_stat,
        null_statistics=null_stats
    )

    summary = {
        "case": case_label,
        "direction": direction_label,
        "source": source,
        "target": target,
        "observed_raw_gc": observed_result["gc_raw"],
        "observed_debiased_deviance": observed_result["deviance_debiased"],
        "observed_statistic_clipped": observed_stat,
        "chi_square_p_value": observed_result["p_value"],
        "empirical_p_value": empirical_p,
        "chi_square_detected": observed_result["p_value"] < ALPHA,
        "empirical_detected": empirical_p < ALPHA,
        "null_mean_statistic": float(np.mean(null_stats)),
        "null_std_statistic": float(np.std(null_stats)),
        "null_median_statistic": float(np.median(null_stats)),
        "null_95_statistic": float(np.percentile(null_stats, 95)),
        "null_99_statistic": float(np.percentile(null_stats, 99)),
        "n_bootstrap": N_BOOTSTRAP,
        "min_possible_empirical_p": 1.0 / (N_BOOTSTRAP + 1),
        "observed_spectral_radius": observed_model.spectral_radius(),
        "observed_em_iterations": len(observed_model.log_likelihoods)
    }

    return summary, null_rows


def run_case(case_label, y_to_x_lag1, y_to_x_lag2, random_seed, base_bootstrap_seed):
    print("\n"+case_label)

    data = simulate_case(
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        y_to_x_lag1=y_to_x_lag1,
        y_to_x_lag2=y_to_x_lag2,
        random_seed=random_seed
    )

    y_obs = data["y_obs"]
    u = data["u"]
    C = data["C"]
    R_fixed = data["R_true"]
    Q_init = data["Q_true"]

    observed_model = fit_em_known_c_fixed_r(
        y_obs=y_obs,
        u=u,
        C=C,
        R_fixed=R_fixed,
        Q_init=Q_init,
        verbose=False
    )

    x_em = observed_model.smoothed_latent_state()

    em_mse = float(np.mean((x_em - data["x_true"]) ** 2))

    print("\nObserved EM model")
    print("EM iterations:", len(observed_model.log_likelihoods))
    print("Spectral radius:", observed_model.spectral_radius())
    print("EM smoothed MSE:", em_mse)

    print("\nEstimated A1")
    print(observed_model.A_matrices[0])

    print("\nEstimated A2")
    print(observed_model.A_matrices[1])

    print("\nEstimated Q")
    print(observed_model.Q)

    summaries = []
    bootstrap_rows = []

    # Y -> X: source=1, target=0
    summary_yx, rows_yx = empirical_calibration_for_direction(
        observed_model=observed_model,
        observed_u=u,
        source=1,
        target=0,
        case_label=case_label,
        direction_label="Y->X",
        base_seed=base_bootstrap_seed
    )

    summaries.append(summary_yx)
    bootstrap_rows.extend(rows_yx)

    # X -> Y: source=0, target=1
    summary_xy, rows_xy = empirical_calibration_for_direction(
        observed_model=observed_model,
        observed_u=u,
        source=0,
        target=1,
        case_label=case_label,
        direction_label="X->Y",
        base_seed=base_bootstrap_seed + 100000
    )

    summaries.append(summary_xy)
    bootstrap_rows.extend(rows_xy)

    for summary in summaries:
        print(summary["case"], summary["direction"])
        print("Observed raw GC:", summary["observed_raw_gc"])
        print("Observed de-biased deviance:", summary["observed_debiased_deviance"])
        print("Observed clipped statistic:", summary["observed_statistic_clipped"])
        print("Chi-square p-value:", summary["chi_square_p_value"])
        print("Empirical p-value:", summary["empirical_p_value"])
        print("Chi-square detected:", summary["chi_square_detected"])
        print("Empirical detected:", summary["empirical_detected"])
        print("Null mean statistic:", summary["null_mean_statistic"])
        print("Null 95% statistic:", summary["null_95_statistic"])
        print("Minimum possible empirical p:", summary["min_possible_empirical_p"])

    return summaries, bootstrap_rows


# Main run
summary_rows = []
all_bootstrap_rows = []

null_summaries, null_boot_rows = run_case(
    case_label="null",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    random_seed=1,
    base_bootstrap_seed=600000
)

summary_rows.extend(null_summaries)
all_bootstrap_rows.extend(null_boot_rows)

true_summaries, true_boot_rows = run_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    random_seed=2,
    base_bootstrap_seed=700000
)

summary_rows.extend(true_summaries)
all_bootstrap_rows.extend(true_boot_rows)


summary_df = pd.DataFrame(summary_rows)
bootstrap_df = pd.DataFrame(all_bootstrap_rows)
# Save results

os.makedirs("results", exist_ok=True)

summary_path = "results/experiment_30c_empirical_calibration_summary.csv"
bootstrap_path = "results/experiment_30c_bootstrap_null_statistics.csv"

summary_df.to_csv(summary_path, index=False)
bootstrap_df.to_csv(bootstrap_path, index=False)

print("Experiment 30C summary table")

columns_to_show = [
    "case",
    "direction",
    "observed_raw_gc",
    "observed_debiased_deviance",
    "observed_statistic_clipped",
    "chi_square_p_value",
    "empirical_p_value",
    "chi_square_detected",
    "empirical_detected",
    "null_mean_statistic",
    "null_95_statistic",
    "n_bootstrap"
]

print(summary_df[columns_to_show].to_string(index=False))

print("\nSaved empirical calibration summary to:")
print(summary_path)

print("\nSaved bootstrap null statistics to:")
print(bootstrap_path)

# Interpretation guide
print("Interpretation guide")
print(
    "\nIf a direction was significant by chi-square but not by empirical p-value, "
    "then the chi-square reference was too liberal for that case."
)

print(
    "\nFor true Y->X, empirical p-value should be small. With N_BOOTSTRAP=50, "
    "the smallest possible empirical p-value is 1/51 = 0.0196."
)

print(
    "\nFor null directions, empirical p-values should ideally be non-significant."
)

print(
    "\nIf empirical calibration removes the false positives while preserving the "
    "true Y->X detection, then the next experiment should be Monte Carlo validation "
    "of empirical p-values."
)