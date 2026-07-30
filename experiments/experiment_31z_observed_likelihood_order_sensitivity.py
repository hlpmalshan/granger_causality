import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.em_varx_p_known_c_warmstart import EMVARXPSSMKnownCConstrainedWarmStart, extract_model_parameters
from src.stats.observed_likelihood_deviance import observed_likelihood_deviance, final_observed_log_likelihood

N_SAMPLES = 1000
BURN_IN = 300

# Start with 10. Increase to 20 after confirming runtime.
N_OUTER_RUNS = 20

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

# True simulation order:
#   na_true = 2
#   nb_true = 3
# Grid tests:
#   endogenous order sensitivity: na = 1,2,3,5 with nb=3
#   exogenous order sensitivity:  nb = 1,2,3,5 with na=2
ORDER_GRID = [
    (1, 3),
    (2, 3),
    (3, 3),
    (5, 3),
    (2, 1),
    (2, 2),
    (2, 5)
]

print("\nExperiment 31D: Observed-likelihood order sensitivity")
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("order grid:", ORDER_GRID)

# Simulate latent VARX(2) state-space data.
# True model:
# na_true = 2
# nb_true = 3
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
        fit_na,
        fit_nb,
        zero_constraints=None,
        initial_parameters=None,
        label="model",
        verbose=False
):
    if zero_constraints is None:
        zero_constraints = []

    model = EMVARXPSSMKnownCConstrainedWarmStart(
        na=fit_na,
        nb=fit_nb,
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

def choose_best_model(models):
    if len(models) < 1:
        raise ValueError("models must contain at least one fitted model.")

    return max(models, key=model_ll)

# Approximate number of free parameters in the full known-C latent VARX model.
#     Counted:
#         A matrices: n_latent * n_latent * fit_na
#         B matrices: n_latent * n_inputs * fit_nb
#         Q symmetric covariance: n_latent * (n_latent + 1) / 2
#         R symmetric covariance: n_obs * (n_obs + 1) / 2

#     Not counted:
#         C, because C is known.
#         D, because D is fixed to zero/None here.
#         Initial state distribution, because it is fixed by implementation.
def count_full_model_parameters(
        n_latent,
        n_obs,
        n_inputs,
        fit_na,
        fit_nb
):
    k_A = n_latent * n_latent * fit_na
    k_B = n_latent * n_inputs * fit_nb
    k_Q = n_latent * (n_latent + 1) // 2
    k_R = n_obs * (n_obs + 1) // 2

    return int(k_A + k_B + k_Q + k_R)

def compute_aic_bic(log_likelihood, n_effective_samples, n_parameters):
    aic = (-2.0 * log_likelihood + 2.0 * n_parameters)
    bic = (-2.0 * log_likelihood + n_parameters * np.log(n_effective_samples))

    return float(aic), float(bic)

def fit_warmstarted_case_models_for_order(y_obs, u, C, fit_na, fit_nb):
    full_candidates = []

    full_proxy = fit_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        fit_na=fit_na,
        fit_nb=fit_nb,
        zero_constraints=[],
        initial_parameters=None,
        label="full_proxy",
        verbose=False
    )

    full_candidates.append(full_proxy)

    direction_specs = [
        {
            "direction": "Y->X",
            "source": 1,
            "target": 0
        },
        {
            "direction": "X->Y",
            "source": 0,
            "target": 1
        }
    ]

    reduced_results = {}

    for spec in direction_specs:
        direction = spec["direction"]
        source = spec["source"]
        target = spec["target"]

        zero_constraints = make_directional_zero_constraint(source=source, target=target)

        reduced_proxy = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            fit_na=fit_na,
            fit_nb=fit_nb,
            zero_constraints=zero_constraints,
            initial_parameters=None,
            label=f"reduced_{direction}_proxy",
            verbose=False
        )

        reduced_from_full = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            fit_na=fit_na,
            fit_nb=fit_nb,
            zero_constraints=zero_constraints,
            initial_parameters=extract_model_parameters(full_proxy),
            label=f"reduced_{direction}_from_full_proxy",
            verbose=False
        )

        best_reduced = choose_best_model([reduced_proxy, reduced_from_full])

        full_from_reduced = fit_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            fit_na=fit_na,
            fit_nb=fit_nb,
            zero_constraints=[],
            initial_parameters=extract_model_parameters(best_reduced),
            label=f"full_from_best_reduced_{direction}",
            verbose=False
        )

        full_candidates.append(full_from_reduced)

        reduced_results[direction] = {
            "source": source,
            "target": target,
            "reduced_proxy": reduced_proxy,
            "reduced_from_full": reduced_from_full,
            "best_reduced": best_reduced,
            "full_from_reduced": full_from_reduced
        }

    best_full = choose_best_model(full_candidates)

    return {
        "best_full": best_full,
        "full_candidates": full_candidates,
        "reduced_results": reduced_results
    }


def collect_order_model_row(
        case_label,
        outer_run,
        fit_na,
        fit_nb,
        best_full,
        full_mse,
        y_obs,
        u
):
    n_obs = y_obs.shape[1]
    n_inputs = u.shape[1] if u.ndim > 1 else 1
    n_latent = best_full.C.shape[1]

    k = count_full_model_parameters(
        n_latent=n_latent,
        n_obs=n_obs,
        n_inputs=n_inputs,
        fit_na=fit_na,
        fit_nb=fit_nb
    )

    ll = model_ll(best_full)

    aic, bic = compute_aic_bic(log_likelihood=ll, n_effective_samples=y_obs.shape[0], n_parameters=k)

    return {
        "case": case_label,
        "outer_run": outer_run,
        "fit_na": fit_na,
        "fit_nb": fit_nb,

        "full_log_likelihood": ll,
        "n_parameters": k,
        "aic": aic,
        "bic": bic,

        "full_mse": full_mse,
        "full_spectral_radius": best_full.spectral_radius(),
        "full_em_iterations": len(best_full.log_likelihoods),

        "full_A1_01": best_full.A_matrices[0][0, 1] if fit_na >= 1 else np.nan,
        "full_A2_01": best_full.A_matrices[1][0, 1] if fit_na >= 2 else np.nan,
        "full_A1_10": best_full.A_matrices[0][1, 0] if fit_na >= 1 else np.nan,
        "full_A2_10": best_full.A_matrices[1][1, 0] if fit_na >= 2 else np.nan,

        "full_R00": best_full.R[0, 0],
        "full_R11": best_full.R[1, 1],
        "full_R01": best_full.R[0, 1]
    }

def collect_direction_result(
        case_label,
        outer_run,
        fit_na,
        fit_nb,
        direction,
        source,
        target,
        expected_significant,
        best_full,
        best_reduced,
        full_mse
):
    obs_result = observed_likelihood_deviance(
        full_model=best_full,
        reduced_model=best_reduced,
        df_removed=fit_na,
        clip_negative=True
    )

    ll_gap = (obs_result["ll_full"] - obs_result["ll_reduced"])

    return {
        "case": case_label,
        "outer_run": outer_run,
        "fit_na": fit_na,
        "fit_nb": fit_nb,

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

        "negative_ll_gap": ll_gap < -1e-8,

        "df": fit_na,

        "full_mse": full_mse,
        "full_spectral_radius": best_full.spectral_radius(),
        "full_em_iterations": len(best_full.log_likelihoods),
        "reduced_spectral_radius": best_reduced.spectral_radius(),
        "reduced_em_iterations": len(best_reduced.log_likelihoods),

        "full_A1_01": best_full.A_matrices[0][0, 1] if fit_na >= 1 else np.nan,
        "full_A2_01": best_full.A_matrices[1][0, 1] if fit_na >= 2 else np.nan,
        "full_A1_10": best_full.A_matrices[0][1, 0] if fit_na >= 1 else np.nan,
        "full_A2_10": best_full.A_matrices[1][1, 0] if fit_na >= 2 else np.nan,

        "full_R00": best_full.R[0, 0],
        "full_R11": best_full.R[1, 1],
        "full_R01": best_full.R[0, 1]
    }

def run_outer_case(
        case_label,
        y_to_x_lag1,
        y_to_x_lag2,
        expected_y_to_x_significant,
        base_seed
):
    direction_rows = []
    order_rows = []

    for outer_run in range(N_OUTER_RUNS):
        print(f"Case={case_label} | outer run {outer_run + 1}/{N_OUTER_RUNS}")
        
        seed = base_seed + outer_run

        data = simulate_case(
            n_samples=N_SAMPLES,
            burn_in=BURN_IN,
            y_to_x_lag1=y_to_x_lag1,
            y_to_x_lag2=y_to_x_lag2,
            random_seed=seed
        )

        x_true = data["x_true"]
        y_obs = data["y_obs"]
        u = data["u"]
        C = data["C"]

        direction_specs = [
            {
                "direction": "Y->X",
                "source": 1,
                "target": 0,
                "expected_significant": expected_y_to_x_significant
            },
            {
                "direction": "X->Y",
                "source": 0,
                "target": 1,
                "expected_significant": False
            }
        ]

        for fit_na, fit_nb in ORDER_GRID:
            print(f"Fitting order fit_na={fit_na}, fit_nb={fit_nb}")
            
            fitted = fit_warmstarted_case_models_for_order(
                y_obs=y_obs,
                u=u,
                C=C,
                fit_na=fit_na,
                fit_nb=fit_nb
            )

            best_full = fitted["best_full"]

            x_full_smooth = best_full.smoothed_latent_state()

            full_mse = float(np.mean((x_full_smooth - x_true) ** 2))

            order_row = collect_order_model_row(
                case_label=case_label,
                outer_run=outer_run,
                fit_na=fit_na,
                fit_nb=fit_nb,
                best_full=best_full,
                full_mse=full_mse,
                y_obs=y_obs,
                u=u
            )

            order_rows.append(order_row)

            print("Best full label:", best_full.fit_label)
            print("Full LL:", order_row["full_log_likelihood"])
            print("AIC:", order_row["aic"])
            print("BIC:", order_row["bic"])
            print("MSE:", order_row["full_mse"])
            print("Spectral radius:", order_row["full_spectral_radius"])
            print("R diag:", order_row["full_R00"], order_row["full_R11"])

            for spec in direction_specs:
                direction = spec["direction"]
                best_reduced = fitted["reduced_results"][direction]["best_reduced"]
                direction_row = collect_direction_result(
                    case_label=case_label,
                    outer_run=outer_run,
                    fit_na=fit_na,
                    fit_nb=fit_nb,
                    direction=direction,
                    source=spec["source"],
                    target=spec["target"],
                    expected_significant=spec["expected_significant"],
                    best_full=best_full,
                    best_reduced=best_reduced,
                    full_mse=full_mse
                )

                direction_rows.append(direction_row)

                print(
                    f"{direction}: "
                    f"D={direction_row['observed_likelihood_deviance']:.4f}, "
                    f"p={direction_row['observed_likelihood_p_value']:.4g}, "
                    f"detected={direction_row['observed_likelihood_detected']}, "
                    f"LL gap={direction_row['ll_gap_full_minus_reduced']:.4f}"
                )

    return direction_rows, order_rows

def summarize_direction_results(direction_df):
    summary_rows = []

    grouped = direction_df.groupby([
        "case",
        "fit_na",
        "fit_nb",
        "direction",
        "expected_significant"
    ])

    for keys, group in grouped:
        case_label, fit_na, fit_nb, direction, expected_significant = keys

        summary_rows.append({
            "case": case_label,
            "fit_na": fit_na,
            "fit_nb": fit_nb,
            "direction": direction,
            "expected_significant": expected_significant,

            "mean_observed_likelihood_deviance": group["observed_likelihood_deviance"].mean(),
            "std_observed_likelihood_deviance": group["observed_likelihood_deviance"].std(),
            "median_observed_likelihood_p_value": group["observed_likelihood_p_value"].median(),
            "observed_likelihood_detection_rate": group["observed_likelihood_detected"].mean(),

            "negative_ll_gap_fraction": group["negative_ll_gap"].mean(),
            "min_ll_gap": group["ll_gap_full_minus_reduced"].min(),
            "mean_ll_gap": group["ll_gap_full_minus_reduced"].mean(),

            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_full_em_iterations": group["full_em_iterations"].mean(),
            "mean_reduced_em_iterations": group["reduced_em_iterations"].mean(),

            "mean_full_A1_01": group["full_A1_01"].mean(),
            "mean_full_A2_01": group["full_A2_01"].mean(),
            "mean_full_A1_10": group["full_A1_10"].mean(),
            "mean_full_A2_10": group["full_A2_10"].mean(),

            "mean_full_R00": group["full_R00"].mean(),
            "mean_full_R11": group["full_R11"].mean(),
            "mean_full_R01": group["full_R01"].mean()
        })

    return pd.DataFrame(summary_rows)

# Select one order per case/outer_run using AIC and BIC.
# Then collect the direction-test rows corresponding to the selected order.
def build_selected_order_rows(order_df, direction_df):
    selected_rows = []
    grouped = order_df.groupby(["case", "outer_run"])

    for keys, group in grouped:
        case_label, outer_run = keys

        best_aic_row = group.loc[group["aic"].idxmin()]
        best_bic_row = group.loc[group["bic"].idxmin()]

        selections = [("AIC", best_aic_row), ("BIC", best_bic_row)]

        for method, selected_order in selections:
            selected_na = int(selected_order["fit_na"])
            selected_nb = int(selected_order["fit_nb"])

            matching_direction_rows = direction_df[
                (direction_df["case"] == case_label)
                & (direction_df["outer_run"] == outer_run)
                & (direction_df["fit_na"] == selected_na)
                & (direction_df["fit_nb"] == selected_nb)
            ]

            for _, row in matching_direction_rows.iterrows():
                row_dict = row.to_dict()
                row_dict["selection_method"] = method
                row_dict["selected_na"] = selected_na
                row_dict["selected_nb"] = selected_nb
                row_dict["selected_aic"] = float(selected_order["aic"])
                row_dict["selected_bic"] = float(selected_order["bic"])

                selected_rows.append(row_dict)

    return pd.DataFrame(selected_rows)

def summarize_selected_order_detection(selected_df):
    summary_rows = []

    grouped = selected_df.groupby([
        "selection_method",
        "case",
        "direction",
        "expected_significant"
    ])

    for keys, group in grouped:
        selection_method, case_label, direction, expected_significant = keys

        summary_rows.append({
            "selection_method": selection_method,
            "case": case_label,
            "direction": direction,
            "expected_significant": expected_significant,

            "detection_rate": group["observed_likelihood_detected"].mean(),
            "median_p_value": group["observed_likelihood_p_value"].median(),
            "mean_deviance": group["observed_likelihood_deviance"].mean(),

            "mean_selected_na": group["selected_na"].mean(),
            "mean_selected_nb": group["selected_nb"].mean(),

            "negative_ll_gap_fraction": group["negative_ll_gap"].mean(),
            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean()
        })

    return pd.DataFrame(summary_rows)

def summarize_order_selection_frequencies(selected_df):
    frequency_rows = []

    grouped = selected_df.drop_duplicates([
        "selection_method",
        "case",
        "outer_run"
    ]).groupby([
        "selection_method",
        "case",
        "selected_na",
        "selected_nb"
    ])

    for keys, group in grouped:
        selection_method, case_label, selected_na, selected_nb = keys

        total = selected_df.drop_duplicates([
            "selection_method",
            "case",
            "outer_run"
        ])

        total_case_method = total[
            (total["selection_method"] == selection_method)
            & (total["case"] == case_label)
        ]

        frequency_rows.append({
            "selection_method": selection_method,
            "case": case_label,
            "selected_na": selected_na,
            "selected_nb": selected_nb,
            "count": len(group),
            "frequency": len(group) / len(total_case_method)
        })

    return pd.DataFrame(frequency_rows)

# Main run
all_direction_rows = []
all_order_rows = []

no_link_direction_rows, no_link_order_rows = run_outer_case(
    case_label="no_link",
    y_to_x_lag1=0.00,
    y_to_x_lag2=0.00,
    expected_y_to_x_significant=False,
    base_seed=1600000
)

all_direction_rows.extend(no_link_direction_rows)
all_order_rows.extend(no_link_order_rows)

true_link_direction_rows, true_link_order_rows = run_outer_case(
    case_label="true_link",
    y_to_x_lag1=0.25,
    y_to_x_lag2=0.12,
    expected_y_to_x_significant=True,
    base_seed=1700000
)

all_direction_rows.extend(true_link_direction_rows)
all_order_rows.extend(true_link_order_rows)
direction_df = pd.DataFrame(all_direction_rows)
order_df = pd.DataFrame(all_order_rows)
sensitivity_summary_df = summarize_direction_results(direction_df)
selected_df = build_selected_order_rows(order_df=order_df, direction_df=direction_df)
selected_detection_summary_df = summarize_selected_order_detection(selected_df)
selection_frequency_df = summarize_order_selection_frequencies(selected_df)

# Print summaries
print("\nExperiment 31D order-sensitivity summary")

sensitivity_columns = [
    "case",
    "fit_na",
    "fit_nb",
    "direction",
    "expected_significant",
    "mean_observed_likelihood_deviance",
    "median_observed_likelihood_p_value",
    "observed_likelihood_detection_rate",
    "negative_ll_gap_fraction",
    "min_ll_gap",
    "mean_full_mse",
    "mean_full_spectral_radius",
    "mean_full_R00",
    "mean_full_R11"
]

print(sensitivity_summary_df[sensitivity_columns].to_string(index=False))
print("\nExperiment 31D AIC/BIC selected-order detection summary")

selected_detection_columns = [
    "selection_method",
    "case",
    "direction",
    "expected_significant",
    "detection_rate",
    "median_p_value",
    "mean_deviance",
    "mean_selected_na",
    "mean_selected_nb",
    "negative_ll_gap_fraction",
    "mean_full_mse",
    "mean_full_spectral_radius"
]

print(selected_detection_summary_df[selected_detection_columns].to_string(index=False))
print("\nExperiment 31D AIC/BIC selected-order frequencies")
print(selection_frequency_df.to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

direction_path = "results/experiment_31d_observed_likelihood_order_direction_results.csv"
order_path = "results/experiment_31d_observed_likelihood_order_model_results.csv"
sensitivity_summary_path = "results/experiment_31d_observed_likelihood_order_sensitivity_summary.csv"
selected_path = "results/experiment_31d_selected_order_direction_results.csv"
selected_detection_path = "results/experiment_31d_selected_order_detection_summary.csv"
selection_frequency_path = "results/experiment_31d_order_selection_frequency_summary.csv"

direction_df.to_csv(direction_path, index=False)
order_df.to_csv(order_path, index=False)

sensitivity_summary_df.to_csv(sensitivity_summary_path, index=False)
selected_df.to_csv(selected_path, index=False)
selected_detection_summary_df.to_csv(selected_detection_path, index=False)
selection_frequency_df.to_csv(selection_frequency_path, index=False)

print("\nSaved direction-level results to:")
print(direction_path)

print("\nSaved order-level model results to:")
print(order_path)

print("\nSaved order-sensitivity summary to:")
print(sensitivity_summary_path)

print("\nSaved selected-order direction results to:")
print(selected_path)

print("\nSaved selected-order detection summary to:")
print(selected_detection_path)

print("\nSaved order-selection frequency summary to:")
print(selection_frequency_path)

# Interpretation guide
print("\nInterpretation guide")

print(
    "\nThe true simulation order is na=2, nb=3."
)

print(
    "\nFor endogenous order sensitivity, compare:"
    "\n    (1,3), (2,3), (3,3), (5,3)"
)

print(
    "\nFor exogenous order sensitivity, compare:"
    "\n    (2,1), (2,2), (2,3), (2,5)"
)

print(
    "\nUnderfitting nb is especially important here because the true confound uses B2: "
    "u drives Y immediately and X after two lags. If nb < 3, the model cannot represent "
    "that delayed exogenous pathway and may create spurious endogenous causality."
)

print(
    "\nA good result should show:"
    "\n    correct/overfit orders control null false positives,"
    "\n    true Y->X remains powerful,"
    "\n    BIC preferably selects or stays close to (2,3),"
    "\n    negative_ll_gap_fraction remains 0."
)

print(
    "\nIf AIC/BIC-selected orders work well, the next step is to use selected orders "
    "in the observed-likelihood pipeline before moving to adaptive/time-varying or spectral GC."
)