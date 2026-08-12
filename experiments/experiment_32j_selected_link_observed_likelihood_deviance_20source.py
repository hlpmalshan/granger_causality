import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
    make_group_zero_constraints
)
from src.ssm.em_varx_p_known_c_warmstart import extract_model_parameters
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.observed_likelihood_deviance import final_observed_log_likelihood
from src.stats.scalable_debiased_varx_network import (
    companion_spectral_radius,
    compute_all_pair_debiased_varx_network,
    compute_confusion_metrics
)
from src.varx.varx_generator import generate_colored_input


# ------------------------------------------------------
# Experiment 32J
# Selected-link observed-likelihood full/reduced diagnostic
# ------------------------------------------------------

N_SOURCES = 20
LINK_DENSITY = 0.05
N_TRUE_LINKS = int(round(LINK_DENSITY * N_SOURCES * (N_SOURCES - 1)))
N_INPUTS = 1

na = 2
nb = 3

N_SAMPLES = 500
BURN_IN = 300

# Full experiment configuration.
N_OUTER_RUNS = 20
LAMBDA_A_GROUP_FRACTION_GRID = [0.003, 0.010, 0.030, 0.100]

ALPHA = 0.05
R_FLOOR = 0.30
FULL_MAX_ITER = 100
FULL_TOL = 1e-6
REDUCED_MAX_ITER = 50
REDUCED_TOL = 1e-5
RIDGE_LAMBDA_DEBIAS = 1.0
BASE_SEED = 3600000

GROUP_SOLVER_MAX_ITER = 5000
GROUP_SOLVER_TOL = 1e-7

CUSTOM_THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
RESULTS_DIR = "results"
RESULTS_STEM = "experiment_32j_selected_link_observed_likelihood_deviance"
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, RESULTS_STEM + "_checkpoint.pkl")


print("\nExperiment 32J: selected-link observed likelihood deviance")
print("----------------------------------------------------------")
print("N outer runs:", N_OUTER_RUNS)
print("lambda fractions:", LAMBDA_A_GROUP_FRACTION_GRID)
print("full max iterations:", FULL_MAX_ITER)
print("reduced max iterations:", REDUCED_MAX_ITER)
print("Q/R fixed at traces:", 0.50 * N_SOURCES, 0.60 * N_SOURCES)
print("True links are used only for diagnostic panel construction/evaluation.")


# ------------------------------------------------------
# Simulation helpers copied from the comparable 32I setup
# ------------------------------------------------------

def make_sparse_stable_var_matrices(
        n_sources,
        n_true_links,
        order,
        random_seed,
        target_radius=0.82
):
    rng = np.random.default_rng(random_seed)
    A_matrices = [np.zeros((n_sources, n_sources)) for _ in range(order)]
    A_matrices[0] += np.diag(rng.uniform(0.25, 0.45, size=n_sources))

    if order >= 2:
        A_matrices[1] += np.diag(rng.uniform(-0.12, -0.04, size=n_sources))

    possible_links = [
        (source, target)
        for source in range(n_sources)
        for target in range(n_sources)
        if source != target
    ]
    selected = rng.choice(len(possible_links), size=n_true_links, replace=False)
    true_link_mask = np.zeros((n_sources, n_sources), dtype=bool)

    for index in selected:
        source, target = possible_links[index]
        sign = rng.choice([-1.0, 1.0])
        A_matrices[0][target, source] = sign * rng.uniform(0.08, 0.16)

        if order >= 2:
            A_matrices[1][target, source] = sign * rng.uniform(0.03, 0.09)

        true_link_mask[target, source] = True

    radius_before = companion_spectral_radius(A_matrices)
    scale_factor = 1.0

    if radius_before >= target_radius:
        scale_factor = target_radius / (radius_before + 1e-12)
        A_matrices = [scale_factor * A for A in A_matrices]

    return {
        "A_matrices": A_matrices,
        "true_link_mask": true_link_mask,
        "radius": companion_spectral_radius(A_matrices),
        "scale_factor": scale_factor
    }


def make_exogenous_filters(n_sources, n_inputs, order, random_seed):
    rng = np.random.default_rng(random_seed)
    scales = (0.45, 0.25, 0.15)

    return [
        rng.normal(
            loc=0.0,
            scale=scales[lag] if lag < len(scales) else scales[-1],
            size=(n_sources, n_inputs)
        )
        for lag in range(order)
    ]


def make_known_mixing_matrix(n_sources, random_seed, mixing_strength=0.05):
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(
        0.0,
        mixing_strength / np.sqrt(n_sources),
        size=(n_sources, n_sources)
    )
    np.fill_diagonal(noise, 0.0)
    return np.eye(n_sources) + noise


def simulate_case(random_seed):
    network = make_sparse_stable_var_matrices(
        n_sources=N_SOURCES,
        n_true_links=N_TRUE_LINKS,
        order=na,
        random_seed=random_seed
    )
    B_matrices = make_exogenous_filters(
        N_SOURCES,
        N_INPUTS,
        nb,
        random_seed + 100
    )
    C = make_known_mixing_matrix(N_SOURCES, random_seed + 200)
    Q = 0.50 * np.eye(N_SOURCES)
    R = 0.60 * np.eye(N_SOURCES)
    u_total = generate_colored_input(
        n_samples=N_SAMPLES + BURN_IN,
        ar_coeff=0.95,
        noise_std=1.0,
        random_seed=random_seed + 300
    )
    simulation = generate_ssm_varx_p_data(
        A_matrices=network["A_matrices"],
        B_matrices=B_matrices,
        u=u_total,
        Q=Q,
        R=R,
        C=C,
        D=None,
        burn_in=BURN_IN,
        random_seed=random_seed + 400,
        return_augmented=True
    )

    return {
        "x_true": simulation["x"],
        "y_obs": simulation["y"],
        "u": simulation["u"],
        "C": C,
        "true_link_mask": network["true_link_mask"],
        "radius": network["radius"],
        "scale_factor": network["scale_factor"]
    }


# ------------------------------------------------------
# Model and diagnostic helpers
# ------------------------------------------------------

def fit_group_model(
        y,
        u,
        C,
        lambda_fraction,
        random_seed,
        max_iter,
        tol,
        zero_constraints=None,
        initial_parameters=None
):
    model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=max_iter,
        tol=tol,
        ridge_m_step=1e-4,
        covariance_floor=1e-6,
        Q_init=0.50 * np.eye(N_SOURCES),
        R_init=0.60 * np.eye(N_SOURCES),
        estimate_Q=False,
        estimate_R=False,
        R_floor=R_FLOOR,
        zero_constraints=zero_constraints or [],
        initial_parameters=initial_parameters,
        jitter_scale=0.0,
        random_seed=random_seed,
        verbose=False,
        alpha_Q=0.0,
        alpha_R=0.0,
        shrinkage_target_Q="spherical",
        shrinkage_target_R="spherical",
        lambda_A_group_fraction=lambda_fraction,
        ridge_A_offdiag=1e-4,
        ridge_A_diag=1e-4,
        ridge_B=1e-4,
        group_solver_max_iter=GROUP_SOLVER_MAX_ITER,
        group_solver_tol=GROUP_SOLVER_TOL,
        stabilize_A=True,
        target_radius=0.98
    )
    model.fit(y=y, u=u)
    return model


def current_latent_posterior_means(model):
    n_sources = model.C.shape[1]
    smooth_result = model.smooth_result
    return {
        "em_filtered": smooth_result["filter"]["x_filt"][:, :n_sources],
        "em_smoothed": smooth_result["smoother"]["x_smooth"][:, :n_sources]
    }


def signal_recovery(signal, x_true):
    correlations = []

    for source in range(x_true.shape[1]):
        correlations.append(np.corrcoef(signal[:, source], x_true[:, source])[0, 1])

    return {
        "mse": float(np.mean((signal - x_true) ** 2)),
        "correlation_mean": float(np.mean(correlations)),
        "correlation_median": float(np.median(correlations))
    }


def posterior_network_tables(model, u, true_link_mask):
    tables = {}

    for signal_type, signal in current_latent_posterior_means(model).items():
        table = compute_all_pair_debiased_varx_network(
            x=signal,
            u=u,
            true_link_mask=true_link_mask,
            na=na,
            nb=nb,
            ridge_lambda=RIDGE_LAMBDA_DEBIAS,
            alpha=ALPHA,
            custom_thresholds=CUSTOM_THRESHOLDS
        )
        tables[signal_type] = table.set_index(["source", "target"])

    return tables


def A_group_table(model, true_link_mask):
    rows = []

    for target in range(N_SOURCES):
        for source in range(N_SOURCES):
            if source == target:
                continue

            coefficients = [float(A[target, source]) for A in model.A_matrices]
            rows.append({
                "source": source,
                "target": target,
                "true_link": bool(true_link_mask[target, source]),
                "A_group_norm_full": float(np.linalg.norm(coefficients)),
                "A_lag1_full": coefficients[0],
                "A_lag2_full": coefficients[1]
            })

    return pd.DataFrame(rows)


def make_selected_link_panel(
        A_table,
        true_link_mask,
        outer_run,
        lambda_fraction
):
    sources_by_link = {}

    def add_links(frame, label):
        for row in frame.itertuples(index=False):
            key = (int(row.source), int(row.target))
            sources_by_link.setdefault(key, set()).add(label)

    true_rows = A_table[A_table["true_link"]]
    false_rows = A_table[~A_table["true_link"]]
    add_links(true_rows, "all_true_links_diagnostic")

    deterministic_seed = (
        BASE_SEED
        + 100000 * int(outer_run)
        + int(round(float(lambda_fraction) * 1e9))
        + 32000
    )
    rng = np.random.default_rng(deterministic_seed)
    false_indices = rng.choice(
        false_rows.index.to_numpy(),
        size=len(true_rows),
        replace=False
    )
    add_links(false_rows.loc[false_indices], "matched_random_false_links")

    top_19 = A_table.nlargest(N_TRUE_LINKS, "A_group_norm_full")
    add_links(top_19, "top_19_A_group_norm")
    add_links(top_19, "top_5_percent_A_group_norm")

    rows = []

    for (source, target), labels in sorted(sources_by_link.items()):
        A_row = A_table[
            (A_table["source"] == source)
            & (A_table["target"] == target)
        ].iloc[0]
        rows.append({
            **A_row.to_dict(),
            "selection_sources": "|".join(sorted(labels)),
            "selected_by_true_diagnostic": "all_true_links_diagnostic" in labels,
            "selected_by_random_false": "matched_random_false_links" in labels,
            "selected_by_top_19_A_group_norm": "top_19_A_group_norm" in labels,
            "selected_by_top_5_percent_A_group_norm": (
                "top_5_percent_A_group_norm" in labels
            ),
            "selected_by_em_filtered_Ddb_gt_10": False
        })

    return pd.DataFrame(rows)


def add_posterior_readout_columns(row, network_tables, source, target):
    for signal_type in ("em_filtered", "em_smoothed"):
        network_row = network_tables[signal_type].loc[(source, target)]
        row[f"{signal_type}_raw_deviance"] = float(network_row["raw_deviance"])
        row[f"{signal_type}_debiased_deviance"] = float(
            network_row["debiased_deviance"]
        )
        row[f"{signal_type}_raw_p_value"] = float(network_row["raw_p_value"])
        row[f"{signal_type}_debiased_p_value"] = float(
            network_row["debiased_p_value"]
        )


def full_model_diagnostics(model):
    return {
        "full_log_likelihood": final_observed_log_likelihood(model),
        "full_spectral_radius": model.spectral_radius(),
        "full_em_iterations": len(model.log_likelihoods),
        "full_Q_trace": float(np.trace(model.Q)),
        "full_R_trace": float(np.trace(model.R)),
        "estimate_Q": model.estimate_Q,
        "estimate_R": model.estimate_R,
        "mean_lambda_A_group_effective": model.mean_lambda_A_group_effective,
        "median_lambda_A_group_effective": model.median_lambda_A_group_effective,
        "min_lambda_A_group_effective": model.min_lambda_A_group_effective,
        "max_lambda_A_group_effective": model.max_lambda_A_group_effective,
        "em_A_nonzero_offdiag_coefficients": (
            model.count_nonzero_offdiag_A_coefficients()
        ),
        "em_A_nonzero_offdiag_groups": model.count_nonzero_offdiag_A_groups(),
        "em_A_mean_offdiag_group_norm": model.mean_offdiag_A_group_norm(),
        "em_A_median_offdiag_group_norm": model.median_offdiag_A_group_norm(),
        "em_A_max_offdiag_group_norm": model.max_offdiag_A_group_norm(),
        "group_solver_converged_last": model.group_solver_converged_history[-1],
        "group_solver_iterations_last": model.group_solver_iterations_history[-1],
        "group_solver_objective_last": model.group_solver_objective_history[-1]
    }


# ------------------------------------------------------
# Summaries
# ------------------------------------------------------

DECISION_COLUMNS = {
    "chi_square": "chi_square_detected",
    "AIC": "AIC_detected",
    "BIC": "BIC_detected",
    "D_gt_8": "D_obs_gt_8p0_detected",
    "D_gt_10": "D_obs_gt_10p0_detected",
    "D_gt_12": "D_obs_gt_12p0_detected",
    "D_gt_15": "D_obs_gt_15p0_detected",
    "D_gt_20": "D_obs_gt_20p0_detected",
    "D_gt_25": "D_obs_gt_25p0_detected"
}


def summarize_decisions(results_df):
    rows = []

    for lambda_fraction, group in results_df.groupby("lambda_A_group_fraction"):
        n_true = int(group["true_link"].sum())
        n_false = int((~group["true_link"]).sum())

        for rule_name, column in DECISION_COLUMNS.items():
            metrics = compute_confusion_metrics(group["true_link"], group[column])
            rows.append({
                "lambda_A_group_fraction": lambda_fraction,
                "selection_panel_type": "union_selected_links_diagnostic",
                "decision_rule": rule_name,
                "n_tested_links": len(group),
                "n_tested_true_links": n_true,
                "n_tested_false_links": n_false,
                **metrics
            })

    return pd.DataFrame(rows)


def summarize_link_strengths(results_df):
    rows = []

    for (lambda_fraction, true_link), group in results_df.groupby(
            ["lambda_A_group_fraction", "true_link"]
    ):
        rows.append({
            "lambda_A_group_fraction": lambda_fraction,
            "true_link": true_link,
            "n_links": len(group),
            "mean_D_obs_raw": group["D_obs_raw"].mean(),
            "median_D_obs_raw": group["D_obs_raw"].median(),
            "std_D_obs_raw": group["D_obs_raw"].std(),
            "mean_D_obs_clipped": group["D_obs_clipped"].mean(),
            "median_D_obs_clipped": group["D_obs_clipped"].median(),
            "std_D_obs_clipped": group["D_obs_clipped"].std(),
            "negative_deviance_rate": group["negative_deviance_flag"].mean(),
            "mean_chi_square_p_value": group["chi_square_p_value"].mean(),
            "median_chi_square_p_value": group["chi_square_p_value"].median(),
            "mean_A_group_norm_full": group["A_group_norm_full"].mean(),
            "median_A_group_norm_full": group["A_group_norm_full"].median(),
            "mean_em_filtered_raw_deviance": (
                group["em_filtered_raw_deviance"].mean()
            ),
            "median_em_filtered_raw_deviance": (
                group["em_filtered_raw_deviance"].median()
            ),
            "mean_em_filtered_debiased_deviance": (
                group["em_filtered_debiased_deviance"].mean()
            ),
            "median_em_filtered_debiased_deviance": (
                group["em_filtered_debiased_deviance"].median()
            ),
            "mean_em_smoothed_raw_deviance": (
                group["em_smoothed_raw_deviance"].mean()
            ),
            "median_em_smoothed_raw_deviance": (
                group["em_smoothed_raw_deviance"].median()
            ),
            "mean_em_smoothed_debiased_deviance": (
                group["em_smoothed_debiased_deviance"].mean()
            ),
            "median_em_smoothed_debiased_deviance": (
                group["em_smoothed_debiased_deviance"].median()
            )
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------
# Atomic, Drive-tolerant checkpoint helpers
# ------------------------------------------------------

def checkpoint_configuration():
    return {
        "version": 1,
        "n_sources": N_SOURCES,
        "n_samples": N_SAMPLES,
        "burn_in": BURN_IN,
        "n_outer_runs": N_OUTER_RUNS,
        "lambda_grid": LAMBDA_A_GROUP_FRACTION_GRID,
        "full_max_iter": FULL_MAX_ITER,
        "full_tol": FULL_TOL,
        "reduced_max_iter": REDUCED_MAX_ITER,
        "reduced_tol": REDUCED_TOL,
        "base_seed": BASE_SEED
    }


def checkpoint_candidates():
    candidates = [CHECKPOINT_PATH] if os.path.exists(CHECKPOINT_PATH) else []
    candidates.extend(glob.glob(CHECKPOINT_PATH + "*.tmp"))
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def save_checkpoint(rows, completed_links, completed_lambdas):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = CHECKPOINT_PATH + f".{os.getpid()}.{time.time_ns()}.tmp"

    with open(path, "wb") as checkpoint_file:
        pickle.dump(
            {
                "configuration": checkpoint_configuration(),
                "rows": rows,
                "completed_links": completed_links,
                "completed_lambdas": completed_lambdas
            },
            checkpoint_file,
            protocol=pickle.HIGHEST_PROTOCOL
        )
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())

    delay = 0.25

    for attempt in range(8):
        try:
            os.replace(path, CHECKPOINT_PATH)
            return
        except PermissionError:
            if attempt < 7:
                time.sleep(delay)
                delay *= 2.0

    print("Warning: main checkpoint locked; valid progress retained in:", path)


def load_checkpoint():
    candidates = checkpoint_candidates()

    if not candidates:
        return [], set(), set()

    for path in candidates:
        try:
            with open(path, "rb") as checkpoint_file:
                checkpoint = pickle.load(checkpoint_file)
        except (EOFError, OSError, pickle.UnpicklingError):
            continue

        if checkpoint.get("configuration") != checkpoint_configuration():
            raise ValueError(
                "32J checkpoint configuration differs from the current run: "
                + path
            )

        print("Resuming from checkpoint:", path)
        print("Completed reduced links:", len(checkpoint["completed_links"]))
        completed_lambdas = set(checkpoint.get("completed_lambdas", []))

        # Backward-compatible migration for checkpoints written before
        # completed-lambda markers were introduced. Every lambda preceding
        # the last observed work unit must have completed, because execution
        # is sequential in outer-run/lambda-grid order. The last unit is
        # conservatively recomputed once.
        if not completed_lambdas and checkpoint["rows"]:
            grid_positions = {
                float(value): position
                for position, value in enumerate(LAMBDA_A_GROUP_FRACTION_GRID)
            }
            last_row = checkpoint["rows"][-1]
            last_outer = int(last_row["outer_run"])
            last_lambda_position = grid_positions[
                float(last_row["lambda_A_group_fraction"])
            ]

            for outer_run in range(last_outer + 1):
                for position, lambda_fraction in enumerate(
                        LAMBDA_A_GROUP_FRACTION_GRID
                ):
                    if outer_run < last_outer or position < last_lambda_position:
                        completed_lambdas.add(
                            (outer_run, float(lambda_fraction))
                        )

        print("Completed lambda units:", len(completed_lambdas))
        return (
            checkpoint["rows"],
            set(checkpoint["completed_links"]),
            completed_lambdas
        )

    raise RuntimeError("No readable 32J checkpoint candidate was found.")


def remove_checkpoint_files():
    for path in checkpoint_candidates():
        try:
            os.remove(path)
        except PermissionError:
            print("Warning: completed checkpoint remains Drive-locked:", path)


# ------------------------------------------------------
# Main experiment
# ------------------------------------------------------

all_rows, completed_links, completed_lambdas = load_checkpoint()

for outer_run in range(N_OUTER_RUNS):
    seed = BASE_SEED + outer_run
    data = simulate_case(seed)

    print("\n" + "=" * 100)
    print(f"Experiment 32J | outer run {outer_run + 1}/{N_OUTER_RUNS}")
    print("=" * 100)

    for lambda_fraction in LAMBDA_A_GROUP_FRACTION_GRID:
        lambda_fraction = float(lambda_fraction)
        lambda_key = (outer_run, lambda_fraction)

        if lambda_key in completed_lambdas:
            print("Skipping completed full/reduced lambda unit:", lambda_fraction)
            continue

        expected_prefix = (outer_run, lambda_fraction)
        already_completed = {
            key[2:]
            for key in completed_links
            if key[:2] == expected_prefix
        }

        print("\nFitting full model, lambda fraction:", lambda_fraction)
        full_model = fit_group_model(
            y=data["y_obs"],
            u=data["u"],
            C=data["C"],
            lambda_fraction=lambda_fraction,
            random_seed=seed + 500,
            max_iter=FULL_MAX_ITER,
            tol=FULL_TOL
        )
        full_diagnostics = full_model_diagnostics(full_model)
        posterior_signals = current_latent_posterior_means(full_model)
        filtered_recovery = signal_recovery(
            posterior_signals["em_filtered"], data["x_true"]
        )
        smoothed_recovery = signal_recovery(
            posterior_signals["em_smoothed"], data["x_true"]
        )
        network_tables = posterior_network_tables(
            full_model, data["u"], data["true_link_mask"]
        )
        A_table = A_group_table(full_model, data["true_link_mask"])
        panel = make_selected_link_panel(
            A_table=A_table,
            true_link_mask=data["true_link_mask"],
            outer_run=outer_run,
            lambda_fraction=lambda_fraction
        )
        full_parameters = extract_model_parameters(full_model)

        print("Full LL:", full_diagnostics["full_log_likelihood"])
        print("Full EM iterations:", full_diagnostics["full_em_iterations"])
        print("Full spectral radius:", full_diagnostics["full_spectral_radius"])
        print("Offdiag A groups:", full_diagnostics["em_A_nonzero_offdiag_groups"])
        print("Selected links:", len(panel))
        print("Selected true links:", int(panel["true_link"].sum()))
        print("Selected false links:", int((~panel["true_link"]).sum()))

        for panel_index, panel_row in enumerate(panel.itertuples(index=False), 1):
            source = int(panel_row.source)
            target = int(panel_row.target)
            completion_key = (outer_run, lambda_fraction, source, target)

            if (source, target) in already_completed:
                print("Skipping completed reduced link:", source, "->", target)
                continue

            print(
                f"Reduced fit {panel_index}/{len(panel)}:",
                source,
                "->",
                target
            )
            reduced_model = fit_group_model(
                y=data["y_obs"],
                u=data["u"],
                C=data["C"],
                lambda_fraction=lambda_fraction,
                random_seed=seed + 500,
                max_iter=REDUCED_MAX_ITER,
                tol=REDUCED_TOL,
                zero_constraints=make_group_zero_constraints(
                    source=source,
                    target=target,
                    na=na
                ),
                initial_parameters=full_parameters
            )
            LL_full = full_diagnostics["full_log_likelihood"]
            LL_reduced = final_observed_log_likelihood(reduced_model)
            D_obs_raw = 2.0 * (LL_full - LL_reduced)
            D_obs_clipped = max(D_obs_raw, 0.0)
            df_removed = na
            T_eff = N_SAMPLES - max(na, nb - 1)
            result = {
                "outer_run": outer_run,
                "random_seed": seed,
                "source": source,
                "target": target,
                "direction": f"{source}->{target}",
                "true_link": bool(panel_row.true_link),
                "diagnostic_post_hoc_panel": True,
                "selection_sources": panel_row.selection_sources,
                "selected_by_true_diagnostic": panel_row.selected_by_true_diagnostic,
                "selected_by_random_false": panel_row.selected_by_random_false,
                "selected_by_top_19_A_group_norm": (
                    panel_row.selected_by_top_19_A_group_norm
                ),
                "selected_by_top_5_percent_A_group_norm": (
                    panel_row.selected_by_top_5_percent_A_group_norm
                ),
                "selected_by_em_filtered_Ddb_gt_10": False,
                "lambda_A_group_fraction": lambda_fraction,
                "A_group_norm_full": panel_row.A_group_norm_full,
                "A_lag1_full": panel_row.A_lag1_full,
                "A_lag2_full": panel_row.A_lag2_full,
                **full_diagnostics,
                "em_filtered_signal_mse": filtered_recovery["mse"],
                "em_filtered_signal_correlation_mean": (
                    filtered_recovery["correlation_mean"]
                ),
                "em_filtered_signal_correlation_median": (
                    filtered_recovery["correlation_median"]
                ),
                "em_smoothed_signal_mse": smoothed_recovery["mse"],
                "em_smoothed_signal_correlation_mean": (
                    smoothed_recovery["correlation_mean"]
                ),
                "em_smoothed_signal_correlation_median": (
                    smoothed_recovery["correlation_median"]
                ),
                "LL_full": LL_full,
                "LL_reduced": LL_reduced,
                "reduced_em_iterations": len(reduced_model.log_likelihoods),
                "reduced_spectral_radius": reduced_model.spectral_radius(),
                "reduced_Q_trace": float(np.trace(reduced_model.Q)),
                "reduced_R_trace": float(np.trace(reduced_model.R)),
                "D_obs_raw": D_obs_raw,
                "D_obs_clipped": D_obs_clipped,
                "negative_deviance_flag": D_obs_raw < 0.0,
                "df_removed": df_removed,
                "T_eff": T_eff,
                "chi_square_p_value": float(chi2.sf(D_obs_clipped, df_removed)),
                "chi_square_detected": bool(
                    chi2.sf(D_obs_clipped, df_removed) < ALPHA
                ),
                "AIC_threshold": 2.0 * df_removed,
                "AIC_detected": D_obs_clipped > 2.0 * df_removed,
                "BIC_threshold": df_removed * np.log(T_eff),
                "BIC_detected": D_obs_clipped > df_removed * np.log(T_eff)
            }

            for threshold in CUSTOM_THRESHOLDS:
                key = str(threshold).replace(".", "p")
                result[f"D_obs_gt_{key}_detected"] = D_obs_clipped > threshold

            add_posterior_readout_columns(
                result,
                network_tables,
                source,
                target
            )
            all_rows.append(result)
            completed_links.add(completion_key)
            save_checkpoint(all_rows, completed_links, completed_lambdas)

        lambda_rows = pd.DataFrame([
            row
            for row in all_rows
            if row["outer_run"] == outer_run
            and row["lambda_A_group_fraction"] == lambda_fraction
        ])
        print("Negative deviance rate:", lambda_rows["negative_deviance_flag"].mean())
        display = summarize_decisions(lambda_rows)
        print(
            display[display["decision_rule"].isin([
                "chi_square", "BIC", "D_gt_8", "D_gt_10", "D_gt_12", "D_gt_15"
            ])].to_string(index=False)
        )
        completed_lambdas.add(lambda_key)
        save_checkpoint(all_rows, completed_links, completed_lambdas)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(RESULTS_DIR, RESULTS_STEM + "_results_partial.csv"),
        index=False
    )


results_df = pd.DataFrame(all_rows)
summary_df = summarize_decisions(results_df)
link_strength_summary_df = summarize_link_strengths(results_df)

results_path = os.path.join(RESULTS_DIR, RESULTS_STEM + "_results.csv")
summary_path = os.path.join(RESULTS_DIR, RESULTS_STEM + "_summary.csv")
link_summary_path = os.path.join(
    RESULTS_DIR,
    RESULTS_STEM + "_link_strength_summary.csv"
)

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
link_strength_summary_df.to_csv(link_summary_path, index=False)
remove_checkpoint_files()

print("\nSaved detailed results:", results_path)
print("Saved decision summary:", summary_path)
print("Saved link-strength summary:", link_summary_path)
