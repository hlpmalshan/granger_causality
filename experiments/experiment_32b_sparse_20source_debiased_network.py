import os
import numpy as np
import pandas as pd

from src.varx.varx_generator import (
    generate_colored_input
)

from src.ssm.ssm_varx_p_simulator import (
    generate_ssm_varx_p_data
)

from src.ssm.em_varx_p_known_c_shrinkage import (
    EMVARXPSSMKnownCConstrainedWarmStartShrinkage
)

from src.stats.observed_likelihood_deviance import (
    final_observed_log_likelihood
)

from src.stats.scalable_debiased_varx_network import (
    companion_spectral_radius,
    compute_all_pair_debiased_varx_network,
    compute_confusion_metrics
)


# ------------------------------------------------------
# Experiment 32B
# Sparse 20-source network with scalable de-biased deviance.
# ------------------------------------------------------

N_SOURCES = 20
LINK_DENSITY = 0.05

N_TRUE_LINKS = int(
    round(
        LINK_DENSITY * N_SOURCES * (N_SOURCES - 1)
    )
)

N_INPUTS = 1

na = 2
nb = 3

N_SAMPLES = 500
BURN_IN = 300

# Start with 10. After confirming stability, set this to 20 or 50.
N_OUTER_RUNS = 20

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

RIDGE_LAMBDA_DEBIAS = 1.0

BASE_SEED = 3200000

CUSTOM_THRESHOLDS = (
    8.0,
    10.0,
    12.0,
    15.0
)

SHRINKAGE_SPECS = [
    {
        "name": "QR_0p20",
        "alpha_Q": 0.20,
        "alpha_R": 0.20,
        "target_Q": "spherical",
        "target_R": "spherical"
    }
]

# For a heavier comparison, later use:
# SHRINKAGE_SPECS = [
#     {"name": "none", "alpha_Q": 0.00, "alpha_R": 0.00, "target_Q": "spherical", "target_R": "spherical"},
#     {"name": "QR_0p20", "alpha_Q": 0.20, "alpha_R": 0.20, "target_Q": "spherical", "target_R": "spherical"}
# ]


print("\nExperiment 32B: sparse 20-source de-biased network")
print("---------------------------------------------------")
print("N sources:", N_SOURCES)
print("Link density:", LINK_DENSITY)
print("N true links:", N_TRUE_LINKS)
print("N samples:", N_SAMPLES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("debiased ridge lambda:", RIDGE_LAMBDA_DEBIAS)
print("shrinkage specs:", [s["name"] for s in SHRINKAGE_SPECS])


def make_sparse_stable_var_matrices(
        n_sources,
        n_true_links,
        na,
        random_seed,
        target_radius=0.82
):
    """
    Generate a sparse VAR(2) network with exactly n_true_links
    directed off-diagonal links.

    Matrix convention:
        A_k[target, source] = coefficient for source -> target.
    """

    rng = np.random.default_rng(
        random_seed
    )

    A_matrices = [
        np.zeros(
            (n_sources, n_sources)
        )
        for _ in range(na)
    ]

    # Stable self dynamics.
    A_matrices[0] += np.diag(
        rng.uniform(
            0.25,
            0.45,
            size=n_sources
        )
    )

    if na >= 2:
        A_matrices[1] += np.diag(
            rng.uniform(
                -0.12,
                -0.04,
                size=n_sources
            )
        )

    possible_links = [
        (source, target)
        for source in range(n_sources)
        for target in range(n_sources)
        if source != target
    ]

    selected_indices = rng.choice(
        len(possible_links),
        size=n_true_links,
        replace=False
    )

    true_links = [
        possible_links[idx]
        for idx in selected_indices
    ]

    true_link_mask = np.zeros(
        (n_sources, n_sources),
        dtype=bool
    )

    true_link_values = []

    for source, target in true_links:

        sign = rng.choice(
            [-1.0, 1.0]
        )

        lag1_value = sign * rng.uniform(
            0.08,
            0.16
        )

        lag2_value = sign * rng.uniform(
            0.03,
            0.09
        )

        A_matrices[0][target, source] = lag1_value

        if na >= 2:
            A_matrices[1][target, source] = lag2_value

        true_link_mask[target, source] = True

        true_link_values.append({
            "source": source,
            "target": target,
            "lag1_value_before_scaling": lag1_value,
            "lag2_value_before_scaling": lag2_value if na >= 2 else 0.0
        })

    radius_before = companion_spectral_radius(
        A_matrices
    )

    scale_factor = 1.0

    if radius_before >= target_radius:

        scale_factor = target_radius / (
            radius_before + 1e-12
        )

        A_matrices = [
            scale_factor * A
            for A in A_matrices
        ]

    radius_after = companion_spectral_radius(
        A_matrices
    )

    for entry in true_link_values:

        source = entry["source"]
        target = entry["target"]

        entry["lag1_value_after_scaling"] = A_matrices[0][target, source]

        if na >= 2:
            entry["lag2_value_after_scaling"] = A_matrices[1][target, source]
        else:
            entry["lag2_value_after_scaling"] = 0.0

        entry["scale_factor"] = scale_factor
        entry["radius_before"] = radius_before
        entry["radius_after"] = radius_after

    return {
        "A_matrices": A_matrices,
        "true_link_mask": true_link_mask,
        "true_link_values": pd.DataFrame(true_link_values),
        "radius_before": radius_before,
        "radius_after": radius_after,
        "scale_factor": scale_factor
    }


def make_exogenous_filters(
        n_sources,
        n_inputs,
        nb,
        random_seed
):
    rng = np.random.default_rng(
        random_seed
    )

    B_matrices = []

    for lag in range(nb):

        if lag == 0:
            scale = 0.45
        elif lag == 1:
            scale = 0.25
        else:
            scale = 0.15

        B = rng.normal(
            loc=0.0,
            scale=scale,
            size=(n_sources, n_inputs)
        )

        B_matrices.append(
            B
        )

    return B_matrices


def make_known_mixing_matrix(
        n_sources,
        random_seed,
        mixing_strength=0.05
):
    """
    Known observation matrix C.

    We keep this close to identity in 32B so that the main
    difficulty is network size, not unknown source localization.
    """

    rng = np.random.default_rng(
        random_seed
    )

    C = np.eye(
        n_sources
    )

    noise = rng.normal(
        loc=0.0,
        scale=mixing_strength / np.sqrt(n_sources),
        size=(n_sources, n_sources)
    )

    np.fill_diagonal(
        noise,
        0.0
    )

    C = C + noise

    return C


def simulate_sparse_20source_case(
        n_samples,
        burn_in,
        random_seed
):
    network = make_sparse_stable_var_matrices(
        n_sources=N_SOURCES,
        n_true_links=N_TRUE_LINKS,
        na=na,
        random_seed=random_seed,
        target_radius=0.82
    )

    B_matrices = make_exogenous_filters(
        n_sources=N_SOURCES,
        n_inputs=N_INPUTS,
        nb=nb,
        random_seed=random_seed + 100
    )

    C = make_known_mixing_matrix(
        n_sources=N_SOURCES,
        random_seed=random_seed + 200,
        mixing_strength=0.05
    )

    Q = 0.50 * np.eye(
        N_SOURCES
    )

    R = 0.60 * np.eye(
        N_SOURCES
    )

    u_total = generate_colored_input(
        n_samples=n_samples + burn_in,
        ar_coeff=0.95,
        noise_std=1.0,
        random_seed=random_seed + 300
    )

    sim = generate_ssm_varx_p_data(
        A_matrices=network["A_matrices"],
        B_matrices=B_matrices,
        u=u_total,
        Q=Q,
        R=R,
        C=C,
        D=None,
        burn_in=burn_in,
        random_seed=random_seed + 400,
        return_augmented=True
    )

    return {
        "x_true": sim["x"],
        "y_obs": sim["y"],
        "u": sim["u"],
        "C": C,
        "Q_true": Q,
        "R_true": R,
        "A_true": network["A_matrices"],
        "B_true": B_matrices,
        "true_link_mask": network["true_link_mask"],
        "true_link_values": network["true_link_values"],
        "radius_before": network["radius_before"],
        "radius_after": network["radius_after"],
        "scale_factor": network["scale_factor"]
    }


def fit_full_em_model(
        y_obs,
        u,
        C,
        shrinkage_spec,
        random_seed
):
    model = EMVARXPSSMKnownCConstrainedWarmStartShrinkage(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=MAX_ITER,
        tol=TOL,
        ridge_m_step=1e-4,
        covariance_floor=1e-6,
        R_init=0.60 * np.eye(N_SOURCES),
        Q_init=0.50 * np.eye(N_SOURCES),
        estimate_R=True,
        R_floor=R_FLOOR,
        zero_constraints=[],
        initial_parameters=None,
        jitter_scale=0.0,
        random_seed=random_seed,
        verbose=False,
        alpha_Q=shrinkage_spec["alpha_Q"],
        alpha_R=shrinkage_spec["alpha_R"],
        shrinkage_target_Q=shrinkage_spec["target_Q"],
        shrinkage_target_R=shrinkage_spec["target_R"]
    )

    model.fit(
        y=y_obs,
        u=u
    )

    model.fit_label = f"full_{shrinkage_spec['name']}"

    return model


def add_run_metadata(
        df,
        outer_run,
        random_seed,
        signal_type,
        shrinkage_spec,
        full_model=None,
        full_mse=np.nan,
        radius_true=np.nan,
        scale_factor=np.nan
):
    df = df.copy()

    df["outer_run"] = outer_run
    df["random_seed"] = random_seed
    df["n_sources"] = N_SOURCES
    df["n_true_links"] = N_TRUE_LINKS
    df["link_density"] = LINK_DENSITY

    df["signal_type"] = signal_type

    df["shrinkage_name"] = shrinkage_spec["name"]
    df["alpha_Q"] = shrinkage_spec["alpha_Q"]
    df["alpha_R"] = shrinkage_spec["alpha_R"]
    df["target_Q"] = shrinkage_spec["target_Q"]
    df["target_R"] = shrinkage_spec["target_R"]

    df["full_mse"] = full_mse
    df["true_network_radius"] = radius_true
    df["true_network_scale_factor"] = scale_factor

    if full_model is not None:

        df["full_log_likelihood"] = final_observed_log_likelihood(
            full_model
        )

        df["full_spectral_radius"] = full_model.spectral_radius()
        df["full_em_iterations"] = len(
            full_model.log_likelihoods
        )

        df["full_Q_trace"] = float(
            np.trace(
                full_model.Q
            )
        )

        df["full_R_trace"] = float(
            np.trace(
                full_model.R
            )
        )

        df["full_Q_offdiag_mean_abs"] = float(
            np.mean(
                np.abs(
                    full_model.Q
                    - np.diag(
                        np.diag(
                            full_model.Q
                        )
                    )
                )
            )
        )

        df["full_R_offdiag_mean_abs"] = float(
            np.mean(
                np.abs(
                    full_model.R
                    - np.diag(
                        np.diag(
                            full_model.R
                        )
                    )
                )
            )
        )

    else:

        df["full_log_likelihood"] = np.nan
        df["full_spectral_radius"] = np.nan
        df["full_em_iterations"] = np.nan
        df["full_Q_trace"] = np.nan
        df["full_R_trace"] = np.nan
        df["full_Q_offdiag_mean_abs"] = np.nan
        df["full_R_offdiag_mean_abs"] = np.nan

    return df


def summarize_link_strengths(
        results_df
):
    grouped = results_df.groupby([
        "signal_type",
        "shrinkage_name",
        "true_link"
    ])

    rows = []

    for keys, group in grouped:

        signal_type, shrinkage_name, true_link = keys

        rows.append({
            "signal_type": signal_type,
            "shrinkage_name": shrinkage_name,
            "true_link": true_link,
            "n_links": len(group),

            # --------------------------------------------------
            # Raw deviance summary
            # --------------------------------------------------
            "mean_raw_deviance": group["raw_deviance"].mean(),
            "median_raw_deviance": group["raw_deviance"].median(),
            "std_raw_deviance": group["raw_deviance"].std(),

            # --------------------------------------------------
            # Full bias term b_f
            # --------------------------------------------------
            "mean_full_bias_term": group["full_bias_term"].mean(),
            "median_full_bias_term": group["full_bias_term"].median(),
            "std_full_bias_term": group["full_bias_term"].std(),

            # --------------------------------------------------
            # Reduced bias term b_r
            # --------------------------------------------------
            "mean_reduced_bias_term": group["reduced_bias_term"].mean(),
            "median_reduced_bias_term": group["reduced_bias_term"].median(),
            "std_reduced_bias_term": group["reduced_bias_term"].std(),

            # --------------------------------------------------
            # Bias correction: -b_r + b_f
            # --------------------------------------------------
            "mean_bias_correction": group["bias_correction"].mean(),
            "median_bias_correction": group["bias_correction"].median(),
            "std_bias_correction": group["bias_correction"].std(),

            # --------------------------------------------------
            # Debiased deviance summary
            # --------------------------------------------------
            "mean_debiased_deviance": group["debiased_deviance"].mean(),
            "median_debiased_deviance": group["debiased_deviance"].median(),
            "std_debiased_deviance": group["debiased_deviance"].std(),

            # --------------------------------------------------
            # P-values
            # --------------------------------------------------
            "mean_raw_p_value": group["raw_p_value"].mean(),
            "median_raw_p_value": group["raw_p_value"].median(),

            "mean_debiased_p_value": group["debiased_p_value"].mean(),
            "median_debiased_p_value": group["debiased_p_value"].median(),

            # --------------------------------------------------
            # Model/recovery diagnostics
            # --------------------------------------------------
            "mean_full_mse": group["full_mse"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_full_Q_offdiag_abs": group["full_Q_offdiag_mean_abs"].mean(),
            "mean_full_R_offdiag_abs": group["full_R_offdiag_mean_abs"].mean()
        })

    return pd.DataFrame(
        rows
    )


def summarize_decision_rules(
        results_df
):
    decision_columns = [
        "raw_chi_detected",
        "debiased_chi_detected",
        "raw_fdr_detected",
        "debiased_fdr_detected",
        "raw_bic_detected",
        "debiased_bic_detected",
        "raw_D_gt_8p0_detected",
        "debiased_D_gt_8p0_detected",
        "raw_D_gt_10p0_detected",
        "debiased_D_gt_10p0_detected",
        "raw_D_gt_12p0_detected",
        "debiased_D_gt_12p0_detected",
        "raw_D_gt_15p0_detected",
        "debiased_D_gt_15p0_detected"
    ]

    grouped = results_df.groupby([
        "signal_type",
        "shrinkage_name"
    ])

    rows = []

    for keys, group in grouped:

        signal_type, shrinkage_name = keys

        for decision_column in decision_columns:

            metrics = compute_confusion_metrics(
                y_true=group["true_link"].values,
                y_pred=group[decision_column].values
            )

            row = {
                "signal_type": signal_type,
                "shrinkage_name": shrinkage_name,
                "decision_rule": decision_column,
                "n_tests": len(group)
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------
# Main run
# ------------------------------------------------------

all_link_rows = []
all_true_link_tables = []

for outer_run in range(N_OUTER_RUNS):

    print("\n" + "=" * 120)
    print(f"Experiment 32B | outer run {outer_run + 1}/{N_OUTER_RUNS}")
    print("=" * 120)

    seed = BASE_SEED + outer_run

    data = simulate_sparse_20source_case(
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        random_seed=seed
    )

    x_true = data["x_true"]
    y_obs = data["y_obs"]
    u = data["u"]
    C = data["C"]

    true_link_mask = data["true_link_mask"]

    print("True links:", int(np.sum(true_link_mask)))
    print("True network radius:", data["radius_after"])
    print("Scale factor:", data["scale_factor"])

    true_link_table = data["true_link_values"].copy()
    true_link_table["outer_run"] = outer_run
    true_link_table["random_seed"] = seed

    all_true_link_tables.append(
        true_link_table
    )

    # --------------------------------------------------
    # Oracle latent benchmark.
    # --------------------------------------------------

    print("\nComputing oracle-latent de-biased network...")

    oracle_df = compute_all_pair_debiased_varx_network(
        x=x_true,
        u=u,
        true_link_mask=true_link_mask,
        na=na,
        nb=nb,
        ridge_lambda=RIDGE_LAMBDA_DEBIAS,
        alpha=ALPHA,
        custom_thresholds=CUSTOM_THRESHOLDS
    )

    oracle_df = add_run_metadata(
        df=oracle_df,
        outer_run=outer_run,
        random_seed=seed,
        signal_type="oracle_latent",
        shrinkage_spec={
            "name": "oracle",
            "alpha_Q": np.nan,
            "alpha_R": np.nan,
            "target_Q": "none",
            "target_R": "none"
        },
        full_model=None,
        full_mse=0.0,
        radius_true=data["radius_after"],
        scale_factor=data["scale_factor"]
    )

    all_link_rows.append(
        oracle_df
    )

    # --------------------------------------------------
    # Smoothed latent benchmark from EM.
    # --------------------------------------------------

    for shrinkage_spec in SHRINKAGE_SPECS:

        print("\nFitting full EM model with shrinkage:", shrinkage_spec["name"])

        full_model = fit_full_em_model(
            y_obs=y_obs,
            u=u,
            C=C,
            shrinkage_spec=shrinkage_spec,
            random_seed=seed + 500
        )

        x_smooth = full_model.smoothed_latent_state()

        full_mse = float(
            np.mean(
                (x_smooth - x_true) ** 2
            )
        )

        print("Full LL:", final_observed_log_likelihood(full_model))
        print("Full EM iterations:", len(full_model.log_likelihoods))
        print("Full spectral radius:", full_model.spectral_radius())
        print("Smoothed latent MSE:", full_mse)
        print("Q offdiag mean abs:", np.mean(np.abs(full_model.Q - np.diag(np.diag(full_model.Q)))))
        print("R offdiag mean abs:", np.mean(np.abs(full_model.R - np.diag(np.diag(full_model.R)))))

        print("Computing smoothed-latent de-biased network...")

        smooth_df = compute_all_pair_debiased_varx_network(
            x=x_smooth,
            u=u,
            true_link_mask=true_link_mask,
            na=na,
            nb=nb,
            ridge_lambda=RIDGE_LAMBDA_DEBIAS,
            alpha=ALPHA,
            custom_thresholds=CUSTOM_THRESHOLDS
        )

        smooth_df = add_run_metadata(
            df=smooth_df,
            outer_run=outer_run,
            random_seed=seed,
            signal_type="smoothed_latent",
            shrinkage_spec=shrinkage_spec,
            full_model=full_model,
            full_mse=full_mse,
            radius_true=data["radius_after"],
            scale_factor=data["scale_factor"]
        )

        all_link_rows.append(
            smooth_df
        )

    # --------------------------------------------------
    # Save partial progress.
    # --------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    partial_df = pd.concat(
        all_link_rows,
        ignore_index=True
    )

    partial_df.to_csv(
        f"results/experiment_32b_sparse_20source_debiased_network_results_partial_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv",
        index=False
    )

    pd.concat(
        all_true_link_tables,
        ignore_index=True
    ).to_csv(
        f"results/experiment_32b_sparse_20source_true_links_partial_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv",
        index=False
    )


# ------------------------------------------------------
# Final summaries
# ------------------------------------------------------

results_df = pd.concat(
    all_link_rows,
    ignore_index=True
)

true_links_df = pd.concat(
    all_true_link_tables,
    ignore_index=True
)

link_strength_summary_df = summarize_link_strengths(
    results_df
)

decision_summary_df = summarize_decision_rules(
    results_df
)


print("\n" + "=" * 160)
print("Experiment 32B link-strength summary")
print("=" * 160)

link_strength_display_columns = [
    "signal_type",
    "shrinkage_name",
    "true_link",
    "n_links",

    "mean_raw_deviance",
    "mean_full_bias_term",
    "mean_reduced_bias_term",
    "mean_bias_correction",
    "mean_debiased_deviance",

    "median_raw_deviance",
    "median_full_bias_term",
    "median_reduced_bias_term",
    "median_bias_correction",
    "median_debiased_deviance",

    "mean_raw_p_value",
    "mean_debiased_p_value",

    "mean_full_mse",
    "mean_full_spectral_radius"
]

print(
    link_strength_summary_df[link_strength_display_columns].to_string(
        index=False
    )
)


print("\n" + "=" * 160)
print("Experiment 32B decision-rule summary")
print("=" * 160)

decision_display_columns = [
    "signal_type",
    "shrinkage_name",
    "decision_rule",
    "fpr",
    "tpr",
    "precision",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn"
]

print(
    decision_summary_df[decision_display_columns].to_string(
        index=False
    )
)


# ------------------------------------------------------
# Save outputs.
# ------------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

results_path = f"results/experiment_32b_sparse_20source_debiased_network_results_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv"
true_links_path = f"results/experiment_32b_sparse_20source_true_links_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv"
link_strength_summary_path = f"results/experiment_32b_sparse_20source_link_strength_summary_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv"
decision_summary_path = f"results/experiment_32b_sparse_20source_decision_summary_{N_SAMPLES}samples_{N_OUTER_RUNS}outerruns.csv"

results_df.to_csv(
    results_path,
    index=False
)

true_links_df.to_csv(
    true_links_path,
    index=False
)

link_strength_summary_df.to_csv(
    link_strength_summary_path,
    index=False
)

decision_summary_df.to_csv(
    decision_summary_path,
    index=False
)

print("\nSaved detailed link results to:")
print(results_path)

print("\nSaved true link table to:")
print(true_links_path)

print("\nSaved link-strength summary to:")
print(link_strength_summary_path)

print("\nSaved decision summary to:")
print(decision_summary_path)


print("\nInterpretation guide")
print("--------------------")
print("oracle_latent is the best-case benchmark using the true latent states.")
print("smoothed_latent is the realistic result after state-space EM recovery.")
print("The main statistic for this experiment is debiased_deviance.")
print("The main decision rules to inspect are debiased_fdr_detected and debiased_D_gt_10p0_detected.")
print("A successful 20-source run should have low FPR and nontrivial TPR.")
print("If oracle_latent works but smoothed_latent fails, the bottleneck is latent-state recovery.")
print("If both fail, the bottleneck is high-dimensional VARX network inference.")