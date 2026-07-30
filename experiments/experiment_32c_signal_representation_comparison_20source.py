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
# Experiment 32C
# Signal-representation comparison for 20-source network.
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

# Use 20 to match 32B. Use 5 or 10 first for a quick smoke test.
N_OUTER_RUNS = 20

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

RIDGE_LAMBDA_DEBIAS = 1.0

BASE_SEED = 3300000

CUSTOM_THRESHOLDS = (
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0
)

SHRINKAGE_SPEC = {
    "name": "QR_0p20",
    "alpha_Q": 0.20,
    "alpha_R": 0.20,
    "target_Q": "spherical",
    "target_R": "spherical"
}


print("\nExperiment 32C: signal representation comparison")
print("------------------------------------------------")
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
print("shrinkage:", SHRINKAGE_SPEC["name"])


def make_sparse_stable_var_matrices(
        n_sources,
        n_true_links,
        na,
        random_seed,
        target_radius=0.82
):
    """
    Generate a sparse stable VAR network.

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

    Observation model:
        y_t = C x_t + v_t
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
        alpha_Q=SHRINKAGE_SPEC["alpha_Q"],
        alpha_R=SHRINKAGE_SPEC["alpha_R"],
        shrinkage_target_Q=SHRINKAGE_SPEC["target_Q"],
        shrinkage_target_R=SHRINKAGE_SPEC["target_R"]
    )

    model.fit(
        y=y_obs,
        u=u
    )

    model.fit_label = f"full_{SHRINKAGE_SPEC['name']}"

    return model


def make_pinv_proxy(
        y_obs,
        C
):
    """
    Convert observation y into source proxy using known C.

    Column convention:
        y_t = C x_t + v_t

    Row-array convention:
        y_obs[t, :] = x[t, :] C^T + v[t, :]

    Therefore:
        x_proxy[t, :] = y_obs[t, :] (C^+)^T
    """

    C_pinv = np.linalg.pinv(
        C
    )

    x_proxy = y_obs @ C_pinv.T

    return x_proxy


def summarize_matrix_condition(
        C
):
    return {
        "C_condition_number": float(
            np.linalg.cond(
                C
            )
        ),
        "C_offdiag_mean_abs": float(
            np.mean(
                np.abs(
                    C
                    - np.diag(
                        np.diag(
                            C
                        )
                    )
                )
            )
        )
    }


def add_signal_metadata(
        df,
        outer_run,
        random_seed,
        signal_type,
        signal_mse,
        signal_correlation_mean,
        signal_correlation_median,
        C_diagnostics,
        full_model=None,
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
    df["signal_mse_to_true_x"] = signal_mse
    df["signal_correlation_mean_to_true_x"] = signal_correlation_mean
    df["signal_correlation_median_to_true_x"] = signal_correlation_median

    df["true_network_radius"] = radius_true
    df["true_network_scale_factor"] = scale_factor

    df["C_condition_number"] = C_diagnostics["C_condition_number"]
    df["C_offdiag_mean_abs"] = C_diagnostics["C_offdiag_mean_abs"]

    df["shrinkage_name"] = SHRINKAGE_SPEC["name"]
    df["alpha_Q"] = SHRINKAGE_SPEC["alpha_Q"]
    df["alpha_R"] = SHRINKAGE_SPEC["alpha_R"]
    df["target_Q"] = SHRINKAGE_SPEC["target_Q"]
    df["target_R"] = SHRINKAGE_SPEC["target_R"]

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


def compute_signal_recovery_metrics(
        signal,
        x_true
):
    """
    Compute MSE and source-wise correlation with true latent states.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    x_true = np.asarray(
        x_true,
        dtype=float
    )

    mse = float(
        np.mean(
            (signal - x_true) ** 2
        )
    )

    correlations = []

    for i in range(x_true.shape[1]):

        a = signal[:, i]
        b = x_true[:, i]

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            correlations.append(
                np.nan
            )
        else:
            correlations.append(
                np.corrcoef(
                    a,
                    b
                )[0, 1]
            )

    correlations = np.asarray(
        correlations,
        dtype=float
    )

    return {
        "mse": mse,
        "corr_mean": float(
            np.nanmean(
                correlations
            )
        ),
        "corr_median": float(
            np.nanmedian(
                correlations
            )
        )
    }


def compute_network_for_signal(
        signal,
        u,
        true_link_mask,
        signal_type,
        outer_run,
        random_seed,
        x_true,
        C_diagnostics,
        full_model,
        radius_true,
        scale_factor
):
    recovery = compute_signal_recovery_metrics(
        signal=signal,
        x_true=x_true
    )

    network_df = compute_all_pair_debiased_varx_network(
        x=signal,
        u=u,
        true_link_mask=true_link_mask,
        na=na,
        nb=nb,
        ridge_lambda=RIDGE_LAMBDA_DEBIAS,
        alpha=ALPHA,
        custom_thresholds=CUSTOM_THRESHOLDS
    )

    network_df = add_signal_metadata(
        df=network_df,
        outer_run=outer_run,
        random_seed=random_seed,
        signal_type=signal_type,
        signal_mse=recovery["mse"],
        signal_correlation_mean=recovery["corr_mean"],
        signal_correlation_median=recovery["corr_median"],
        C_diagnostics=C_diagnostics,
        full_model=full_model,
        radius_true=radius_true,
        scale_factor=scale_factor
    )

    return network_df


def summarize_link_strengths(
        results_df
):
    grouped = results_df.groupby([
        "signal_type",
        "true_link"
    ])

    rows = []

    for keys, group in grouped:

        signal_type, true_link = keys

        rows.append({
            "signal_type": signal_type,
            "true_link": true_link,
            "n_links": len(group),

            # Raw deviance
            "mean_raw_deviance": group["raw_deviance"].mean(),
            "median_raw_deviance": group["raw_deviance"].median(),
            "std_raw_deviance": group["raw_deviance"].std(),

            # Full bias term b_f
            "mean_full_bias_term": group["full_bias_term"].mean(),
            "median_full_bias_term": group["full_bias_term"].median(),
            "std_full_bias_term": group["full_bias_term"].std(),

            # Reduced bias term b_r
            "mean_reduced_bias_term": group["reduced_bias_term"].mean(),
            "median_reduced_bias_term": group["reduced_bias_term"].median(),
            "std_reduced_bias_term": group["reduced_bias_term"].std(),

            # Bias correction -b_r + b_f
            "mean_bias_correction": group["bias_correction"].mean(),
            "median_bias_correction": group["bias_correction"].median(),
            "std_bias_correction": group["bias_correction"].std(),

            # Debiased deviance
            "mean_debiased_deviance": group["debiased_deviance"].mean(),
            "median_debiased_deviance": group["debiased_deviance"].median(),
            "std_debiased_deviance": group["debiased_deviance"].std(),

            # P-values
            "mean_raw_p_value": group["raw_p_value"].mean(),
            "median_raw_p_value": group["raw_p_value"].median(),
            "mean_debiased_p_value": group["debiased_p_value"].mean(),
            "median_debiased_p_value": group["debiased_p_value"].median(),

            # Signal recovery
            "mean_signal_mse_to_true_x": group["signal_mse_to_true_x"].mean(),
            "mean_signal_corr_to_true_x": group["signal_correlation_mean_to_true_x"].mean(),
            "median_signal_corr_to_true_x": group["signal_correlation_median_to_true_x"].mean(),

            # State-space diagnostics
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_full_Q_offdiag_abs": group["full_Q_offdiag_mean_abs"].mean(),
            "mean_full_R_offdiag_abs": group["full_R_offdiag_mean_abs"].mean(),

            # Mixing matrix
            "mean_C_condition_number": group["C_condition_number"].mean(),
            "mean_C_offdiag_abs": group["C_offdiag_mean_abs"].mean()
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
        "debiased_D_gt_15p0_detected",
        "raw_D_gt_20p0_detected",
        "debiased_D_gt_20p0_detected",
        "raw_D_gt_25p0_detected",
        "debiased_D_gt_25p0_detected"
    ]

    grouped = results_df.groupby([
        "signal_type"
    ])

    rows = []

    for signal_type, group in grouped:

        for decision_column in decision_columns:

            metrics = compute_confusion_metrics(
                y_true=group["true_link"].values,
                y_pred=group[decision_column].values
            )

            row = {
                "signal_type": signal_type,
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
    print(f"Experiment 32C | outer run {outer_run + 1}/{N_OUTER_RUNS}")
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

    C_diagnostics = summarize_matrix_condition(
        C
    )

    print("True links:", int(np.sum(true_link_mask)))
    print("True network radius:", data["radius_after"])
    print("Scale factor:", data["scale_factor"])
    print("C condition number:", C_diagnostics["C_condition_number"])
    print("C offdiag mean abs:", C_diagnostics["C_offdiag_mean_abs"])

    true_link_table = data["true_link_values"].copy()
    true_link_table["outer_run"] = outer_run
    true_link_table["random_seed"] = seed

    all_true_link_tables.append(
        true_link_table
    )

    # --------------------------------------------------
    # Fit EM once per outer run.
    # --------------------------------------------------

    print("\nFitting full EM model...")

    full_model = fit_full_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        random_seed=seed + 500
    )

    x_smooth = full_model.smoothed_latent_state()

    print("Full LL:", final_observed_log_likelihood(full_model))
    print("Full EM iterations:", len(full_model.log_likelihoods))
    print("Full spectral radius:", full_model.spectral_radius())
    print("Q offdiag mean abs:", np.mean(np.abs(full_model.Q - np.diag(np.diag(full_model.Q)))))
    print("R offdiag mean abs:", np.mean(np.abs(full_model.R - np.diag(np.diag(full_model.R)))))

    # --------------------------------------------------
    # Build pseudo-inverse proxy.
    # --------------------------------------------------

    x_pinv_proxy = make_pinv_proxy(
        y_obs=y_obs,
        C=C
    )

    # --------------------------------------------------
    # Compare signal representations.
    # --------------------------------------------------

    signal_specs = [
        {
            "signal_type": "oracle_latent",
            "signal": x_true,
            "full_model": None
        },
        {
            "signal_type": "observed_y",
            "signal": y_obs,
            "full_model": None
        },
        {
            "signal_type": "pinv_proxy",
            "signal": x_pinv_proxy,
            "full_model": None
        },
        {
            "signal_type": "smoothed_latent",
            "signal": x_smooth,
            "full_model": full_model
        }
    ]

    for spec in signal_specs:

        print("\nComputing network for:", spec["signal_type"])

        network_df = compute_network_for_signal(
            signal=spec["signal"],
            u=u,
            true_link_mask=true_link_mask,
            signal_type=spec["signal_type"],
            outer_run=outer_run,
            random_seed=seed,
            x_true=x_true,
            C_diagnostics=C_diagnostics,
            full_model=spec["full_model"],
            radius_true=data["radius_after"],
            scale_factor=data["scale_factor"]
        )

        all_link_rows.append(
            network_df
        )

        temp_summary = summarize_decision_rules(
            network_df
        )

        temp_display = temp_summary[
            temp_summary["decision_rule"].isin([
                "debiased_chi_detected",
                "debiased_fdr_detected",
                "debiased_D_gt_10p0_detected",
                "debiased_D_gt_15p0_detected",
                "debiased_D_gt_20p0_detected"
            ])
        ]

        print(
            temp_display[
                [
                    "signal_type",
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
            ].to_string(
                index=False
            )
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
        "results/experiment_32c_signal_representation_comparison_results_partial.csv",
        index=False
    )

    pd.concat(
        all_true_link_tables,
        ignore_index=True
    ).to_csv(
        "results/experiment_32c_signal_representation_comparison_true_links_partial.csv",
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
print("Experiment 32C link-strength summary")
print("=" * 160)

link_strength_display_columns = [
    "signal_type",
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

    "mean_signal_mse_to_true_x",
    "mean_signal_corr_to_true_x",
    "median_signal_corr_to_true_x",

    "mean_full_spectral_radius",
    "mean_full_Q_offdiag_abs",
    "mean_full_R_offdiag_abs"
]

print(
    link_strength_summary_df[link_strength_display_columns].to_string(
        index=False
    )
)


print("\n" + "=" * 160)
print("Experiment 32C decision-rule summary")
print("=" * 160)

decision_display_columns = [
    "signal_type",
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

results_path = "results/experiment_32c_signal_representation_comparison_results.csv"
true_links_path = "results/experiment_32c_signal_representation_comparison_true_links.csv"
link_strength_summary_path = "results/experiment_32c_signal_representation_comparison_link_strength_summary.csv"
decision_summary_path = "results/experiment_32c_signal_representation_comparison_decision_summary.csv"

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
print("oracle_latent: true latent state, best-case network inference.")
print("observed_y: noisy/mixed observation used directly as if it were sources.")
print("pinv_proxy: C-pseudo-inverse source proxy.")
print("smoothed_latent: EM/Kalman-smoothed latent state.")
print("If observed_y and pinv_proxy perform better than smoothed_latent, EM smoothing is the main source of spurious links.")
print("If all non-oracle signals fail, then observation noise/source leakage is the bottleneck.")
print("Compare raw_deviance, debiased_deviance, full_bias_term, and reduced_bias_term in the link-strength summary.")