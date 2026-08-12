import os

# Each thread handles a complete EM fit. Limiting BLAS threads prevents
# nested parallelism from multiplying OpenBLAS/MKL worker threads.
BLAS_THREADS_PER_WORKER = int(
    os.environ.get("EXPERIMENT_33A_BLAS_THREADS", "1")
)
os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

import pickle
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("EXPERIMENT_33A_RESULTS_DIR", "results")

from src.varx.varx_generator import (
    generate_colored_input
)

from src.ssm.ssm_varx_p_simulator import (
    generate_ssm_varx_p_data
)

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov
)
from src.ssm.em_varx_p_known_c_l1_mstep import NUMBA_AVAILABLE

from src.stats.observed_likelihood_deviance import (
    final_observed_log_likelihood
)

from src.stats.scalable_debiased_varx_network import (
    companion_spectral_radius,
    compute_all_pair_debiased_varx_network,
    compute_confusion_metrics
)


# ------------------------------------------------------
# Experiment 33A
# Sample-size and parameter-recovery diagnostic for the 32I estimator
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

T_VALUES = [500, 1000, 2000]
BURN_IN = 300

# Use 5 or 10 for a quick smoke test.
# Use 20 for the full experiment.
N_OUTER_RUNS = 20

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

RIDGE_LAMBDA_DEBIAS = 1.0

BASE_SEED = 3700000

CUSTOM_THRESHOLDS = (
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0
)

LAMBDA_A_GROUP_FRACTION_GRID = [
    0.003,
    0.010,
    0.030,
    0.100
]

# Four workers is a conservative default for the memory-heavy EM fits.
# Override explicitly for a particular machine, for example:
#   $env:EXPERIMENT_33A_WORKERS = "8"
N_WORKERS = int(
    os.environ.get(
        "EXPERIMENT_33A_WORKERS",
        str(min(4, os.cpu_count() or 1, len(LAMBDA_A_GROUP_FRACTION_GRID)))
    )
)

if N_WORKERS < 1:
    raise ValueError("EXPERIMENT_33A_WORKERS must be at least 1.")

SHRINKAGE_SPEC = {
    "name": "fixed_Q0p50_R0p60",
    "alpha_Q": 0.0,
    "alpha_R": 0.0,
    "target_Q": "spherical",
    "target_R": "spherical"
}


print("\nExperiment 33A: sample-size and parameter-recovery diagnostic")
print("----------------------------------------------------------")
print("N sources:", N_SOURCES)
print("Link density:", LINK_DENSITY)
print("N true links:", N_TRUE_LINKS)
print("T values:", T_VALUES)
print("Burn-in:", BURN_IN)
print("N outer runs:", N_OUTER_RUNS)
print("na:", na)
print("nb:", nb)
print("alpha:", ALPHA)
print("R floor:", R_FLOOR)
print("max EM iterations:", MAX_ITER)
print("tol:", TOL)
print("debiased ridge lambda:", RIDGE_LAMBDA_DEBIAS)
print("lambda_A_group_fraction grid:", LAMBDA_A_GROUP_FRACTION_GRID)
print("covariance control:", SHRINKAGE_SPEC["name"])
print("parallel worker threads:", N_WORKERS)
print("BLAS threads per worker:", BLAS_THREADS_PER_WORKER)
print("Numba group solver:", "enabled" if NUMBA_AVAILABLE else "fallback")


# ------------------------------------------------------
# Simulation helpers
# ------------------------------------------------------

def make_sparse_stable_var_matrices(
        n_sources,
        n_true_links,
        na,
        random_seed,
        target_radius=0.82
):
    """
    Generate sparse stable VAR network.

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


# ------------------------------------------------------
# Companion Kalman filter and RTS smoother
# ------------------------------------------------------

def build_companion_state_space(
        A_matrices,
        B_matrices,
        Q,
        C,
        R
):
    """
    Build augmented companion-form state-space matrices.

    State:
        z_t = [x_t, x_{t-1}, ..., x_{t-na+1}]

    Dynamics:
        x_t = A1 x_{t-1} + ... + Ap x_{t-p}
              + B0 u_t + B1 u_{t-1} + ...
              + w_t

    Observation:
        y_t = C x_t + v_t
    """

    A_matrices = [
        np.asarray(A, dtype=float)
        for A in A_matrices
    ]

    B_matrices = [
        np.asarray(B, dtype=float)
        for B in B_matrices
    ]

    Q = np.asarray(
        Q,
        dtype=float
    )

    C = np.asarray(
        C,
        dtype=float
    )

    R = np.asarray(
        R,
        dtype=float
    )

    p = len(
        A_matrices
    )

    n_sources = A_matrices[0].shape[0]
    n_state = n_sources * p

    F = np.zeros(
        (n_state, n_state)
    )

    F[:n_sources, :n_state] = np.hstack(
        A_matrices
    )

    if p > 1:
        F[n_sources:, :-n_sources] = np.eye(
            n_sources * (p - 1)
        )

    Q_aug = np.zeros(
        (n_state, n_state)
    )

    Q_aug[:n_sources, :n_sources] = Q

    H = np.zeros(
        (C.shape[0], n_state)
    )

    H[:, :n_sources] = C

    return {
        "F": F,
        "Q_aug": Q_aug,
        "H": H,
        "R": R,
        "B_matrices": B_matrices,
        "n_sources": n_sources,
        "n_state": n_state,
        "p": p
    }


def exogenous_augmented_input(
        u,
        t,
        B_matrices,
        n_sources,
        n_state
):
    """
    Build augmented exogenous input vector for state time t.
    """

    if u.ndim == 1:
        u = u[:, None]

    d = np.zeros(
        n_state
    )

    top = np.zeros(
        n_sources
    )

    for lag, B in enumerate(B_matrices):

        idx = t - lag

        if idx >= 0:
            top += B @ u[idx]

    d[:n_sources] = top

    return d


def kalman_filter_and_rts_smoother(
        y,
        u,
        A_matrices,
        B_matrices,
        Q,
        C,
        R,
        initial_covariance_scale=10.0
):
    """
    Run causal Kalman filtering and acausal RTS smoothing
    using supplied parameters.

    Returns:
        filtered_x:
            E[x_t | y_1, ..., y_t]

        smoothed_x:
            E[x_t | y_1, ..., y_T]
    """

    y = np.asarray(
        y,
        dtype=float
    )

    u = np.asarray(
        u,
        dtype=float
    )

    if u.ndim == 1:
        u = u[:, None]

    system = build_companion_state_space(
        A_matrices=A_matrices,
        B_matrices=B_matrices,
        Q=Q,
        C=C,
        R=R
    )

    F = system["F"]
    Q_aug = system["Q_aug"]
    H = system["H"]
    R = system["R"]
    B_matrices = system["B_matrices"]
    n_sources = system["n_sources"]
    n_state = system["n_state"]

    T = y.shape[0]

    I = np.eye(
        n_state
    )

    m_pred = np.zeros(
        (T, n_state)
    )

    P_pred = np.zeros(
        (T, n_state, n_state)
    )

    m_filt = np.zeros(
        (T, n_state)
    )

    P_filt = np.zeros(
        (T, n_state, n_state)
    )

    m_prev = np.zeros(
        n_state
    )

    P_prev = initial_covariance_scale * np.eye(
        n_state
    )

    for t in range(T):

        d_t = exogenous_augmented_input(
            u=u,
            t=t,
            B_matrices=B_matrices,
            n_sources=n_sources,
            n_state=n_state
        )

        m_pred[t] = F @ m_prev + d_t

        P_pred[t] = (
            F @ P_prev @ F.T
            + Q_aug
        )

        P_pred[t] = 0.5 * (
            P_pred[t]
            + P_pred[t].T
        )

        innovation = y[t] - H @ m_pred[t]

        S = (
            H @ P_pred[t] @ H.T
            + R
        )

        S = 0.5 * (
            S
            + S.T
        )

        try:
            K = np.linalg.solve(
                S.T,
                (P_pred[t] @ H.T).T
            ).T
        except np.linalg.LinAlgError:
            K = (
                P_pred[t]
                @ H.T
                @ np.linalg.pinv(S)
            )

        m_filt[t] = (
            m_pred[t]
            + K @ innovation
        )

        P_filt[t] = (
            (I - K @ H)
            @ P_pred[t]
            @ (I - K @ H).T
            + K @ R @ K.T
        )

        P_filt[t] = 0.5 * (
            P_filt[t]
            + P_filt[t].T
        )

        m_prev = m_filt[t]
        P_prev = P_filt[t]

    m_smooth = np.zeros_like(
        m_filt
    )

    P_smooth = np.zeros_like(
        P_filt
    )

    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):

        try:
            J = np.linalg.solve(
                P_pred[t + 1].T,
                (P_filt[t] @ F.T).T
            ).T
        except np.linalg.LinAlgError:
            J = (
                P_filt[t]
                @ F.T
                @ np.linalg.pinv(
                    P_pred[t + 1]
                )
            )

        m_smooth[t] = (
            m_filt[t]
            + J @ (
                m_smooth[t + 1]
                - m_pred[t + 1]
            )
        )

        P_smooth[t] = (
            P_filt[t]
            + J @ (
                P_smooth[t + 1]
                - P_pred[t + 1]
            ) @ J.T
        )

        P_smooth[t] = 0.5 * (
            P_smooth[t]
            + P_smooth[t].T
        )

    return {
        "filtered_x": m_filt[:, :n_sources],
        "smoothed_x": m_smooth[:, :n_sources],
        "filtered_state": m_filt,
        "smoothed_state": m_smooth,
        "filtered_covariance": P_filt,
        "smoothed_covariance": P_smooth
    }


# ------------------------------------------------------
# Model fitting and signal helpers
# ------------------------------------------------------

def fit_posterior_moment_group_lasso_fixed_cov_em_model(
        y_obs,
        u,
        C,
        lambda_A_group_fraction,
        random_seed
):
    model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=MAX_ITER,
        tol=TOL,

        # Kept for compatibility with inherited class.
        # The main A/B update is done by the posterior-moment group solver.
        ridge_m_step=1e-4,

        covariance_floor=1e-6,
        R_init=0.60 * np.eye(N_SOURCES),
        Q_init=0.50 * np.eye(N_SOURCES),
        estimate_Q=False,
        estimate_R=False,
        R_floor=R_FLOOR,
        zero_constraints=[],
        initial_parameters=None,
        jitter_scale=0.0,
        random_seed=random_seed,
        verbose=False,

        # Inert while Q/R are fixed; retained for constructor compatibility.
        alpha_Q=SHRINKAGE_SPEC["alpha_Q"],
        alpha_R=SHRINKAGE_SPEC["alpha_R"],
        shrinkage_target_Q=SHRINKAGE_SPEC["target_Q"],
        shrinkage_target_R=SHRINKAGE_SPEC["target_R"],

        # Target-relative posterior-moment group-lasso M-step.
        lambda_A_group_fraction=lambda_A_group_fraction,

        # Weak ridge for numerical stability.
        ridge_A_offdiag=1e-4,
        ridge_A_diag=1e-4,
        ridge_B=1e-4,

        group_solver_max_iter=5000,
        group_solver_tol=1e-7,

        stabilize_A=True,
        target_radius=0.98
    )

    model.fit(
        y=y_obs,
        u=u
    )

    model.fit_label = (
        "posterior_moment_group_lasso_fixed_cov_fraction_"
        f"{lambda_A_group_fraction}"
    )

    return model


def make_pinv_proxy(
        y_obs,
        C
):
    """
    Convert observation y into source proxy using known C.

    Observation:
        y_t = C x_t + v_t

    Row-array convention:
        y_obs[t, :] = x[t, :] C.T + v[t, :]

    Therefore:
        x_proxy[t, :] = y_obs[t, :] (C^+)^T
    """

    C_pinv = np.linalg.pinv(
        C
    )

    return y_obs @ C_pinv.T


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


def compute_signal_recovery_metrics(
        signal,
        x_true
):
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


"""Legacy 32I helper block omitted from Experiment 33A.

def safe_model_diagnostic(
        full_model,
        attribute_name,
        default_value=np.nan
):
    if full_model is None:
        return default_value

    # The remainder of this quoted block is retained only as inert historical text.
        f"lambda_A_group_fraction={lambda_A_group_fraction}"
    )
    print("-" * 100)

    full_model = fit_posterior_moment_group_lasso_fixed_cov_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        lambda_A_group_fraction=lambda_A_group_fraction,
        random_seed=seed + 500
    )

    print("Full LL:", final_observed_log_likelihood(full_model))
    print("EM iterations:", len(full_model.log_likelihoods))
    print("Spectral radius:", full_model.spectral_radius())
    print("Q trace:", float(np.trace(full_model.Q)))
    print("R trace:", float(np.trace(full_model.R)))
    print(
        "Nonzero offdiag A coefficients:",
        full_model.count_nonzero_offdiag_A_coefficients()
    )
    print(
        "Nonzero total A coefficients:",
        full_model.count_nonzero_total_A_coefficients()
    )
    print(
        "Nonzero offdiag A groups:",
        full_model.count_nonzero_offdiag_A_groups()
    )
    print("Nonzero total A groups:", full_model.count_nonzero_total_A_groups())
    print("Mean offdiag group norm:", full_model.mean_offdiag_A_group_norm())
    print("Median offdiag group norm:", full_model.median_offdiag_A_group_norm())
    print("Max offdiag group norm:", full_model.max_offdiag_A_group_norm())
    print("Mean abs offdiag A:", full_model.mean_abs_offdiag_A())
    print("Max abs offdiag A:", full_model.max_abs_offdiag_A())
    print("Mean effective group lambda:", full_model.mean_lambda_A_group_effective)
    print("Median effective group lambda:", full_model.median_lambda_A_group_effective)
    print("Min effective group lambda:", full_model.min_lambda_A_group_effective)
    print("Max effective group lambda:", full_model.max_lambda_A_group_effective)
    print(
        "Mean moment condition number:",
        full_model.moment_solver_condition_number_mean
    )
    print(
        "Max moment condition number:",
        full_model.moment_solver_condition_number_max
    )

    if len(full_model.group_solver_converged_history) > 0:
        print(
            "Group solver converged last:",
            full_model.group_solver_converged_history[-1]
        )
        print(
            "Group solver mean iterations last:",
            full_model.group_solver_iterations_history[-1]
        )
        print(
            "Group solver objective last:",
            full_model.group_solver_objective_history[-1]
        )

    A_support_df = build_A_support_table(
        model=full_model,
        true_link_mask=task["true_link_mask"],
        outer_run=task["outer_run"],
        random_seed=seed,
        lambda_A_group_fraction=lambda_A_group_fraction
    )
    A_support_display = summarize_A_support(A_support_df)
    A_support_display = A_support_display[
        A_support_display["selection_rule"].isin([
            "selected_top_19_group_norm",
            "selected_group_norm_gt_0p01",
            "selected_group_norm_gt_0p03"
        ])
    ]
    print("\nSelected A-support diagnostics:")
    print(
        A_support_display[
            [
                "lambda_A_group_fraction",
                "selection_rule",
                "fpr",
                "tpr",
                "precision",
                "f1",
                "tp",
                "fp",
                "tn",
                "fn"
            ]
        ].to_string(index=False)
    )

    posterior = kalman_filter_and_rts_smoother(
        y=y_obs,
        u=u,
        A_matrices=full_model.A_matrices,
        B_matrices=full_model.B_matrices,
        Q=full_model.Q,
        C=C,
        R=full_model.R
    )

    signal_specs = [
        {
            "signal_type": "em_filtered",
            "signal": posterior["filtered_x"]
        },
        {
            "signal_type": "em_smoothed",
            "signal": posterior["smoothed_x"]
        }
    ]

    network_dfs = []

    for spec in signal_specs:
        print(
            "\nComputing network for:",
            spec["signal_type"],
            "lambda_A_group_fraction:",
            lambda_A_group_fraction
        )

        network_df = compute_network_for_signal(
            signal=spec["signal"],
            u=u,
            true_link_mask=task["true_link_mask"],
            signal_type=spec["signal_type"],
            lambda_A_group_fraction=lambda_A_group_fraction,
            outer_run=task["outer_run"],
            random_seed=seed,
            x_true=task["x_true"],
            C_diagnostics=task["C_diagnostics"],
            full_model=full_model,
            radius_true=task["radius_true"],
            scale_factor=task["scale_factor"]
        )
        network_dfs.append(network_df)

    return float(lambda_A_group_fraction), network_dfs, A_support_df


"""


def add_metadata(
        df, outer_run, random_seed, signal_type, lambda_A_group_fraction,
        signal_mse, signal_corr_mean, signal_corr_median, C_diagnostics,
        full_model, radius_true, scale_factor
):
    df = df.copy()
    values = {
        "outer_run": outer_run, "random_seed": random_seed,
        "n_sources": N_SOURCES, "n_true_links": N_TRUE_LINKS,
        "link_density": LINK_DENSITY, "signal_type": signal_type,
        "lambda_A_group_fraction": lambda_A_group_fraction,
        "signal_mse_to_true_x": signal_mse,
        "signal_correlation_mean_to_true_x": signal_corr_mean,
        "signal_correlation_median_to_true_x": signal_corr_median,
        "true_network_radius": radius_true,
        "true_network_scale_factor": scale_factor,
        "C_condition_number": C_diagnostics["C_condition_number"],
        "C_offdiag_mean_abs": C_diagnostics["C_offdiag_mean_abs"]
    }
    for key, value in values.items():
        df[key] = value
    if full_model is None:
        for key in ("full_log_likelihood", "full_spectral_radius",
                    "full_em_iterations", "full_Q_trace", "full_R_trace",
                    "estimate_Q", "estimate_R"):
            df[key] = np.nan
    else:
        df["full_log_likelihood"] = final_observed_log_likelihood(full_model)
        df["full_spectral_radius"] = full_model.spectral_radius()
        df["full_em_iterations"] = len(full_model.log_likelihoods)
        df["full_Q_trace"] = float(np.trace(full_model.Q))
        df["full_R_trace"] = float(np.trace(full_model.R))
        df["estimate_Q"] = full_model.estimate_Q
        df["estimate_R"] = full_model.estimate_R
    return df


def compute_network_for_signal(
        signal, u, true_link_mask, signal_type, lambda_A_group_fraction,
        outer_run, random_seed, x_true, C_diagnostics, full_model,
        radius_true, scale_factor
):
    recovery = compute_signal_recovery_metrics(signal, x_true)
    network_df = compute_all_pair_debiased_varx_network(
        x=signal, u=u, true_link_mask=true_link_mask, na=na, nb=nb,
        ridge_lambda=RIDGE_LAMBDA_DEBIAS, alpha=ALPHA,
        custom_thresholds=CUSTOM_THRESHOLDS
    )
    return add_metadata(
        network_df, outer_run, random_seed, signal_type,
        lambda_A_group_fraction, recovery["mse"], recovery["corr_mean"],
        recovery["corr_median"], C_diagnostics, full_model, radius_true,
        scale_factor
    )


# ------------------------------------------------------
# Experiment 33A diagnostics and summaries
# ------------------------------------------------------

OUTPUT_PREFIX = "experiment_33a_sample_size_parameter_recovery"
NETWORK_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_network_results.csv")
PARAMETER_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_parameter_summary.csv")
A_SUPPORT_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_A_support_results.csv")
B_RECOVERY_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_B_recovery_results.csv")
LINK_SUMMARY_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_link_strength_summary.csv")
DECISION_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_decision_summary.csv")
ROC_SUMMARY_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_roc_summary.csv")
ROC_POINTS_PATH = os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_roc_curve_points.csv")

PARTIAL_PATHS = {
    "network": os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_network_results_partial.csv"),
    "parameter": os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_parameter_summary_partial.csv"),
    "A_support": os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_A_support_results_partial.csv"),
    "roc": os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_roc_summary_partial.csv"),
    "B_recovery": os.path.join(RESULTS_DIR, OUTPUT_PREFIX + "_B_recovery_results_partial.csv")
}


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 2 or np.std(x[keep]) < 1e-12 or np.std(y[keep]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def safe_spearman(x, y):
    return safe_pearson(
        pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy(),
        pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    )


def safe_relative_error(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    denominator = np.linalg.norm(truth)
    return float(np.linalg.norm(estimate - truth) / denominator) if denominator > 0 else np.nan


def parameter_recovery(model, data, T, outer_run, lambda_fraction):
    A_hat = np.asarray(model.A_matrices, dtype=float)
    A_true = np.asarray(data["A_true"], dtype=float)
    B_hat = np.asarray(model.B_matrices, dtype=float)
    B_true = np.asarray(data["B_true"], dtype=float)
    offdiag = ~np.eye(N_SOURCES, dtype=bool)
    diagonal = ~offdiag
    offdiag_3d = np.broadcast_to(offdiag, A_true.shape)
    diagonal_3d = np.broadcast_to(diagonal, A_true.shape)

    true_norms, estimated_norms, labels = [], [], []
    for target in range(N_SOURCES):
        for source in range(N_SOURCES):
            if source == target:
                continue
            true_norms.append(np.linalg.norm(A_true[:, target, source]))
            estimated_norms.append(np.linalg.norm(A_hat[:, target, source]))
            labels.append(bool(data["true_link_mask"][target, source]))
    true_norms = np.asarray(true_norms)
    estimated_norms = np.asarray(estimated_norms)
    labels = np.asarray(labels, dtype=bool)
    true_mean = float(np.mean(estimated_norms[labels]))
    false_mean = float(np.mean(estimated_norms[~labels]))

    filtered_recovery = compute_signal_recovery_metrics(
        data["posterior"]["filtered_x"], data["x_true"]
    )
    smoothed_recovery = compute_signal_recovery_metrics(
        data["posterior"]["smoothed_x"], data["x_true"]
    )
    converged = model.group_solver_converged_history[-1] if model.group_solver_converged_history else np.nan
    solver_iterations = model.group_solver_iterations_history[-1] if model.group_solver_iterations_history else np.nan
    solver_objective = model.group_solver_objective_history[-1] if model.group_solver_objective_history else np.nan

    row = {
        "T": T, "outer_run": outer_run, "random_seed": data["seed"],
        "lambda_A_group_fraction": lambda_fraction,
        "A_relative_frobenius_error": safe_relative_error(A_hat, A_true),
        "A_offdiag_relative_frobenius_error": safe_relative_error(
            A_hat[offdiag_3d], A_true[offdiag_3d]
        ),
        "A_diagonal_relative_frobenius_error": safe_relative_error(
            A_hat[diagonal_3d], A_true[diagonal_3d]
        ),
        "A_mean_absolute_error": float(np.mean(np.abs(A_hat - A_true))),
        "A_offdiag_mean_absolute_error": float(np.mean(np.abs(A_hat[offdiag_3d] - A_true[offdiag_3d]))),
        "A_diagonal_mean_absolute_error": float(np.mean(np.abs(A_hat[diagonal_3d] - A_true[diagonal_3d]))),
        "A_pearson_correlation": safe_pearson(A_true.ravel(), A_hat.ravel()),
        "A_offdiag_pearson_correlation": safe_pearson(A_true[offdiag_3d], A_hat[offdiag_3d]),
        "A_offdiag_spearman_correlation": safe_spearman(A_true[offdiag_3d], A_hat[offdiag_3d]),
        "A_group_norm_pearson_correlation": safe_pearson(true_norms, estimated_norms),
        "A_group_norm_spearman_correlation": safe_spearman(true_norms, estimated_norms),
        "mean_estimated_A_group_norm_true_links": true_mean,
        "mean_estimated_A_group_norm_false_links": false_mean,
        "estimated_A_group_norm_true_false_ratio": true_mean / false_mean if false_mean > 0 else np.inf,
        "B_relative_frobenius_error": safe_relative_error(B_hat, B_true),
        "B_mean_absolute_error": float(np.mean(np.abs(B_hat - B_true))),
        "B_pearson_correlation": safe_pearson(B_true.ravel(), B_hat.ravel()),
        "B_spearman_correlation": safe_spearman(B_true.ravel(), B_hat.ravel()),
        "full_log_likelihood": final_observed_log_likelihood(model),
        "full_spectral_radius": model.spectral_radius(),
        "full_em_iterations": len(model.log_likelihoods),
        "Q_trace": float(np.trace(model.Q)), "R_trace": float(np.trace(model.R)),
        "estimate_Q": model.estimate_Q, "estimate_R": model.estimate_R,
        "offdiag_A_group_count": model.count_nonzero_offdiag_A_groups(),
        "offdiag_A_coefficient_count": model.count_nonzero_offdiag_A_coefficients(),
        "mean_offdiag_A_group_norm": model.mean_offdiag_A_group_norm(),
        "median_offdiag_A_group_norm": model.median_offdiag_A_group_norm(),
        "max_offdiag_A_group_norm": model.max_offdiag_A_group_norm(),
        "group_solver_converged_last": converged,
        "group_solver_mean_iterations_last": solver_iterations,
        "group_solver_objective_last": solver_objective,
        "mean_lambda_A_group_effective": model.mean_lambda_A_group_effective,
        "median_lambda_A_group_effective": model.median_lambda_A_group_effective,
        "min_lambda_A_group_effective": model.min_lambda_A_group_effective,
        "max_lambda_A_group_effective": model.max_lambda_A_group_effective,
        "moment_solver_condition_number_mean": model.moment_solver_condition_number_mean,
        "moment_solver_condition_number_max": model.moment_solver_condition_number_max,
        "filtered_signal_mse": filtered_recovery["mse"],
        "filtered_signal_correlation": filtered_recovery["corr_mean"],
        "smoothed_signal_mse": smoothed_recovery["mse"],
        "smoothed_signal_correlation": smoothed_recovery["corr_mean"]
    }
    for lag in range(nb):
        row[f"B_lag{lag}_relative_frobenius_error"] = safe_relative_error(B_hat[lag], B_true[lag])
    return row


def build_detailed_support(model, data, T, outer_run, lambda_fraction):
    rows = []
    for target in range(N_SOURCES):
        for source in range(N_SOURCES):
            if source == target:
                continue
            true_values = np.asarray([A[target, source] for A in data["A_true"]])
            estimated_values = np.asarray([A[target, source] for A in model.A_matrices])
            row = {
                "T": T, "outer_run": outer_run, "random_seed": data["seed"],
                "lambda_A_group_fraction": lambda_fraction,
                "source": source, "target": target,
                "true_link": bool(data["true_link_mask"][target, source]),
                "true_A_group_norm": float(np.linalg.norm(true_values)),
                "estimated_A_group_norm": float(np.linalg.norm(estimated_values))
            }
            for lag in range(na):
                row[f"true_A_lag{lag + 1}"] = true_values[lag]
                row[f"estimated_A_lag{lag + 1}"] = estimated_values[lag]
            rows.append(row)
    df = pd.DataFrame(rows)
    score = df["estimated_A_group_norm"].to_numpy()
    order = np.argsort(-score, kind="mergesort")
    for name, count in [
        ("top_19_group_norms", N_TRUE_LINKS),
        ("top_5_percent_group_norms", max(1, int(round(.05 * len(df))))),
        ("top_10_percent_group_norms", max(1, int(round(.10 * len(df)))))
    ]:
        selected = np.zeros(len(df), dtype=bool)
        selected[order[:count]] = True
        df[name] = selected
    for threshold in (1e-8, .001, .003, .01, .03):
        df[f"group_norm_gt_{threshold:g}"] = score > threshold

    run_roc, _ = roc_analysis_table(
        df, ["T", "outer_run", "lambda_A_group_fraction"],
        "estimated_A_group_norm", "A_support"
    )
    if len(run_roc):
        summary = run_roc.iloc[0]
        df["best_youden_threshold"] = score >= summary["best_youden_threshold"]
        df["threshold_at_best_F1"] = score >= summary["threshold_at_best_F1"]
    return df


def build_B_detail(model, data, T, outer_run, lambda_fraction):
    rows = []
    for lag, (truth, estimate) in enumerate(zip(data["B_true"], model.B_matrices)):
        for target in range(N_SOURCES):
            for input_index in range(N_INPUTS):
                rows.append({
                    "T": T, "outer_run": outer_run, "random_seed": data["seed"],
                    "lambda_A_group_fraction": lambda_fraction,
                    "B_lag": lag, "target": target, "input": input_index,
                    "B_true": truth[target, input_index],
                    "B_hat": estimate[target, input_index],
                    "absolute_error": abs(estimate[target, input_index] - truth[target, input_index])
                })
    return pd.DataFrame(rows)


def _fallback_curves(y_true, score):
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1]]
    points = []
    for threshold in thresholds:
        pred = score >= threshold
        m = compute_confusion_metrics(y_true, pred)
        points.append((threshold, m))
    return points


def roc_analysis_table(df, group_columns, score_column, score_type):
    summary_rows, point_rows = [], []
    for keys, group in df.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_columns, keys))
        y = group["true_link"].astype(bool).to_numpy()
        score = pd.to_numeric(group[score_column], errors="coerce").to_numpy()
        keep = np.isfinite(score)
        y, score = y[keep], score[keep]
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        try:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                precision_recall_curve, roc_curve
            )
            fpr, tpr, thresholds = roc_curve(y, score)
            precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y, score)
            roc_auc = float(roc_auc_score(y, score))
            auprc = float(average_precision_score(y, score))
            points = []
            for threshold, fpr_value, tpr_value in zip(thresholds, fpr, tpr):
                pred = score >= threshold
                m = compute_confusion_metrics(y, pred)
                points.append((threshold, m))
            if len(thresholds_pr):
                f1_pr = 2 * precision_pr[:-1] * recall_pr[:-1] / (
                    precision_pr[:-1] + recall_pr[:-1] + 1e-15
                )
                best_pr_index = int(np.nanargmax(f1_pr))
                best_f1_threshold = float(thresholds_pr[best_pr_index])
                best_f1 = float(f1_pr[best_pr_index])
            else:
                best_f1_threshold, best_f1 = np.nan, np.nan
        except ImportError:
            points = _fallback_curves(y, score)
            fpr_values = np.asarray([m["fpr"] for _, m in points])
            tpr_values = np.asarray([m["tpr"] for _, m in points])
            trapezoid = getattr(np, "trapezoid", None)
            if trapezoid is None:
                trapezoid = np.trapz
            roc_auc = float(trapezoid(tpr_values, fpr_values))
            order = np.argsort(-score, kind="mergesort")
            yy = y[order].astype(int)
            tp = np.cumsum(yy)
            precision_values = tp / np.arange(1, len(yy) + 1)
            auprc = float(np.sum(precision_values * yy) / max(1, np.sum(yy)))
            f1_values = np.asarray([m["f1"] for _, m in points])
            best_f1_index = int(np.nanargmax(f1_values))
            best_f1_threshold = float(points[best_f1_index][0])
            best_f1 = float(f1_values[best_f1_index])

        youdens = np.asarray([m["tpr"] - m["fpr"] for _, m in points])
        best_index = int(np.nanargmax(youdens))
        best_threshold, best_metrics = points[best_index]
        for threshold, metrics in points:
            point_rows.append({
                **meta, "score_type": score_type, "threshold": threshold,
                "fpr": metrics["fpr"], "tpr": metrics["tpr"],
                "precision": metrics["precision"], "recall": metrics["tpr"],
                "specificity": metrics["specificity"], "f1": metrics["f1"],
                "youden_j": metrics["tpr"] - metrics["fpr"]
            })
        summary_rows.append({
            **meta, "score_type": score_type, "n_links": len(y),
            "ROC_AUC": roc_auc, "average_precision": auprc, "AUPRC": auprc,
            "best_youden_J": youdens[best_index],
            "best_youden_threshold": best_threshold,
            "FPR_at_best_youden": best_metrics["fpr"],
            "TPR_at_best_youden": best_metrics["tpr"],
            "precision_at_best_youden": best_metrics["precision"],
            "F1_at_best_youden": best_metrics["f1"],
            "threshold_at_best_F1": best_f1_threshold,
            "best_F1": best_f1
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(point_rows)


def all_roc_outputs(network_df, support_df):
    summaries, points = [], []
    for score_column in ("debiased_deviance", "raw_deviance"):
        summary, curve = roc_analysis_table(
            network_df, ["T", "signal_type", "lambda_A_group_fraction"],
            score_column, score_column
        )
        summaries.append(summary)
        points.append(curve)
    summary, curve = roc_analysis_table(
        support_df, ["T", "lambda_A_group_fraction"],
        "estimated_A_group_norm", "estimated_A_group_norm"
    )
    summaries.append(summary)
    points.append(curve)
    return pd.concat(summaries, ignore_index=True), pd.concat(points, ignore_index=True)


def decision_summary_33a(network_df):
    rules = [
        "raw_chi_detected", "debiased_chi_detected",
        "raw_fdr_detected", "debiased_fdr_detected",
        "raw_bic_detected", "debiased_bic_detected"
    ]
    for threshold in CUSTOM_THRESHOLDS:
        label = str(threshold).replace(".", "p")
        rules.extend([f"raw_D_gt_{label}_detected", f"debiased_D_gt_{label}_detected"])
    rows = []
    for keys, group in network_df.groupby(
        ["T", "signal_type", "lambda_A_group_fraction"], dropna=False
    ):
        for rule in rules:
            if rule not in group:
                continue
            metrics = compute_confusion_metrics(group["true_link"], group[rule])
            rows.append({
                "T": keys[0], "signal_type": keys[1],
                "lambda_A_group_fraction": keys[2],
                "decision_rule": rule, "n_tests": len(group), **metrics
            })
    return pd.DataFrame(rows)


def support_summary_33a(support_df):
    rule_columns = [
        "group_norm_gt_1e-08", "group_norm_gt_0.001", "group_norm_gt_0.003",
        "group_norm_gt_0.01", "group_norm_gt_0.03", "top_19_group_norms",
        "top_5_percent_group_norms", "top_10_percent_group_norms",
        "best_youden_threshold", "threshold_at_best_F1"
    ]
    rows = []
    for keys, group in support_df.groupby(["T", "lambda_A_group_fraction"], dropna=False):
        for rule in rule_columns:
            metrics = compute_confusion_metrics(group["true_link"], group[rule])
            rows.append({
                "T": keys[0], "lambda_A_group_fraction": keys[1],
                "selection_rule": rule, "n_tests": len(group), **metrics
            })
    return pd.DataFrame(rows)


def link_strength_summary_33a(network_df):
    metric_columns = [
        "raw_deviance", "full_bias_term", "reduced_bias_term",
        "bias_correction", "debiased_deviance"
    ]
    recovery_columns = [
        "A_relative_frobenius_error", "A_offdiag_relative_frobenius_error",
        "A_diagonal_relative_frobenius_error", "B_relative_frobenius_error",
        "A_group_norm_pearson_correlation", "A_group_norm_spearman_correlation",
        "mean_estimated_A_group_norm_true_links",
        "mean_estimated_A_group_norm_false_links",
        "estimated_A_group_norm_true_false_ratio"
    ]
    rows = []
    for keys, group in network_df.groupby(
        ["T", "signal_type", "lambda_A_group_fraction", "true_link"], dropna=False
    ):
        row = dict(zip(
            ["T", "signal_type", "lambda_A_group_fraction", "true_link"], keys
        ))
        row["n_links"] = len(group)
        for column in metric_columns:
            row[f"mean_{column}"] = group[column].mean()
            row[f"median_{column}"] = group[column].median()
            row[f"std_{column}"] = group[column].std()
        for column in ("raw_p_value", "debiased_p_value"):
            row[f"mean_{column}"] = group[column].mean()
            row[f"median_{column}"] = group[column].median()
        for column in (
            "signal_mse_to_true_x", "signal_correlation_mean_to_true_x",
            "signal_correlation_median_to_true_x"
        ):
            row[f"mean_{column}"] = group[column].mean()
        for column in recovery_columns:
            row[f"mean_{column}"] = group[column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_run_metadata(network, T, parameter_row=None):
    network = network.copy()
    network["T"] = T
    if parameter_row is not None:
        for key, value in parameter_row.items():
            if key not in ("T", "outer_run", "random_seed", "lambda_A_group_fraction"):
                network[key] = value
    return network


def atomic_csv(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_partial(name):
    path = PARTIAL_PATHS[name]
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def save_partials(network_df, parameter_df, support_df, b_detail_df):
    atomic_csv(network_df, PARTIAL_PATHS["network"])
    atomic_csv(parameter_df, PARTIAL_PATHS["parameter"])
    atomic_csv(support_df, PARTIAL_PATHS["A_support"])
    atomic_csv(b_detail_df, PARTIAL_PATHS["B_recovery"])
    if len(network_df) and len(support_df):
        roc_summary, _ = all_roc_outputs(network_df, support_df)
        atomic_csv(roc_summary, PARTIAL_PATHS["roc"])


def selected_readout(network):
    wanted = [
        "debiased_chi_detected", "debiased_fdr_detected",
        "debiased_bic_detected", "debiased_D_gt_10p0_detected",
        "debiased_D_gt_15p0_detected", "debiased_D_gt_20p0_detected"
    ]
    rows = []
    for rule in wanted:
        m = compute_confusion_metrics(network["true_link"], network[rule])
        rows.append((rule, m["fpr"], m["tpr"], m["precision"], m["f1"]))
    return pd.DataFrame(rows, columns=["rule", "fpr", "tpr", "precision", "f1"])


# ------------------------------------------------------
# Main experiment
# ------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    network_df = load_partial("network")
    parameter_df = load_partial("parameter")
    support_df = load_partial("A_support")
    b_detail_df = load_partial("B_recovery")
    completed = set()
    if len(parameter_df):
        completed = set(zip(
            parameter_df["T"].astype(int),
            parameter_df["outer_run"].astype(int),
            parameter_df["lambda_A_group_fraction"].astype(float)
        ))

    for T in T_VALUES:
        for outer_run in range(N_OUTER_RUNS):
            # Pair sample-size conditions within an outer run: A, B, C, and
            # random streams use the same seed, so changes across T diagnose
            # sample size rather than a different synthetic network.
            seed = BASE_SEED + outer_run
            data = simulate_sparse_20source_case(T, BURN_IN, seed)
            data["seed"] = seed
            C_diagnostics = summarize_matrix_condition(data["C"])
            pinv_proxy = make_pinv_proxy(data["y_obs"], data["C"])

            # Baselines are lambda-independent and are computed once per T/run.
            baseline_done = (
                len(network_df)
                and np.any(
                    (network_df["T"] == T)
                    & (network_df["outer_run"] == outer_run)
                    & (network_df["signal_type"] == "oracle_latent")
                )
            )
            if not baseline_done:
                baseline_frames = []
                for signal_type, signal in [
                    ("oracle_latent", data["x_true"]),
                    ("observed_y", data["y_obs"]),
                    ("pinv_proxy", pinv_proxy)
                ]:
                    frame = compute_network_for_signal(
                        signal, data["u"], data["true_link_mask"], signal_type,
                        np.nan, outer_run, seed, data["x_true"], C_diagnostics,
                        None, data["radius_after"], data["scale_factor"]
                    )
                    baseline_frames.append(add_run_metadata(frame, T))
                network_df = pd.concat([network_df, *baseline_frames], ignore_index=True)

            for lambda_fraction in LAMBDA_A_GROUP_FRACTION_GRID:
                key = (T, outer_run, float(lambda_fraction))
                if key in completed:
                    print("Skipping completed work unit:", key)
                    continue
                model = fit_posterior_moment_group_lasso_fixed_cov_em_model(
                    data["y_obs"], data["u"], data["C"], lambda_fraction, seed + 500
                )
                data["posterior"] = kalman_filter_and_rts_smoother(
                    data["y_obs"], data["u"], model.A_matrices, model.B_matrices,
                    model.Q, data["C"], model.R
                )
                parameter_row = parameter_recovery(
                    model, data, T, outer_run, lambda_fraction
                )
                parameter_df = pd.concat(
                    [parameter_df, pd.DataFrame([parameter_row])], ignore_index=True
                )
                run_support = build_detailed_support(
                    model, data, T, outer_run, lambda_fraction
                )
                support_df = pd.concat([support_df, run_support], ignore_index=True)
                b_detail_df = pd.concat([
                    b_detail_df,
                    build_B_detail(model, data, T, outer_run, lambda_fraction)
                ], ignore_index=True)

                run_networks = {}
                for signal_type, signal in [
                    ("em_filtered", data["posterior"]["filtered_x"]),
                    ("em_smoothed", data["posterior"]["smoothed_x"])
                ]:
                    frame = compute_network_for_signal(
                        signal, data["u"], data["true_link_mask"], signal_type,
                        lambda_fraction, outer_run, seed, data["x_true"],
                        C_diagnostics, model, data["radius_after"], data["scale_factor"]
                    )
                    frame = add_run_metadata(frame, T, parameter_row)
                    network_df = pd.concat([network_df, frame], ignore_index=True)
                    run_networks[signal_type] = frame

                support_roc, _ = roc_analysis_table(
                    run_support, ["T", "outer_run", "lambda_A_group_fraction"],
                    "estimated_A_group_norm", "estimated_A_group_norm"
                )
                sr = support_roc.iloc[0]
                print(
                    f"\n33A T={T} run={outer_run + 1}/{N_OUTER_RUNS} "
                    f"lambda={lambda_fraction:g} LL={parameter_row['full_log_likelihood']:.3f} "
                    f"iterations={parameter_row['full_em_iterations']} "
                    f"radius={parameter_row['full_spectral_radius']:.3f} "
                    f"Q/R traces={parameter_row['Q_trace']:.3f}/{parameter_row['R_trace']:.3f}\n"
                    f"A rel={parameter_row['A_relative_frobenius_error']:.3f}, "
                    f"A offdiag rel={parameter_row['A_offdiag_relative_frobenius_error']:.3f}, "
                    f"B rel={parameter_row['B_relative_frobenius_error']:.3f}, "
                    f"A-group corr={parameter_row['A_group_norm_pearson_correlation']:.3f}, "
                    f"true/false norms={parameter_row['mean_estimated_A_group_norm_true_links']:.4f}/"
                    f"{parameter_row['mean_estimated_A_group_norm_false_links']:.4f}, "
                    f"A ROC_AUC={sr['ROC_AUC']:.3f}, AUPRC={sr['AUPRC']:.3f}, "
                    f"best J={sr['best_youden_J']:.3f}"
                )
                for signal_type in ("em_filtered", "em_smoothed"):
                    print("\n", signal_type)
                    print(selected_readout(run_networks[signal_type]).to_string(index=False))
                observed = network_df[
                    (network_df["T"] == T)
                    & (network_df["outer_run"] == outer_run)
                    & (network_df["signal_type"] == "observed_y")
                ]
                print("\n observed_y baseline")
                print(selected_readout(observed).to_string(index=False))
                completed.add(key)

            # Outer-run boundary is a scientifically safe checkpoint: all
            # lambda fits and readouts for this simulated dataset are complete.
            save_partials(network_df, parameter_df, support_df, b_detail_df)

    roc_summary_df, roc_points_df = all_roc_outputs(network_df, support_df)
    link_summary_df = link_strength_summary_33a(network_df)
    decision_df = decision_summary_33a(network_df)
    support_summary_df = support_summary_33a(support_df)

    atomic_csv(network_df, NETWORK_PATH)
    atomic_csv(parameter_df, PARAMETER_PATH)
    atomic_csv(support_df, A_SUPPORT_PATH)
    atomic_csv(b_detail_df, B_RECOVERY_PATH)
    atomic_csv(link_summary_df, LINK_SUMMARY_PATH)
    atomic_csv(decision_df, DECISION_PATH)
    atomic_csv(roc_summary_df, ROC_SUMMARY_PATH)
    atomic_csv(roc_points_df, ROC_POINTS_PATH)
    atomic_csv(support_summary_df, os.path.join(
        RESULTS_DIR, OUTPUT_PREFIX + "_A_support_summary.csv"
    ))
    print("\nExperiment 33A complete. Final outputs written to", RESULTS_DIR)


if __name__ == "__main__":
    main()


# Interpretation guide:
# 1. If T=1000/2000 reduces A/B error and improves EM GC, sample size is
#    the main limitation; size future high-lag experiments by parameter/sample ratio.
# 2. If recovery stays poor at T=2000, return to controlled two-source models.
# 3. If A/B recovery improves but GC does not, diagnose readout/testing using
#    ROC/Youden, observed likelihood, and the paper's Eq. 7 metric.
# 4. Poor B recovery plus persistent A false links suggests exogenous leakage;
#    test stronger B regularization or fixed-B diagnostics next.
#
# TODO Experiment 33C: Locate “Non-Asymptotic Guarantees for Reliable
# Identification of Granger Causality via the LASSO”, extract Eq. 7 exactly into
# documentation, and implement it only after 33A establishes recovery behavior.
#
# TODO Experiment 33F: adapt density/coverage as exploratory edge-distribution
# diagnostics. They must not replace FPR/TPR/precision/F1/ROC/AUPRC.
