import os

# Each thread handles a complete EM fit. Limiting BLAS threads prevents
# nested parallelism from multiplying OpenBLAS/MKL worker threads.
BLAS_THREADS_PER_WORKER = int(
    os.environ.get("EXPERIMENT_32G_BLAS_THREADS", "1")
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

from src.varx.varx_generator import (
    generate_colored_input
)

from src.ssm.ssm_varx_p_simulator import (
    generate_ssm_varx_p_data
)

from src.ssm.em_varx_p_known_c_l1_mstep_fixed_cov import (
    EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov
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
# Experiment 32G
# Covariance-controlled row-wise L1 M-step diagnostic
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

# Use 5 or 10 for a quick smoke test.
# Use 20 for the full experiment.
N_OUTER_RUNS = 20

ALPHA = 0.05

R_FLOOR = 0.30

MAX_ITER = 100
TOL = 1e-6

RIDGE_LAMBDA_DEBIAS = 1.0

BASE_SEED = 3600000

CUSTOM_THRESHOLDS = (
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0
)

# Small penalties diagnose the abrupt sparsity transition seen in 32F.
LAMBDA_A_GRID = [
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3
]

# Four workers is a conservative default for the memory-heavy EM fits.
# Override explicitly for a particular machine, for example:
#   $env:EXPERIMENT_32G_WORKERS = "8"
N_WORKERS = int(
    os.environ.get(
        "EXPERIMENT_32G_WORKERS",
        str(min(4, os.cpu_count() or 1, len(LAMBDA_A_GRID)))
    )
)

if N_WORKERS < 1:
    raise ValueError("EXPERIMENT_32G_WORKERS must be at least 1.")

SHRINKAGE_SPEC = {
    "name": "fixed_Q0p50_R0p60",
    "alpha_Q": 0.0,
    "alpha_R": 0.0,
    "target_Q": "spherical",
    "target_R": "spherical"
}


print("\nExperiment 32G: covariance-controlled row-wise L1 M-step")
print("----------------------------------------------------------")
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
print("lambda_A grid:", LAMBDA_A_GRID)
print("covariance control:", SHRINKAGE_SPEC["name"])
print("parallel worker threads:", N_WORKERS)
print("BLAS threads per worker:", BLAS_THREADS_PER_WORKER)
print("Numba coordinate descent:", "enabled" if NUMBA_AVAILABLE else "fallback")


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

def fit_l1_mstep_fixed_cov_em_model(
        y_obs,
        u,
        C,
        lambda_A,
        random_seed
):
    model = EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov(
        na=na,
        nb=nb,
        C=C,
        D=None,
        max_iter=MAX_ITER,
        tol=TOL,

        # Kept for compatibility with inherited class.
        # The main A/B update is now done inside the row-wise L1 M-step.
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

        # Row-wise L1 M-step.
        lambda_A_offdiag=lambda_A,
        lambda_A_diag=0.0,
        lambda_B=0.0,

        # Weak ridge for numerical stability.
        ridge_A_offdiag=1e-4,
        ridge_A_diag=1e-4,
        ridge_B=1e-4,

        lambda_A_lag_decay=1.0,
        lasso_max_iter=1000,
        lasso_tol=1e-6,

        stabilize_A=True,
        target_radius=0.98
    )

    model.fit(
        y=y_obs,
        u=u
    )

    model.fit_label = f"l1_mstep_fixed_cov_lambda_{lambda_A}"

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


def safe_model_diagnostic(
        full_model,
        attribute_name,
        default_value=np.nan
):
    if full_model is None:
        return default_value

    if not hasattr(full_model, attribute_name):
        return default_value

    value = getattr(
        full_model,
        attribute_name
    )

    return value


def add_metadata(
        df,
        outer_run,
        random_seed,
        signal_type,
        lambda_A,
        signal_mse,
        signal_corr_mean,
        signal_corr_median,
        C_diagnostics,
        full_model,
        radius_true,
        scale_factor
):
    df = df.copy()

    df["outer_run"] = outer_run
    df["random_seed"] = random_seed

    df["n_sources"] = N_SOURCES
    df["n_true_links"] = N_TRUE_LINKS
    df["link_density"] = LINK_DENSITY

    df["signal_type"] = signal_type
    df["lambda_A"] = lambda_A

    df["signal_mse_to_true_x"] = signal_mse
    df["signal_correlation_mean_to_true_x"] = signal_corr_mean
    df["signal_correlation_median_to_true_x"] = signal_corr_median

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

        df["em_A_nonzero_offdiag"] = full_model.count_nonzero_offdiag_A()
        df["em_A_nonzero_total"] = full_model.count_nonzero_total_A()
        df["em_A_mean_abs_offdiag"] = full_model.mean_abs_offdiag_A()
        df["em_A_max_abs_offdiag"] = full_model.max_abs_offdiag_A()

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

        df["fixed_Q_trace"] = full_model.fixed_Q_trace
        df["fixed_R_trace"] = full_model.fixed_R_trace
        df["estimate_Q"] = full_model.estimate_Q
        df["estimate_R"] = full_model.estimate_R

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

        if hasattr(full_model, "lasso_converged_history"):

            df["lasso_converged_last"] = bool(
                full_model.lasso_converged_history[-1]
            ) if len(full_model.lasso_converged_history) > 0 else np.nan

            df["lasso_mean_iterations_last"] = float(
                full_model.lasso_iterations_history[-1]
            ) if len(full_model.lasso_iterations_history) > 0 else np.nan

            df["lasso_objective_last"] = float(
                full_model.lasso_objective_history[-1]
            ) if len(full_model.lasso_objective_history) > 0 else np.nan

        else:

            df["lasso_converged_last"] = np.nan
            df["lasso_mean_iterations_last"] = np.nan
            df["lasso_objective_last"] = np.nan

    else:

        df["full_log_likelihood"] = np.nan
        df["full_spectral_radius"] = np.nan
        df["full_em_iterations"] = np.nan

        df["em_A_nonzero_offdiag"] = np.nan
        df["em_A_nonzero_total"] = np.nan
        df["em_A_mean_abs_offdiag"] = np.nan
        df["em_A_max_abs_offdiag"] = np.nan

        df["full_Q_trace"] = np.nan
        df["full_R_trace"] = np.nan
        df["fixed_Q_trace"] = np.nan
        df["fixed_R_trace"] = np.nan
        df["estimate_Q"] = np.nan
        df["estimate_R"] = np.nan
        df["full_Q_offdiag_mean_abs"] = np.nan
        df["full_R_offdiag_mean_abs"] = np.nan

        df["lasso_converged_last"] = np.nan
        df["lasso_mean_iterations_last"] = np.nan
        df["lasso_objective_last"] = np.nan

    return df


def compute_network_for_signal(
        signal,
        u,
        true_link_mask,
        signal_type,
        lambda_A,
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

    network_df = add_metadata(
        df=network_df,
        outer_run=outer_run,
        random_seed=random_seed,
        signal_type=signal_type,
        lambda_A=lambda_A,
        signal_mse=recovery["mse"],
        signal_corr_mean=recovery["corr_mean"],
        signal_corr_median=recovery["corr_median"],
        C_diagnostics=C_diagnostics,
        full_model=full_model,
        radius_true=radius_true,
        scale_factor=scale_factor
    )

    return network_df


# ------------------------------------------------------
# Summary helpers
# ------------------------------------------------------

def summarize_link_strengths(
        results_df
):
    grouped = results_df.groupby(
        [
            "signal_type",
            "lambda_A",
            "true_link"
        ],
        dropna=False
    )

    rows = []

    for keys, group in grouped:

        signal_type, lambda_A, true_link = keys

        rows.append({
            "signal_type": signal_type,
            "lambda_A": lambda_A,
            "true_link": true_link,
            "n_links": len(group),

            # Deviance and bias terms
            "mean_raw_deviance": group["raw_deviance"].mean(),
            "median_raw_deviance": group["raw_deviance"].median(),
            "std_raw_deviance": group["raw_deviance"].std(),

            "mean_full_bias_term": group["full_bias_term"].mean(),
            "median_full_bias_term": group["full_bias_term"].median(),
            "std_full_bias_term": group["full_bias_term"].std(),

            "mean_reduced_bias_term": group["reduced_bias_term"].mean(),
            "median_reduced_bias_term": group["reduced_bias_term"].median(),
            "std_reduced_bias_term": group["reduced_bias_term"].std(),

            "mean_bias_correction": group["bias_correction"].mean(),
            "median_bias_correction": group["bias_correction"].median(),
            "std_bias_correction": group["bias_correction"].std(),

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

            # EM diagnostics
            "mean_full_log_likelihood": group["full_log_likelihood"].mean(),
            "mean_full_spectral_radius": group["full_spectral_radius"].mean(),
            "mean_full_em_iterations": group["full_em_iterations"].mean(),

            "mean_em_A_nonzero_offdiag": group["em_A_nonzero_offdiag"].mean(),
            "mean_em_A_nonzero_total": group["em_A_nonzero_total"].mean(),
            "mean_em_A_mean_abs_offdiag": group["em_A_mean_abs_offdiag"].mean(),
            "mean_em_A_max_abs_offdiag": group["em_A_max_abs_offdiag"].mean(),

            "mean_full_Q_trace": group["full_Q_trace"].mean(),
            "mean_full_R_trace": group["full_R_trace"].mean(),
            "mean_fixed_Q_trace": group["fixed_Q_trace"].mean(),
            "mean_fixed_R_trace": group["fixed_R_trace"].mean(),
            "mean_estimate_Q": group["estimate_Q"].mean(),
            "mean_estimate_R": group["estimate_R"].mean(),
            "mean_full_Q_offdiag_abs": group["full_Q_offdiag_mean_abs"].mean(),
            "mean_full_R_offdiag_abs": group["full_R_offdiag_mean_abs"].mean(),

            "mean_lasso_converged_last": group["lasso_converged_last"].mean(),
            "mean_lasso_iterations_last": group["lasso_mean_iterations_last"].mean(),
            "mean_lasso_objective_last": group["lasso_objective_last"].mean(),

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

    grouped = results_df.groupby(
        [
            "signal_type",
            "lambda_A"
        ],
        dropna=False
    )

    rows = []

    for keys, group in grouped:

        signal_type, lambda_A = keys

        for decision_column in decision_columns:

            metrics = compute_confusion_metrics(
                y_true=group["true_link"].values,
                y_pred=group[decision_column].values
            )

            row = {
                "signal_type": signal_type,
                "lambda_A": lambda_A,
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
# Checkpoint helpers
# ------------------------------------------------------

CHECKPOINT_PATH = (
    "data/"
    "experiment_32g_l1_mstep_fixed_covariance_checkpoint.pkl"
)

CHECKPOINT_REPLACE_ATTEMPTS = 8
CHECKPOINT_RETRY_INITIAL_SECONDS = 0.25


def checkpoint_configuration():
    """
    Settings that determine the numerical experiment.

    A checkpoint is only reusable when these settings match exactly.
    """

    return {
        "checkpoint_version": 1,
        "n_sources": N_SOURCES,
        "link_density": LINK_DENSITY,
        "n_true_links": N_TRUE_LINKS,
        "n_inputs": N_INPUTS,
        "na": na,
        "nb": nb,
        "n_samples": N_SAMPLES,
        "burn_in": BURN_IN,
        "n_outer_runs": N_OUTER_RUNS,
        "alpha": ALPHA,
        "R_floor": R_FLOOR,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "ridge_lambda_debias": RIDGE_LAMBDA_DEBIAS,
        "base_seed": BASE_SEED,
        "custom_thresholds": CUSTOM_THRESHOLDS,
        "lambda_A_grid": LAMBDA_A_GRID,
        "shrinkage_spec": SHRINKAGE_SPEC,
        "estimate_Q": False,
        "estimate_R": False,
        "Q_init_scale": 0.50,
        "R_init_scale": 0.60
    }


def save_checkpoint(
        all_link_rows,
        all_true_link_tables,
        completed_units,
        true_link_outer_runs
):
    """
    Atomically save completed work.

    os.replace exposes either the previous complete checkpoint or the new
    complete checkpoint, never a partially written pickle.
    """

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True
    )

    checkpoint = {
        "configuration": checkpoint_configuration(),
        "all_link_rows": all_link_rows,
        "all_true_link_tables": all_true_link_tables,
        "completed_units": completed_units,
        "true_link_outer_runs": true_link_outer_runs
    }

    temporary_path = (
        CHECKPOINT_PATH
        + f".{os.getpid()}.{time.time_ns()}.tmp"
    )

    with open(temporary_path, "wb") as checkpoint_file:
        pickle.dump(
            checkpoint,
            checkpoint_file,
            protocol=pickle.HIGHEST_PROTOCOL
        )
        checkpoint_file.flush()
        os.fsync(
            checkpoint_file.fileno()
        )

    # Google Drive for desktop can briefly lock the existing synchronized
    # destination. Retry promotion, but do not fail the experiment if Drive
    # keeps the destination locked: the complete uniquely named temporary
    # file is itself a valid restart checkpoint and load_checkpoint considers
    # it on the next launch.
    retry_seconds = CHECKPOINT_RETRY_INITIAL_SECONDS

    for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            os.replace(
                temporary_path,
                CHECKPOINT_PATH
            )
            return
        except PermissionError:
            if attempt + 1 < CHECKPOINT_REPLACE_ATTEMPTS:
                time.sleep(
                    retry_seconds
                )
                retry_seconds *= 2.0

    print(
        "\nWarning: Google Drive kept the main checkpoint locked. "
        "Progress remains safely saved in:"
    )
    print(temporary_path)


def checkpoint_candidates():
    candidates = []

    if os.path.exists(CHECKPOINT_PATH):
        candidates.append(
            CHECKPOINT_PATH
        )

    candidates.extend(
        glob.glob(
            CHECKPOINT_PATH + "*.tmp"
        )
    )

    return sorted(
        candidates,
        key=os.path.getmtime,
        reverse=True
    )


def load_checkpoint():
    candidates = checkpoint_candidates()

    if not candidates:
        return {
            "all_link_rows": [],
            "all_true_link_tables": [],
            "completed_units": set(),
            "true_link_outer_runs": set()
        }

    checkpoint = None
    loaded_path = None

    for candidate_path in candidates:
        try:
            with open(candidate_path, "rb") as checkpoint_file:
                checkpoint = pickle.load(
                    checkpoint_file
                )
            loaded_path = candidate_path
            break
        except (EOFError, OSError, pickle.UnpicklingError) as error:
            print(
                "Ignoring unreadable checkpoint candidate:",
                candidate_path,
                "|",
                error
            )

    if checkpoint is None:
        raise RuntimeError(
            "Experiment 32G found checkpoint files, but none could be "
            "read. Preserve them for diagnosis and inspect: "
            f"{CHECKPOINT_PATH}*.tmp"
        )

    if checkpoint.get("configuration") != checkpoint_configuration():
        raise ValueError(
            "Experiment 32G checkpoint settings do not match the current "
            "experiment settings. Move or delete the checkpoint before "
            "starting a different configuration: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint["completed_units"] = set(
        checkpoint["completed_units"]
    )
    checkpoint["true_link_outer_runs"] = set(
        checkpoint["true_link_outer_runs"]
    )

    print("\nResuming Experiment 32G from checkpoint:")
    print(loaded_path)
    print(
        "Completed work units:",
        len(checkpoint["completed_units"])
    )

    return checkpoint


def remove_checkpoint_files():
    """Best-effort cleanup after every final output is safely written."""

    for checkpoint_path in checkpoint_candidates():
        try:
            os.remove(
                checkpoint_path
            )
        except PermissionError:
            print(
                "Warning: Google Drive temporarily locked completed "
                "checkpoint; it may be deleted manually later:"
            )
            print(checkpoint_path)


def run_l1_work_unit(task):
    """Fit and evaluate one lambda value without writing shared outputs."""

    lambda_A = task["lambda_A"]
    y_obs = task["y_obs"]
    u = task["u"]
    C = task["C"]
    seed = task["seed"]

    print("\n" + "-" * 100)
    print(f"Fitting fixed-covariance L1-M-step EM with lambda_A={lambda_A}")
    print("-" * 100)

    full_model = fit_l1_mstep_fixed_cov_em_model(
        y_obs=y_obs,
        u=u,
        C=C,
        lambda_A=lambda_A,
        random_seed=seed + 500
    )

    print("Full LL:", final_observed_log_likelihood(full_model))
    print("EM iterations:", len(full_model.log_likelihoods))
    print("Spectral radius:", full_model.spectral_radius())
    print("Q trace:", float(np.trace(full_model.Q)))
    print("R trace:", float(np.trace(full_model.R)))
    print("Nonzero offdiag A:", full_model.count_nonzero_offdiag_A())
    print("Nonzero total A:", full_model.count_nonzero_total_A())
    print("Mean abs offdiag A:", full_model.mean_abs_offdiag_A())
    print("Max abs offdiag A:", full_model.max_abs_offdiag_A())

    if (
            hasattr(full_model, "lasso_converged_history")
            and len(full_model.lasso_converged_history) > 0
    ):
        print("Lasso converged last:", full_model.lasso_converged_history[-1])
        print("Lasso mean iterations last:", full_model.lasso_iterations_history[-1])
        print("Lasso objective last:", full_model.lasso_objective_history[-1])

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
            "lambda_A:",
            lambda_A
        )

        network_df = compute_network_for_signal(
            signal=spec["signal"],
            u=u,
            true_link_mask=task["true_link_mask"],
            signal_type=spec["signal_type"],
            lambda_A=lambda_A,
            outer_run=task["outer_run"],
            random_seed=seed,
            x_true=task["x_true"],
            C_diagnostics=task["C_diagnostics"],
            full_model=full_model,
            radius_true=task["radius_true"],
            scale_factor=task["scale_factor"]
        )
        network_dfs.append(network_df)

    return float(lambda_A), network_dfs


# ------------------------------------------------------
# Main experiment
# ------------------------------------------------------

checkpoint = load_checkpoint()

all_link_rows = checkpoint["all_link_rows"]
all_true_link_tables = checkpoint["all_true_link_tables"]
completed_units = checkpoint["completed_units"]
true_link_outer_runs = checkpoint["true_link_outer_runs"]

for outer_run in range(N_OUTER_RUNS):

    print("\n" + "=" * 120)
    print(f"Experiment 32G | outer run {outer_run + 1}/{N_OUTER_RUNS}")
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

    if outer_run not in true_link_outer_runs:
        all_true_link_tables.append(
            true_link_table
        )
        true_link_outer_runs.add(
            outer_run
        )
        save_checkpoint(
            all_link_rows=all_link_rows,
            all_true_link_tables=all_true_link_tables,
            completed_units=completed_units,
            true_link_outer_runs=true_link_outer_runs
        )

    # --------------------------------------------------
    # Baseline signal representations
    # --------------------------------------------------

    x_pinv_proxy = make_pinv_proxy(
        y_obs=y_obs,
        C=C
    )

    baseline_specs = [
        {
            "signal_type": "oracle_latent",
            "signal": x_true,
            "lambda_A": np.nan,
            "full_model": None
        },
        {
            "signal_type": "observed_y",
            "signal": y_obs,
            "lambda_A": np.nan,
            "full_model": None
        },
        {
            "signal_type": "pinv_proxy",
            "signal": x_pinv_proxy,
            "lambda_A": np.nan,
            "full_model": None
        }
    ]

    for spec in baseline_specs:

        unit_key = (
            outer_run,
            "baseline",
            spec["signal_type"]
        )

        if unit_key in completed_units:
            print(
                "\nSkipping completed baseline network:",
                spec["signal_type"]
            )
            continue

        print("\nComputing baseline network:", spec["signal_type"])

        network_df = compute_network_for_signal(
            signal=spec["signal"],
            u=u,
            true_link_mask=true_link_mask,
            signal_type=spec["signal_type"],
            lambda_A=spec["lambda_A"],
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
        completed_units.add(
            unit_key
        )
        save_checkpoint(
            all_link_rows=all_link_rows,
            all_true_link_tables=all_true_link_tables,
            completed_units=completed_units,
            true_link_outer_runs=true_link_outer_runs
        )

    # --------------------------------------------------
    # L1-M-step EM comparisons
    # --------------------------------------------------

    pending_l1_tasks = [
        {
            "lambda_A": float(lambda_A),
            "y_obs": y_obs,
            "u": u,
            "C": C,
            "seed": seed,
            "true_link_mask": true_link_mask,
            "outer_run": outer_run,
            "x_true": x_true,
            "C_diagnostics": C_diagnostics,
            "radius_true": data["radius_after"],
            "scale_factor": data["scale_factor"]
        }
        for lambda_A in LAMBDA_A_GRID
        if (
            outer_run,
            "l1_mstep",
            float(lambda_A)
        ) not in completed_units
    ]

    skipped_lambdas = len(LAMBDA_A_GRID) - len(pending_l1_tasks)

    if skipped_lambdas > 0:
        print("\nSkipping completed L1-M-step fits:", skipped_lambdas)

    if pending_l1_tasks:
        print(
            "\nRunning",
            len(pending_l1_tasks),
            "L1-M-step fits with",
            min(N_WORKERS, len(pending_l1_tasks)),
            "worker threads."
        )

        with ThreadPoolExecutor(
                max_workers=min(N_WORKERS, len(pending_l1_tasks))
        ) as executor:
            work_results = executor.map(
                run_l1_work_unit,
                pending_l1_tasks
            )

            # executor.map yields in lambda-grid order. This preserves the
            # reference row ordering while later fits run concurrently.
            for lambda_A, network_dfs in work_results:
                for network_df in network_dfs:
                    all_link_rows.append(network_df)

                    temp_summary = summarize_decision_rules(network_df)
                    temp_display = temp_summary[
                        temp_summary["decision_rule"].isin([
                            "debiased_chi_detected",
                            "debiased_fdr_detected",
                            "debiased_bic_detected",
                            "debiased_D_gt_10p0_detected",
                            "debiased_D_gt_15p0_detected",
                            "debiased_D_gt_20p0_detected",
                            "debiased_D_gt_25p0_detected"
                        ])
                    ]

                    print(
                        temp_display[
                            [
                                "signal_type",
                                "lambda_A",
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
                        ].to_string(index=False)
                    )

                completed_units.add(
                    (
                        outer_run,
                        "l1_mstep",
                        float(lambda_A)
                    )
                )

                # Only this parent thread mutates restart state or files.
                save_checkpoint(
                    all_link_rows=all_link_rows,
                    all_true_link_tables=all_true_link_tables,
                    completed_units=completed_units,
                    true_link_outer_runs=true_link_outer_runs
                )

    # --------------------------------------------------
    # Save partial progress after each outer run
    # --------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    partial_df = pd.concat(
        all_link_rows,
        ignore_index=True
    )

    partial_df.to_csv(
        "data/experiment_32g_l1_mstep_fixed_covariance_results_partial.csv",
        index=False
    )

    pd.concat(
        all_true_link_tables,
        ignore_index=True
    ).to_csv(
        "data/experiment_32g_l1_mstep_fixed_covariance_true_links_partial.csv",
        index=False
    )

    print("\nPartial results saved after outer run:", outer_run + 1)


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
print("Experiment 32G link-strength summary")
print("=" * 160)

link_strength_display_columns = [
    "signal_type",
    "lambda_A",
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
    "mean_full_em_iterations",

    "mean_em_A_nonzero_offdiag",
    "mean_em_A_nonzero_total",
    "mean_em_A_mean_abs_offdiag",
    "mean_em_A_max_abs_offdiag",

    "mean_full_Q_trace",
    "mean_full_R_trace",
    "mean_fixed_Q_trace",
    "mean_fixed_R_trace",
    "mean_estimate_Q",
    "mean_estimate_R",
    "mean_full_Q_offdiag_abs",
    "mean_full_R_offdiag_abs",

    "mean_lasso_converged_last",
    "mean_lasso_iterations_last"
]

available_link_strength_display_columns = [
    col
    for col in link_strength_display_columns
    if col in link_strength_summary_df.columns
]

print(
    link_strength_summary_df[available_link_strength_display_columns].to_string(
        index=False
    )
)


print("\n" + "=" * 160)
print("Experiment 32G decision-rule summary")
print("=" * 160)

decision_display_columns = [
    "signal_type",
    "lambda_A",
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

available_decision_display_columns = [
    col
    for col in decision_display_columns
    if col in decision_summary_df.columns
]

print(
    decision_summary_df[available_decision_display_columns].to_string(
        index=False
    )
)


# ------------------------------------------------------
# Save outputs
# ------------------------------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

results_path = "data/experiment_32g_l1_mstep_fixed_covariance_results.csv"
true_links_path = "data/experiment_32g_l1_mstep_fixed_covariance_true_links.csv"
link_strength_summary_path = "data/experiment_32g_l1_mstep_fixed_covariance_link_strength_summary.csv"
decision_summary_path = "data/experiment_32g_l1_mstep_fixed_covariance_decision_summary.csv"

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

# Remove restart checkpoints only after every final output is written.
remove_checkpoint_files()

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
print("oracle_latent: true latent state, best-case benchmark.")
print("observed_y: noisy mixed observation used directly.")
print("pinv_proxy: source proxy using known C pseudo-inverse.")
print("em_filtered: causal filtered posterior mean from L1-M-step EM parameters.")
print("em_smoothed: fixed-interval smoothed posterior mean from L1-M-step EM parameters.")
print("lambda_A=0.00 is the no-L1 baseline under the same L1-M-step framework.")
print("lambda_A>0 applies L1 sparsity only to off-diagonal endogenous A coefficients.")
print("B coefficients are not L1-penalized because they explain exogenous/common input.")
print("Check whether false-link deviance decreases faster than true-link deviance as lambda_A increases.")
print("The key output columns are raw_deviance, full_bias_term, reduced_bias_term, debiased_deviance, FPR, TPR, precision, F1, and em_A_nonzero_offdiag.")
