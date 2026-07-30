import numpy as np
import pandas as pd
from scipy.stats import chi2


def companion_spectral_radius(A_matrices):
    A_matrices = [
        np.asarray(A, dtype=float)
        for A in A_matrices
    ]

    p = len(A_matrices)
    n = A_matrices[0].shape[0]

    companion = np.zeros(
        (n * p, n * p)
    )

    companion[:n, :n * p] = np.hstack(
        A_matrices
    )

    if p > 1:
        companion[n:, :-n] = np.eye(
            n * (p - 1)
        )

    eigenvalues = np.linalg.eigvals(
        companion
    )

    return float(
        np.max(
            np.abs(
                eigenvalues
            )
        )
    )


def build_varx_design(
        x,
        u,
        na,
        nb,
        target
):
    """
    Build row-wise VARX design for one target source.

    x[t, :] are endogenous source signals.
    u[t, :] are exogenous inputs.

    Model:
        x_i[t] =
            sum_k A_k[i, :] x[t-k, :]
            + sum_l B_l[i, :] u[t-l, :]
            + error
    """

    x = np.asarray(
        x,
        dtype=float
    )

    u = np.asarray(
        u,
        dtype=float
    )

    if u.ndim == 1:
        u = u[:, None]

    T, n_sources = x.shape
    _, n_inputs = u.shape

    start = max(
        na,
        nb - 1
    )

    y = []
    rows = []

    for t in range(start, T):

        row = []

        for lag in range(1, na + 1):
            row.extend(
                x[t - lag, :]
            )

        for lag in range(0, nb):
            row.extend(
                u[t - lag, :]
            )

        rows.append(
            row
        )

        y.append(
            x[t, target]
        )

    X = np.asarray(
        rows,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    return X, y, start


def source_lag_columns(
        source,
        n_sources,
        na
):
    """
    Column indices in the full design matrix corresponding to
    one source's lagged history.
    """

    return [
        (lag - 1) * n_sources + source
        for lag in range(1, na + 1)
    ]


def fit_tikhonov_regression_with_bias(
        X,
        y,
        ridge_gamma,
        pinv_rcond=1e-8,
        variance_floor=1e-12
):
    """
    Fit ridge/Tikhonov regression and compute the L2 de-biasing term.

    The regularized estimator is:

        h_hat = (X^T X + gamma Gamma)^(-1) X^T y

    where Gamma = diag(X^T X).

    The de-biasing term is:

        b = 1/(2 * e^T e) * r_xe^T (X^T X)^(-1) r_xe

    with:

        r_xe = X^T e.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    Rxx = X.T @ X
    Rxy = X.T @ y

    Gamma = np.diag(
        np.diag(
            Rxx
        )
    )

    diag_gamma = np.diag(
        Gamma
    )

    if np.any(diag_gamma <= 0):
        replacement = np.median(
            diag_gamma[diag_gamma > 0]
        ) if np.any(diag_gamma > 0) else 1.0

        diag_gamma = np.where(
            diag_gamma > 0,
            diag_gamma,
            replacement
        )

        Gamma = np.diag(
            diag_gamma
        )

    regularized_matrix = (
        Rxx
        + ridge_gamma * Gamma
    )

    try:
        h_hat = np.linalg.solve(
            regularized_matrix,
            Rxy
        )
    except np.linalg.LinAlgError:
        h_hat = np.linalg.pinv(
            regularized_matrix,
            rcond=pinv_rcond
        ) @ Rxy

    residual = (
        y
        - X @ h_hat
    )

    sse = float(
        residual.T @ residual
    )

    sigma2 = float(
        max(
            sse / len(y),
            variance_floor
        )
    )

    rxe = X.T @ residual

    Rxx_pinv = np.linalg.pinv(
        Rxx,
        rcond=pinv_rcond
    )

    bias_numerator = float(
        rxe.T @ Rxx_pinv @ rxe
    )

    bias = float(
        0.5 * bias_numerator / max(
            sse,
            variance_floor
        )
    )

    if not np.isfinite(bias):
        bias = np.nan

    return {
        "h_hat": h_hat,
        "residual": residual,
        "sse": sse,
        "sigma2": sigma2,
        "bias": bias,
        "n_parameters": X.shape[1],
        "n_samples_effective": len(y)
    }


def bh_fdr_mask(
        p_values,
        alpha=0.05
):
    """
    Benjamini-Hochberg FDR detection mask.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    mask = np.zeros(
        len(p_values),
        dtype=bool
    )

    finite_mask = np.isfinite(
        p_values
    )

    finite_indices = np.where(
        finite_mask
    )[0]

    if len(finite_indices) == 0:
        return mask, np.nan

    p_finite = p_values[
        finite_indices
    ]

    order = np.argsort(
        p_finite
    )

    sorted_p = p_finite[
        order
    ]

    m = len(
        sorted_p
    )

    thresholds = alpha * (
        np.arange(1, m + 1) / m
    )

    passed = sorted_p <= thresholds

    if not np.any(passed):
        return mask, np.nan

    k_max = np.max(
        np.where(
            passed
        )[0]
    )

    cutoff = sorted_p[
        k_max
    ]

    mask = p_values <= cutoff

    return mask, float(cutoff)


def compute_all_pair_debiased_varx_network(
        x,
        u,
        true_link_mask,
        na,
        nb,
        ridge_lambda=1.0,
        alpha=0.05,
        custom_thresholds=(8.0, 10.0, 12.0, 15.0),
        pinv_rcond=1e-8
):
    """
    Compute raw and de-biased row-wise VARX Granger/deviance statistics
    for all directed links j -> i, i != j.

    This is scalable because each target row is fit once as a full model,
    and then reduced regressions are fit source-by-source.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    u = np.asarray(
        u,
        dtype=float
    )

    if u.ndim == 1:
        u = u[:, None]

    T, n_sources = x.shape

    true_link_mask = np.asarray(
        true_link_mask,
        dtype=bool
    )

    assert true_link_mask.shape == (
        n_sources,
        n_sources
    )

    rows = []

    for target in range(n_sources):

        X_full, y_target, start = build_varx_design(
            x=x,
            u=u,
            na=na,
            nb=nb,
            target=target
        )

        T_eff = len(
            y_target
        )

        ridge_gamma = float(
            ridge_lambda / np.sqrt(
                T_eff
            )
        )

        full_fit = fit_tikhonov_regression_with_bias(
            X=X_full,
            y=y_target,
            ridge_gamma=ridge_gamma,
            pinv_rcond=pinv_rcond
        )

        for source in range(n_sources):

            if source == target:
                continue

            remove_cols = source_lag_columns(
                source=source,
                n_sources=n_sources,
                na=na
            )

            keep_cols = [
                c
                for c in range(X_full.shape[1])
                if c not in remove_cols
            ]

            X_reduced = X_full[
                :,
                keep_cols
            ]

            reduced_fit = fit_tikhonov_regression_with_bias(
                X=X_reduced,
                y=y_target,
                ridge_gamma=ridge_gamma,
                pinv_rcond=pinv_rcond
            )

            sigma2_full = full_fit["sigma2"]
            sigma2_reduced = reduced_fit["sigma2"]

            raw_deviance = float(
                T_eff * np.log(
                    sigma2_reduced / sigma2_full
                )
            )

            bias_full = float(
                full_fit["bias"]
            )

            bias_reduced = float(
                reduced_fit["bias"]
            )

            debiased_deviance = float(
                raw_deviance
                - bias_reduced
                + bias_full
            )

            raw_deviance_for_p = max(
                raw_deviance,
                0.0
            )

            debiased_deviance_for_p = max(
                debiased_deviance,
                0.0
            )

            df_removed = na

            raw_p_value = float(
                chi2.sf(
                    raw_deviance_for_p,
                    df=df_removed
                )
            )

            debiased_p_value = float(
                chi2.sf(
                    debiased_deviance_for_p,
                    df=df_removed
                )
            )

            bic_threshold = float(
                df_removed * np.log(
                    T_eff
                )
            )

            aic_threshold = float(
                2.0 * df_removed
            )

            row = {
                "target": target,
                "source": source,
                "direction": f"{source}->{target}",

                "true_link": bool(
                    true_link_mask[target, source]
                ),

                "T": T,
                "T_eff": T_eff,
                "na": na,
                "nb": nb,
                "df_removed": df_removed,
                "ridge_lambda": ridge_lambda,
                "ridge_gamma": ridge_gamma,

                "sigma2_full": sigma2_full,
                "sigma2_reduced": sigma2_reduced,

                "raw_deviance": raw_deviance,
                "raw_deviance_for_p": raw_deviance_for_p,
                "raw_p_value": raw_p_value,
                "raw_chi_detected": raw_p_value < alpha,

                "full_bias_term": bias_full,
                "reduced_bias_term": bias_reduced,
                "bias_full": bias_full,
                "bias_reduced": bias_reduced,
                "bias_correction": -bias_reduced + bias_full,
                "negative_reduced_bias": -bias_reduced,
                "positive_full_bias": bias_full,

                "debiased_deviance": debiased_deviance,
                "debiased_deviance_for_p": debiased_deviance_for_p,
                "debiased_p_value": debiased_p_value,
                "debiased_chi_detected": debiased_p_value < alpha,

                "aic_threshold": aic_threshold,
                "bic_threshold": bic_threshold,

                "raw_aic_detected": raw_deviance_for_p > aic_threshold,
                "raw_bic_detected": raw_deviance_for_p > bic_threshold,

                "debiased_aic_detected": debiased_deviance_for_p > aic_threshold,
                "debiased_bic_detected": debiased_deviance_for_p > bic_threshold
            }

            for threshold in custom_thresholds:

                key = str(
                    threshold
                ).replace(
                    ".",
                    "p"
                )

                row[f"raw_D_gt_{key}_detected"] = (
                    raw_deviance_for_p > threshold
                )

                row[f"debiased_D_gt_{key}_detected"] = (
                    debiased_deviance_for_p > threshold
                )

            rows.append(
                row
            )

    results_df = pd.DataFrame(
        rows
    )

    raw_fdr_mask, raw_fdr_cutoff = bh_fdr_mask(
        results_df["raw_p_value"].values,
        alpha=alpha
    )

    debiased_fdr_mask, debiased_fdr_cutoff = bh_fdr_mask(
        results_df["debiased_p_value"].values,
        alpha=alpha
    )

    results_df["raw_fdr_detected"] = raw_fdr_mask
    results_df["debiased_fdr_detected"] = debiased_fdr_mask

    results_df["raw_fdr_cutoff"] = raw_fdr_cutoff
    results_df["debiased_fdr_cutoff"] = debiased_fdr_cutoff

    return results_df


def compute_confusion_metrics(
        y_true,
        y_pred
):
    y_true = np.asarray(
        y_true,
        dtype=bool
    )

    y_pred = np.asarray(
        y_pred,
        dtype=bool
    )

    tp = int(
        np.sum(
            y_true & y_pred
        )
    )

    fp = int(
        np.sum(
            ~y_true & y_pred
        )
    )

    tn = int(
        np.sum(
            ~y_true & ~y_pred
        )
    )

    fn = int(
        np.sum(
            y_true & ~y_pred
        )
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if np.isfinite(precision)
        and np.isfinite(tpr)
        and (precision + tpr) > 0
        else np.nan
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "specificity": specificity,
        "f1": f1
    }