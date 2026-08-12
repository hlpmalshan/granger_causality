import os
import time

# One complete EM fit is handled by one process.  Keep BLAS from adding nested
# workers when users later parallelize independent work units.
BLAS_THREADS = int(os.environ.get("EXPERIMENT_33B_BLAS_THREADS", "1"))
os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS)
os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS)
os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS)

import numpy as np
import pandas as pd

from src.ssm.em_varx_p_known_c_group_lasso_controlled import (
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedA,
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB,
)
from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
)
from src.ssm.kalman_varx_p import (
    extract_current_latent_state,
    kalman_smooth_varx_p_companion,
)
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.observed_likelihood_deviance import final_observed_log_likelihood
from src.stats.scalable_debiased_varx_network import (
    companion_spectral_radius,
    compute_all_pair_debiased_varx_network,
    compute_confusion_metrics,
)
from src.varx.varx_generator import generate_colored_input
from src.varx.varx_order_selection import fit_varx_multitarget


# ---------------------------------------------------------------------------
# Experiment 33B configuration
# ---------------------------------------------------------------------------

N_SOURCES = 2
N_INPUTS = 1
na = 2
nb = 3
BURN_IN = 300
Q_TRUE = 0.50 * np.eye(N_SOURCES)
R_TRUE = 0.60 * np.eye(N_SOURCES)
ESTIMATE_Q = False
ESTIMATE_R = False
ALPHA = 0.05
RIDGE_LAMBDA_DEBIAS = 1.0
MAX_ITER = 100
TOL = 1e-6
R_FLOOR = 0.30
BASE_SEED = 3800000

CASE_LIST = ["no_link", "true_link"]
C_MODE_LIST = ["identity", "mild_mixing", "strong_mixing"]
T_VALUES = [100, 250, 500, 1000, 2000]
N_OUTER_RUNS = 100
LAMBDA_A_GROUP_FRACTION_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.10]
MODEL_VARIANTS = [
    "direct_latent_varx",
    "state_space_em_estimate_A_B",
    "state_space_em_fixed_B_true",
    "state_space_em_fixed_A_true",
]
CUSTOM_THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)

RESULTS_DIR = os.environ.get("EXPERIMENT_33B_RESULTS_DIR", "results/experiment_33b")
PREFIX = "experiment_33b_two_source"
PATHS = {
    "parameter": os.path.join(RESULTS_DIR, PREFIX + "_parameter_recovery_results.csv"),
    "gc": os.path.join(RESULTS_DIR, PREFIX + "_gc_readout_results.csv"),
    "parameter_summary": os.path.join(RESULTS_DIR, PREFIX + "_parameter_recovery_summary.csv"),
    "decision": os.path.join(RESULTS_DIR, PREFIX + "_gc_decision_summary.csv"),
    "roc": os.path.join(RESULTS_DIR, PREFIX + "_roc_summary.csv"),
    "roc_points": os.path.join(RESULTS_DIR, PREFIX + "_roc_curve_points.csv"),
    "parameter_partial": os.path.join(RESULTS_DIR, PREFIX + "_parameter_recovery_results_partial.csv"),
    "gc_partial": os.path.join(RESULTS_DIR, PREFIX + "_gc_readout_results_partial.csv"),
}


def case_parameters(case_name):
    A1 = np.array([[0.60, 0.00], [0.00, 0.55]], dtype=float)
    A2 = np.array([[-0.10, 0.00], [0.00, -0.08]], dtype=float)
    if case_name == "true_link":
        A1[0, 1] = 0.25
        A2[0, 1] = 0.10
    elif case_name != "no_link":
        raise ValueError(f"Unknown case: {case_name}")
    B = [
        np.array([[0.50], [0.40]]),
        np.array([[0.25], [0.20]]),
        np.array([[0.10], [0.08]]),
    ]
    mask = np.zeros((N_SOURCES, N_SOURCES), dtype=bool)
    mask[0, 1] = case_name == "true_link"
    return [A1, A2], B, mask


def mixing_matrix(c_mode):
    matrices = {
        "identity": np.eye(2),
        "mild_mixing": np.array([[1.00, 0.20], [0.15, 1.00]]),
        "strong_mixing": np.array([[1.00, 0.45], [0.35, 1.00]]),
    }
    if c_mode not in matrices:
        raise ValueError(f"Unknown C mode: {c_mode}")
    return matrices[c_mode].copy()


def simulate_case(case_name, T, c_mode, seed):
    A_true, B_true, true_link_mask = case_parameters(case_name)
    C = mixing_matrix(c_mode)
    u_total = generate_colored_input(
        n_samples=T + BURN_IN, ar_coeff=0.95, noise_std=1.0,
        random_seed=seed + 100,
    )
    simulation = generate_ssm_varx_p_data(
        A_matrices=A_true, B_matrices=B_true, u=u_total,
        Q=Q_TRUE, R=R_TRUE, C=C, D=None, burn_in=BURN_IN,
        random_seed=seed + 200, return_augmented=True,
    )
    return {
        "x_true": simulation["x"], "y_obs": simulation["y"],
        "u": simulation["u"], "C": C, "A_true": A_true,
        "B_true": B_true, "true_link_mask": true_link_mask,
    }


def direct_varx_fit(x, u):
    fit = fit_varx_multitarget(
        y=x, u=u, na=na, nb=nb, gamma=0.0, penalty="diag",
        include_intercept=False, start_lag=max(na, nb - 1),
    )
    beta = fit["beta"]
    cursor = 0
    A_hat, B_hat = [], []
    for _ in range(na):
        A_hat.append(beta[cursor:cursor + N_SOURCES].T.copy())
        cursor += N_SOURCES
    for _ in range(nb):
        B_hat.append(beta[cursor:cursor + N_INPUTS].T.copy())
        cursor += N_INPUTS
    return A_hat, B_hat


def em_kwargs(C, lambda_fraction, seed):
    return dict(
        na=na, nb=nb, C=C, D=None, max_iter=MAX_ITER, tol=TOL,
        ridge_m_step=1e-4, covariance_floor=1e-6,
        R_init=R_TRUE.copy(), Q_init=Q_TRUE.copy(),
        estimate_Q=ESTIMATE_Q, estimate_R=ESTIMATE_R, R_floor=R_FLOOR,
        zero_constraints=[], initial_parameters=None, jitter_scale=0.0,
        random_seed=seed, verbose=False, alpha_Q=0.0, alpha_R=0.0,
        shrinkage_target_Q="spherical", shrinkage_target_R="spherical",
        lambda_A_group_fraction=lambda_fraction,
        ridge_A_offdiag=1e-4, ridge_A_diag=1e-4, ridge_B=1e-4,
        group_solver_max_iter=5000, group_solver_tol=1e-7,
        stabilize_A=True, target_radius=0.98,
    )


def fit_em_variant(variant, data, lambda_fraction, seed):
    kwargs = em_kwargs(data["C"], lambda_fraction, seed)
    if variant == "state_space_em_estimate_A_B":
        model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(**kwargs)
    elif variant == "state_space_em_fixed_B_true":
        model = EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB(
            fixed_B_matrices=data["B_true"], **kwargs
        )
    elif variant == "state_space_em_fixed_A_true":
        model = EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedA(
            fixed_A_matrices=data["A_true"], **kwargs
        )
    else:
        raise ValueError(f"Not an EM variant: {variant}")
    model.fit(data["y_obs"], data["u"])
    posterior = kalman_smooth_varx_p_companion(
        y=data["y_obs"], u=data["u"], F=model.F, G=model.G,
        Q_aug=model.Q_aug, R=model.R, C_aug=model.C_aug, nb=nb, D=model.D,
    )
    filtered = extract_current_latent_state(posterior["filter"]["x_filt"], N_SOURCES)
    smoothed = extract_current_latent_state(posterior["smoother"]["x_smooth"], N_SOURCES)
    return model, filtered, smoothed


def safe_relative_error(estimate, truth):
    denominator = np.linalg.norm(truth)
    return float(np.linalg.norm(estimate - truth) / denominator) if denominator > 0 else np.nan


def safe_pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 2 or np.std(x[keep]) < 1e-12 or np.std(y[keep]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def safe_spearman(x, y):
    return safe_pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def signal_metrics(signal, truth):
    correlations = [safe_pearson(signal[:, i], truth[:, i]) for i in range(N_SOURCES)]
    return float(np.mean((signal - truth) ** 2)), float(np.nanmean(correlations))


def parameter_row(data, A_hat, B_hat, metadata, model=None, filtered=None, smoothed=None):
    A_true = np.asarray(data["A_true"])
    B_true = np.asarray(data["B_true"])
    A_hat_array = np.asarray(A_hat)
    B_hat_array = np.asarray(B_hat)
    offdiag = np.broadcast_to(~np.eye(N_SOURCES, dtype=bool), A_true.shape)
    diagonal = ~offdiag
    filtered_mse, filtered_corr = (np.nan, np.nan) if filtered is None else signal_metrics(filtered, data["x_true"])
    smoothed_mse, smoothed_corr = (np.nan, np.nan) if smoothed is None else signal_metrics(smoothed, data["x_true"])
    row = {
        **metadata,
        "A_relative_frobenius_error": safe_relative_error(A_hat_array, A_true),
        "A_offdiag_relative_frobenius_error": safe_relative_error(A_hat_array[offdiag], A_true[offdiag]),
        "A_diagonal_relative_frobenius_error": safe_relative_error(A_hat_array[diagonal], A_true[diagonal]),
        "A_mean_absolute_error": float(np.mean(np.abs(A_hat_array - A_true))),
        "A_offdiag_mean_absolute_error": float(np.mean(np.abs(A_hat_array[offdiag] - A_true[offdiag]))),
        "A_diagonal_mean_absolute_error": float(np.mean(np.abs(A_hat_array[diagonal] - A_true[diagonal]))),
        "A_pearson_correlation": safe_pearson(A_true.ravel(), A_hat_array.ravel()),
        "A_offdiag_pearson_correlation": safe_pearson(A_true[offdiag], A_hat_array[offdiag]),
        "A_offdiag_spearman_correlation": safe_spearman(A_true[offdiag], A_hat_array[offdiag]),
        "Y_to_X_group_norm_true": float(np.linalg.norm(A_true[:, 0, 1])),
        "Y_to_X_group_norm_hat": float(np.linalg.norm(A_hat_array[:, 0, 1])),
        "X_to_Y_group_norm_true": float(np.linalg.norm(A_true[:, 1, 0])),
        "X_to_Y_group_norm_hat": float(np.linalg.norm(A_hat_array[:, 1, 0])),
        "B_relative_frobenius_error": safe_relative_error(B_hat_array, B_true),
        "B_mean_absolute_error": float(np.mean(np.abs(B_hat_array - B_true))),
        "B_pearson_correlation": safe_pearson(B_true.ravel(), B_hat_array.ravel()),
        "B_spearman_correlation": safe_spearman(B_true.ravel(), B_hat_array.ravel()),
        "filtered_signal_mse": filtered_mse, "smoothed_signal_mse": smoothed_mse,
        "filtered_signal_correlation": filtered_corr,
        "smoothed_signal_correlation": smoothed_corr,
        "C_condition_number": float(np.linalg.cond(data["C"])),
    }
    labels = ("X", "Y")
    for lag in range(na):
        for target in range(N_SOURCES):
            for source in range(N_SOURCES):
                stem = f"A{lag + 1}_{labels[target]}_from_{labels[source]}"
                row[stem + "_true"] = A_true[lag, target, source]
                row[stem + "_hat"] = A_hat_array[lag, target, source]
    for lag in range(nb):
        row[f"B_lag{lag}_relative_frobenius_error"] = safe_relative_error(B_hat_array[lag], B_true[lag])
        for target in range(N_SOURCES):
            stem = f"B{lag}_{labels[target]}"
            row[stem + "_true"] = B_true[lag, target, 0]
            row[stem + "_hat"] = B_hat_array[lag, target, 0]
    if model is None:
        row.update({
            "full_log_likelihood": np.nan,
            "full_spectral_radius": companion_spectral_radius(A_hat),
            "full_em_iterations": np.nan, "Q_trace": np.nan, "R_trace": np.nan,
            "estimate_Q": False, "estimate_R": False,
            "offdiag_A_group_count": int(sum(np.linalg.norm(A_hat_array[:, t, s]) > 1e-8 for t in range(2) for s in range(2) if t != s)),
            "offdiag_A_coefficient_count": int(np.sum(np.abs(A_hat_array[:, ~np.eye(2, dtype=bool)]) > 1e-8)),
            "group_solver_converged_last": np.nan,
            "group_solver_mean_iterations_last": np.nan,
            "group_solver_objective_last": np.nan,
        })
    else:
        history = model.group_solver_converged_history
        row.update({
            "full_log_likelihood": final_observed_log_likelihood(model),
            "full_spectral_radius": model.spectral_radius(),
            "full_em_iterations": len(model.log_likelihoods),
            "Q_trace": float(np.trace(model.Q)), "R_trace": float(np.trace(model.R)),
            "estimate_Q": model.estimate_Q, "estimate_R": model.estimate_R,
            "offdiag_A_group_count": model.count_nonzero_offdiag_A_groups(),
            "offdiag_A_coefficient_count": model.count_nonzero_offdiag_A_coefficients(),
            "group_solver_converged_last": history[-1] if history else np.nan,
            "group_solver_mean_iterations_last": model.group_solver_iterations_history[-1] if history else np.nan,
            "group_solver_objective_last": model.group_solver_objective_history[-1] if history else np.nan,
        })
    return row


def gc_frames(data, metadata, filtered=None, smoothed=None):
    pinv_proxy = data["y_obs"] @ np.linalg.pinv(data["C"]).T
    signals = [("x_true", data["x_true"]), ("y_obs", data["y_obs"]), ("pinv_proxy", pinv_proxy)]
    if filtered is not None:
        signals.extend([("em_filtered", filtered), ("em_smoothed", smoothed)])
    frames = []
    for signal_type, signal in signals:
        frame = compute_all_pair_debiased_varx_network(
            x=signal, u=data["u"], true_link_mask=data["true_link_mask"],
            na=na, nb=nb, ridge_lambda=RIDGE_LAMBDA_DEBIAS, alpha=ALPHA,
            custom_thresholds=CUSTOM_THRESHOLDS,
        )
        for key, value in metadata.items():
            frame[key] = value
        frame["signal_type"] = signal_type
        frame["direction_label"] = np.where(
            (frame["source"] == 1) & (frame["target"] == 0), "Y_to_X", "X_to_Y"
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    # Synced Windows drives can briefly lock a checkpoint while uploading it.
    last_error = None
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            wait_seconds = min(0.25 * (2 ** attempt), 5.0)
            print(
                f"Checkpoint locked; retry {attempt + 1}/12 in "
                f"{wait_seconds:.2f}s: {path}"
            )
            time.sleep(wait_seconds)
    raise PermissionError(
        f"Could not promote checkpoint. Recovery file remains at {temporary}."
    ) from last_error


def load_partial(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


WORK_KEY_COLUMNS = [
    "outer_run", "case_name", "T", "C_MODE", "MODEL_VARIANT",
    "lambda_A_group_fraction",
]


def reconcile_partial_checkpoints(parameter_results, gc_results):
    """Retain only work units complete in both independently atomic CSVs."""
    if len(parameter_results) == 0 or len(gc_results) == 0:
        return parameter_results.iloc[0:0].copy(), gc_results.iloc[0:0].copy(), set()

    parameter_keys = set(map(
        tuple,
        parameter_results[WORK_KEY_COLUMNS].itertuples(index=False, name=None),
    ))
    gc_counts = gc_results.groupby(WORK_KEY_COLUMNS, dropna=False).size()
    complete_gc_keys = set()
    for key, count in gc_counts.items():
        key = key if isinstance(key, tuple) else (key,)
        expected = 6 if key[4] == "direct_latent_varx" else 10
        if int(count) == expected:
            complete_gc_keys.add(tuple(key))
    complete = parameter_keys & complete_gc_keys

    parameter_mask = parameter_results[WORK_KEY_COLUMNS].apply(tuple, axis=1).isin(complete)
    gc_mask = gc_results[WORK_KEY_COLUMNS].apply(tuple, axis=1).isin(complete)
    removed_parameter = int((~parameter_mask).sum())
    removed_gc = int((~gc_mask).sum())
    if removed_parameter or removed_gc:
        print(
            "Reconciled interrupted checkpoint: dropping "
            f"{removed_parameter} parameter rows and {removed_gc} GC rows. "
            "Only those incomplete work units will be recomputed."
        )
    return (
        parameter_results.loc[parameter_mask].reset_index(drop=True),
        gc_results.loc[gc_mask].reset_index(drop=True),
        complete,
    )


def save_partials(parameter_results, gc_results):
    atomic_csv(parameter_results, PATHS["parameter_partial"])
    atomic_csv(gc_results, PATHS["gc_partial"])


def aggregate_statistics(frame, group_columns, metric_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["n_runs"] = len(group)
        for metric in metric_columns:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"median_{metric}"] = group[metric].median()
            row[f"std_{metric}"] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_summary(parameter_results):
    metrics = [
        "A_relative_frobenius_error", "A_offdiag_relative_frobenius_error",
        "A_diagonal_relative_frobenius_error", "Y_to_X_group_norm_hat",
        "X_to_Y_group_norm_hat", "B_relative_frobenius_error",
        "B_lag0_relative_frobenius_error", "B_lag1_relative_frobenius_error",
        "B_lag2_relative_frobenius_error", "filtered_signal_mse",
        "smoothed_signal_mse", "filtered_signal_correlation",
        "smoothed_signal_correlation", "full_spectral_radius",
        "full_em_iterations", "group_solver_converged_last",
        "group_solver_mean_iterations_last", "group_solver_objective_last",
    ]
    return aggregate_statistics(
        parameter_results,
        ["case_name", "T", "C_MODE", "MODEL_VARIANT", "lambda_A_group_fraction"],
        metrics,
    )


def decision_summary(gc_results):
    rules = [
        "raw_chi_detected", "debiased_chi_detected", "raw_fdr_detected",
        "debiased_fdr_detected", "raw_bic_detected", "debiased_bic_detected",
    ]
    for threshold in CUSTOM_THRESHOLDS:
        label = str(threshold).replace(".", "p")
        rules.extend([f"raw_D_gt_{label}_detected", f"debiased_D_gt_{label}_detected"])
    detailed_groups = [
        "case_name", "T", "C_MODE", "MODEL_VARIANT", "signal_type",
        "lambda_A_group_fraction", "direction_label",
    ]
    confusion_groups = [
        "T", "C_MODE", "MODEL_VARIANT", "signal_type", "lambda_A_group_fraction",
    ]
    rows = []
    for keys, group in gc_results.groupby(detailed_groups, dropna=False):
        metadata = dict(zip(detailed_groups, keys))
        for rule in rules:
            rows.append({
                **metadata, "summary_type": "direction_detection_rate",
                "decision_rule": rule, "n_tests": len(group),
                "detection_rate": group[rule].mean(),
                "mean_debiased_deviance": group["debiased_deviance"].mean(),
                "median_debiased_deviance": group["debiased_deviance"].median(),
                "mean_p_value": group["debiased_p_value"].mean(),
                "median_p_value": group["debiased_p_value"].median(),
            })
    for keys, group in gc_results.groupby(confusion_groups, dropna=False):
        metadata = dict(zip(confusion_groups, keys))
        for rule in rules:
            metrics = compute_confusion_metrics(group["true_link"], group[rule])
            rows.append({
                **metadata, "summary_type": "confusion_across_cases",
                "decision_rule": rule, "n_tests": len(group), **metrics,
            })
    return pd.DataFrame(rows)


def _threshold_points(y, scores):
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    return [(threshold, compute_confusion_metrics(y, scores >= threshold)) for threshold in thresholds]


def roc_tables(score_frame):
    # Keep signal representations separate; pooling x_true, y_obs, proxy, and
    # posterior scores would make a threshold curve scientifically ambiguous.
    group_columns = [
        "T", "C_MODE", "MODEL_VARIANT", "lambda_A_group_fraction",
        "signal_type", "score_type",
    ]
    summaries, curves = [], []
    for keys, group in score_frame.groupby(group_columns, dropna=False):
        metadata = dict(zip(group_columns, keys))
        y = group["true_link"].astype(bool).to_numpy()
        scores = group["score"].astype(float).to_numpy()
        keep = np.isfinite(scores)
        y, scores = y[keep], scores[keep]
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        points = _threshold_points(y, scores)
        try:
            from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
            roc_auc = float(roc_auc_score(y, scores))
            auprc = float(average_precision_score(y, scores))
            precision, recall, pr_thresholds = precision_recall_curve(y, scores)
            f1_pr = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-15)
            best_f1_index = int(np.nanargmax(f1_pr))
            best_f1, best_f1_threshold = float(f1_pr[best_f1_index]), float(pr_thresholds[best_f1_index])
        except ImportError:
            fpr = np.asarray([metrics["fpr"] for _, metrics in points])
            tpr = np.asarray([metrics["tpr"] for _, metrics in points])
            trapezoid = getattr(np, "trapezoid", None)
            if trapezoid is None:
                trapezoid = np.trapz
            roc_auc = float(trapezoid(tpr, fpr))
            order = np.argsort(-scores, kind="mergesort")
            labels = y[order].astype(int)
            precision_rank = np.cumsum(labels) / np.arange(1, len(labels) + 1)
            auprc = float(np.sum(precision_rank * labels) / max(1, np.sum(labels)))
            f1_values = np.asarray([metrics["f1"] for _, metrics in points])
            best_f1_index = int(np.nanargmax(f1_values))
            best_f1_threshold, best_f1 = float(points[best_f1_index][0]), float(f1_values[best_f1_index])
        youden = np.asarray([metrics["tpr"] - metrics["fpr"] for _, metrics in points])
        best_index = int(np.nanargmax(youden))
        best_threshold, best_metrics = points[best_index]
        summaries.append({
            **metadata, "n_links": len(y), "ROC_AUC": roc_auc,
            "average_precision": auprc, "AUPRC": auprc,
            "best_youden_J": youden[best_index],
            "best_youden_threshold": best_threshold,
            "FPR_at_best_youden": best_metrics["fpr"],
            "TPR_at_best_youden": best_metrics["tpr"],
            "precision_at_best_youden": best_metrics["precision"],
            "best_F1": best_f1, "threshold_at_best_F1": best_f1_threshold,
        })
        for threshold, metrics in points:
            curves.append({
                **metadata, "threshold": threshold, "fpr": metrics["fpr"],
                "tpr": metrics["tpr"], "precision": metrics["precision"],
                "recall": metrics["tpr"], "specificity": metrics["specificity"],
                "f1": metrics["f1"], "youden_j": metrics["tpr"] - metrics["fpr"],
            })
    return pd.DataFrame(summaries), pd.DataFrame(curves)


def build_score_frame(parameter_results, gc_results):
    parameter_scores = []
    for _, row in parameter_results.iterrows():
        common = {
            "T": row["T"], "C_MODE": row["C_MODE"],
            "MODEL_VARIANT": row["MODEL_VARIANT"],
            "lambda_A_group_fraction": row["lambda_A_group_fraction"],
            "signal_type": "model_A",
        }
        parameter_scores.extend([
            {**common, "true_link": row["case_name"] == "true_link", "score_type": "estimated_A_group_norm", "score": row["Y_to_X_group_norm_hat"]},
            {**common, "true_link": False, "score_type": "estimated_A_group_norm", "score": row["X_to_Y_group_norm_hat"]},
        ])
    gc_scores = []
    for score_type in ("debiased_deviance", "raw_deviance"):
        part = gc_results[[
            "T", "C_MODE", "MODEL_VARIANT", "lambda_A_group_fraction",
            "signal_type", "true_link", score_type
        ]].copy()
        part["score_type"] = score_type
        part["score"] = part[score_type]
        gc_scores.append(part.drop(columns=[score_type]))
    return pd.concat([pd.DataFrame(parameter_scores), *gc_scores], ignore_index=True)


def selected_detection(gc_block):
    rows = []
    for direction in ("Y_to_X", "X_to_Y"):
        group = gc_block[(gc_block["signal_type"] == "em_filtered") & (gc_block["direction_label"] == direction)]
        if len(group):
            rows.append(f"{direction}: chi={group['debiased_chi_detected'].mean():.2f}, BIC={group['debiased_bic_detected'].mean():.2f}, D>10={group['debiased_D_gt_10p0_detected'].mean():.2f}")
    return "; ".join(rows) if rows else "direct baseline (no EM posterior)"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parameter_results = load_partial(PATHS["parameter_partial"])
    gc_results = load_partial(PATHS["gc_partial"])
    parameter_results, gc_results, completed = reconcile_partial_checkpoints(
        parameter_results, gc_results
    )

    for outer_run in range(N_OUTER_RUNS):
        seed = BASE_SEED + outer_run
        for case_name in CASE_LIST:
            for T in T_VALUES:
                for c_mode in C_MODE_LIST:
                    # Common random numbers pair all C modes and both cases as
                    # closely as their differing dynamics permit.
                    data = simulate_case(case_name, T, c_mode, seed)
                    for lambda_fraction in LAMBDA_A_GROUP_FRACTION_GRID:
                        for variant in MODEL_VARIANTS:
                            key = (outer_run, case_name, T, c_mode, variant, float(lambda_fraction))
                            if key in completed:
                                continue
                            metadata = {
                                "outer_run": outer_run, "random_seed": seed,
                                "case_name": case_name, "T": T, "C_MODE": c_mode,
                                "MODEL_VARIANT": variant,
                                "lambda_A_group_fraction": float(lambda_fraction),
                            }
                            if variant == "direct_latent_varx":
                                A_hat, B_hat = direct_varx_fit(data["x_true"], data["u"])
                                model, filtered, smoothed = None, None, None
                            else:
                                model, filtered, smoothed = fit_em_variant(
                                    variant, data, lambda_fraction, seed + 500
                                )
                                A_hat, B_hat = model.A_matrices, model.B_matrices
                            row = parameter_row(
                                data, A_hat, B_hat, metadata, model, filtered, smoothed
                            )
                            parameter_results = pd.concat(
                                [parameter_results, pd.DataFrame([row])], ignore_index=True
                            )
                            gc_block = gc_frames(data, metadata, filtered, smoothed)
                            gc_results = pd.concat([gc_results, gc_block], ignore_index=True)
                            print(
                                f"33B case={case_name} T={T} C={c_mode} variant={variant} "
                                f"lambda={lambda_fraction:g} Aoff={row['A_offdiag_relative_frobenius_error']:.3g} "
                                f"Brel={row['B_relative_frobenius_error']:.3g} "
                                f"Y->X={row['Y_to_X_group_norm_true']:.3g}/{row['Y_to_X_group_norm_hat']:.3g} "
                                f"X->Y={row['X_to_Y_group_norm_true']:.3g}/{row['X_to_Y_group_norm_hat']:.3g} "
                                f"filt/smooth MSE={row['filtered_signal_mse']:.3g}/{row['smoothed_signal_mse']:.3g} "
                                f"{selected_detection(gc_block)}"
                            )
                            completed.add(key)
                    # Keep accumulating this outer run in memory. Rewriting the
                    # entire growing checkpoint for every case/T/C block causes
                    # excessive I/O and lock contention on synced drives.
        save_partials(parameter_results, gc_results)
        print(f"Completed outer run {outer_run + 1}/{N_OUTER_RUNS}")

    parameter_summary_frame = parameter_summary(parameter_results)
    decision_frame = decision_summary(gc_results)
    roc_summary_frame, roc_points_frame = roc_tables(
        build_score_frame(parameter_results, gc_results)
    )
    atomic_csv(parameter_results, PATHS["parameter"])
    atomic_csv(gc_results, PATHS["gc"])
    atomic_csv(parameter_summary_frame, PATHS["parameter_summary"])
    atomic_csv(decision_frame, PATHS["decision"])
    atomic_csv(roc_summary_frame, PATHS["roc"])
    atomic_csv(roc_points_frame, PATHS["roc_points"])

    print("\nInterpretation guide")
    print("A. Direct latent succeeds -> the row-wise VARX estimator is sound.")
    print("B. Joint EM fails while direct succeeds -> latent estimation/EM is the bottleneck.")
    print("C. Fixed-B improves A -> B error leaks into A.")
    print("D. Fixed-A recovers B -> joint A/B optimization is the problem.")
    print("E. Recovery worsens with C mixing -> source mixing/leakage is important.")
    print("F. Larger T solves 2D -> 20D is primarily sample complexity; otherwise EM is structural.")


if __name__ == "__main__":
    main()


# TODO Experiment 33C: implement Eq. 7 only after locating the exact equation
# in the uploaded “Non-Asymptotic Guarantees...” paper.
# TODO Experiment 33D: source-count scaling N = 2, 5, 10, 20.
# TODO Experiment 33E: increased lag ratio p/T, as requested by the advisor.
# TODO Experiment 33F: exploratory density/coverage edge-distribution diagnostics.
# Density/coverage and Eq. 7 are intentionally not implemented in 33B.
