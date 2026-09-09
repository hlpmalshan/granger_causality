import os

BLAS_THREADS_PER_WORKER = int(os.environ.get("EXPERIMENT_33C_BLAS_THREADS", "1"))
os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
)
from src.ssm.kalman_varx_p import extract_current_latent_state, kalman_smooth_varx_p_companion
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.lasso_gc_statistic import NUMBA_AVAILABLE, compute_lasso_gc_eq7_network
from src.stats.observed_likelihood_deviance import final_observed_log_likelihood
from src.stats.scalable_debiased_varx_network import (
    companion_spectral_radius, compute_all_pair_debiased_varx_network,
    compute_confusion_metrics,
)
from src.varx.varx_generator import generate_colored_input


SOURCE_COUNT_VALUES = [2, 5, 10, 20]
T_VALUES = [500, 1000, 2000]
C_MODE_LIST = ["identity", "mild_mixing"]
N_OUTER_RUNS = 20
LAMBDA_GRIDS = {
    2: [0.0, 0.001, 0.003],
    5: [0.0, 0.001, 0.003, 0.01],
    10: [0.0, 0.003, 0.01, 0.03],
    20: [0.003, 0.01, 0.03, 0.10],
}
LGC_LAMBDA_GRID = [0.0, 0.003, 0.01]
N_INPUTS, na, nb, BURN_IN = 1, 2, 3, 300
LINK_DENSITY = 0.05
ALPHA, RIDGE_LAMBDA_DEBIAS = 0.05, 1.0
MAX_ITER, TOL, BASE_SEED = 100, 1e-6, 3900000
CUSTOM_THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
FPR_LEVELS = (0.01, 0.03, 0.05, 0.10)
RESULTS_DIR = os.environ.get("EXPERIMENT_33C_RESULTS_DIR", "results/experiment_33c")
N_WORKERS = int(os.environ.get("EXPERIMENT_33C_WORKERS", str(min(4, os.cpu_count() or 1))))


def make_network(n, seed, target_radius=0.82):
    rng = np.random.default_rng(seed)
    A = [np.zeros((n, n)) for _ in range(na)]
    A[0] += np.diag(rng.uniform(0.25, 0.45, n))
    A[1] += np.diag(rng.uniform(-0.12, -0.04, n))
    pairs = [(source, target) for source in range(n) for target in range(n) if source != target]
    n_links = max(1, round(LINK_DENSITY * n * (n - 1)))
    selected = rng.choice(len(pairs), n_links, replace=False)
    mask = np.zeros((n, n), dtype=bool)
    for index in selected:
        source, target = pairs[index]
        sign = rng.choice((-1.0, 1.0))
        A[0][target, source] = sign * rng.uniform(0.08, 0.16)
        A[1][target, source] = sign * rng.uniform(0.03, 0.09)
        mask[target, source] = True
    radius = companion_spectral_radius(A)
    if radius >= target_radius:
        scale = target_radius / (radius + 1e-12)
        A = [matrix * scale for matrix in A]
    return A, mask


def simulate(n, T, c_mode, seed):
    A, mask = make_network(n, seed)
    rng = np.random.default_rng(seed + 100)
    B = [rng.normal(0, scale, (n, 1)) for scale in (0.45, 0.25, 0.15)]
    C = np.eye(n)
    if c_mode == "mild_mixing":
        noise = rng.normal(0, 0.05 / np.sqrt(n), (n, n))
        np.fill_diagonal(noise, 0.0)
        C += noise
    u = generate_colored_input(T + BURN_IN, 0.95, 1.0, seed + 200)
    Q, R = 0.50 * np.eye(n), 0.60 * np.eye(n)
    result = generate_ssm_varx_p_data(
        A_matrices=A, B_matrices=B, u=u, Q=Q, R=R, C=C, D=None,
        burn_in=BURN_IN, random_seed=seed + 300, return_augmented=True,
    )
    return dict(x=result["x"], y=result["y"], u=result["u"], A=A, B=B,
                C=C, Q=Q, R=R, mask=mask)


def correlations(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return correlations(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def relative(estimate, truth):
    denominator = np.linalg.norm(truth)
    return float(np.linalg.norm(estimate - truth) / denominator) if denominator else np.nan


def curve_metrics(y, score):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1]]
    points = [(threshold, compute_confusion_metrics(y, score >= threshold)) for threshold in thresholds]
    fpr = np.array([m["fpr"] for _, m in points])
    tpr = np.array([m["tpr"] for _, m in points])
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    roc_auc = float(trapezoid(tpr, fpr))
    order = np.argsort(-score, kind="mergesort")
    labels = y[order].astype(int)
    precision_rank = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    auprc = float(np.sum(precision_rank * labels) / max(1, labels.sum()))
    youden = tpr - fpr
    best_j = int(np.nanargmax(youden))
    f1s = np.array([m["f1"] for _, m in points])
    f1_safe = np.where(np.isfinite(f1s), f1s, -np.inf)
    best_f1 = int(np.argmax(f1_safe))
    summary = {
        "ROC_AUC": roc_auc, "AUPRC": auprc, "best_youden_J": youden[best_j],
        "best_youden_threshold": points[best_j][0],
        "FPR_at_best_youden": points[best_j][1]["fpr"],
        "TPR_at_best_youden": points[best_j][1]["tpr"],
        "precision_at_best_youden": points[best_j][1]["precision"],
        "F1_at_best_youden": points[best_j][1]["f1"],
        "best_F1": f1s[best_f1], "threshold_at_best_F1": points[best_f1][0],
    }
    for level in FPR_LEVELS:
        eligible = [i for i, (_, m) in enumerate(points) if m["fpr"] <= level]
        index = max(eligible, key=lambda i: (points[i][1]["tpr"], -points[i][0]))
        label = str(level).replace(".", "p")
        summary[f"threshold_for_FPR_leq_{label}"] = points[index][0]
        for metric, name in (("tpr", "TPR"), ("precision", "precision"), ("f1", "F1")):
            summary[f"{name}_at_FPR_leq_{label}"] = points[index][1][metric]
    return summary, points


def fit_model(data, n, lambda_fraction, seed):
    model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
        na=na, nb=nb, C=data["C"], D=None, max_iter=MAX_ITER, tol=TOL,
        ridge_m_step=1e-4, covariance_floor=1e-6, R_init=data["R"], Q_init=data["Q"],
        estimate_Q=False, estimate_R=False, R_floor=0.30, zero_constraints=[],
        initial_parameters=None, jitter_scale=0.0, random_seed=seed, verbose=False,
        alpha_Q=0.0, alpha_R=0.0, shrinkage_target_Q="spherical",
        shrinkage_target_R="spherical", lambda_A_group_fraction=lambda_fraction,
        ridge_A_offdiag=1e-4, ridge_A_diag=1e-4, ridge_B=1e-4,
        group_solver_max_iter=5000, group_solver_tol=1e-7,
        stabilize_A=True, target_radius=0.98,
    )
    model.fit(data["y"], data["u"])
    posterior = kalman_smooth_varx_p_companion(
        data["y"], data["u"], model.F, model.G, model.Q_aug, model.R,
        model.C_aug, nb, model.D
    )
    filtered = extract_current_latent_state(posterior["filter"]["x_filt"], n)
    smoothed = extract_current_latent_state(posterior["smoother"]["x_smooth"], n)
    return model, filtered, smoothed


def recovery_rows(model, data, meta, filtered, smoothed):
    A, Ah, B, Bh = map(np.asarray, (data["A"], model.A_matrices, data["B"], model.B_matrices))
    n = A.shape[1]
    off = np.broadcast_to(~np.eye(n, dtype=bool), A.shape)
    diag = ~off
    support = []
    for target in range(n):
        for source in range(n):
            if source != target:
                support.append({**meta, "target": target, "source": source,
                    "true_link": data["mask"][target, source],
                    "true_A_group_norm": np.linalg.norm(A[:, target, source]),
                    "estimated_A_group_norm": np.linalg.norm(Ah[:, target, source])})
    support = pd.DataFrame(support)
    support_curve, _ = curve_metrics(support.true_link, support.estimated_A_group_norm)
    true_mean = support.loc[support.true_link, "estimated_A_group_norm"].mean()
    false_mean = support.loc[~support.true_link, "estimated_A_group_norm"].mean()
    def signal(signal):
        return float(np.mean((signal-data["x"])**2)), float(np.nanmean([
            correlations(signal[:, i], data["x"][:, i]) for i in range(n)]))
    fmse, fcorr = signal(filtered); smse, scorr = signal(smoothed)
    history = model.group_solver_converged_history
    row = {**meta,
        "A_relative_frobenius_error": relative(Ah, A),
        "A_offdiag_relative_frobenius_error": relative(Ah[off], A[off]),
        "A_diagonal_relative_frobenius_error": relative(Ah[diag], A[diag]),
        "A_mean_absolute_error": np.mean(np.abs(Ah-A)),
        "A_offdiag_mean_absolute_error": np.mean(np.abs(Ah[off]-A[off])),
        "A_diagonal_mean_absolute_error": np.mean(np.abs(Ah[diag]-A[diag])),
        "A_pearson_correlation": correlations(A.ravel(), Ah.ravel()),
        "A_offdiag_pearson_correlation": correlations(A[off], Ah[off]),
        "A_offdiag_spearman_correlation": spearman(A[off], Ah[off]),
        "A_group_norm_pearson_correlation": correlations(support.true_A_group_norm, support.estimated_A_group_norm),
        "A_group_norm_spearman_correlation": spearman(support.true_A_group_norm, support.estimated_A_group_norm),
        "mean_estimated_group_norm_true_links": true_mean,
        "mean_estimated_group_norm_false_links": false_mean,
        "group_norm_true_false_ratio": true_mean / false_mean if false_mean > 0 else np.inf,
        "B_relative_frobenius_error": relative(Bh, B), "B_mean_absolute_error": np.mean(np.abs(Bh-B)),
        "B_pearson_correlation": correlations(B.ravel(), Bh.ravel()), "B_spearman_correlation": spearman(B.ravel(), Bh.ravel()),
        "filtered_signal_mse": fmse, "smoothed_signal_mse": smse,
        "filtered_signal_correlation": fcorr, "smoothed_signal_correlation": scorr,
        "full_log_likelihood": final_observed_log_likelihood(model), "full_spectral_radius": model.spectral_radius(),
        "full_em_iterations": len(model.log_likelihoods), "Q_trace": np.trace(model.Q), "R_trace": np.trace(model.R),
        "estimate_Q": model.estimate_Q, "estimate_R": model.estimate_R,
        "C_condition_number": np.linalg.cond(data["C"]),
        "offdiag_A_group_count": model.count_nonzero_offdiag_A_groups(),
        "offdiag_A_coefficient_count": model.count_nonzero_offdiag_A_coefficients(),
        "mean_offdiag_A_group_norm": model.mean_offdiag_A_group_norm(),
        "median_offdiag_A_group_norm": model.median_offdiag_A_group_norm(),
        "max_offdiag_A_group_norm": model.max_offdiag_A_group_norm(),
        "group_solver_converged_last": history[-1] if history else np.nan,
        "group_solver_mean_iterations_last": model.group_solver_iterations_history[-1] if history else np.nan,
        "group_solver_objective_last": model.group_solver_objective_history[-1] if history else np.nan,
        "mean_lambda_A_group_effective": model.mean_lambda_A_group_effective,
        "median_lambda_A_group_effective": model.median_lambda_A_group_effective,
        "min_lambda_A_group_effective": model.min_lambda_A_group_effective,
        "max_lambda_A_group_effective": model.max_lambda_A_group_effective,
    }
    for lag in range(nb): row[f"B_lag{lag}_relative_frobenius_error"] = relative(Bh[lag], B[lag])
    for key, value in support_curve.items(): row["A_support_" + key] = value
    return row, support


def readouts(signal, signal_type, data, meta, lgc_lambdas):
    network = compute_all_pair_debiased_varx_network(
        signal, data["u"], data["mask"], na, nb, RIDGE_LAMBDA_DEBIAS,
        ALPHA, CUSTOM_THRESHOLDS
    )
    for key, value in {**meta, "signal_type": signal_type}.items(): network[key] = value
    lgc_frames = []
    for lgc_lambda in lgc_lambdas:
        frame = compute_lasso_gc_eq7_network(signal, data["u"], data["mask"], na, nb, lgc_lambda)
        for key, value in {**meta, "signal_type": signal_type}.items(): frame[key] = value
        lgc_frames.append(frame)
    return network, pd.concat(lgc_frames, ignore_index=True)


def work_unit(args):
    data, meta, lambda_fraction, seed = args
    n = meta["n_sources"]
    local = {**meta, "lambda_A_group_fraction": lambda_fraction}
    model, filtered, smoothed = fit_model(data, n, lambda_fraction, seed)
    parameter, support = recovery_rows(model, data, local, filtered, smoothed)
    networks, lgcs = [], []
    for name, signal in (("em_filtered", filtered), ("em_smoothed", smoothed)):
        network, lgc = readouts(signal, name, data, local, [lambda_fraction])
        networks.append(network); lgcs.append(lgc)
    return parameter, support, pd.concat(networks), pd.concat(lgcs)


def atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    for attempt in range(12):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            time.sleep(min(0.25 * 2**attempt, 5.0))
    raise PermissionError(f"Could not promote {temporary}; recovery file is preserved.")


def roc_outputs(network, lgc, support):
    score_frames = []
    for score in ("raw_deviance", "debiased_deviance"):
        part = network.copy(); part["score_type"] = score; part["score"] = part[score]; part["lgc_lambda"] = np.nan
        score_frames.append(part)
    for score in ("lgc_raw", "lgc_clipped"):
        part = lgc.copy(); part["score_type"] = score; part["score"] = part[score]
        score_frames.append(part)
    part = support.copy(); part["signal_type"] = "model_A"; part["score_type"] = "estimated_A_group_norm"
    part["score"] = part.estimated_A_group_norm; part["lgc_lambda"] = np.nan
    score_frames.append(part)
    scores = pd.concat(score_frames, ignore_index=True)
    groups = ["n_sources", "T", "C_MODE", "signal_type", "lambda_A_group_fraction", "score_type", "lgc_lambda"]
    summaries, points = [], []
    for keys, group in scores.groupby(groups, dropna=False):
        summary, curve = curve_metrics(group.true_link, group.score)
        meta = dict(zip(groups, keys)); summaries.append({**meta, "n_links": len(group), **summary})
        for threshold, metrics in curve:
            points.append({**meta, "threshold": threshold, **metrics, "recall": metrics["tpr"],
                           "youden_j": metrics["tpr"]-metrics["fpr"]})
    return pd.DataFrame(summaries), pd.DataFrame(points)


def grouped_stats(frame, groups, metrics):
    aggregations = {metric: ["mean", "median", "std"] for metric in metrics}
    result = frame.groupby(groups, dropna=False).agg(aggregations).reset_index()
    result.columns = ["_".join(filter(None, map(str, column))) if isinstance(column, tuple) else column for column in result.columns]
    return result


def decisions(frame, rules, groups):
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        for rule in rules:
            metrics = compute_confusion_metrics(group.true_link, group[rule])
            rows.append({**dict(zip(groups, keys)), "decision_rule": rule, **metrics,
                "acceptable_fpr_0p05": metrics["fpr"] <= .05,
                "acceptable_fpr_0p10": metrics["fpr"] <= .10,
                "conservative_good": metrics["fpr"] <= .05 and metrics["tpr"] >= .30,
                "balanced_good": metrics["fpr"] <= .10 and metrics["tpr"] >= .50})
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    partial = {
        "lgc_eq7": os.path.join(RESULTS_DIR, "lgc_eq7_results_partial.csv"),
        "network": os.path.join(RESULTS_DIR, "network_results_partial.csv"),
        "A_support": os.path.join(RESULTS_DIR, "A_support_results_partial.csv"),
        # Save the run-level completion ledger last. Its presence certifies that
        # all three detailed tables for that work unit were promoted first.
        "parameter_summary": os.path.join(RESULTS_DIR, "parameter_summary_partial.csv"),
    }
    dataframes = {name: pd.read_csv(path) if os.path.exists(path) else pd.DataFrame() for name, path in partial.items()}
    detail_keys = {
        "network": ["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction", "signal_type", "target", "source"],
        "lgc_eq7": ["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction", "signal_type", "lgc_lambda", "target", "source"],
        "A_support": ["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction", "target", "source"],
        "parameter_summary": ["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction"],
    }
    for name, keys in detail_keys.items():
        if len(dataframes[name]):
            dataframes[name] = dataframes[name].drop_duplicates(keys, keep="last").reset_index(drop=True)
    completed = set()
    if len(dataframes["parameter_summary"]):
        completed = set(map(tuple, dataframes["parameter_summary"][["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction"]].itertuples(index=False, name=None)))
    for name in ("network", "lgc_eq7", "A_support"):
        table = dataframes[name]
        if len(table):
            keys = table[["outer_run", "n_sources", "T", "C_MODE", "lambda_A_group_fraction"]].apply(tuple, axis=1)
            baseline = table["lambda_A_group_fraction"].isna()
            dataframes[name] = table.loc[baseline | keys.isin(completed)].reset_index(drop=True)
    config = {"SOURCE_COUNT_VALUES": SOURCE_COUNT_VALUES, "T_VALUES": T_VALUES, "C_MODE_LIST": C_MODE_LIST,
              "N_OUTER_RUNS": N_OUTER_RUNS, "LAMBDA_GRIDS": LAMBDA_GRIDS, "LGC_LAMBDA_GRID": LGC_LAMBDA_GRID,
              "workers": N_WORKERS, "numba_available": NUMBA_AVAILABLE}
    with open(os.path.join(RESULTS_DIR, "experiment_config.json"), "w", encoding="utf-8") as handle: json.dump(config, handle, indent=2)
    run_rows = []
    for outer_run in range(N_OUTER_RUNS):
        for n in SOURCE_COUNT_VALUES:
            for T in T_VALUES:
                for c_mode in C_MODE_LIST:
                    seed = BASE_SEED + outer_run
                    data = simulate(n, T, c_mode, seed)
                    base = {"outer_run": outer_run, "random_seed": seed, "n_sources": n, "T": T, "C_MODE": c_mode}
                    network_baseline_done = len(dataframes["network"]) and np.any(
                        (dataframes["network"].outer_run == outer_run) & (dataframes["network"].n_sources == n) &
                        (dataframes["network"]["T"] == T) & (dataframes["network"].C_MODE == c_mode) &
                        (dataframes["network"].signal_type == "oracle_latent"))
                    lgc_baseline_done = len(dataframes["lgc_eq7"]) and np.any(
                        (dataframes["lgc_eq7"].outer_run == outer_run) & (dataframes["lgc_eq7"].n_sources == n) &
                        (dataframes["lgc_eq7"]["T"] == T) & (dataframes["lgc_eq7"].C_MODE == c_mode) &
                        (dataframes["lgc_eq7"].signal_type == "oracle_latent"))
                    baseline_done = network_baseline_done and lgc_baseline_done
                    if not baseline_done:
                        # An interrupted prior save may contain only one side;
                        # remove that incomplete baseline before recomputation.
                        for table_name in ("network", "lgc_eq7"):
                            table = dataframes[table_name]
                            if len(table):
                                incomplete = (
                                    (table.outer_run == outer_run) &
                                    (table.n_sources == n) & (table["T"] == T) &
                                    (table.C_MODE == c_mode) &
                                    table.signal_type.isin(("oracle_latent", "observed_y", "pinv_proxy"))
                                )
                                dataframes[table_name] = table.loc[~incomplete].reset_index(drop=True)
                        proxy = data["y"] @ np.linalg.pinv(data["C"]).T
                        for name, signal in (("oracle_latent", data["x"]), ("observed_y", data["y"]), ("pinv_proxy", proxy)):
                            meta = {**base, "lambda_A_group_fraction": np.nan}
                            network, lgc = readouts(signal, name, data, meta, LGC_LAMBDA_GRID)
                            dataframes["network"] = pd.concat([dataframes["network"], network], ignore_index=True)
                            dataframes["lgc_eq7"] = pd.concat([dataframes["lgc_eq7"], lgc], ignore_index=True)
                    tasks = [(data, base, lam, seed + 500) for lam in LAMBDA_GRIDS[n]
                             if (outer_run, n, T, c_mode, float(lam)) not in completed]
                    if tasks:
                        with ThreadPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as executor:
                            for parameter, support, network, lgc in executor.map(work_unit, tasks):
                                dataframes["parameter_summary"] = pd.concat([dataframes["parameter_summary"], pd.DataFrame([parameter])], ignore_index=True)
                                dataframes["A_support"] = pd.concat([dataframes["A_support"], support], ignore_index=True)
                                dataframes["network"] = pd.concat([dataframes["network"], network], ignore_index=True)
                                dataframes["lgc_eq7"] = pd.concat([dataframes["lgc_eq7"], lgc], ignore_index=True)
                                completed.add((outer_run, n, T, c_mode, float(parameter["lambda_A_group_fraction"])))
                                label = "0p05"
                                print(f"33C N={n} T={T} C={c_mode} lambda={parameter['lambda_A_group_fraction']:g} "
                                      f"LL={parameter['full_log_likelihood']:.2f} iter={parameter['full_em_iterations']} "
                                      f"radius={parameter['full_spectral_radius']:.3f} Q/R={parameter['Q_trace']:.2f}/{parameter['R_trace']:.2f} "
                                      f"A/Aoff/B={parameter['A_relative_frobenius_error']:.3f}/"
                                      f"{parameter['A_offdiag_relative_frobenius_error']:.3f}/{parameter['B_relative_frobenius_error']:.3f} "
                                      f"AUPRC={parameter['A_support_AUPRC']:.3f} "
                                      f"TPR@FPR.05={parameter['A_support_TPR_at_FPR_leq_'+label]:.3f}")
                                for signal_type in ("em_filtered", "em_smoothed"):
                                    selected = network[network.signal_type == signal_type]
                                    lgc_selected = lgc[lgc.signal_type == signal_type]
                                    lgc_curve, _ = curve_metrics(
                                        lgc_selected.true_link, lgc_selected.lgc_clipped
                                    )
                                    print(
                                        f"  {signal_type}: debiased chi/FDR/BIC/D>10="
                                        f"{selected.debiased_chi_detected.mean():.3f}/"
                                        f"{selected.debiased_fdr_detected.mean():.3f}/"
                                        f"{selected.debiased_bic_detected.mean():.3f}/"
                                        f"{selected.debiased_D_gt_10p0_detected.mean():.3f}; "
                                        f"LGC AUPRC={lgc_curve['AUPRC']:.3f}, "
                                        f"TPR@FPR.05={lgc_curve['TPR_at_FPR_leq_0p05']:.3f}, "
                                        f"threshold={lgc_curve['threshold_for_FPR_leq_0p05']:.6g}"
                                    )
        for name, path in partial.items(): atomic_csv(dataframes[name], path)
        if len(dataframes["network"]) and len(dataframes["A_support"]):
            partial_roc, _ = roc_outputs(
                dataframes["network"], dataframes["lgc_eq7"],
                dataframes["A_support"],
            )
            atomic_csv(partial_roc, os.path.join(RESULTS_DIR, "roc_summary_partial.csv"))
        run_rows.append({"outer_run": outer_run, "completed_work_units": len(completed), "timestamp": time.time()})
        atomic_csv(pd.DataFrame(run_rows), os.path.join(RESULTS_DIR, "run_summary.csv"))

    roc, roc_points = roc_outputs(dataframes["network"], dataframes["lgc_eq7"], dataframes["A_support"])
    parameter_metrics = [column for column in dataframes["parameter_summary"] if column.startswith(("A_", "B_", "filtered_", "smoothed_", "full_", "offdiag_")) and pd.api.types.is_numeric_dtype(dataframes["parameter_summary"][column])]
    parameter_aggregate = grouped_stats(dataframes["parameter_summary"], ["n_sources", "T", "C_MODE", "lambda_A_group_fraction"], parameter_metrics)
    link_metrics = ["raw_deviance", "full_bias_term", "reduced_bias_term", "bias_correction", "debiased_deviance", "raw_p_value", "debiased_p_value"]
    link_summary = grouped_stats(dataframes["network"], ["n_sources", "T", "C_MODE", "signal_type", "lambda_A_group_fraction", "true_link"], link_metrics)
    conventional = [column for column in dataframes["network"] if column.endswith("_detected")]
    decision = decisions(dataframes["network"], conventional, ["n_sources", "T", "C_MODE", "signal_type", "lambda_A_group_fraction"])
    lgc_rules = [column for column in dataframes["lgc_eq7"] if column.startswith("lgc_detected_threshold_")]
    lgc_decision = decisions(dataframes["lgc_eq7"], lgc_rules, ["n_sources", "T", "C_MODE", "signal_type", "lambda_A_group_fraction", "lgc_lambda"])
    outputs = {"network_results.csv": dataframes["network"], "lgc_eq7_results.csv": dataframes["lgc_eq7"],
        "parameter_summary.csv": parameter_aggregate, "A_support_results.csv": dataframes["A_support"],
        "link_strength_summary.csv": link_summary, "decision_summary.csv": decision,
        "lgc_decision_summary.csv": lgc_decision, "roc_summary.csv": roc,
        "roc_curve_points.csv": roc_points, "fixed_fpr_operating_points.csv": roc}
    for filename, frame in outputs.items(): atomic_csv(frame, os.path.join(RESULTS_DIR, filename))
    print("\nInterpretation: compare A-offdiag scaling, T compensation, mixing collapse, and each score's TPR at FPR<=0.05; determine where EM ceases to beat observed/proxy and whether A support exceeds posterior readout.")


if __name__ == "__main__": main()

# TODO Experiment 33F: add density/coverage edge-distribution diagnostics later.
