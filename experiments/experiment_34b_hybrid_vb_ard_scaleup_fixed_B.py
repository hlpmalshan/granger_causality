"""Experiment 34B: reduced M=10/20 scale-up of Hybrid Kalman + VB-ARD.

This retains the Level-1 34A approximation: an exact companion Kalman/RTS
smoother at E[A], followed by q(A) and q(alpha) updates.  It is not full
structured variational inference.

Set ``EXPERIMENT_34B_SMOKE=1`` for the requested two-run M=10 smoke test.
Final defaults are 20 outer runs for M=10 and 10 for M=20 at T=2000.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34B_BLAS_THREADS", "1")

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, grouped_stats
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import (
    ALPHA, CUSTOM_THRESHOLDS, RIDGE_LAMBDA_DEBIAS, fit_em, fit_vb,
    materialize_tables, recovery, safe_curve, simulate, uncertainty_summary,
    vb_details, vb_edge_summary,
)
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.stats.lasso_gc_statistic import compute_lasso_gc_eq7_network
from src.stats.scalable_debiased_varx_network import compute_all_pair_debiased_varx_network


na, nb = 2, 3
SOURCE_COUNT_VALUES = [10, 20]
T_VALUES = [2000]
N_OUTER_RUNS_BY_SOURCE_COUNT = {10: 20, 20: 10}
METHODS = ["standard_em_fixed_B", "group_lasso_em_fixed_B", "hybrid_vb_ard_fixed_B"]
LAMBDA_A_GROUP_FRACTION_GRID = {10: [0.003, 0.01, 0.03], 20: [0.003, 0.03]}
VB_HYPERPARAM = {"a0": 1e-3, "b0": 1e-3}
LGC_LAMBDA_GRID = [0.0, 0.003]
FPR_LEVELS = (0.01, 0.03, 0.05, 0.10)
BASE_SEED = 4200000
SMOKE_TEST = os.environ.get("EXPERIMENT_34B_SMOKE", "0") == "1"
if SMOKE_TEST:
    SOURCE_COUNT_VALUES = [10]
    N_OUTER_RUNS_BY_SOURCE_COUNT = {10: 2}
    METHODS = ["group_lasso_em_fixed_B", "hybrid_vb_ard_fixed_B"]
N_OUTER_OVERRIDE = os.environ.get("EXPERIMENT_34B_N_OUTER_RUNS")
if N_OUTER_OVERRIDE is not None:
    N_OUTER_RUNS_BY_SOURCE_COUNT = {n: int(N_OUTER_OVERRIDE) for n in SOURCE_COUNT_VALUES}
RESULTS_DIR = os.environ.get("EXPERIMENT_34B_RESULTS_DIR", "results/experiment_34b")


def add_meta(frame, meta):
    frame = frame.copy()
    for key, value in meta.items():
        frame[key] = value
    return frame


def network_scores(signal, signal_type, data, meta, compute_lgc=False):
    network = compute_all_pair_debiased_varx_network(
        signal, data["u"], data["mask"], na, nb, RIDGE_LAMBDA_DEBIAS,
        ALPHA, CUSTOM_THRESHOLDS,
    )
    network = add_meta(network, {**meta, "signal_type": signal_type})
    score_parts = []
    for score in ("raw_deviance", "debiased_deviance"):
        part = network.copy()
        part["score_type"], part["score"], part["lgc_lambda"] = score, part[score], np.nan
        score_parts.append(part)
    lgc_frames = []
    if compute_lgc:
        for value in LGC_LAMBDA_GRID:
            part = compute_lasso_gc_eq7_network(
                signal, data["u"], data["mask"], na, nb, value
            )
            part = add_meta(part, {**meta, "signal_type": signal_type})
            lgc_frames.append(part)
            for score in ("lgc_raw", "lgc_clipped"):
                scored = part.copy()
                scored["score_type"], scored["score"] = "Eq7_" + score, scored[score]
                score_parts.append(scored)
    return network, lgc_frames, score_parts


def support_score_rows(support, meta):
    part = add_meta(support, meta)
    part["signal_type"] = "model_A"
    part["score_type"] = "estimated_A_group_norm"
    part["score"] = part["estimated_A_group_norm"]
    part["lgc_lambda"] = np.nan
    return part


def vb_score_rows(edges, meta):
    names = {
        "group_snr_score": "vb_group_snr",
        "inverse_alpha_score": "vb_inverse_alpha",
        "posterior_second_moment_group_norm": "vb_second_moment_group_norm",
        "posterior_mean_group_norm": "vb_posterior_mean_group_norm",
    }
    rows = []
    for column, signal_type in names.items():
        part = add_meta(edges, meta)
        part["signal_type"], part["score_type"] = signal_type, column
        part["score"], part["lgc_lambda"] = part[column], np.nan
        rows.append(part)
    return rows


def summarize_scores(scores):
    groups = ["n_sources", "T", "method", "signal_type", "score_type",
              "lambda_A_group_fraction", "a0", "b0", "lgc_lambda"]
    summaries, points, fixed = [], [], []
    for keys, group in scores.groupby(groups, dropna=False):
        meta = dict(zip(groups, keys))
        metrics, curve = safe_curve(group.true_link, group.score)
        summaries.append({**meta, "n_edges": len(group), **metrics})
        for threshold, item in curve:
            points.append({**meta, "threshold": threshold, **item})
        for level in FPR_LEVELS:
            eligible = [(threshold, item) for threshold, item in curve
                        if np.isfinite(item["fpr"]) and item["fpr"] <= level]
            threshold, item = max(
                eligible,
                key=lambda pair: (np.nan_to_num(pair[1]["tpr"], nan=-1), -pair[0]),
            ) if eligible else curve[0]
            fixed.append({**meta, "target_fpr_level": level, "threshold": threshold,
                "actual_fpr": item["fpr"], **{key: item[key] for key in
                ("tpr", "precision", "f1", "tp", "fp", "tn", "fn")}})
    return pd.DataFrame(summaries), pd.DataFrame(points), pd.DataFrame(fixed)


def em_diagnostics(model):
    history = np.asarray(model.log_likelihoods, dtype=float)
    converged = len(history) < model.max_iter
    final_change = np.nan
    if len(history) > 1:
        final_change = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1.0)
    return {"n_iter": len(history), "converged": converged,
            "final_kalman_log_likelihood": float(model.smooth_result["log_likelihood"]),
            "surrogate_objective": np.nan, "final_A_change_norm": final_change,
            "final_alpha_change_norm": np.nan,
            "spectral_radius_A": float(model.spectral_radius()),
            "stability_rescaling_flag": bool(any(np.asarray(
                getattr(model, "A_sparse_scale_history", [1.0])) != 1.0)),
            "numerical_warning_flag": not np.all(np.isfinite(history))}


def vb_diagnostics(model):
    return {"n_iter": model.n_iter_, "converged": model.converged_,
        "final_kalman_log_likelihood": model.objective_history_[-1],
        "surrogate_objective": model.objective_history_[-1],
        "final_A_change_norm": model.A_change_history_[-1],
        "final_alpha_change_norm": model.alpha_change_history_[-1],
        "spectral_radius_A": var_companion_spectral_radius(model.A_mean_matrices_),
        "stability_rescaling_flag": bool(any(model.rescaling_history_)),
        "numerical_warning_flag": not model.diagnostics_["all_finite"]}


def partial_summaries(frames):
    groups = ["n_sources", "T", "method", "lambda_A_group_fraction", "a0", "b0"]
    metrics = ["A_relative_frobenius_error", "A_offdiag_relative_frobenius_error",
        "A_diagonal_relative_frobenius_error", "A_group_norm_pearson_correlation",
        "A_group_norm_spearman_correlation", "A_support_ROC_AUC", "A_support_AUPRC",
        "A_support_TPR_at_FPR_0p01", "A_support_TPR_at_FPR_0p03",
        "A_support_TPR_at_FPR_0p05", "A_support_TPR_at_FPR_0p10",
        "precision_at_FPR_0p05", "F1_at_FPR_0p05", "filtered_signal_mse",
        "smoothed_signal_mse", "runtime_seconds", "n_iter", "converged",
        "spectral_radius_A"]
    parameter = grouped_stats(frames["parameter"], groups, metrics)
    support = grouped_stats(frames["parameter"], groups,
        ["A_support_ROC_AUC", "A_support_AUPRC", "A_support_best_youden_J",
         "A_support_best_F1", "A_support_TPR_at_FPR_0p01",
         "A_support_TPR_at_FPR_0p03", "A_support_TPR_at_FPR_0p05",
         "A_support_TPR_at_FPR_0p10", "precision_at_FPR_0p05",
         "F1_at_FPR_0p05", "mean_estimated_group_norm_true_links",
         "mean_estimated_group_norm_false_links", "group_norm_true_false_ratio"])
    vb_summary = vb_edge_summary(frames["vb_edges"])
    calibration = uncertainty_summary(frames["uncertainty"])
    roc, points, fixed = summarize_scores(frames["scores"])
    return parameter, support, vb_summary, calibration, roc, points, fixed


def save_plots(parameter, vb_summary, edges, calibration, runtime):
    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    def bars(frame, value, filename, ylabel):
        pivot = frame.groupby(["n_sources", "method"])[value].mean().unstack()
        fig, ax = plt.subplots(); pivot.plot.bar(ax=ax)
        ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, filename)); plt.close(fig)
    bars(parameter, "A_support_AUPRC", "A_support_AUPRC_by_method_source_count.png", "A-support AUPRC")
    bars(parameter, "A_support_TPR_at_FPR_0p05", "TPR_FPR_0p05_by_method_source_count.png", "TPR at FPR <= 0.05")
    bars(parameter, "A_offdiag_relative_frobenius_error", "A_offdiag_error_by_method_source_count.png", "off-diagonal relative error")
    bars(runtime, "runtime_seconds", "runtime_by_method_source_count.png", "seconds")
    if len(vb_summary):
        pivot = vb_summary.groupby(["n_sources", "vb_score_type"]).AUPRC.mean().unstack()
        fig, ax = plt.subplots(); pivot.plot.bar(ax=ax); ax.set_ylabel("AUPRC"); ax.tick_params(axis="x",rotation=0)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir,"VB_edge_score_AUPRC.png")); plt.close(fig)
    if len(edges):
        fig, ax = plt.subplots()
        ax.boxplot([edges.loc[edges.true_link,"group_snr_score"], edges.loc[~edges.true_link,"group_snr_score"]], labels=["true","false"])
        ax.set_ylabel("VB group-SNR"); fig.tight_layout(); fig.savefig(os.path.join(plot_dir,"VB_group_SNR_true_false.png")); plt.close(fig)
    if len(calibration):
        selected = calibration.loc[calibration.coefficient_type != "all"]
        pivot = selected.groupby(["n_sources","coefficient_type"]).empirical_coverage_95.mean().unstack()
        fig, ax = plt.subplots(); pivot.plot.bar(ax=ax); ax.set_ylabel("empirical 95% coverage"); ax.tick_params(axis="x",rotation=0)
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir,"calibration_by_type_source_count.png")); plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = {"experiment": "34B", "method_label": "Hybrid Kalman + VB-ARD",
        "is_full_structured_vi": False, "SOURCE_COUNT_VALUES": SOURCE_COUNT_VALUES,
        "T_VALUES": T_VALUES, "N_OUTER_RUNS_BY_SOURCE_COUNT": N_OUTER_RUNS_BY_SOURCE_COUNT,
        "METHODS": METHODS, "na": na, "nb": nb, "Q_true": "0.50 I",
        "R_true": "0.60 I", "C_MODE": "identity", "B": "fixed true",
        "LINK_DENSITY": 0.05, "lambda_grids": LAMBDA_A_GROUP_FRACTION_GRID,
        "VB_HYPERPARAM": VB_HYPERPARAM, "LGC_LAMBDA_GRID": LGC_LAMBDA_GRID,
        "smoke_test": SMOKE_TEST,
        "TODO_34C": "full structured q_x with block-banded precision and E[A^T Q^-1 A]",
        "TODO_34D": "sample A from q(A) and propagate uncertainty to spectral GC"}
    config = json.loads(json.dumps(config))
    config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    ledger_path = os.path.join(RESULTS_DIR, "run_summary_partial.csv")
    if os.path.exists(config_path) and os.path.exists(ledger_path):
        with open(config_path, encoding="utf-8") as handle: previous = json.load(handle)
        keys = ("SOURCE_COUNT_VALUES", "T_VALUES", "N_OUTER_RUNS_BY_SOURCE_COUNT",
                "METHODS", "na", "nb", "lambda_grids", "VB_HYPERPARAM", "LGC_LAMBDA_GRID")
        if any(previous.get(key) != config.get(key) for key in keys):
            raise ValueError("Existing 34B checkpoint has a different numerical configuration.")
    with open(config_path, "w", encoding="utf-8") as handle: json.dump(config, handle, indent=2)

    checkpoint = {"parameter":"parameter_results_partial.csv", "support":"_A_support_checkpoint.csv",
        "vb_edges":"vb_edge_scores_partial.csv", "uncertainty":"_uncertainty_checkpoint.csv",
        "objective":"_objective_checkpoint.csv", "network":"_network_checkpoint.csv",
        "lgc":"_lgc_checkpoint.csv", "scores":"_scores_checkpoint.csv",
        "runtime":"_runtime_checkpoint.csv", "run":"run_summary_partial.csv"}
    tables = {key: [] for key in checkpoint if key != "run"}; run_rows = []
    for key, filename in checkpoint.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            loaded = pd.read_csv(path)
            if key == "run": run_rows = loaded.to_dict("records")
            elif len(loaded): tables[key] = [loaded]
    completed = {(int(row["n_sources"]), int(row["outer_run"])) for row in run_rows}

    for n in SOURCE_COUNT_VALUES:
        for outer in range(N_OUTER_RUNS_BY_SOURCE_COUNT[n]):
            if (n, outer) in completed: continue
            for T in T_VALUES:
                seed = BASE_SEED + n * 100000 + outer * 1000 + T
                data = simulate(n, T, "sparse_random", seed)
                base = {"n_sources":n, "T":T, "outer_run":outer, "case_name":"sparse_random", "C_MODE":"identity"}
                baseline_meta = {**base, "method":"dataset_baseline", "lambda_A_group_fraction":np.nan, "a0":np.nan, "b0":np.nan}
                proxy = data["y"] @ np.linalg.pinv(data["C"]).T
                # Dataset-invariant readouts are deliberately computed once.
                for signal_type, signal in (("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy)):
                    network, lgcs, scores = network_scores(signal, signal_type, data, baseline_meta, compute_lgc=signal_type=="observed_y")
                    tables["network"].append(network); tables["lgc"].extend(lgcs); tables["scores"].extend(scores)

                specs = []
                if "standard_em_fixed_B" in METHODS: specs.append(("standard_em_fixed_B", {"lambda_A_group_fraction":0.0}))
                if "group_lasso_em_fixed_B" in METHODS:
                    specs.extend(("group_lasso_em_fixed_B", {"lambda_A_group_fraction":value}) for value in LAMBDA_A_GROUP_FRACTION_GRID[n])
                if "hybrid_vb_ard_fixed_B" in METHODS: specs.append(("hybrid_vb_ard_fixed_B", VB_HYPERPARAM))
                for method, hyper in specs:
                    meta = {**base, "method":method,
                        "lambda_A_group_fraction":hyper.get("lambda_A_group_fraction",np.nan),
                        "a0":hyper.get("a0",np.nan), "b0":hyper.get("b0",np.nan)}
                    started = time.perf_counter()
                    if method == "hybrid_vb_ard_fixed_B":
                        model, Ah, filtered, smoothed = fit_vb(data, hyper, seed+500)
                        diagnostics = vb_diagnostics(model)
                    else:
                        model, Ah, filtered, smoothed = fit_em(data, hyper["lambda_A_group_fraction"], seed+500)
                        diagnostics = em_diagnostics(model)
                    runtime = time.perf_counter() - started
                    row, support = recovery(Ah, data, filtered, smoothed, meta, runtime,
                                             diagnostics["n_iter"], diagnostics["converged"])
                    row.update(diagnostics)
                    row["precision_at_FPR_0p05"] = row.get("A_support_precision_at_FPR_0p05", np.nan)
                    row["F1_at_FPR_0p05"] = row.get("A_support_F1_at_FPR_0p05", np.nan)
                    tables["parameter"].append(row); tables["support"].append(support)
                    tables["runtime"].append({**meta, "runtime_seconds":runtime, **diagnostics})
                    for signal_type, signal in (("method_filtered",filtered),("method_smoothed",smoothed)):
                        network, lgcs, scores = network_scores(signal,signal_type,data,meta,compute_lgc=True)
                        tables["network"].append(network);tables["lgc"].extend(lgcs);tables["scores"].extend(scores)
                    tables["scores"].append(support_score_rows(support,meta))
                    if method == "hybrid_vb_ard_fixed_B":
                        coeff, edges, history = vb_details(model,data,meta)
                        edges["true_A_group_norm"] = edges["true_group_norm"]
                        tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["objective"].append(history)
                        tables["scores"].extend(vb_score_rows(edges,meta))
                        snr_metrics,_ = safe_curve(edges.true_link,edges.group_snr_score)
                        print({"mean_alpha_true":edges.loc[edges.true_link,"alpha_mean"].mean(),
                            "mean_alpha_false":edges.loc[~edges.true_link,"alpha_mean"].mean(),
                            "mean_group_norm_true":edges.loc[edges.true_link,"posterior_mean_group_norm"].mean(),
                            "mean_group_norm_false":edges.loc[~edges.true_link,"posterior_mean_group_norm"].mean(),
                            "VB_group_SNR_AUPRC":snr_metrics["AUPRC"],
                            "VB_group_SNR_TPR_at_FPR_0p05":snr_metrics["TPR_at_FPR_0p05"],
                            "coverage_offdiag_nonzero":coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean(),
                            "coverage_offdiag_zero":coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean(),
                            "final_surrogate_objective":diagnostics["surrogate_objective"]})
                    print({key: row.get(key) for key in ("n_sources","outer_run","method",
                        "lambda_A_group_fraction","a0","b0","runtime_seconds","converged","n_iter",
                        "spectral_radius_A","A_offdiag_relative_frobenius_error","A_support_AUPRC",
                        "A_support_TPR_at_FPR_0p05","precision_at_FPR_0p05","F1_at_FPR_0p05",
                        "filtered_signal_mse","smoothed_signal_mse")})

            run_rows.append({"n_sources":n,"outer_run":outer,"completed":True,"timestamp":time.time()})
            frames = materialize_tables(tables)
            parameter_summary,support_summary,vb_summary,calibration,roc,_,fixed = partial_summaries(frames)
            direct = {"parameter_results_partial.csv":frames["parameter"],
                "parameter_summary_partial.csv":parameter_summary,"A_support_summary_partial.csv":support_summary,
                "vb_edge_scores_partial.csv":frames["vb_edges"],"vb_edge_score_summary_partial.csv":vb_summary,
                "vb_uncertainty_calibration_summary_partial.csv":calibration,
                "fixed_fpr_operating_points_partial.csv":fixed,"roc_summary_partial.csv":roc}
            for filename, frame in direct.items(): atomic_csv(frame,os.path.join(RESULTS_DIR,filename))
            for key, filename in checkpoint.items():
                if key != "run": atomic_csv(frames[key],os.path.join(RESULTS_DIR,filename))
            # Ledger promotion last certifies all detailed and summary checkpoints.
            atomic_csv(pd.DataFrame(run_rows),ledger_path)

    frames = materialize_tables(tables)
    parameter_summary,support_summary,vb_summary,calibration,roc,points,fixed = partial_summaries(frames)
    decision = fixed.copy(); decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
    groups=["n_sources","T","method","lambda_A_group_fraction","a0","b0"]
    outputs={"run_summary.csv":pd.DataFrame(run_rows),
        "runtime_summary.csv":grouped_stats(frames["runtime"],groups,["runtime_seconds","n_iter","converged","spectral_radius_A"]),
        "parameter_results.csv":frames["parameter"],"parameter_summary.csv":parameter_summary,
        "A_support_summary.csv":support_summary,"network_results.csv":frames["network"],
        "link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","method","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
        "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,
        "decision_summary.csv":decision,"lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],
        "vb_edge_scores.csv":frames["vb_edges"],"vb_edge_score_summary.csv":vb_summary,
        "vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":calibration,
        "objective_history.csv":frames["objective"]}
    for filename,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,filename))
    save_plots(frames["parameter"],vb_summary,frames["vb_edges"],calibration,frames["runtime"])
    print("\nInterpretation guide:\n1. Does VB-ARD improve A-support AUPRC at M=10?\n2. Does it improve at M=20?\n3. Does group-SNR outperform posterior mean group norm?\n4. Does VB improve TPR at FPR <= 0.05 versus group-lasso EM?\n5. Does VB reduce false A-edge scores?\n6. Does model_A close the gap with score-based GC?\n7. Is M=20 runtime acceptable?\n8. Are intervals still under-covering?\n9. Should VB remain an estimator/ranker only?\n\nOutcomes: improvement at both scales supports practical B work next; M=10-only improvement indicates an M=20 sample/runtime barrier; no gain indicates the Level-1 approximation does not scale; support gains with under-coverage mean use VB as an estimator/ranker, not a significance test.")


if __name__ == "__main__":
    main()
