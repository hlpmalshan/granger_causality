"""Experiment 34A: Hybrid Kalman + VB-ARD for A with fixed B/C/Q/R.

Final defaults are the requested 50 runs at T=2000.  Set
``EXPERIMENT_34A_SMOKE=1`` for the documented 3-run, M=2, T=(100, 250)
smoke test.  This is a Level-1 hybrid method, not full structured VI.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34A_BLAS_THREADS", "1")

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import (
    atomic_csv, correlations, grouped_stats, make_network, relative, spearman,
)
from src.ssm.em_varx_p_known_c_group_lasso_controlled import (
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB,
)
from src.ssm.kalman_varx_p import extract_current_latent_state, kalman_smooth_varx_p_companion
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.ssm.vb_ard_varx_ssm import HybridVBARDVARXSSMFixedBKnownC
from src.stats.lasso_gc_statistic import compute_lasso_gc_eq7_network
from src.stats.scalable_debiased_varx_network import (
    compute_all_pair_debiased_varx_network, compute_confusion_metrics,
)
from src.varx.varx_generator import generate_colored_input


na, nb, BURN_IN = 2, 3, 300
SOURCE_COUNT_VALUES, T_VALUES, C_MODE_LIST, N_OUTER_RUNS = [2, 5], [2000], ["identity"], 50
SMOKE = os.environ.get("EXPERIMENT_34A_SMOKE", "0") == "1"
if SMOKE:
    SOURCE_COUNT_VALUES, T_VALUES, N_OUTER_RUNS = [2], [100, 250], 3
# Developer/CI override; absent in scientific runs and documented smoke runs.
N_OUTER_RUNS = int(os.environ.get("EXPERIMENT_34A_N_OUTER_RUNS", N_OUTER_RUNS))
LAMBDA_GRIDS = {2: [0.0, 0.001, 0.003], 5: [0.0, 0.001, 0.003, 0.01]}
VB_HYPERPARAM_GRID = [{"a0": 1e-3, "b0": 1e-3}, {"a0": 1e-2, "b0": 1e-2}]
LGC_LAMBDA_GRID = [0.0, 0.003]
FPR_LEVELS = (0.01, 0.03, 0.05, 0.10)
RESULTS_DIR = os.environ.get("EXPERIMENT_34A_RESULTS_DIR", "results/experiment_34a")
BASE_SEED, MAX_ITER, TOL = 4100000, 100, 1e-6
ALPHA, RIDGE_LAMBDA_DEBIAS = 0.05, 1.0
CUSTOM_THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)


def simulate(n, T, case_name, seed):
    if n == 2:
        linked = case_name == "true_Y_to_X"
        A = [np.array([[.60, .25 if linked else 0.], [0., .55]]),
             np.array([[-.10, .10 if linked else 0.], [0., -.08]])]
        mask = np.zeros((2, 2), bool); mask[0, 1] = linked
    else:
        # Reuse the exact Experiment 33C/33D sparse generator.
        A, mask = make_network(n, seed)
    rng = np.random.default_rng(seed + 100)
    B = [rng.normal(0, scale, (n, 1)) for scale in (.45, .25, .15)]
    C, Q, R = np.eye(n), .50 * np.eye(n), .60 * np.eye(n)
    u = generate_colored_input(T + BURN_IN, .95, 1., seed + 200)
    result = generate_ssm_varx_p_data(A, B, u, Q, R, C=C, D=None,
                                      burn_in=BURN_IN, random_seed=seed + 300,
                                      return_augmented=True)
    return {"A": A, "B": B, "C": C, "Q": Q, "R": R, "mask": mask,
            "x": result["x"], "y": result["y"], "u": result["u"]}


def cases_for(n):
    return ["null_Y_to_X", "true_Y_to_X"] if n == 2 else ["sparse_random"]


def em_kwargs(data, fraction, seed):
    return dict(na=na, nb=nb, C=data["C"], D=None, max_iter=MAX_ITER, tol=TOL,
        ridge_m_step=1e-4, covariance_floor=1e-6, R_init=data["R"], Q_init=data["Q"],
        estimate_Q=False, estimate_R=False, R_floor=.30, zero_constraints=[],
        initial_parameters=None, jitter_scale=0., random_seed=seed, verbose=False,
        alpha_Q=0., alpha_R=0., shrinkage_target_Q="spherical", shrinkage_target_R="spherical",
        lambda_A_group_fraction=fraction, ridge_A_offdiag=1e-4, ridge_A_diag=1e-4,
        ridge_B=1e-4, group_solver_max_iter=5000, group_solver_tol=1e-7,
        stabilize_A=True, target_radius=.98, fixed_B_matrices=data["B"])


def fit_em(data, fraction, seed):
    model = EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB(**em_kwargs(data, fraction, seed))
    model.fit(data["y"], data["u"])
    posterior = kalman_smooth_varx_p_companion(data["y"], data["u"], model.F, model.G,
        model.Q_aug, model.R, model.C_aug, nb, model.D)
    n = data["C"].shape[1]
    return model, np.asarray(model.A_matrices), extract_current_latent_state(
        posterior["filter"]["x_filt"], n), extract_current_latent_state(
        posterior["smoother"]["x_smooth"], n)


def fit_vb(data, hyper, seed):
    model = HybridVBARDVARXSSMFixedBKnownC(na, nb, data["C"], data["B"], data["Q"],
        data["R"], max_iter=MAX_ITER, tol_objective=TOL, tol_A_change=TOL,
        a0=hyper["a0"], b0=hyper["b0"], diagonal_prior_precision=1e-4,
        posterior_jitter=1e-8, random_state=seed).fit(data["y"], data["u"])
    return model, model.get_A_posterior_mean(), model.filtered_state_mean_, model.smoothed_state_mean_


def safe_curve(y, score):
    y, score = np.asarray(y, bool), np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1]]
    points = [(float(th), compute_confusion_metrics(y, score >= th)) for th in thresholds]
    fpr = np.asarray([p[1]["fpr"] for p in points]); tpr = np.asarray([p[1]["tpr"] for p in points])
    valid = np.isfinite(fpr) & np.isfinite(tpr)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    roc = float(trapezoid(tpr[valid], fpr[valid])) if valid.sum() > 1 else np.nan
    order = np.argsort(-score, kind="mergesort"); labels = y[order].astype(int)
    auprc = float(np.sum(np.cumsum(labels) / np.arange(1, len(labels)+1) * labels) / labels.sum()) if labels.sum() else np.nan
    youden = tpr - fpr; f1 = np.asarray([p[1]["f1"] for p in points])
    ij = int(np.nanargmax(youden)) if np.any(np.isfinite(youden)) else 0
    iff = int(np.nanargmax(f1)) if np.any(np.isfinite(f1)) else 0
    out = {"ROC_AUC": roc, "AUPRC": auprc, "best_youden_J": youden[ij],
           "best_youden_threshold": points[ij][0], "best_F1": f1[iff],
           "FPR_at_best_youden": points[ij][1]["fpr"],
           "TPR_at_best_youden": points[ij][1]["tpr"],
           "precision_at_best_youden": points[ij][1]["precision"],
           "F1_at_best_youden": points[ij][1]["f1"],
           "threshold_at_best_F1": points[iff][0]}
    for level in FPR_LEVELS:
        eligible = [i for i,p in enumerate(points) if np.isfinite(p[1]["fpr"]) and p[1]["fpr"] <= level]
        idx = max(eligible, key=lambda i: (np.nan_to_num(points[i][1]["tpr"], nan=-1), -points[i][0])) if eligible else 0
        label = f"{level:.2f}".replace(".", "p")
        for key, name in (("tpr", "TPR"), ("precision", "precision"), ("f1", "F1")):
            out[f"{name}_at_FPR_{label}"] = points[idx][1][key]
        out[f"threshold_at_FPR_{label}"] = points[idx][0]
    return out, points


def recovery(Ah, data, filtered, smoothed, meta, runtime, n_iter, converged):
    A = np.asarray(data["A"]); off = np.broadcast_to(~np.eye(A.shape[1], dtype=bool), A.shape); diag = ~off
    support = []
    for target in range(A.shape[1]):
        for source in range(A.shape[1]):
            if target != source:
                support.append({**meta, "target": target, "source": source,
                    "true_link": bool(data["mask"][target, source]),
                    "true_A_group_norm": np.linalg.norm(A[:, target, source]),
                    "estimated_A_group_norm": np.linalg.norm(Ah[:, target, source])})
    support = pd.DataFrame(support); metrics, _ = safe_curve(support.true_link, support.estimated_A_group_norm)
    def latent(value):
        return float(np.mean((value-data["x"])**2)), float(np.nanmean([
            correlations(value[:, j], data["x"][:, j]) for j in range(A.shape[1])]))
    fmse, fcorr = latent(filtered); smse, scorr = latent(smoothed)
    true_mean = support.loc[support.true_link, "estimated_A_group_norm"].mean()
    false_mean = support.loc[~support.true_link, "estimated_A_group_norm"].mean()
    row = {**meta, "A_relative_frobenius_error": relative(Ah,A),
        "A_offdiag_relative_frobenius_error": relative(Ah[off],A[off]),
        "A_diagonal_relative_frobenius_error": relative(Ah[diag],A[diag]),
        "A_mean_absolute_error": np.mean(abs(Ah-A)), "A_offdiag_mean_absolute_error": np.mean(abs(Ah[off]-A[off])),
        "A_diagonal_mean_absolute_error": np.mean(abs(Ah[diag]-A[diag])),
        "A_pearson_correlation": correlations(A.ravel(),Ah.ravel()),
        "A_offdiag_pearson_correlation": correlations(A[off],Ah[off]), "A_offdiag_spearman_correlation": spearman(A[off],Ah[off]),
        "A_group_norm_pearson_correlation": correlations(support.true_A_group_norm,support.estimated_A_group_norm),
        "A_group_norm_spearman_correlation": spearman(support.true_A_group_norm,support.estimated_A_group_norm),
        "mean_estimated_group_norm_true_links": true_mean, "mean_estimated_group_norm_false_links": false_mean,
        "group_norm_true_false_ratio": true_mean/false_mean if false_mean > 0 else np.nan,
        "filtered_signal_mse": fmse, "smoothed_signal_mse": smse,
        "filtered_signal_correlation": fcorr, "smoothed_signal_correlation": scorr,
        "runtime_seconds": runtime, "n_iter": n_iter, "converged": converged}
    row.update({"A_support_"+key: value for key,value in metrics.items()})
    return row, support


def vb_details(model, data, meta):
    A, mean, std = np.asarray(data["A"]), model.get_A_posterior_mean(), model.get_A_posterior_std()
    coeff = []
    for lag in range(na):
        for target in range(A.shape[1]):
            for source in range(A.shape[1]):
                truth, mu, sigma = A[lag,target,source], mean[lag,target,source], std[lag,target,source]
                ctype = "diagonal" if target == source else ("offdiag_nonzero" if data["mask"][target,source] else "offdiag_zero")
                coeff.append({**meta,"lag":lag+1,"target":target,"source":source,"posterior_mean":mu,
                    "posterior_std":sigma,"true_value":truth,"error":mu-truth,"abs_error":abs(mu-truth),
                    "standardized_abs_error":abs(mu-truth)/sigma,"ci95_lower":mu-1.96*sigma,
                    "ci95_upper":mu+1.96*sigma,"ci95_contains_true":mu-1.96*sigma <= truth <= mu+1.96*sigma,
                    "coefficient_type":ctype})
    edges = []
    for row in model.get_edge_scores():
        target,source=row["target"],row["source"]
        edges.append({**meta,**row,"true_link":bool(data["mask"][target,source]),
                      "true_group_norm":np.linalg.norm(A[:,target,source])})
    history = [{**meta,"iteration":i+1,"surrogate_objective":value,
        "kalman_log_likelihood":value,"expected_transition_sse":model.expected_transition_sse_history_[i],
        "A_change_norm":model.A_change_history_[i],"alpha_change_norm":model.alpha_change_history_[i],
        "rescaled_A":model.rescaling_history_[i]} for i,value in enumerate(model.objective_history_)]
    return pd.DataFrame(coeff), pd.DataFrame(edges), pd.DataFrame(history)


def network_readout(signal, signal_type, data, meta):
    frame = compute_all_pair_debiased_varx_network(signal,data["u"],data["mask"],na,nb,
        RIDGE_LAMBDA_DEBIAS,ALPHA,CUSTOM_THRESHOLDS)
    for key,value in {**meta,"signal_type":signal_type}.items(): frame[key]=value
    lgcs=[]
    for value in LGC_LAMBDA_GRID:
        part=compute_lasso_gc_eq7_network(signal,data["u"],data["mask"],na,nb,value)
        for key,item in {**meta,"signal_type":signal_type}.items():part[key]=item
        lgcs.append(part)
    return frame,pd.concat(lgcs,ignore_index=True)


def score_tables(network, lgc, support):
    parts=[]
    for score in ("raw_deviance","debiased_deviance"):
        x=network.copy();x["score_type"]=score;x["score"]=x[score];x["lgc_lambda"]=np.nan;parts.append(x)
    for score in ("lgc_raw","lgc_clipped"):
        x=lgc.copy();x["score_type"]="Eq7_"+score;x["score"]=x[score];parts.append(x)
    x=support.copy();x["signal_type"]="model_A";x["score_type"]="estimated_A_group_norm";x["score"]=x.estimated_A_group_norm;x["lgc_lambda"]=np.nan;parts.append(x)
    return pd.concat(parts,ignore_index=True,sort=False)


def summarize_scores(scores):
    groups=["n_sources","T","case_name","method","signal_type","score_type","lambda_A_group_fraction","a0","b0","lgc_lambda"]
    summaries=[];points=[];fixed=[]
    for keys,g in scores.groupby(groups,dropna=False):
        meta=dict(zip(groups,keys)); metrics,curve=safe_curve(g.true_link,g.score); summaries.append({**meta,**metrics})
        for threshold,item in curve:points.append({**meta,"threshold":threshold,**item})
        for level in FPR_LEVELS:
            eligible=[(th,item) for th,item in curve if np.isfinite(item["fpr"]) and item["fpr"]<=level]
            th,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
            fixed.append({**meta,"target_fpr_level":level,"threshold":th,"actual_fpr":item["fpr"],
                          **{k:item[k] for k in ("tpr","precision","f1","tp","fp","tn","fn")}})
    return pd.DataFrame(summaries),pd.DataFrame(points),pd.DataFrame(fixed)


def uncertainty_summary(frame):
    rows=[]; groups=["n_sources","T"]
    if "case_name" in frame.columns: groups.append("case_name")
    if "C_MODE" in frame.columns: groups.append("C_MODE")
    if "B_MODEL_VARIANT" in frame.columns: groups.append("B_MODEL_VARIANT")
    if "Q_VARIANT" in frame.columns: groups.append("Q_VARIANT")
    if "B_ridge_lambda" in frame.columns: groups.append("B_ridge_lambda")
    if "convergence_config_name" in frame.columns: groups.append("convergence_config_name")
    groups += ["a0","b0","coefficient_type"]
    expanded=pd.concat([frame.assign(coefficient_type="all"),frame],ignore_index=True)
    for keys,g in expanded.groupby(groups,dropna=False):
        rows.append({**dict(zip(groups,keys)),"n_coefficients":len(g),"empirical_coverage_95":g.ci95_contains_true.mean(),
            "mean_posterior_std":g.posterior_std.mean(),"median_posterior_std":g.posterior_std.median(),
            "mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),
            "mean_standardized_abs_error":g.standardized_abs_error.mean(),"median_standardized_abs_error":g.standardized_abs_error.median(),
            "posterior_std_abs_error_correlation":correlations(g.posterior_std,g.abs_error)})
    return pd.DataFrame(rows)


def vb_edge_summary(frame):
    rows=[];groups=["n_sources","T","case_name","a0","b0"]
    for keys,g in frame.groupby(groups,dropna=False):
        for score in ("posterior_mean_group_norm","posterior_second_moment_group_norm","inverse_alpha_score","group_snr_score"):
            metrics,_=safe_curve(g.true_link,g[score]);rows.append({**dict(zip(groups,keys)),"vb_score_type":score,**metrics})
    return pd.DataFrame(rows)


def save_plots(parameter, uncertainty, edges, objective, runtime):
    plot_dir=os.path.join(RESULTS_DIR,"plots");os.makedirs(plot_dir,exist_ok=True)
    def line(metric,name,ylabel):
        fig,ax=plt.subplots()
        for method,g in parameter.groupby("method"):
            s=g.groupby("T")[metric].mean();ax.plot(s.index,s.values,marker="o",label=method)
        ax.set(xlabel="T",ylabel=ylabel);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(os.path.join(plot_dir,name));plt.close(fig)
    line("A_offdiag_relative_frobenius_error","A_error_vs_T.png","off-diagonal relative error")
    line("A_support_AUPRC","A_support_AUPRC_vs_T.png","A-support AUPRC")
    line("A_support_TPR_at_FPR_0p05","TPR_at_FPR_0p05_vs_T.png","TPR at FPR <= 0.05")
    if len(objective):
        fig,ax=plt.subplots();
        for _,g in objective.groupby(["outer_run","n_sources","T","case_name","a0","b0"]):ax.plot(g.iteration,g.surrogate_objective,alpha=.25)
        ax.set(xlabel="VB iteration",ylabel="surrogate objective (Kalman log likelihood)");fig.tight_layout();fig.savefig(os.path.join(plot_dir,"surrogate_objective_vs_iteration.png"));plt.close(fig)
    if len(uncertainty):
        fig,ax=plt.subplots();ax.scatter(uncertainty.true_value,uncertainty.posterior_mean,s=6,alpha=.25);ax.set(xlabel="true A",ylabel="posterior mean A");fig.tight_layout();fig.savefig(os.path.join(plot_dir,"VB_A_mean_vs_true.png"));plt.close(fig)
        fig,ax=plt.subplots();ax.scatter(uncertainty.posterior_std,uncertainty.abs_error,s=6,alpha=.25);ax.set(xlabel="posterior std",ylabel="absolute error");fig.tight_layout();fig.savefig(os.path.join(plot_dir,"posterior_std_vs_abs_error.png"));plt.close(fig)
        cov=uncertainty.groupby("T").ci95_contains_true.mean();fig,ax=plt.subplots();ax.plot(cov.index,cov.values,marker="o");ax.set(xlabel="T",ylabel="empirical 95% coverage");fig.tight_layout();fig.savefig(os.path.join(plot_dir,"coverage_vs_T.png"));plt.close(fig)
    for column,name,label in (("alpha_mean","alpha_true_false.png","alpha mean"),("posterior_mean_group_norm","group_norm_true_false.png","posterior mean group norm")):
        if len(edges):
            fig,ax=plt.subplots();values=[edges.loc[edges.true_link,column],edges.loc[~edges.true_link,column]];ax.boxplot(values,labels=["true","false"]);ax.set_ylabel(label);fig.tight_layout();fig.savefig(os.path.join(plot_dir,name));plt.close(fig)
    if len(runtime):
        means=runtime.groupby("method").runtime_seconds.mean();fig,ax=plt.subplots();ax.bar(means.index,means.values);ax.tick_params(axis="x",rotation=25);ax.set_ylabel("seconds");fig.tight_layout();fig.savefig(os.path.join(plot_dir,"runtime_by_method.png"));plt.close(fig)


def materialize_tables(tables):
    """Convert mixed resumed DataFrames and newly appended row dicts to frames."""
    frames = {}
    for key, items in tables.items():
        frame_parts = []
        pending_rows = []
        for item in items:
            if isinstance(item, pd.DataFrame):
                if pending_rows:
                    frame_parts.append(pd.DataFrame(pending_rows))
                    pending_rows = []
                frame_parts.append(item)
            elif isinstance(item, pd.Series):
                pending_rows.append(item.to_dict())
            elif isinstance(item, dict):
                pending_rows.append(item)
            else:
                raise TypeError(
                    f"Unsupported checkpoint item for table {key!r}: "
                    f"{type(item).__name__}"
                )
        if pending_rows:
            frame_parts.append(pd.DataFrame(pending_rows))
        frames[key] = (
            pd.concat(frame_parts, ignore_index=True, sort=False)
            if frame_parts else pd.DataFrame()
        )
    return frames


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    config={"experiment":"34A","method_label":"Hybrid Kalman + VB-ARD","is_full_structured_vi":False,
        "SOURCE_COUNT_VALUES":SOURCE_COUNT_VALUES,"T_VALUES":T_VALUES,"C_MODE_LIST":C_MODE_LIST,
        "N_OUTER_RUNS":N_OUTER_RUNS,"na":na,"nb":nb,"Q_true":"0.50 I","R_true":"0.60 I",
        "B":"fixed true","lambda_grids":LAMBDA_GRIDS,"VB_HYPERPARAM_GRID":VB_HYPERPARAM_GRID,
        "LGC_LAMBDA_GRID":LGC_LAMBDA_GRID,"smoke":SMOKE,
        "TODO_34C":"full structured mean-field q_x using block-banded precision and E[A^T Q^-1 A]",
        "TODO_34D":"sample A from q(A) and propagate uncertainty to spectral GC"}
    config=json.loads(json.dumps(config))
    config_path=os.path.join(RESULTS_DIR,"experiment_config.json")
    if os.path.exists(config_path) and os.path.exists(os.path.join(RESULTS_DIR,"run_summary_partial.csv")):
        with open(config_path,encoding="utf-8") as handle: previous=json.load(handle)
        checked=("SOURCE_COUNT_VALUES","T_VALUES","N_OUTER_RUNS","na","nb","Q_true","R_true","B","lambda_grids","VB_HYPERPARAM_GRID","LGC_LAMBDA_GRID")
        if any(previous.get(key)!=config.get(key) for key in checked):
            raise ValueError("Existing Experiment 34A checkpoint has a different numerical configuration.")
    with open(config_path,"w",encoding="utf-8") as handle:json.dump(config,handle,indent=2)
    partial={"parameter":"parameter_results_partial.csv","support":"_A_support_checkpoint.csv",
      "vb_edges":"vb_edge_scores_partial.csv","uncertainty":"vb_coefficient_uncertainty_partial.csv",
      "objective":"_objective_checkpoint.csv","network":"_network_checkpoint.csv","lgc":"_lgc_checkpoint.csv",
      "scores":"_scores_checkpoint.csv","runtime":"_runtime_checkpoint.csv","run":"run_summary_partial.csv"}
    tables={key:[] for key in partial if key!="run"};run_rows=[]
    for key,name in partial.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            loaded=pd.read_csv(path)
            if key=="run":run_rows=loaded.to_dict("records")
            elif len(loaded):tables[key]=[loaded]
    completed={(int(row["outer_run"]),int(row["n_sources"]),int(row["T"]),str(row["case_name"])) for row in run_rows}
    for outer in range(N_OUTER_RUNS):
      for n in SOURCE_COUNT_VALUES:
       for T in T_VALUES:
        for case_name in cases_for(n):
         if (outer,n,T,case_name) in completed:
          continue
         seed=BASE_SEED+outer*10000+n*100+T+(0 if case_name.startswith("null") else 1);data=simulate(n,T,case_name,seed)
         base={"outer_run":outer,"n_sources":n,"T":T,"case_name":case_name,"C_MODE":"identity"}
         specs=[("standard_em_fixed_B",{"lambda_A_group_fraction":0.0})]
         specs += [("group_lasso_em_fixed_B",{"lambda_A_group_fraction":v}) for v in LAMBDA_GRIDS[n]]
         specs += [("hybrid_vb_ard_fixed_B",h) for h in VB_HYPERPARAM_GRID]
         for method,hyper in specs:
          meta={**base,"method":method,"lambda_A_group_fraction":hyper.get("lambda_A_group_fraction",np.nan),"a0":hyper.get("a0",np.nan),"b0":hyper.get("b0",np.nan)}
          started=time.perf_counter()
          if method=="hybrid_vb_ard_fixed_B":model,Ah,filtered,smoothed=fit_vb(data,hyper,seed+500)
          else:model,Ah,filtered,smoothed=fit_em(data,hyper["lambda_A_group_fraction"],seed+500)
          runtime=time.perf_counter()-started;n_iter=model.n_iter_ if hasattr(model,"n_iter_") else len(model.log_likelihoods);converged=model.converged_ if hasattr(model,"converged_") else bool(getattr(model,"converged",False))
          row,support=recovery(Ah,data,filtered,smoothed,meta,runtime,n_iter,converged);tables["parameter"].append(row);tables["support"].append(support);tables["runtime"].append({**meta,"runtime_seconds":runtime,"n_iter":n_iter,"converged":converged})
          if method=="hybrid_vb_ard_fixed_B":
            coeff,edges,history=vb_details(model,data,meta);tables["uncertainty"].append(coeff);tables["vb_edges"].append(edges);tables["objective"].append(history)
          proxy=data["y"]@np.linalg.pinv(data["C"]).T
          signals=[("oracle_latent",data["x"]),("observed_y",data["y"]),("pinv_proxy",proxy),("method_filtered",filtered),("method_smoothed",smoothed)]
          networks=[];lgcs=[]
          for signal_name,signal in signals:
            net,lgc=network_readout(signal,signal_name,data,meta);networks.append(net);lgcs.append(lgc)
          network=pd.concat(networks,ignore_index=True);lgc=pd.concat(lgcs,ignore_index=True);tables["network"].append(network);tables["lgc"].append(lgc);tables["scores"].append(score_tables(network,lgc,support))
          print({k:row.get(k) for k in ("n_sources","T","case_name","method","lambda_A_group_fraction","a0","b0","runtime_seconds","converged","n_iter","A_offdiag_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p05","filtered_signal_mse","smoothed_signal_mse")})
          if method=="hybrid_vb_ard_fixed_B":
            print({"mean_alpha_true":edges.loc[edges.true_link,"alpha_mean"].mean(),"mean_alpha_false":edges.loc[~edges.true_link,"alpha_mean"].mean(),"mean_group_norm_true":edges.loc[edges.true_link,"posterior_mean_group_norm"].mean(),"mean_group_norm_false":edges.loc[~edges.true_link,"posterior_mean_group_norm"].mean(),"coverage_offdiag_nonzero":coeff.loc[coeff.coefficient_type=="offdiag_nonzero","ci95_contains_true"].mean(),"coverage_offdiag_zero":coeff.loc[coeff.coefficient_type=="offdiag_zero","ci95_contains_true"].mean(),"surrogate_objective_final":model.objective_history_[-1]})
         run_rows.append({**base,"completed":True,"timestamp":time.time()})
         frames=materialize_tables(tables)
         # Promote the completion ledger last: it certifies every detailed
         # checkpoint table for this scientifically safe simulated-data unit.
         for key,name in partial.items():
          if key!="run":atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
         atomic_csv(pd.DataFrame(run_rows),os.path.join(RESULTS_DIR,partial["run"]))
         roc,_,fixed=summarize_scores(frames["scores"]);atomic_csv(roc,os.path.join(RESULTS_DIR,"roc_summary_partial.csv"));atomic_csv(fixed,os.path.join(RESULTS_DIR,"fixed_fpr_operating_points_partial.csv"))
    frames=materialize_tables(tables)
    parameter_metrics=["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error","A_group_norm_pearson_correlation","A_group_norm_spearman_correlation","A_support_ROC_AUC","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p03","A_support_TPR_at_FPR_0p05","A_support_TPR_at_FPR_0p10","filtered_signal_mse","smoothed_signal_mse","filtered_signal_correlation","smoothed_signal_correlation","runtime_seconds","n_iter","converged"]
    groups=["n_sources","T","case_name","method","lambda_A_group_fraction","a0","b0"]
    parameter_summary=grouped_stats(frames["parameter"],groups,parameter_metrics)
    roc,points,fixed=summarize_scores(frames["scores"]);unc_summary=uncertainty_summary(frames["uncertainty"]);edge_summary=vb_edge_summary(frames["vb_edges"])
    decision=fixed.copy();decision["FPR"]=decision.actual_fpr;decision["TPR"]=decision.tpr;decision["specificity"]=1-decision.actual_fpr
    outputs={"run_summary.csv":pd.DataFrame(run_rows),"parameter_summary.csv":parameter_summary,"parameter_results.csv":frames["parameter"],
      "A_support_summary.csv":grouped_stats(frames["support"],groups,["true_A_group_norm","estimated_A_group_norm"]),"vb_edge_scores.csv":frames["vb_edges"],"vb_edge_score_summary.csv":edge_summary,
      "vb_coefficient_uncertainty.csv":frames["uncertainty"],"vb_uncertainty_calibration_summary.csv":unc_summary,"network_results.csv":frames["network"],
      "link_strength_summary.csv":grouped_stats(frames["network"],["n_sources","T","case_name","method","signal_type","true_link"],["raw_deviance","debiased_deviance"]),
      "roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":fixed,"decision_summary.csv":decision,
      "lgc_decision_summary.csv":decision.loc[decision.score_type.astype(str).str.startswith("Eq7")],"objective_history.csv":frames["objective"],"runtime_summary.csv":grouped_stats(frames["runtime"],groups,["runtime_seconds","n_iter","converged"])}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    save_plots(frames["parameter"],frames["uncertainty"],frames["vb_edges"],frames["objective"],frames["runtime"])
    print("\nInterpretation guide / key questions:\n1. Does hybrid VB-ARD improve A_offdiag error versus standard EM fixed-B?\n2. Does it improve A-support AUPRC versus group-lasso EM fixed-B?\n3. Does it improve TPR at FPR <= 0.05?\n4. Do alpha values separate true and false edges?\n5. Is empirical interval coverage acceptable?\n6. Does uncertainty decrease with T?\n7. Is VB useful uncertainty quantification or only shrinkage?\n8. Is runtime feasible beyond M=5?\n\nOutcomes: recovery plus coverage -> proceed to 34B; recovery with poor coverage -> treat as regularization only; no recovery gain -> ARD does not solve this bottleneck; excessive runtime -> optimize before scaling.")


if __name__ == "__main__":
    main()
