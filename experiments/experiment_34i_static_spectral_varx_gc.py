"""Experiment 34I: static spectral VARX-GC under validated square-C modes."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34I_BLAS_THREADS", "1")
import json, time, traceback, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats, make_network, relative
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import recovery, safe_curve, vb_details
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import B_metrics
from experiments.experiment_34f_vb_ard_Q_estimation_shrinkage import Q_metrics, diagnostics, fit_vb
from experiments.experiment_34g_vb_ard_C_mixing_robustness import make_C, C_metrics
from src.gc.spectral_varx_gc import (
    compute_A_frequency_matrix, compute_A_transfer_function,
    compute_B_frequency_matrix, integrate_spectral_score,
    validate_time_frequency_gc_2source,
)
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data, var_companion_spectral_radius
from src.varx.varx_generator import generate_colored_input

M, na, nb, BURN_IN = 20, 2, 3, 300
C_MODE_LIST = ["identity", "mild_mixing", "strong_mixing"]
T_VALUES = [1000, 2000]
N_OUTER_RUNS_BY_CONDITION = {mode: {1000: 5, 2000: 5} for mode in C_MODE_LIST}
METHOD = "hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25"
Q_VARIANT = "estimate_Q_diag_shrink_scalar_rho_0p25"
N_FREQS = 128; BASE_SEED = 4900000
BANDS = {"low_band": (0., .2*np.pi), "mid_band": (.2*np.pi, .5*np.pi),
         "high_band": (.5*np.pi, np.pi), "full_band": (0., np.pi)}
SMOKE_TEST = os.environ.get("EXPERIMENT_34I_SMOKE", "0") == "1"
if SMOKE_TEST:
    C_MODE_LIST = ["identity", "mild_mixing"]; T_VALUES = [1000]
    N_OUTER_RUNS_BY_CONDITION = {mode: {1000: 1} for mode in C_MODE_LIST}
override = os.environ.get("EXPERIMENT_34I_N_OUTER_RUNS")
if override is not None:
    N_OUTER_RUNS_BY_CONDITION = {mode: {T: int(override) for T in T_VALUES} for mode in C_MODE_LIST}
RESULTS_DIR = os.environ.get("EXPERIMENT_34I_RESULTS_DIR", "results/experiment_34i")


class ProgressBar:
    def __init__(self, total, width=30):
        self.total, self.width, self.done = max(int(total), 0), width, 0
        if not self.total: print("Experiment 34I: all checkpointed work is already complete.")
    def update(self, label):
        self.done += 1; f = min(self.done/max(self.total, 1), 1.); n = round(self.width*f)
        print(f"\rExperiment 34I [{'#'*n}{'-'*(self.width-n)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:62]}",
              end="\n" if self.done >= self.total else "", flush=True)


def simulate(T, mode, seed):
    A, mask = make_network(M, seed); rng = np.random.default_rng(seed+100)
    B = [rng.normal(0, scale, (M, 1)) for scale in (.45, .25, .15)]
    C = np.eye(M) if mode == "identity" else make_C(M, mode, seed)
    Q, R = .5*np.eye(M), .6*np.eye(M); u = generate_colored_input(T+BURN_IN, .95, 1., seed+200)
    result = generate_ssm_varx_p_data(A, B, u, Q, R, C=C, D=None, burn_in=BURN_IN,
                                      random_seed=seed+300, return_augmented=True)
    return {"A": A, "B": B, "C": C, "Q": Q, "R": R, "mask": mask,
            "x": result["x"], "y": result["y"], "u": result["u"]}


def C_diagnostics(C):
    s = np.linalg.svd(C, compute_uv=False); off = C.copy(); np.fill_diagonal(off, 0.)
    return {"C_rank": np.linalg.matrix_rank(C), "C_condition_number": s[0]/s[-1],
            "C_min_singular_value": s[-1], "C_max_singular_value": s[0],
            "C_spectral_norm": s[0], "C_frobenius_norm": np.linalg.norm(C),
            "C_offdiag_energy_ratio": np.sum(off**2)/np.sum(C**2)}


def _edge_deletion_from_transfer(A, Q, omega, H, full_spectrum, target, source, eps=1e-12):
    # A_reduced(omega) = A_full(omega) + c e_target e_source^T.
    c = sum(A[k, target, source]*np.exp(-1j*omega*(k+1)) for k in range(len(A)))
    column, row = H[:, target], H[source, :]
    denominator = 1. + c*H[source, target]
    if abs(denominator) < eps: return 0., 1
    H_reduced = H - (c/denominator)*np.outer(column, row)
    reduced_ii = float(np.real((H_reduced @ Q @ H_reduced.conj().T)[target, target]))
    full_ii = max(float(np.real(full_spectrum[target, target])), eps)
    raw = np.log(max(reduced_ii, eps)/full_ii)
    return max(float(np.real(raw)), 0.), int(raw < 0)


def compute_spectral_tables(A_mats, B_mats, Q, omega, data, meta, parameter_source):
    A, B, Q = np.asarray(A_mats), np.asarray(B_mats), np.asarray(Q)
    edges = [(i, j) for i in range(M) for j in range(M) if i != j]
    transfer = np.zeros((len(edges), len(omega))); deletion = np.zeros_like(transfer)
    exog = np.zeros((M, B.shape[2], len(omega))); conditions = []
    clips = denom_bad = inversions = 0
    for f, w in enumerate(omega):
        matrix = compute_A_frequency_matrix(A, w); condition = np.linalg.cond(matrix); conditions.append(condition)
        try: H = compute_A_transfer_function(A, w)
        except np.linalg.LinAlgError:
            H = compute_A_transfer_function(A, w, jitter=1e-8); inversions += 1
        spectrum = H @ Q @ H.conj().T; total = np.maximum(np.real(np.diag(spectrum)), 1e-12)
        contribution = np.abs(H)**2*np.diag(Q)[None, :]; raw_den = total[:, None]-contribution
        denom_bad += int(np.sum(raw_den < 1e-12)); values = np.maximum(np.real(np.log(total[:, None]/np.maximum(raw_den, 1e-12))), 0.)
        for e, (target, source) in enumerate(edges):
            transfer[e, f] = values[target, source]
            deletion[e, f], clipped = _edge_deletion_from_transfer(A, Q, w, H, spectrum, target, source); clips += clipped
        Hu = H @ compute_B_frequency_matrix(B, w); exog[:, :, f] = np.abs(Hu)**2
    common = {**meta, "parameter_source": parameter_source}; spectra, bands = [], []
    true_norm = np.sqrt(np.sum(np.asarray(data["A"])**2, axis=0))
    for score_type, curves in (("transfer_spectral_gc_diagQ", transfer), ("edge_deletion_spectral_contrast", deletion)):
        for e, (target, source) in enumerate(edges):
            truth = bool(data["mask"][target, source]); norm = true_norm[target, source]
            for f, w in enumerate(omega):
                spectra.append({**common, "score_type": score_type, "source": source, "target": target,
                    "omega_rad": w, "omega_over_pi": w/np.pi, "spectral_gc_value": curves[e, f],
                    "true_direct_link": truth, "true_A_group_norm": norm})
            for name, band in BANDS.items():
                value = integrate_spectral_score(omega, curves[e], band)
                bands.append({**common, "score_type": score_type, "source": source, "target": target,
                    "band_name": name, "band_start_omega_over_pi": band[0]/np.pi,
                    "band_end_omega_over_pi": band[1]/np.pi, "integrated_score": value["integral"],
                    "average_score": value["average"], "peak_score": value["peak"],
                    "peak_omega_over_pi": value["peak_frequency"]/np.pi,
                    "true_direct_link": truth, "true_A_group_norm": norm})
    exog_spectra, exog_summary = [], []
    for source in range(M):
        for input_index in range(B.shape[2]):
            for f, w in enumerate(omega):
                exog_spectra.append({**common, "source_index": source, "input_index": input_index,
                    "omega_rad": w, "omega_over_pi": w/np.pi, "response_power": exog[source, input_index, f]})
            for name, band in BANDS.items():
                value = integrate_spectral_score(omega, exog[source, input_index], band)
                exog_summary.append({**common, "source_index": source, "input_index": input_index,
                    "band_name": name, "integrated_response_power": value["integral"],
                    "average_response_power": value["average"], "peak_response_power": value["peak"],
                    "peak_omega_over_pi": value["peak_frequency"]/np.pi})
    numerical = {**common, "spectral_radius_A": var_companion_spectral_radius(A),
        "spectral_inversion_warning_count": inversions, "number_of_frequency_points_with_bad_condition": int(np.sum(np.asarray(conditions)>1e10)),
        "max_condition_number_Aomega": np.max(conditions), "median_condition_number_Aomega": np.median(conditions),
        "spectral_clipping_count": clips, "denominator_violation_count": denom_bad,
        "condition_numbers_json": json.dumps(conditions)}
    return pd.DataFrame(spectra), pd.DataFrame(bands), pd.DataFrame(exog_spectra), pd.DataFrame(exog_summary), numerical


def metric_rows(bands):
    rows, fixed = [], []
    keys = ["run_id", "C_MODE", "T", "parameter_source", "score_type", "band_name"]
    for values, group in bands.groupby(keys, dropna=False):
        meta = dict(zip(keys, values)); metrics, curve = safe_curve(group.true_direct_link, group.integrated_score)
        rows.append({**meta, **metrics})
        for level in (.01, .03, .05, .10):
            eligible = [item for item in curve if np.isfinite(item[1]["fpr"]) and item[1]["fpr"] <= level]
            threshold, item = max(eligible, key=lambda x: (np.nan_to_num(x[1]["tpr"], nan=-1), -x[0])) if eligible else curve[0]
            fixed.append({**meta, "target_fpr_level": level, "threshold": threshold, "actual_fpr": item["fpr"],
                          **{k: item[k] for k in ("tpr", "precision", "f1", "tp", "fp", "tn", "fn")}})
    return pd.DataFrame(rows), pd.DataFrame(fixed)


def similarity_rows(oracle, estimated):
    rows = []; keys = ["run_id", "C_MODE", "T", "score_type", "source", "target"]
    merged = oracle.merge(estimated, on=keys+["omega_rad", "true_direct_link"], suffixes=("_oracle", "_estimated"))
    for values, group in merged.groupby(keys+["true_direct_link"], dropna=False):
        meta = dict(zip(keys+["true_direct_link"], values)); x, y = group.spectral_gc_value_oracle.values, group.spectral_gc_value_estimated.values
        corr = correlations(x, y) if np.std(x)>1e-12 and np.std(y)>1e-12 else np.nan
        io = np.trapezoid(x, group.omega_rad); ie = np.trapezoid(y, group.omega_rad)
        po, pe = np.argmax(x), np.argmax(y)
        rows.append({**meta, "edge_class": "true_direct_edge" if meta["true_direct_link"] else "false_direct_edge",
            "spectral_curve_correlation": corr, "spectral_curve_rmse": np.sqrt(np.mean((x-y)**2)),
            "spectral_curve_mae": np.mean(np.abs(x-y)), "integrated_score_absolute_error": abs(ie-io),
            "integrated_score_relative_error": abs(ie-io)/max(abs(io),1e-12),
            "peak_frequency_absolute_error": abs(group.omega_over_pi_oracle.iloc[pe]-group.omega_over_pi_oracle.iloc[po]),
            "peak_score_absolute_error": abs(y[pe]-x[po]), "oracle_integrated_score": io,
            "estimated_integrated_score": ie, "oracle_peak_score": x[po], "estimated_peak_score": y[pe]})
    frame = pd.DataFrame(rows); all_edges=[]
    for _, group in frame.groupby(["run_id","C_MODE","T","score_type"],dropna=False):
        copy=group.copy();copy["edge_class"]="all_edges"
        copy["integrated_rank_spearman"]=correlations(group.oracle_integrated_score.rank(),group.estimated_integrated_score.rank())
        copy["peak_rank_spearman"]=correlations(group.oracle_peak_score.rank(),group.estimated_peak_score.rank())
        n_true=max(int(group.true_direct_link.sum()),1)
        for label,k0 in (("top_k_true_links_overlap",n_true),("top_2k_true_links_overlap",2*n_true),("top_50_overlap",50)):
            k=min(k0,len(group));oracle_top=set(group.nlargest(k,"oracle_integrated_score").index);estimated_top=set(group.nlargest(k,"estimated_integrated_score").index)
            copy[label]=len(oracle_top&estimated_top)/k
        all_edges.append(copy)
    return pd.concat([frame]+all_edges,ignore_index=True)


def aggregate_mean_median_std(frame, groups, metrics):
    return frame.groupby(groups, dropna=False)[metrics].agg(["mean", "median", "std"]).reset_index().set_axis(
        groups+[f"{metric}_{stat}" for metric in metrics for stat in ("mean","median","std")], axis=1)


def score_comparison(run_metrics, vb_edges):
    rows = []
    for _, row in run_metrics.iterrows():
        if row.parameter_source == "estimated_AQ":
            rows.append({"run_id": row.run_id, "C_MODE": row.C_MODE, "T": row["T"],
                         "score_family": f"{row.score_type}_{row.band_name}", **{k: row[k] for k in
                         ("ROC_AUC","AUPRC","TPR_at_FPR_0p01","TPR_at_FPR_0p05","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p05")}})
    for values, group in vb_edges.groupby(["run_id","C_MODE","T"]):
        for family, column in (("model_A_group_norm","posterior_mean_group_norm"),("vb_group_snr","group_snr_score")):
            metrics, _ = safe_curve(group.true_link, group[column]); rows.append({"run_id":values[0],"C_MODE":values[1],"T":values[2],"score_family":family,**metrics})
    frame = pd.DataFrame(rows); metrics = ["ROC_AUC","AUPRC","TPR_at_FPR_0p01","TPR_at_FPR_0p05","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p05"]
    return frame.groupby(["C_MODE","T","score_family"],dropna=False)[metrics].mean().reset_index()


def save_plots(spectra, recovery, fixed, similarity, comparison, exog, numerical):
    path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
    selected=spectra.loc[spectra.score_type=="transfer_spectral_gc_diagQ"]
    for truth,name in ((True,"selected_true_edge_curves.png"),(False,"selected_false_edge_curves.png")):
        group=selected.loc[selected.true_direct_link==truth]; edge=group[["run_id","source","target"]].drop_duplicates().head(1)
        fig,ax=plt.subplots()
        if len(edge):
            key=edge.iloc[0]; subset=group.loc[(group.run_id==key.run_id)&(group.source==key.source)&(group.target==key.target)]
            for source,g in subset.groupby("parameter_source"):ax.plot(g.omega_over_pi,g.spectral_gc_value,label=source)
        ax.set(xlabel="omega / pi",ylabel="spectral score");ax.legend();fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    def bars(frame,index,columns,value,name,label):
        frame=frame.copy();frame[value]=pd.to_numeric(frame[value],errors="coerce")
        pivot=frame.pivot_table(index=index,columns=columns,values=value);fig,ax=plt.subplots()
        if not pivot.empty:pivot.plot.bar(ax=ax)
        ax.set_ylabel(label);ax.tick_params(axis="x",rotation=15);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
    est=recovery.loc[recovery.parameter_source=="estimated_AQ"]
    bars(est.loc[est.band_name=="full_band"],["C_MODE","T"],"score_type","AUPRC_mean","full_band_AUPRC.png","AUPRC")
    bands=est.groupby(["C_MODE","band_name"]).AUPRC_mean.mean().reset_index();bars(bands,"C_MODE","band_name","AUPRC_mean","band_AUPRC.png","AUPRC")
    for level,name in ((.01,"TPR01.png"),(.05,"TPR05.png")):
        frame=fixed.loc[(fixed.parameter_source=="estimated_AQ")&(fixed.band_name=="full_band")&(fixed.target_fpr_level==level)]
        bars(frame,"C_MODE","score_type","tpr",name,f"TPR at FPR <= {level}")
    frame=fixed.loc[(fixed.parameter_source=="estimated_AQ")&(fixed.band_name=="full_band")&(fixed.target_fpr_level==.05)]
    bars(frame,"C_MODE","score_type","precision","precision05.png","precision")
    sim=similarity.loc[similarity.edge_class=="true_direct_edge"]
    bars(sim,["C_MODE","T"],"score_type","spectral_curve_correlation_mean","curve_correlation.png","correlation")
    bars(sim,["C_MODE","T"],"score_type","peak_frequency_absolute_error_mean","peak_frequency_error.png","|peak error| / pi")
    chosen=comparison.loc[comparison.score_family.isin(["vb_group_snr","transfer_spectral_gc_diagQ_full_band","edge_deletion_spectral_contrast_full_band"])]
    bars(chosen,["C_MODE","T"],"score_family","AUPRC","score_comparison.png","AUPRC")
    chosen_exog=exog.loc[(exog.band_name=="full_band")&(exog.parameter_source=="estimated_AQ")].head(M)
    fig,ax=plt.subplots();ax.bar(chosen_exog.source_index,chosen_exog.integrated_response_power);ax.set(xlabel="source",ylabel="exogenous response power");fig.tight_layout();fig.savefig(os.path.join(path,"exogenous_response_power.png"));plt.close(fig)
    fig,ax=plt.subplots()
    if len(numerical):
        values=json.loads(numerical.iloc[0].condition_numbers_json);ax.plot(np.linspace(0,1,len(values)),values)
    ax.set(xlabel="omega / pi",ylabel="condition number A(omega)");fig.tight_layout();fig.savefig(os.path.join(path,"Aomega_condition.png"));plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True); omega=np.linspace(0,np.pi,N_FREQS)
    validation=pd.DataFrame([validate_time_frequency_gc_2source()]);atomic_csv(validation,os.path.join(RESULTS_DIR,"spectral_validation_2source.csv"))
    config={"experiment":"34I","C_MODE_LIST":C_MODE_LIST,"T_VALUES":T_VALUES,"N_OUTER_RUNS_BY_CONDITION":N_OUTER_RUNS_BY_CONDITION,
        "M":M,"na":na,"nb":nb,"METHODS":[METHOD],"Q_VARIANT":Q_VARIANT,"N_FREQS":N_FREQS,
        "frequency_grid":"[0, pi] inclusive","BANDS":{k:[a/np.pi,b/np.pi] for k,(a,b) in BANDS.items()},
        "score_labels":{"transfer_spectral_gc_diagQ":"total innovation-transfer influence; indirect paths possible",
                        "edge_deletion_spectral_contrast":"direct deletion diagnostic; not refitted Geweke conditional GC"},
        "exogenous_note":"H_u is stimulus response and is not source-source GC","smoke_test":SMOKE_TEST,
        "TODO":"Experiment 35A sliding-window adaptive spectral VARX-GC; rectangular C remains deferred"}
    config=json.loads(json.dumps(config))
    cp=os.path.join(RESULTS_DIR,"experiment_config.json");ledger=os.path.join(RESULTS_DIR,"run_summary_partial.csv")
    if os.path.exists(cp) and os.path.exists(ledger):
        with open(cp,encoding="utf-8") as h:old=json.load(h)
        for key in ("C_MODE_LIST","T_VALUES","N_OUTER_RUNS_BY_CONDITION","M","N_FREQS","METHODS"):
            if old.get(key)!=config.get(key):raise ValueError("Existing 34I checkpoint has a different numerical configuration.")
    with open(cp,"w",encoding="utf-8") as h:json.dump(config,h,indent=2)
    files={"spectra":"spectral_gc_spectra_partial.csv","bands":"band_integrated_spectral_gc_partial.csv",
           "exog_spectra":"_exogenous_response_spectra_partial.csv","exog":"exogenous_response_summary_partial.csv",
           "similarity":"spectral_oracle_similarity_partial.csv","vb":"_vb_edges_partial.csv","uncertainty":"_uncertainty_partial.csv",
           "parameter":"_parameter_partial.csv","numerical":"spectral_numerical_diagnostics_partial.csv","run":"run_summary_partial.csv"}
    tables={k:[] for k in files};
    for key,name in files.items():
        path=os.path.join(RESULTS_DIR,name)
        if os.path.exists(path):
            try:frame=pd.read_csv(path)
            except pd.errors.EmptyDataError:frame=pd.DataFrame()
            if len(frame):tables[key]=[frame]
    existing=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame()
    completed=set(zip(existing.C_MODE,existing["T"].astype(int),existing.outer_run.astype(int))) if len(existing) else set()
    total=sum((mode,T,o) not in completed for mode in C_MODE_LIST for T in T_VALUES for o in range(N_OUTER_RUNS_BY_CONDITION[mode][T]));progress=ProgressBar(total)
    for mode in C_MODE_LIST:
      for T in T_VALUES:
       for outer in range(N_OUTER_RUNS_BY_CONDITION[mode][T]):
        if (mode,T,outer) in completed:continue
        run_id=f"{mode}_T{T}_run{outer:03d}";seed=BASE_SEED+C_MODE_LIST.index(mode)*1000000+T*100+outer;started=time.perf_counter();phase="simulate";data=None
        meta={"run_id":run_id,"C_MODE":mode,"T":T,"outer_run":outer,"method":METHOD,"Q_VARIANT":Q_VARIANT,"a0":1e-3,"b0":1e-3}
        try:
            data=simulate(T,mode,seed);proxy=data["y"]@np.linalg.pinv(data["C"]).T;fit_started=time.perf_counter();phase="fit"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always");model,Ah,Bh,filtered,smoothed=fit_vb(data,Q_VARIANT,seed+500)
            fit_runtime=time.perf_counter()-fit_started;diag=diagnostics(model);phase="spectral";spectral_started=time.perf_counter()
            oracle=compute_spectral_tables(data["A"],data["B"],data["Q"],omega,data,meta,"oracle_AQ")
            estimated=compute_spectral_tables(Ah,Bh,model.Q,omega,data,meta,"estimated_AQ")
            spectral_diag_hat=estimated[4]
            spectral_runtime=time.perf_counter()-spectral_started
            spectra=pd.concat([oracle[0],estimated[0]],ignore_index=True);bands=pd.concat([oracle[1],estimated[1]],ignore_index=True)
            exog_spectra=pd.concat([oracle[2],estimated[2]],ignore_index=True);exog=pd.concat([oracle[3],estimated[3]],ignore_index=True)
            similarity=similarity_rows(oracle[0],estimated[0]);row,_=recovery(Ah,data,filtered,smoothed,meta,fit_runtime,diag["n_iter"],diag["converged_all"])
            row.update(diag);row.update(B_metrics(Bh,data,model.B_initialization_mode_,diag["final_B_change_norm"]));row.update(Q_metrics(model.Q,data,model));row.update(C_diagnostics(data["C"]))
            row.update({"pinv_proxy_mse":np.mean((proxy-data["x"])**2),"pinv_proxy_correlation":np.nanmean([correlations(proxy[:,i],data["x"][:,i]) for i in range(M)]),
                        "runtime_fit_seconds":fit_runtime,"runtime_spectral_seconds":spectral_runtime,"total_runtime_seconds":time.perf_counter()-started,"run_status":"success"})
            coeff,edges,_=vb_details(model,data,meta);edges["run_id"]=run_id;status="success"
            for key,value in (("spectra",spectra),("bands",bands),("exog_spectra",exog_spectra),("exog",exog),("similarity",similarity),("vb",edges),("uncertainty",coeff),("parameter",pd.DataFrame([row])),("numerical",pd.DataFrame([oracle[4],estimated[4]]))):tables[key].append(value)
        except Exception as error:
            spectral_diag_hat={}
            fit_runtime=locals().get("fit_runtime",np.nan);spectral_runtime=time.perf_counter()-locals().get("spectral_started",time.perf_counter()) if phase=="spectral" else np.nan;status="failed_"+phase
            row={**meta,"run_status":status,"runtime_fit_seconds":fit_runtime,"runtime_spectral_seconds":spectral_runtime,"total_runtime_seconds":time.perf_counter()-started,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()};tables["parameter"].append(pd.DataFrame([row]))
        tables["run"].append(pd.DataFrame([{**meta,"run_status":status,"runtime_fit_seconds":fit_runtime,
            "runtime_spectral_seconds":spectral_runtime,"total_runtime_seconds":time.perf_counter()-started,
            **{key:row.get(key,np.nan) for key in ("n_iter","converged_all","practically_converged","hit_max_iter",
              "final_relative_A_change_norm","final_relative_B_change_norm","final_relative_Q_change_norm","final_relative_alpha_change_norm")},
            "spectral_radius_A_true":var_companion_spectral_radius(data["A"]) if data is not None else np.nan,
            "spectral_radius_A_hat":row.get("spectral_radius_A",np.nan),
            **{key:spectral_diag_hat.get(key,np.nan) for key in ("spectral_inversion_warning_count",
              "number_of_frequency_points_with_bad_condition","max_condition_number_Aomega","median_condition_number_Aomega")}}]))
        frames={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in tables.items()}
        for key,name in files.items():atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
        if len(frames["bands"]):
            run_metrics,fixed=metric_rows(frames["bands"]);atomic_csv(run_metrics,os.path.join(RESULTS_DIR,"spectral_gc_recovery_summary_partial.csv"));atomic_csv(fixed,os.path.join(RESULTS_DIR,"spectral_gc_fixed_fpr_summary_partial.csv"));atomic_csv(score_comparison(run_metrics,frames["vb"]),os.path.join(RESULTS_DIR,"score_comparison_summary_partial.csv"))
        runtime=frames["run"].groupby(["C_MODE","T"])[["runtime_fit_seconds","runtime_spectral_seconds","total_runtime_seconds"]].mean().reset_index();atomic_csv(runtime,os.path.join(RESULTS_DIR,"runtime_summary_partial.csv"))
        progress.update(f"{mode} T={T} run={outer+1}/{N_OUTER_RUNS_BY_CONDITION[mode][T]} {status}")
    frames={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in tables.items()};run_metrics,fixed=metric_rows(frames["bands"])
    recovery_summary=aggregate_mean_median_std(run_metrics,["C_MODE","T","parameter_source","score_type","band_name"],["ROC_AUC","AUPRC","best_youden_J","best_F1","TPR_at_FPR_0p01","TPR_at_FPR_0p03","TPR_at_FPR_0p05","TPR_at_FPR_0p10","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05"])
    similarity_summary=aggregate_mean_median_std(frames["similarity"],["C_MODE","T","score_type","edge_class"],["spectral_curve_correlation","spectral_curve_rmse","spectral_curve_mae","integrated_score_absolute_error","integrated_score_relative_error","peak_frequency_absolute_error","peak_score_absolute_error","integrated_rank_spearman","peak_rank_spearman","top_k_true_links_overlap","top_2k_true_links_overlap","top_50_overlap"])
    exog=frames["exog"];oracle_exog=exog.loc[exog.parameter_source=="oracle_AQ"];estimated_exog=exog.loc[exog.parameter_source=="estimated_AQ"]
    join=["run_id","C_MODE","T","source_index","input_index","band_name"];errors=estimated_exog.merge(oracle_exog,on=join,suffixes=("","_oracle"));errors["oracle_estimated_absolute_error"]=(errors.integrated_response_power-errors.integrated_response_power_oracle).abs();errors["oracle_estimated_relative_error"]=errors.oracle_estimated_absolute_error/errors.integrated_response_power_oracle.abs().clip(lower=1e-12)
    oracle_exog=oracle_exog.copy();oracle_exog["oracle_estimated_absolute_error"]=np.nan;oracle_exog["oracle_estimated_relative_error"]=np.nan
    exog_with_errors=pd.concat([oracle_exog,errors[oracle_exog.columns]],ignore_index=True)
    exog_summary=exog_with_errors.groupby(["C_MODE","T","parameter_source","source_index","input_index","band_name"],dropna=False)[["integrated_response_power","average_response_power","peak_response_power","peak_omega_over_pi","oracle_estimated_absolute_error","oracle_estimated_relative_error"]].mean().reset_index()
    fixed_summary=fixed.groupby(["C_MODE","T","parameter_source","score_type","band_name","target_fpr_level"],dropna=False)[["threshold","actual_fpr","tpr","precision","f1","tp","fp","tn","fn"]].mean().reset_index()
    parameter=frames["parameter"];success=parameter.loc[parameter.run_status=="success"];groups=["C_MODE","T","method"]
    outputs={"run_summary.csv":frames["run"],"runtime_summary.csv":frames["run"].groupby(["C_MODE","T"])[["runtime_fit_seconds","runtime_spectral_seconds","total_runtime_seconds"]].agg(["mean","median","std"]).reset_index(),
      "parameter_summary.csv":grouped_stats(success,groups,["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p05"]),
      "B_recovery_summary.csv":grouped_stats(success,groups,["B_relative_frobenius_error","B_energy_ratio_hat_to_true"]),"Q_recovery_summary.csv":grouped_stats(success,groups,["Q_relative_frobenius_error","Q_trace_ratio_hat_to_true","Q_mean_diag","Q_diag_coefficient_of_variation"]),
      "latent_recovery_summary.csv":grouped_stats(success,groups,["filtered_signal_mse","smoothed_signal_mse","filtered_signal_correlation","smoothed_signal_correlation","pinv_proxy_mse","pinv_proxy_correlation"]),
      "spectral_gc_spectra.csv":frames["spectra"],"band_integrated_spectral_gc.csv":frames["bands"],"spectral_gc_recovery_summary.csv":recovery_summary,
      "spectral_gc_fixed_fpr_summary.csv":fixed_summary,"spectral_oracle_similarity.csv":similarity_summary,"score_comparison_summary.csv":score_comparison(run_metrics,frames["vb"]),
      "exogenous_response_spectra.csv":frames["exog_spectra"],"exogenous_response_summary.csv":exog_summary,
      "vb_edge_score_summary.csv":score_comparison(run_metrics,frames["vb"]).loc[lambda x:x.score_family.isin(["model_A_group_norm","vb_group_snr"])],
      "vb_uncertainty_calibration_summary.csv":frames["uncertainty"].groupby(["C_MODE","T","coefficient_type"])[["ci95_contains_true","posterior_std","error","standardized_abs_error"]].agg(["mean","std"]).reset_index(),
      "spectral_numerical_diagnostics.csv":frames["numerical"],"decision_summary.csv":fixed}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    save_plots(frames["spectra"],recovery_summary,fixed,similarity_summary,outputs["score_comparison_summary.csv"],exog_summary,frames["numerical"])
    print(f"Experiment 34I complete. Outputs saved under {RESULTS_DIR}/")
    print("Interpretation guide: assess oracle direct-support alignment, estimated/oracle profile agreement, transfer versus edge-deletion scores, band localization, identity-to-strong-C degradation, T=2000 gains, complementarity with group-SNR, and separate exogenous-response recovery before moving to Experiment 35A adaptive spectral GC.")


if __name__=="__main__":main()
