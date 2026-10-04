"""Experiment 34UC: theoretically grounded greedy Stage-B selector validation.

The default run is a read-only replay of completed 34UA legacy Stage-B
references.  New approximately-90-edge legacy references are opt-in because a
single reference costs many hours.  Matrix-free LRVB is intentionally absent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UC_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34u_a_fullrun_observability_calibration as ua
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.greedy_stageb_selection import (
    exact_greedy_ratios,
    facility_objective,
    facility_order,
    logdet_objective,
    logdet_order,
    median_bandwidth,
    pivoted_cholesky_order,
    purpose_order,
    randomized_submodularity_check,
    rbf_similarity,
    stratified_random_order,
)

EPS = 1e-12
BASE_FEATURES = [
    "estimated_group_norm", "vb_cov_trace", "louis_cov_trace",
    "log_louis_cov_determinant", "louis_group_snr",
    "spectral_transfer_full_band_score",
]
OBSERVABILITY_FEATURES = [
    "log_edge_o_inst_product", "source_observability",
    "target_observability", "edge_o_dyn_estA_mean_K10",
]
METHODS = ["B0_current", "G1_purpose", "G2_facility", "G3_logdet", "G4_pivoted_cholesky"]
FRACTIONS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]


def deployable_features(frame):
    features=list(BASE_FEATURES)
    if "log_ard_group_precision" in frame and frame.log_ard_group_precision.notna().all():
        features.append("log_ard_group_precision")
    return features+OBSERVABILITY_FEATURES


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UC_SMOKE", "0") == "1")
    run_new = os.environ.get("EXPERIMENT_34UC_RUN_NEW_FULL_REFERENCES", "0") == "1"
    run_my15 = os.environ.get("EXPERIMENT_34UC_RUN_MY15_CONFIRMATION", "0") == "1"
    root = Path("results/experiment_34uc_greedy_selector_validation")
    if args.results_dir:
        output = Path(args.results_dir)
    elif "EXPERIMENT_34UC_RESULTS_DIR" in os.environ:
        output = Path(os.environ["EXPERIMENT_34UC_RESULTS_DIR"])
    elif smoke:
        output = root / "smoke_test"
    elif run_new:
        output = root / "prospective_validation"
    else:
        output = root / "discovery_replay"
    return {
        "experiment": "34UC_greedy_selector_validation",
        "title": "Theoretically grounded greedy Stage-B edge selection: algorithm and budget validation",
        "smoke_test": smoke,
        "results_directory": str(output),
        "source_34UA": "results/experiment_34u_a_fullrun_observability_calibration",
        "source_34UB": "results/experiment_34ub_validation_6edges",
        "M_x": 20, "M_y_primary": 40, "T": 1000, "na": 2, "nb": 3,
        "calibration_references": env_int("EXPERIMENT_34UC_CAL_REPS", 1 if smoke else 2),
        "evaluation_references": env_int("EXPERIMENT_34UC_EVAL_REPS", 1 if smoke else 2),
        "discovery_calibration_references": 1 if smoke else 2,
        "discovery_evaluation_references": 1 if smoke else 2,
        "random_repeats": env_int("EXPERIMENT_34UC_RANDOM_REPEATS", 10 if smoke else 100),
        "workers": max(1, env_int("EXPERIMENT_34UC_WORKERS", 1)),
        "inner_threads": max(1, env_int("EXPERIMENT_34UC_INNER_THREADS", 1)),
        "run_new_full_references": run_new,
        "max_new_full_references": env_int("EXPERIMENT_34UC_MAX_NEW_FULL_REFERENCES", 1),
        "run_My15_confirmation": run_my15,
        "deployable_candidate_pool_size": env_int("EXPERIMENT_34UC_CANDIDATE_POOL_SIZE", 90),
        "ridge_penalty_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "facility_lambda_grid": [0.25, 0.50, 0.75, 0.90],
        "facility_weight_modes": ["purpose", "uniform"],
        "logdet_delta": 1e-3,
        "fraction_grid": FRACTIONS,
        "elbow_delta_threshold": 0.01,
        "elbow_consecutive_edges": 5,
        "A_center": "A_VB",
        "matrixfree_LRVB_used": False,
        "observability_used_as_edge_evidence": False,
        "discovery_reference_limitation": "34UA pool has 64 truth-stratified simulation-benchmark edge groups; it is screening evidence only",
        "identity_of_atomic_item": "directed off-diagonal p=2 edge group [A1[target,source], A2[target,source]]",
        "diagonal_groups": "excluded from candidate pool and off-diagonal budget",
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def initialize(config):
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    marker = output / "_COMPLETED.json"
    if marker.exists():
        raise FileExistsError(f"Completed 34UC directory exists: {output}")
    path = output / "experiment_config.json"
    protected = ["smoke_test", "calibration_references", "evaluation_references",
                 "discovery_calibration_references", "discovery_evaluation_references", "random_repeats",
                 "run_new_full_references", "max_new_full_references", "run_My15_confirmation",
                 "deployable_candidate_pool_size"]
    if path.exists():
        with path.open(encoding="utf8") as handle: old = json.load(handle)
        changed = [key for key in protected if old.get(key) != config.get(key)]
        if changed: raise ValueError(f"Existing checkpoint configuration differs: {changed}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Non-empty output directory has no compatible 34UC configuration: {output}")
    atomic_json(config, path)
    (output / "plots").mkdir(exist_ok=True)
    return output


class Progress:
    def __init__(self, total):
        self.total=max(int(total),1); self.done=0; self.started=time.perf_counter()
    def update(self, label=""):
        self.done += 1; fraction=min(self.done/self.total,1.0); filled=round(28*fraction)
        eta=(time.perf_counter()-self.started)/max(self.done,1)*(self.total-self.done)
        print(f"\r34UC [{'#'*filled}{'-'*(28-filled)}] {self.done}/{self.total} {100*fraction:5.1f}% ETA {eta/60:5.1f}m {label[:34]}",
              end="\n" if self.done>=self.total else "", flush=True)


def reconstruct_observability(frame, source_config):
    frame = frame.copy(); source_values=pd.Series(index=frame.index,dtype=float); target_values=pd.Series(index=frame.index,dtype=float)
    for _, group in frame.groupby("run_id", sort=False):
        first=group.iloc[0]
        meta={key:first[key] for key in ("M_y","true_network_id","replicate_id")}
        data,_=ua.simulate(source_config, meta)
        node=np.diag(data["C"].T @ np.linalg.solve(data["R"], data["C"]))
        source_values.loc[group.index]=node[group.source.astype(int)]
        target_values.loc[group.index]=node[group.target.astype(int)]
    frame["source_observability"]=source_values; frame["target_observability"]=target_values
    return frame


def load_discovery(config):
    source=Path(config["source_34UA"])
    edges=read_csv(source/"calibration_edge_rows_partial.csv")
    runtime=read_csv(source/"runtime_summary_partial.csv")
    if not len(edges): raise FileNotFoundError("No completed 34UA calibration edge references were found.")
    success=set(runtime.loc[runtime.run_status.eq("success"),"run_id"]) if len(runtime) else set(edges.run_id)
    eligible=edges[(edges.M_y==40)&edges.candidate_support.eq("full_candidate")&edges.run_id.isin(success)].copy()
    runs=sorted(eligible.run_id.unique()); needed=config["discovery_calibration_references"]+config["discovery_evaluation_references"]
    if len(runs)<needed: raise RuntimeError(f"34UC needs {needed} reusable references; only {len(runs)} are complete.")
    cal=runs[:config["discovery_calibration_references"]]
    evaluation=runs[config["discovery_calibration_references"]:needed]
    eligible=eligible[eligible.run_id.isin(cal+evaluation)].copy()
    eligible["reference_split"]=np.where(eligible.run_id.isin(cal),"calibration","evaluation")
    eligible["evidence_stage"]="discovery"
    eligible["candidate_pool_truth_stratified"]=True
    eligible["source_row_order"]=eligible.groupby("run_id").cumcount()+1
    with (source/"experiment_config.json").open(encoding="utf8") as handle: source_config=json.load(handle)
    eligible=reconstruct_observability(eligible,source_config)
    eligible["log_louis_cov_determinant"]=np.log(np.maximum(eligible.louis_cov_determinant,EPS))
    eligible["log_ard_group_precision"]=np.nan
    manifest=[]
    for run_id, group in eligible.groupby("run_id",sort=False):
        row=runtime[runtime.run_id.eq(run_id)].iloc[-1] if len(runtime[runtime.run_id.eq(run_id)]) else pd.Series()
        manifest.append({"run_id":run_id,"source_experiment":"34UA","source_path":str(source),
            "reference_split":group.reference_split.iloc[0],"evidence_stage":"discovery",
            "M_y":40,"candidate_pool_size":len(group),"atomic_edge_groups":True,
            "individual_coefficient_directions":2*len(group),"diagonal_groups":0,
            "candidate_pool_semantics":"truth-stratified simulation benchmark over observability quartiles and SNR",
            "deployable_candidate_pool":False,"full_legacy_stageB_available":True,
            "stageB_runtime_seconds":row.get("stageB_total_runtime_seconds",row.get("stageb_runtime_seconds",np.nan))})
    validation=read_csv(Path(config["source_34UB"])/"matrixfree_solver_validation.csv")
    manifest.append({"run_id":"34UB_validation_6edges","source_experiment":"34UB",
        "source_path":config["source_34UB"],"reference_split":"diagnostic_only","evidence_stage":"discovery",
        "M_y":40,"candidate_pool_size":len(validation),"atomic_edge_groups":True,
        "individual_coefficient_directions":2*len(validation),"diagonal_groups":0,
        "candidate_pool_semantics":"six matrix-free validation edges; not a full legacy reference",
        "deployable_candidate_pool":False,"full_legacy_stageB_available":False,"stageB_runtime_seconds":np.nan})
    return eligible,pd.DataFrame(manifest),source_config


def prospective_metadata(config, replicate):
    split="calibration" if replicate<config["calibration_references"] else "evaluation"
    network=20000+replicate
    return {"experiment_name":config["experiment"],"split":split,"M_x":20,"M_y":40,
        "observation_ratio":2.0,"T":1000,"C_family":"gaussian_isotropic","C_MODE":"gaussian_isotropic",
        "method":"hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_C_known_R",
        "candidate_support":"deployable_current_top90","true_network_id":network,"replicate_id":replicate,
        "seed_group":"34UC_PROSPECTIVE_CALIBRATION" if split=="calibration" else "34UC_PROSPECTIVE_EVALUATION",
        "run_id":f"34UC_{split}_Mx20_My40_deployable_top90_net{network}_rep{replicate}"}


def all_deployable_candidates(model,data,louis,meta,edge_scores,allocator_model):
    M=model.n_states; J=data["C"].T@np.linalg.solve(data["R"],data["C"]); node=np.diag(J)
    dynamic=ua.dynamic_observability(np.asarray(model.A_mean_matrices_),data["C"],data["R"],10)
    Ahat=np.asarray(model.A_mean_matrices_); Atrue=np.asarray(data["A"]); rows=[]
    for target in range(M):
        for source in range(M):
            if target==source: continue
            columns=[source,M+source]; center=Ahat[:,target,source]
            block,_=ua.safe_covariance(louis[target][np.ix_(columns,columns)])
            vb,_=ua.safe_covariance(model.A_row_covariances_[target][np.ix_(columns,columns)])
            raw=float(node[target]*node[source])
            rows.append({**meta,"target":target,"source":source,"true_edge":bool(data["mask"][target,source]),
                "louis_group_snr":float(np.sqrt(max(center@np.linalg.pinv(block)@center,0.0))),
                "edge_o_inst_product":raw,"log_edge_o_inst_product":float(np.log(raw+EPS)),
                "source_observability":float(node[source]),"target_observability":float(node[target]),
                "edge_o_dyn_estA_mean_K10":float((dynamic[target]+dynamic[source])/2),
                "estimated_group_norm":float(np.linalg.norm(center)),"true_group_norm":float(np.linalg.norm(Atrue[:,target,source])),
                "group_error_norm":float(np.linalg.norm(center-Atrue[:,target,source])),
                "vb_cov_trace":float(np.trace(vb)),"louis_cov_trace":float(np.trace(block)),
                "louis_cov_determinant":float(max(np.linalg.det(block),EPS)),
                "log_louis_cov_determinant":float(np.log(max(np.linalg.det(block),EPS))),
                "log_ard_group_precision":float(np.log(max(model.alpha_mean_[target,source],EPS))),
                "spectral_transfer_full_band_score":float(edge_scores[(target,source)][1]),
                "benchmark_truth_used_for_sampling_only":False,
                "benchmark_selection_reason":"deployable_current_snr_observability_ranking"})
    candidates=pd.DataFrame(rows)
    candidates["normalized_edge_o_inst_product"]=(candidates.log_edge_o_inst_product-candidates.log_edge_o_inst_product.mean())/max(candidates.log_edge_o_inst_product.std(ddof=0),EPS)
    candidates["observability_quartile"]=pd.qcut(candidates.edge_o_inst_product.rank(method="first"),4,labels=["Q1_low","Q2","Q3","Q4_high"])
    candidates["current_production_score"]=ua.predict(allocator_model,candidates,"snr_plus_observability")
    return candidates.sort_values(["current_production_score","louis_group_snr"],ascending=False).reset_index(drop=True)


def atomic_npz(path,**arrays):
    path=Path(path); temporary=Path(str(path)+".tmp")
    with temporary.open("wb") as handle: np.savez(handle,**arrays)
    os.replace(temporary,path)


def checkpoint_token(meta):
    prefix="c" if str(meta["split"]).startswith("calibration") else "e"
    return f"{prefix}{int(meta['true_network_id'])}r{int(meta['replicate_id'])}"


def direction_checkpoint_path(output,meta,index):
    directory=Path(output)/"ckpt"/checkpoint_token(meta); directory.mkdir(parents=True,exist_ok=True)
    return directory/f"d{int(index)}.npy"


def direction_diagnostic_path(output,meta,index):
    return direction_checkpoint_path(output,meta,index).with_suffix(".json")


def save_ready_edge_checkpoint(output,meta,target,source,M):
    indices=[ua.global_index(target,lag,source,M,2) for lag in range(2)]
    paths=[direction_checkpoint_path(output,meta,index) for index in indices]
    if not all(path.exists() for path in paths): return False
    columns=np.column_stack([np.load(path) for path in paths])
    raw=columns[np.ix_(indices,[0,1])]; projected,_=ua.safe_covariance(raw)
    directory=output/"edge_ckpt"/checkpoint_token(meta); directory.mkdir(parents=True,exist_ok=True)
    atomic_npz(directory/f"e{target}_{source}.npz",target=target,source=source,
               direction_indices=np.asarray(indices),covariance_block_raw=raw,covariance_block_psd=projected)
    return True


def run_stageb_checkpointed(config,source_config,output,model,data,selected,meta):
    M=model.n_states
    indices=[ua.global_index(int(row.target),lag,int(row.source),M,2) for row in selected.itertuples() for lag in range(2)]
    pair_for_index={ua.global_index(int(row.target),lag,int(row.source),M,2):(int(row.target),int(row.source))
                    for row in selected.itertuples() for lag in range(2)}
    columns={index:np.load(direction_checkpoint_path(output,meta,index)) for index in indices
             if direction_checkpoint_path(output,meta,index).exists()}
    diagnostics=[]
    for index in indices:
        path=direction_diagnostic_path(output,meta,index)
        if path.exists():
            with path.open(encoding="utf8") as handle: diagnostics.append({**meta,**json.load(handle)})
    pending=[index for index in indices if index not in columns]; completed=len(columns); started=time.perf_counter()
    progress=Progress(len(indices)); progress.done=completed
    if pending:
        with ProcessPoolExecutor(max_workers=min(config["workers"],len(pending))) as executor:
            futures={executor.submit(ua.perturb_one,(model,data["y"],data["u"],index,
                source_config["PERTURBED_VB_MAX_ITER"],source_config["PERTURBED_RESCUE_MAX_ITER"])):index for index in pending}
            for future in as_completed(futures):
                index,column,diagnostic=future.result()
                if np.all(np.isfinite(column)):
                    columns[index]=column; ua.atomic_array(direction_checkpoint_path(output,meta,index),column)
                    ua.atomic_json(diagnostic,direction_diagnostic_path(output,meta,index))
                diagnostics.append({**meta,**diagnostic}); target,source=pair_for_index[index]
                save_ready_edge_checkpoint(output,meta,target,source,M); progress.update("legacy Stage-B directions")
    valid=[index for index in indices if index in columns and np.all(np.isfinite(columns[index]))]
    if len(valid)!=len(indices): raise FloatingPointError(f"Only {len(valid)}/{len(indices)} Stage-B directions are valid")
    matrix=np.column_stack([columns[index] for index in valid]); raw,projected,projection=ua.project_selected_covariance(matrix,valid)
    position={index:offset for offset,index in enumerate(valid)}
    for row in selected.itertuples():
        target,source=int(row.target),int(row.source); edge_indices=[ua.global_index(target,lag,source,M,2) for lag in range(2)]
        positions=[position[index] for index in edge_indices]
        directory=output/"edge_ckpt"/checkpoint_token(meta); directory.mkdir(parents=True,exist_ok=True)
        atomic_npz(directory/f"e{target}_{source}.npz",target=target,source=source,
                   direction_indices=np.asarray(edge_indices),covariance_block_raw=raw[np.ix_(positions,positions)],
                   covariance_block_psd=projected[np.ix_(positions,positions)])
    measured=sum(float(row.get("runtime_seconds",0.0)) for row in diagnostics)
    seconds=measured if measured>0 else time.perf_counter()-started
    return valid,position,raw,projected,projection,pd.DataFrame(diagnostics),seconds


def generate_prospective_references(config,source_config,output,discovery):
    allocator_models,_=ua.fit_calibration_models(source_config,discovery[discovery.reference_split.eq("calibration")])
    allocator=allocator_models[(40,"full_candidate")]
    edge_path=output/"new_reference_edge_rows_partial.csv"; runtime_path=output/"new_reference_runtime_partial.csv"
    existing=read_csv(edge_path); runtime=read_csv(runtime_path)
    done=set(runtime.loc[runtime.run_status.eq("success"),"run_id"]) if len(runtime) else set()
    for replicate in range(config["max_new_full_references"]):
        meta=prospective_metadata(config,replicate)
        if meta["run_id"] in done: continue
        started=time.perf_counter()
        try:
            data,seed=ua.simulate(source_config,meta); full=np.ones((20,20),bool)
            vb_started=time.perf_counter(); model=ua.fit_model(source_config,data,full,seed+500); vb_seconds=time.perf_counter()-vb_started
            louis_started=time.perf_counter(); louis=ua.compute_louis(source_config,model,data,seed+700); louis_seconds=time.perf_counter()-louis_started
            point,spectral,edge_scores=ua.point_and_spectral_metrics(source_config,model,data,full,louis,meta)
            candidates=all_deployable_candidates(model,data,louis,meta,edge_scores,allocator)
            selected=candidates.head(min(config["deployable_candidate_pool_size"],len(candidates))).copy()
            selected["source_row_order"]=np.arange(1,len(selected)+1)
            valid,position,raw,stage,projection,perturbations,stage_seconds=run_stageb_checkpointed(
                config,source_config,output,model,data,selected,meta)
            reference=ua.build_reference_rows(source_config,selected,model,data,louis,valid,position,stage,projection,edge_scores,meta)
            reference["evidence_stage"]="prospective"; reference["candidate_pool_truth_stratified"]=False
            ua.append_checkpoint(edge_path,reference,["run_id","target","source"])
            ua.append_checkpoint(output/"new_reference_perturbation_diagnostics_partial.csv",perturbations,["run_id","direction_index"])
            run=pd.DataFrame([{**meta,"baseline_VB_runtime_seconds":vb_seconds,"louis_runtime_seconds":louis_seconds,
                "stageB_runtime_seconds":stage_seconds,"total_runtime_seconds":time.perf_counter()-started,
                "candidate_pool_size":len(selected),"n_stageb_directions":len(valid),"run_status":"success"}])
            ua.append_checkpoint(runtime_path,run,["run_id"])
        except Exception as error:
            failed=pd.DataFrame([{**meta,"run_status":"failed","error_type":type(error).__name__,"error_message":str(error),
                                  "traceback":traceback.format_exc(),"total_runtime_seconds":time.perf_counter()-started}])
            ua.append_checkpoint(runtime_path,failed,["run_id"])
    return read_csv(edge_path),read_csv(runtime_path)


def robust_scale(frame, calibration, features):
    output=frame.copy(); rows=[]
    for feature in features:
        values=calibration[feature].to_numpy(float); median=float(np.nanmedian(values))
        mad=float(np.nanmedian(np.abs(values-median))); scale=1.4826*mad
        if not np.isfinite(scale) or scale<=EPS: scale=float(np.nanstd(values))
        scale=max(scale,EPS); output[f"z__{feature}"]=(output[feature]-median)/scale
        rows.append({"feature":feature,"calibration_median":median,"calibration_robust_scale":scale})
    return output,pd.DataFrame(rows)


def ridge_coefficients(X,y,penalty):
    matrix=penalty*np.eye(X.shape[1]); matrix[0,0]=0.0
    return np.linalg.solve(X.T@X+matrix,X.T@y)


def fit_purpose_models(frame, config):
    calibration=frame[frame.reference_split.eq("calibration")]
    non_observability=[name for name in deployable_features(frame) if name not in OBSERVABILITY_FEATURES]
    feature_sets={"without_observability":non_observability,
                  "with_observability":deployable_features(frame)}
    coefficient_rows=[]; summary=[]; output=frame.copy()
    for variant,features in feature_sets.items():
        columns=[f"z__{name}" for name in features]
        candidates=[]; run_ids=sorted(calibration.run_id.unique())
        for penalty in config["ridge_penalty_grid"]:
            losses=[]
            for held_out in run_ids:
                train=calibration[~calibration.run_id.eq(held_out)]; valid=calibration[calibration.run_id.eq(held_out)]
                if not len(train): continue
                X=np.c_[np.ones(len(train)),train[columns].to_numpy(float)]
                coef=ridge_coefficients(X,train.relative_stageb_inflation.to_numpy(float),penalty)
                pred=np.c_[np.ones(len(valid)),valid[columns].to_numpy(float)]@coef
                losses.append(float(np.mean((pred-valid.relative_stageb_inflation.to_numpy(float))**2)))
            candidates.append((float(np.mean(losses)) if losses else np.inf,penalty))
        _,penalty=min(candidates); X=np.c_[np.ones(len(calibration)),calibration[columns].to_numpy(float)]
        coef=ridge_coefficients(X,calibration.relative_stageb_inflation.to_numpy(float),penalty)
        raw=np.c_[np.ones(len(output)),output[columns].to_numpy(float)]@coef
        output[f"purpose_raw__{variant}"]=raw; output[f"u__{variant}"]=np.maximum(raw,0.0)
        for term,value in zip(["intercept",*features],coef):
            coefficient_rows.append({"purpose_variant":variant,"term":term,"coefficient":value,
                                     "ridge_penalty":penalty,"target":"relative_stageb_inflation",
                                     "transformation":"u=max(raw,0)","calibration_only":True})
        summary.append({"component":"purpose_model","variant":variant,"selected_ridge_penalty":penalty,
                        "leave_one_reference_out_MSE":min(candidates)[0],"n_calibration_rows":len(calibration)})
    output["u_e"]=output["u__with_observability"]
    return output,pd.DataFrame(coefficient_rows),pd.DataFrame(summary)


def covariance(row,prefix):
    p=prefix.lower()
    value=lambda name:row[name] if isinstance(row,pd.Series) else getattr(row,name)
    return np.asarray([[value(f"Sigma_{p}_00"),value(f"Sigma_{p}_01")],
                       [value(f"Sigma_{p}_10"),value(f"Sigma_{p}_11")]],float)


def block_stats(row,block):
    block=ua.safe_covariance(block)[0]
    difference=np.asarray([row.beta_hat_lag1-row.beta_true_lag1,row.beta_hat_lag2-row.beta_true_lag2])
    D2=float(difference@np.linalg.solve(block,difference)); sign,logdet=np.linalg.slogdet(block)
    area=float(np.pi*ua.CHI2_95_DF2*np.exp(.5*logdet)) if sign>0 else np.nan
    return bool(D2<=ua.CHI2_95_DF2),float(np.trace(block)),area


def order_frame(group,method,ordering,**metadata):
    base=group.reset_index(drop=True); rows=[]
    for rank,item in enumerate(ordering,1):
        edge=base.iloc[item["index"]]
        row={"run_id":edge.run_id,"reference_split":edge.reference_split,"evidence_stage":edge.evidence_stage,
             "method":method,"greedy_rank":rank,"target":int(edge.target),"source":int(edge.source),
             "marginal_gain":item.get("marginal_gain",np.nan),
             "cumulative_objective":item.get("cumulative_objective",np.nan),
             "normalized_marginal_gain":item.get("normalized_marginal_gain",np.nan),
             "selected_reason":metadata.pop("selected_reason",method) if False else metadata.get("selected_reason",method),
             "true_edge_evaluation_only":bool(edge.true_edge),"u_e":edge.u_e}
        for name in deployable_features(base): row[name]=edge[name]
        row.update({key:value for key,value in metadata.items() if key!="selected_reason"})
        for key in ("residual_trace_fraction","maximum_residual_block_trace","kernel_reconstruction_error","minimum_information_eigenvalue"):
            row[key]=item.get(key,np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_orderings(frame, config, bandwidth):
    rows=[]
    for run_number,(run_id,group) in enumerate(frame.groupby("run_id",sort=False)):
        group=group.reset_index(drop=True); Z=group[[f"z__{name}" for name in deployable_features(group)]].to_numpy(float)
        similarity=rbf_similarity(Z,bandwidth); utility=group.u_e.to_numpy(float)
        current=np.argsort(group.source_row_order.to_numpy(),kind="stable")
        current_rows=[{"index":int(index),"marginal_gain":np.nan,"cumulative_objective":np.nan,"normalized_marginal_gain":np.nan} for index in current]
        rows.append(order_frame(group,"B0_current",current_rows,selected_reason="existing 34UA benchmark order; diagnostic only"))
        rows.append(order_frame(group,"G1_purpose",purpose_order(utility),selected_reason="modular top-k purpose score"))
        rows.append(order_frame(group,"G1_without_observability",purpose_order(group.u__without_observability.to_numpy(float)),selected_reason="purpose ablation excluding observability"))
        for weight_mode in config["facility_weight_modes"]:
            for lam in config["facility_lambda_grid"]:
                rows.append(order_frame(group,"G2_facility",facility_order(similarity,utility,lam,weight_mode),
                    facility_lambda=lam,facility_weight_mode=weight_mode,selected_reason="maximum monotone submodular facility marginal gain"))
        rows.append(order_frame(group,"G3_logdet",logdet_order(Z,utility,config["logdet_delta"]),
            logdet_delta=config["logdet_delta"],selected_reason="maximum purpose-weighted PSD log-det marginal gain"))
        kernel=similarity+1e-10*np.eye(len(group))
        rows.append(order_frame(group,"G4_pivoted_cholesky",pivoted_cholesky_order(kernel),
            selected_reason="maximum residual block trace of RBF feature kernel"))
        for repeat in range(config["random_repeats"]):
            ordering=stratified_random_order(group.louis_group_snr,group.edge_o_inst_product,340000+1000*run_number+repeat)
            rows.append(order_frame(group,"B1_stratified_random",ordering,random_repeat=repeat,
                selected_reason="round-robin random within deployable SNR x observability quartiles"))
    return pd.concat(rows,ignore_index=True)


def prepare_replay(group):
    louis_cover=[]; stage_cover=[]; cov_error=[]; trace_error=[]; ellipse_error=[]
    for row in group.itertuples(index=False):
        louis=covariance(row,"louis"); stage=covariance(row,"stageb")
        covered_louis,trace_louis,area_louis=block_stats(row,louis)
        covered_stage,_,_=block_stats(row,stage)
        louis_cover.append(covered_louis); stage_cover.append(covered_stage)
        cov_error.append(float(np.linalg.norm(louis-stage)/max(np.linalg.norm(stage),EPS)))
        trace_error.append(abs(trace_louis-float(np.trace(stage)))/max(float(np.trace(stage)),EPS))
        ellipse_error.append(abs(area_louis-float(row.stageb_ellipse_area_95))/max(float(row.stageb_ellipse_area_95),EPS))
    Z=group[[f"z__{name}" for name in deployable_features(group)]].to_numpy(float)
    distances=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=2)
    return {"pairs":list(zip(group.target.astype(int),group.source.astype(int))),
            "true":group.true_edge.to_numpy(bool),"quartile":group.observability_quartile.astype(str).to_numpy(),
            "louis_cover":np.asarray(louis_cover,bool),"stage_cover":np.asarray(stage_cover,bool),
            "covariance_error":np.asarray(cov_error),"trace_error":np.asarray(trace_error),
            "ellipse_error":np.asarray(ellipse_error),"positive_inflation":np.maximum(group.delta_trace.to_numpy(float),0),
            "observability":group.edge_o_inst_product.to_numpy(float),"snr":group.louis_group_snr.to_numpy(float),
            "purpose":group.u_e.to_numpy(float),"feature_distances":distances}


def replay_one(group,prepared,ordering,k,method,metadata):
    selected=set(zip(ordering.head(k).target.astype(int),ordering.head(k).source.astype(int)))
    chosen=np.asarray([pair in selected for pair in prepared["pairs"]],bool)
    covered=np.where(chosen,prepared["stage_cover"],prepared["louis_cover"])
    values=pd.DataFrame({"true_edge":prepared["true"],"quartile":prepared["quartile"],"selected":chosen,
        "covered":covered,"covariance_error":np.where(chosen,0.0,prepared["covariance_error"]),
        "trace_error":np.where(chosen,0.0,prepared["trace_error"]),
        "ellipse_error":np.where(chosen,0.0,prepared["ellipse_error"]),
        "positive_inflation":np.where(chosen,prepared["positive_inflation"],0.0),
        "observability":prepared["observability"],"snr":prepared["snr"],"purpose":prepared["purpose"]})
    active=values[values.true_edge]; zero=values[~values.true_edge]
    louis_active=group[group.true_edge].louis_group_covered_95.mean(); full_active=group[group.true_edge].stageb_group_covered_95.mean()
    louis_zero=group[~group.true_edge].louis_group_covered_95.mean(); full_zero=group[~group.true_edge].stageb_group_covered_95.mean()
    chosen=values[values.selected]; indices=np.flatnonzero(values.selected)
    diversity=float(np.mean(prepared["feature_distances"][np.ix_(indices,indices)][np.triu_indices(len(indices),1)])) if len(indices)>1 else 0.0
    output={"run_id":group.run_id.iloc[0],"reference_split":group.reference_split.iloc[0],"evidence_stage":group.evidence_stage.iloc[0],
        "method":method,"k":int(k),"N":len(group),"fraction_k_over_N":k/len(group),
        "active_group_coverage":active.covered.mean(),"zero_group_coverage":zero.covered.mean(),
        "louis_active_group_coverage":louis_active,"full_active_group_coverage":full_active,
        "louis_zero_group_coverage":louis_zero,"full_zero_group_coverage":full_zero,
        "retained_active_coverage_gain":(active.covered.mean()-louis_active)/max(full_active-louis_active,EPS),
        "zero_coverage_deterioration_vs_full":max(float(full_zero-zero.covered.mean()),0.0),
        "captured_inflation_fraction":chosen.positive_inflation.sum()/max(prepared["positive_inflation"].sum(),EPS),
        "mean_relative_frobenius_error":values.covariance_error.mean(),"mean_trace_relative_error":values.trace_error.mean(),
        "mean_ellipse_area_error":values.ellipse_error.mean(),"active_edge_selected_fraction":active.selected.mean(),
        "zero_edge_selected_fraction":zero.selected.mean(),"mean_selected_observability":chosen.observability.mean(),
        "mean_selected_snr":chosen.snr.mean(),"mean_selected_purpose":chosen.purpose.mean(),"pairwise_feature_diversity":diversity,
        "mean_marginal_gain":ordering.head(k).marginal_gain.mean(),"median_marginal_gain":ordering.head(k).marginal_gain.median(),
        "surrogate_objective_value":ordering.head(k).cumulative_objective.iloc[-1]}
    for quartile in ("Q1_low","Q2","Q3","Q4_high"):
        subset=active[active.quartile.eq(quartile)]; base=group[group.true_edge & group.observability_quartile.astype(str).eq(quartile)]
        output[f"{quartile}_active_group_coverage"]=subset.covered.mean()
        output[f"{quartile}_full_active_group_coverage"]=base.stageb_group_covered_95.mean()
        output[f"{quartile}_coverage_deterioration_vs_full"]=max(float(base.stageb_group_covered_95.mean()-subset.covered.mean()),0.0)
        output[f"{quartile}_retained_coverage_gain"]=(subset.covered.mean()-base.louis_group_covered_95.mean())/max(base.stageb_group_covered_95.mean()-base.louis_group_covered_95.mean(),EPS)
    output.update(metadata); return output


def k_values(N):
    fractions={max(1,min(N,int(round(value*N)))) for value in FRACTIONS}
    local={value for value in (50,54,58,60,62,64,66,68,70,72,75,80,90) if value<=N} if 80<=N<=110 else set()
    return sorted(set(range(5,N+1))|fractions|local|{N})


def replay(frame,orderings):
    rows=[]
    for run_id,group in frame.groupby("run_id",sort=False):
        prepared=prepare_replay(group)
        candidate=orderings[orderings.run_id.eq(run_id)]
        keys=["method","facility_lambda","facility_weight_mode","logdet_delta","random_repeat"]
        descriptors=candidate[[key for key in keys if key in candidate]].drop_duplicates()
        for descriptor in descriptors.to_dict("records"):
            selected=candidate.copy()
            for key,value in descriptor.items():
                if pd.isna(value): selected=selected[selected[key].isna()]
                elif isinstance(value,float): selected=selected[np.isclose(selected[key],value)]
                else: selected=selected[selected[key].eq(value)]
            selected=selected.sort_values("greedy_rank")
            for k in k_values(len(group)):
                rows.append(replay_one(group,prepared,selected,k,descriptor["method"],descriptor))
    return pd.DataFrame(rows)


def choose_hyperparameters(curve):
    calibration=curve[curve.reference_split.eq("calibration")]
    target=[]
    for (lam,weight),group in calibration[calibration.method.eq("G2_facility")].groupby(["facility_lambda","facility_weight_mode"]):
        near=group.iloc[(group.fraction_k_over_N-.70).abs().argsort()].groupby("run_id").head(1)
        score=np.nanmean([near.retained_active_coverage_gain.mean(),near.captured_inflation_fraction.mean()])
        target.append({"component":"G2_facility","facility_lambda":lam,"facility_weight_mode":weight,
                       "calibration_fraction":near.fraction_k_over_N.mean(),"calibration_score":score,
                       "retained_gain":near.retained_active_coverage_gain.mean(),"captured_inflation":near.captured_inflation_fraction.mean()})
    table=pd.DataFrame(target); best=table.sort_values(["calibration_score","facility_lambda"],ascending=[False,False]).iloc[0]
    table["selected_on_calibration"]=(np.isclose(table.facility_lambda,best.facility_lambda)&table.facility_weight_mode.eq(best.facility_weight_mode))
    return float(best.facility_lambda),str(best.facility_weight_mode),table


def freeze_curve(curve,lam,weight):
    keep=(~curve.method.eq("G2_facility"))|(np.isclose(curve.facility_lambda,lam)&curve.facility_weight_mode.eq(weight))
    return curve[keep].copy()


def freeze_orderings(orderings,lam,weight):
    keep=(~orderings.method.eq("G2_facility"))|(
        np.isclose(orderings.facility_lambda,lam,equal_nan=False)&orderings.facility_weight_mode.eq(weight))
    return orderings[keep].copy()


def bootstrap_interval(values,seed=0):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return np.nan,np.nan
    rng=np.random.default_rng(seed); means=[rng.choice(values,len(values),replace=True).mean() for _ in range(1000)]
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def aggregate_curves(curve):
    evaluation=curve[curve.reference_split.eq("evaluation")].copy()
    random=evaluation[evaluation.method.eq("B1_stratified_random")].groupby(["run_id","method","k","N"],as_index=False).mean(numeric_only=True)
    deterministic=evaluation[~evaluation.method.eq("B1_stratified_random")]
    use=pd.concat([deterministic,random],ignore_index=True,sort=False)
    metrics=[name for name in use.select_dtypes(include=[np.number]) if name not in ("k","N","random_repeat","facility_lambda","logdet_delta")]
    rows=[]
    for (method,k,N),group in use.groupby(["method","k","N"]):
        row={"method":method,"k":k,"N":N,"fraction_k_over_N":k/N,"n_evaluation_references":group.run_id.nunique()}
        for metric in metrics:
            row[f"{metric}_mean"]=group[metric].mean(); row[f"{metric}_std"]=group[metric].std()
            row[f"{metric}_median"]=group[metric].median()
        lower,upper=bootstrap_interval(group.retained_active_coverage_gain,44000+k)
        row["retained_active_coverage_gain_ci95_lower"]=lower; row["retained_active_coverage_gain_ci95_upper"]=upper
        rows.append(row)
    return pd.DataFrame(rows),use


def theoretical_checks(frame,config,bandwidth,lam,weight):
    group=frame[frame.reference_split.eq("calibration")].groupby("run_id",sort=False).__iter__().__next__()[1].reset_index(drop=True)
    Z=group[[f"z__{name}" for name in deployable_features(group)]].to_numpy(float)
    utility=group.u_e.to_numpy(float); similarity=rbf_similarity(Z,bandwidth)
    facility=lambda selected:facility_objective(selected,similarity,utility,lam,weight)
    logdet=lambda selected:logdet_objective(selected,Z,utility,config["logdet_delta"])
    rows=[]
    for method,objective in (("G2_facility",facility),("G3_logdet",logdet)):
        result=randomized_submodularity_check(objective,len(group),1000,7300 if method=="G2_facility" else 7400)
        argument=("Nonnegative weighted facility location is monotone submodular; adding a nonnegative modular purpose term preserves both properties."
                  if method=="G2_facility" else
                  "log det(delta I + sum_e q_e phi_e phi_e^T), q_e>=0 and delta>0, is the standard monotone submodular PSD information-gain form.")
        rows.append({"method":method,"check":"randomized_diminishing_returns_and_monotonicity",
                     "mathematical_argument":argument,"guarantee_scope":"(1-1/e) greedy guarantee for the surrogate objective only",**result})
    q=EPS+np.maximum(utility,0)/max(float(np.maximum(utility,0).max()),EPS)
    min_me=min(float(np.linalg.eigvalsh(q[i]*np.outer(Z[i],Z[i])).min()) for i in range(len(Z)))
    rows.append({"method":"G3_logdet","check":"PSD_and_positive_definite","minimum_Me_eigenvalue":min_me,
                 "minimum_base_eigenvalue":config["logdet_delta"],"all_Me_PSD":bool(min_me>=-1e-10)})
    n_small=min(12,len(group)); small_similarity=similarity[:n_small,:n_small]; small_Z=Z[:n_small]; small_u=utility[:n_small]
    facility_small=lambda selected:facility_objective(selected,small_similarity,small_u,lam,weight)
    logdet_small=lambda selected:logdet_objective(selected,small_Z,small_u,config["logdet_delta"])
    exact=[]
    exact.extend({"method":"G2_facility",**row} for row in exact_greedy_ratios(facility_small,facility_order(small_similarity,small_u,lam,weight),n_small))
    exact.extend({"method":"G3_logdet",**row} for row in exact_greedy_ratios(logdet_small,logdet_order(small_Z,small_u,config["logdet_delta"]),n_small))
    return pd.DataFrame(rows),pd.DataFrame(exact)


def selector_overlap(orderings):
    rows=[]
    deterministic=orderings[orderings.method.isin(METHODS)]
    for run_id,group in deterministic.groupby("run_id"):
        N=group.groupby("method").size().min(); k=max(1,round(.70*N)); sets={}
        for method,part in group.groupby("method"):
            if method=="G2_facility" and len(part)>N: continue
            sets[method]=set(zip(part.nsmallest(k,"greedy_rank").target,part.nsmallest(k,"greedy_rank").source))
        for left,right in ((a,b) for index,a in enumerate(sets) for b in list(sets)[index+1:]):
            rows.append({"run_id":run_id,"k":k,"fraction_k_over_N":k/N,"method_left":left,"method_right":right,
                         "jaccard":len(sets[left]&sets[right])/max(len(sets[left]|sets[right]),1)})
    return pd.DataFrame(rows)


def elbow_for(group,config):
    ordered=group.sort_values("k"); smooth=ordered.retained_active_coverage_gain_mean.rolling(3,min_periods=1,center=True).mean()
    delta=smooth.diff(); threshold=config["elbow_delta_threshold"]; consecutive=config["elbow_consecutive_edges"]
    for start in range(1,max(len(delta)-consecutive+1,1)):
        window=delta.iloc[start:start+consecutive]
        if len(window)==consecutive and np.all(window<=threshold): return int(ordered.k.iloc[start])
    return int(ordered.k.iloc[-1])


def decision(summary,config,evidence_stage="discovery"):
    deployable=summary[summary.method.isin(["G1_purpose","G2_facility","G3_logdet","G4_pivoted_cholesky"])]
    method_rows=[]
    for method,group in deployable.groupby("method"):
        group=group.sort_values("k")
        def first(threshold):
            q=group[group.retained_active_coverage_gain_mean>=threshold]; return int(q.k.min()) if len(q) else np.nan
        robust=group[(group.retained_active_coverage_gain_mean>=.95)&(group.retained_active_coverage_gain_ci95_lower>=.90)&
                     (group.zero_coverage_deterioration_vs_full_mean<=.02)&
                     (group.Q1_low_coverage_deterioration_vs_full_mean<=.05)]
        method_rows.append({"method":method,"k80":first(.80),"k90":first(.90),"k95":first(.95),
                            "robust_k":int(robust.k.min()) if len(robust) else np.nan,"elbow_k":elbow_for(group,config)})
    candidates=pd.DataFrame(method_rows)
    candidates["ranking_k"]=candidates.robust_k.fillna(candidates.k95).fillna(np.inf)
    simplicity={"G1_purpose":0,"G2_facility":1,"G3_logdet":2,"G4_pivoted_cholesky":3}
    candidates["simplicity"]=candidates.method.map(simplicity)
    winner=candidates.sort_values(["ranking_k","simplicity"]).iloc[0]
    win_curve=deployable[deployable.method.eq(winner.method)].sort_values("k")
    recommended_k=int(winner.robust_k if np.isfinite(winner.robust_k) else (winner.k95 if np.isfinite(winner.k95) else win_curve.N.iloc[0]))
    selected=win_curve[win_curve.k.eq(recommended_k)].iloc[0]; N=int(selected.N)
    lookup=candidates.set_index("method")
    g1=lookup.loc["G1_purpose","k95"]
    def kval(method): return lookup.loc[method,"k95"]
    def gain(method):
        candidate=deployable[(deployable.method.eq(method))&(deployable.k.eq(recommended_k))]
        baseline=deployable[(deployable.method.eq("G1_purpose"))&(deployable.k.eq(recommended_k))]
        return candidate.retained_active_coverage_gain_mean.mean()-baseline.retained_active_coverage_gain_mean.mean()
    ready=bool(evidence_stage=="prospective" and np.isfinite(winner.robust_k) and N>=80 and selected.n_evaluation_references>=3)
    reason=("smallest robust k satisfying mean/CI/zero-edge criteria" if np.isfinite(winner.robust_k)
            else "preliminary smallest k95; robust criterion not established")
    row={"winning_algorithm":winner.method,"winning_algorithm_reason":reason,
         "candidate_pool_size_mean":deployable.N.mean(),"recommended_k_absolute_if_N_near_90":int(round(90*recommended_k/N)),
         "recommended_fraction_k_over_N":recommended_k/N,"k80":winner.k80,"k90":winner.k90,"k95":winner.k95,
         "elbow_k":winner.elbow_k,"elbow_fraction":winner.elbow_k/N,
         "G1_k95":kval("G1_purpose"),"G2_k95":kval("G2_facility"),"G3_k95":kval("G3_logdet"),"G4_k95":kval("G4_pivoted_cholesky"),
         "G2_reduces_budget_vs_G1":kval("G2_facility")<g1,"G3_reduces_budget_vs_G1":kval("G3_logdet")<g1,
         "G4_reduces_budget_vs_G1":kval("G4_pivoted_cholesky")<g1,
         "G2_downstream_gain_vs_G1":gain("G2_facility"),"G3_downstream_gain_vs_G1":gain("G3_logdet"),
         "G4_downstream_gain_vs_G1":gain("G4_pivoted_cholesky"),
         "low_observability_behavior_acceptable":bool(selected.Q1_low_coverage_deterioration_vs_full_mean<=.05),
         "zero_edge_behavior_acceptable":bool(selected.zero_coverage_deterioration_vs_full_mean<=.02),
         "winner_stable_across_replicates":bool(selected.retained_active_coverage_gain_std<=.10),
         "robust_stress_transfer_My15_if_run":np.nan,
         "ready_for_future_stageB_default":ready,
         "evidence_limitation":("none for the pre-specified M_y=40 prospective validation: two calibration and three held-out deployable approximately-90-edge references completed"
             if evidence_stage=="prospective" else
             "discovery-only 64-edge truth-stratified pool; requires independent deployable approximately-90-edge references")}
    return pd.DataFrame([row]),candidates


def summaries(frame,curve,summary,orderings):
    retained=summary[[column for column in summary if column.startswith("retained_") or column in ("method","k","N","fraction_k_over_N")]].copy()
    covariance=summary[[column for column in summary if "frobenius" in column or "trace_relative" in column or "ellipse" in column or column in ("method","k","N","fraction_k_over_N")]].copy()
    quartile_columns=[column for column in summary if any(q in column for q in ("Q1_low","Q2_","Q3_","Q4_high"))]
    quartile=summary[["method","k","N","fraction_k_over_N",*quartile_columns]].copy()
    random=curve[curve.method.eq("B1_stratified_random")].groupby(["reference_split","k","N"]).agg(
        retained_gain_mean=("retained_active_coverage_gain","mean"),retained_gain_sd=("retained_active_coverage_gain","std"),
        retained_gain_q05=("retained_active_coverage_gain",lambda x:x.quantile(.05)),retained_gain_q50=("retained_active_coverage_gain","median"),
        retained_gain_q95=("retained_active_coverage_gain",lambda x:x.quantile(.95)),n_replays=("random_repeat","count")).reset_index()
    target=orderings[orderings.greedy_rank<=orderings.groupby(["run_id","method"]).greedy_rank.transform("max")*.70]
    stability=target.groupby("method").agg(
        estimated_group_norm_mean=("estimated_group_norm","mean"),estimated_group_norm_std=("estimated_group_norm","std"),
        louis_group_snr_mean=("louis_group_snr","mean"),louis_group_snr_std=("louis_group_snr","std"),
        louis_cov_trace_mean=("louis_cov_trace","mean"),louis_cov_trace_std=("louis_cov_trace","std"),
        observability_mean=("log_edge_o_inst_product","mean"),observability_std=("log_edge_o_inst_product","std"),
        spectral_score_mean=("spectral_transfer_full_band_score","mean"),spectral_score_std=("spectral_transfer_full_band_score","std"),
        purpose_mean=("u_e","mean"),purpose_std=("u_e","std")).reset_index()
    return retained,quartile,covariance,random,stability


def make_plots(output,summary,curve,orderings,decision_row):
    path=output/"plots"
    def save(name): plt.tight_layout(); plt.savefig(path/name,dpi=160); plt.close()
    use=summary[summary.method.isin(METHODS)]
    def lines(x,y,name,xlabel,ylabel):
        for method,group in use.groupby("method"):
            plt.plot(group[x],group[y],label=method)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend(fontsize=7); save(name)
    lines("k","retained_active_coverage_gain_mean","01_retained_gain_vs_k.png","k","retained active coverage gain")
    lines("fraction_k_over_N","retained_active_coverage_gain_mean","02_retained_gain_vs_fraction.png","k/N","retained active coverage gain")
    lines("fraction_k_over_N","active_group_coverage_mean","03_active_coverage_vs_fraction.png","k/N","active group coverage")
    lines("fraction_k_over_N","mean_relative_frobenius_error_mean","04_covariance_error_vs_fraction.png","k/N","mean relative covariance error")
    lines("fraction_k_over_N","captured_inflation_fraction_mean","05_captured_inflation_vs_fraction.png","k/N","captured positive inflation")
    winner=str(decision_row.winning_algorithm.iloc[0]); win=use[use.method.eq(winner)]
    for q in ("Q1_low","Q2","Q3","Q4_high"):
        plt.plot(win.fraction_k_over_N,win[f"{q}_active_group_coverage_mean"],label=q)
    plt.xlabel("k/N"); plt.ylabel("active coverage"); plt.legend(); save("06_quartile_coverage_vs_fraction.png")
    near=use[np.abs(use.fraction_k_over_N-float(decision_row.recommended_fraction_k_over_N.iloc[0]))<=.05]
    near.groupby("method").retained_active_coverage_gain_mean.mean().plot.bar(); plt.ylabel("retained gain near recommendation"); save("07_methods_near_elbow.png")
    for method,group in use.groupby("method"):
        ordered=group.sort_values("k"); plt.plot(ordered.k,ordered.retained_active_coverage_gain_mean.diff(),label=method)
    plt.xlabel("k"); plt.ylabel("marginal downstream gain"); plt.legend(fontsize=7); save("08_downstream_marginal_gain.png")
    evaluation=curve[(curve.reference_split.eq("evaluation"))&(~curve.method.eq("B1_stratified_random"))]
    plt.scatter(evaluation.mean_marginal_gain,evaluation.captured_inflation_fraction,s=8,alpha=.5); plt.xlabel("surrogate marginal gain"); plt.ylabel("actual captured inflation"); save("09_surrogate_vs_actual.png")
    chosen=orderings[orderings.greedy_rank<=20]; data=[g.estimated_group_norm.to_numpy() for _,g in chosen.groupby("method")]; labels=[m for m,_ in chosen.groupby("method")]
    plt.boxplot(data,tick_labels=labels); plt.xticks(rotation=25,ha="right"); plt.ylabel("selected estimated group norm"); save("10_selected_feature_distributions.png")


def prospective_design(config, manifest):
    legacy=manifest.loc[manifest.full_legacy_stageB_available,"stageB_runtime_seconds"].astype(float)
    mean_seconds=legacy.mean(); target=max(3,config["max_new_full_references"])
    return pd.DataFrame([{"phase":"proposed_independent_deployable_pool_validation","M_y":40,
        "candidate_pool_size_target":config["deployable_candidate_pool_size"],"calibration_references":2,
        "heldout_evaluation_references":max(3,target),"legacy_reference_runtime_seconds_estimate":mean_seconds*config["deployable_candidate_pool_size"]/64,
        "total_runtime_seconds_serial_estimate":mean_seconds*config["deployable_candidate_pool_size"]/64*(2+max(3,target)),
        "truth_used_to_construct_candidate_pool":False,"checkpoint_boundary":"every coefficient direction / atomic edge completed after both directions",
        "status":"not_started_without_explicit_user_request"}])


def main():
    args=parse_args(); config=configuration(args); output=initialize(config); started=time.perf_counter(); progress=Progress(8)
    discovery,manifest,source_config=load_discovery(config)
    source_config=dict(source_config); source_config["workers"]=config["workers"]
    frame=discovery; new_runtime=pd.DataFrame()
    if config["run_new_full_references"]:
        new_rows,new_runtime=generate_prospective_references(config,source_config,output,discovery)
        if len(new_rows) and "reference_split" not in new_rows.columns and "split" in new_rows.columns:
            new_rows=new_rows.copy()
            new_rows["reference_split"]=new_rows["split"].astype(str)
        successful=set(new_runtime.loc[new_runtime.run_status.eq("success"),"run_id"]) if len(new_runtime) else set()
        for run_id,group in new_rows[new_rows.run_id.isin(successful)].groupby("run_id"):
            runtime_row=new_runtime[new_runtime.run_id.eq(run_id)].iloc[-1]
            manifest=pd.concat([manifest,pd.DataFrame([{"run_id":run_id,"source_experiment":"34UC",
                "source_path":str(output),"reference_split":group.reference_split.iloc[0],"evidence_stage":"prospective",
                "M_y":40,"candidate_pool_size":len(group),"atomic_edge_groups":True,
                "individual_coefficient_directions":2*len(group),"diagonal_groups":0,
                "candidate_pool_semantics":"deployable top-N current snr+observability ranking; no truth used",
                "deployable_candidate_pool":True,"full_legacy_stageB_available":True,
                "stageB_runtime_seconds":runtime_row.get("stageB_runtime_seconds",np.nan)}])],ignore_index=True,sort=False)
        required_columns={"reference_split","run_id"}
        usable_new_rows=bool(len(new_rows) and required_columns.issubset(new_rows.columns))
        n_cal=new_rows[new_rows.reference_split.eq("calibration")].run_id.nunique() if usable_new_rows else 0
        n_eval=new_rows[new_rows.reference_split.eq("evaluation")].run_id.nunique() if usable_new_rows else 0
        if not usable_new_rows and not len(successful):
            raise RuntimeError(
                f"No prospective reference completed. Inspect {output/'new_reference_runtime_partial.csv'}; "
                "the run remains resumable and no completion marker was written."
            )
        if n_cal>=config["calibration_references"] and n_eval>=config["evaluation_references"]:
            frame=new_rows.copy(); frame["reference_split"]=frame.reference_split.astype(str)
            frame["evidence_stage"]="prospective"; frame["candidate_pool_truth_stratified"]=False
            frame["source_row_order"]=frame.groupby("run_id").cumcount()+1
    atomic_csv(manifest,str(output/"reference_replicate_manifest.csv")); progress.update("loaded references")
    all_features=deployable_features(frame)
    frame,scaling=robust_scale(frame,frame[frame.reference_split.eq("calibration")],all_features)
    frame,coefficients,purpose_summary=fit_purpose_models(frame,config)
    atomic_csv(frame,str(output/"candidate_edge_features.csv")); atomic_csv(coefficients,str(output/"purpose_model_coefficients.csv")); progress.update("fit purpose models")
    bandwidth=median_bandwidth(frame[frame.reference_split.eq("calibration")][[f"z__{name}" for name in all_features]])
    orderings=build_orderings(frame,config,bandwidth); atomic_csv(orderings,str(output/"selected_edge_orderings_partial.csv")); progress.update("built orderings")
    curve=replay(frame,orderings); atomic_csv(curve,str(output/"greedy_k_curve_partial.csv")); progress.update("replayed every k")
    lam,weight,facility_summary=choose_hyperparameters(curve); curve=freeze_curve(curve,lam,weight)
    hyper=pd.concat([purpose_summary,facility_summary.assign(component="G2_facility",RBF_bandwidth=bandwidth)],ignore_index=True,sort=False)
    atomic_csv(hyper,str(output/"selector_hyperparameter_summary.csv")); progress.update("froze hyperparameters")
    theory,exact=theoretical_checks(frame,config,bandwidth,lam,weight); atomic_csv(theory,str(output/"theoretical_validation_summary.csv")); atomic_csv(exact,str(output/"exact_optimum_sanity.csv")); progress.update("checked theory")
    summary,evaluation=aggregate_curves(curve); decision_row,method_decisions=decision(summary,config,frame.evidence_stage.iloc[0])
    frozen_orderings=freeze_orderings(orderings,lam,weight)
    retained,quartile,covariance_summary,random_summary,stability=summaries(frame,curve,summary,frozen_orderings)
    overlaps=selector_overlap(frozen_orderings); design=prospective_design(config,manifest)
    completed_new=int(new_runtime.run_status.eq("success").sum()) if len(new_runtime) else 0
    runtime=pd.DataFrame([{"component":"34UC_selector_validation","runtime_seconds":time.perf_counter()-started,
        "analysis_evidence_stage":frame.evidence_stage.iloc[0],
        "new_full_legacy_references_run":completed_new,"legacy_stageB_refits_run":completed_new,"matrixfree_LRVB_run":False,
        "workers_configured":config["workers"],"inner_threads":config["inner_threads"]}])
    atomic_csv(orderings,str(output/"selected_edge_orderings.csv")); atomic_csv(curve,str(output/"greedy_k_curve.csv"))
    fraction=curve[curve.apply(lambda row:any(abs(row.fraction_k_over_N-f)<=.5/max(row.N,1) for f in FRACTIONS),axis=1)]
    atomic_csv(fraction,str(output/"greedy_fraction_curve.csv")); atomic_csv(summary,str(output/"algorithm_comparison_summary.csv"))
    atomic_csv(retained,str(output/"retained_gain_summary.csv")); atomic_csv(quartile,str(output/"observability_quartile_summary.csv"))
    atomic_csv(covariance_summary,str(output/"covariance_fidelity_summary.csv")); atomic_csv(random_summary,str(output/"random_baseline_summary.csv"))
    atomic_csv(stability,str(output/"selection_stability_summary.csv")); atomic_csv(overlaps,str(output/"selection_overlap_summary.csv"))
    selector_runtime=pd.DataFrame([{"selector_replay_total_seconds":time.perf_counter()-started,"n_ordering_rows":len(orderings),"n_curve_rows":len(curve)}])
    atomic_csv(selector_runtime,str(output/"selector_runtime_summary.csv")); atomic_csv(design,str(output/"prospective_validation_design.csv"))
    atomic_csv(method_decisions,str(output/"method_budget_diagnostics.csv")); atomic_csv(decision_row,str(output/"decision_summary.csv")); atomic_csv(runtime,str(output/"runtime_summary.csv"))
    make_plots(output,summary,curve,orderings,decision_row); progress.update("saved summaries/plots")
    atomic_json({"completed":True,"completed_at_unix":time.time(),
                 "phase":"smoke" if config["smoke_test"] else ("prospective_validation" if config["run_new_full_references"] else "cheap_discovery_replay"),
                 "new_full_references_started":config["run_new_full_references"],"new_full_references_completed":completed_new,
                 "analysis_evidence_stage":frame.evidence_stage.iloc[0]},output/"_COMPLETED.json")
    progress.update("complete")


if __name__ == "__main__":
    main()
