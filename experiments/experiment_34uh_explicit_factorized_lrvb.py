"""Experiment 34UH: explicit-Jacobian factorized implicit LRVB validation.

The experiment reuses paired 34UG Stage-A networks and legacy references.
Dense Jacobians are assembled exclusively from the validated analytic tangent
JVP, factorized once, and reused across all right-hand sides.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34UH_INNER_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import lu_solve
from scipy.stats import pearsonr, spearmanr

import experiments.experiment_34ug_efficient_lrvb_stageb_comparison as ug
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.stats.exact_implicit_lrvb import HybridFixedPointTangentMap
from src.stats.factorized_implicit_lrvb import (
    assemble_exact_jacobian, benchmark_column_workers,
    block_jacobi_preconditioner, explicit_gmres,
    factorize_response_matrix, solve_multiple_rhs,
)
from src.stats.lrvb_stageB_smoother_feedback import global_index, sensitivity_column


OUTPUT_ROOT = Path("results/experiment_34uh_explicit_factorized_lrvb")
SOURCE_34UG = Path("results/experiment_34ug_efficient_lrvb_stageb_comparison")
EPS = 1e-12
CHI2_95_DF2 = 5.991464547107979
REQUIRED = [
    "source_34ug_manifest.csv", "network_manifest.csv", "validation_edge_manifest.csv",
    "jacobian_parallel_benchmark.csv", "jacobian_column_runtime.csv",
    "jacobian_assembly_summary.csv", "dense_jacobian_diagnostics.csv",
    "lu_factorization_diagnostics.csv", "direct_solve_residuals.csv",
    "direct_full_results.csv", "gmres_vs_direct_response.csv",
    "solver_error_decomposition.csv", "covariance_fidelity.csv",
    "coverage_agreement.csv", "stageb_inflation_agreement.csv",
    "block_jacobi_gmres.csv", "reduced_direct_results.csv",
    "legacy_epsilon_sensitivity.csv", "multi_rhs_runtime.csv",
    "amortized_runtime_projection.csv", "runtime_summary.csv",
    "method_summary.csv", "decision_summary.csv",
]


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def configuration(args):
    smoke = bool(args.smoke or os.environ.get("EXPERIMENT_34UH_SMOKE", "0") == "1")
    output = Path(args.results_dir or os.environ.get(
        "EXPERIMENT_34UH_RESULTS_DIR", str(OUTPUT_ROOT / "smoke_test" if smoke else OUTPUT_ROOT)))
    mx = 3 if smoke else 20
    return {
        "experiment": "34UH_explicit_factorized_lrvb", "smoke_test": smoke,
        "results_directory": str(output), "source_34UG": str(SOURCE_34UG),
        "M_x": mx, "M_y_values": [5, 2] if smoke else [40, 15],
        "T": 40 if smoke else 1000, "na": 2, "nb": 3,
        "C_family": "gaussian_isotropic", "C_known": True, "R_fixed_true": True,
        "B_update_mode": "free", "Q_update_mode": "diag_shrink_scalar",
        "Q_shrinkage_rho": .25, "Q_update_damping": .5,
        "include_A_posterior_uncertainty_in_Q": True, "A_center": "A_VB",
        "a0": 1e-3, "b0": 1e-3,
        "VB_MAX_ITER": env_int("EXPERIMENT_34UH_VB_MAX_ITER", 5 if smoke else 75),
        "N_FFBS_SAMPLES": env_int("EXPERIMENT_34UH_N_FFBS", 3 if smoke else 50),
        "Louis_eta": .7, "Louis_tau": .9,
        "finite_difference_epsilon": 1e-4,
        "PERTURBED_VB_MAX_ITER": env_int("EXPERIMENT_34UH_PERTURB_MAX_ITER", 3 if smoke else 30),
        "PERTURBED_RESCUE_MAX_ITER": env_int("EXPERIMENT_34UH_PERTURB_RESCUE_MAX_ITER", 4 if smoke else 50),
        "validation_groups_per_network": 2 if smoke else 12,
        "stageA_network_ids": {"5" if smoke else "40": 348040, "2" if smoke else "15": 348015} if smoke else {"40": 347040, "15": 347015},
        "stageB_network_ids": {"5" if smoke else "40": 348140, "2" if smoke else "15": 348115} if smoke else {"40": 347140, "15": 347115},
        "jacobian_workers_max": max(1, env_int("EXPERIMENT_34UH_JACOBIAN_WORKERS", 1)),
        "workers": max(1, env_int("EXPERIMENT_34UH_LEGACY_WORKERS", 1)),
        "inner_threads": max(1, env_int("EXPERIMENT_34UH_INNER_THREADS", 1)),
        "jacobian_checkpoint_columns": 5 if smoke else 20,
        "parallel_benchmark_columns": 4 if smoke else 32,
        "analytic_tangent_only": True, "dense_dtype": "float64",
        "gmres_rtol": 1e-5, "gmres_restart": 50, "gmres_total_iterations": 100,
        "run_block_jacobi": os.environ.get("EXPERIMENT_34UH_BLOCK_JACOBI", "1") == "1",
        "run_all_A_rhs_benchmark": os.environ.get("EXPERIMENT_34UH_ALL_A_RHS", "0" if smoke else "1") == "1",
        "run_stageB_after_gate": not smoke,
        "legacy_epsilon_grid": [1e-4, 3e-5] if smoke else [3e-4, 1e-4, 3e-5, 1e-5],
        "legacy_iteration_grid": [3, 4, 5] if smoke else [30, 50, 75],
        "fidelity_gate": {"median_frobenius": .05, "p90_frobenius": .10,
            "median_trace": .05, "inflation_spearman": .95,
            "coverage_agreement": .95, "finite_fraction": 1., "PSD_fraction": 1.},
        "direct_residual_target": 1e-10,
        "truth_used_by_computational_method": False,
        "full_380_legacy_stageB_prohibited": True,
    }


def atomic_json(value, path):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
    os.replace(temporary, path)


def read_csv(path):
    try: return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError): return pd.DataFrame()


def append_rows(path, rows, keys):
    if not rows: return
    new = pd.DataFrame(rows); old = read_csv(path)
    frame = pd.concat([old, new], ignore_index=True) if len(old) else new
    subset = [key for key in keys if key in frame]
    if subset: frame = frame.drop_duplicates(subset=subset, keep="last")
    atomic_csv(frame, str(path))


def atomic_array(path, value):
    path = Path(path); temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle: np.save(handle, np.asarray(value))
    os.replace(temporary, path)


class Progress:
    def __init__(self): self.started = time.perf_counter(); self.phase_started=self.started; self.label = ""; self.total = 1; self.done = 0
    def start(self, label, total, completed=0):
        self.label, self.total, self.done = label, max(int(total), 1), int(completed); self.phase_started=time.perf_counter(); self._show()
    def update(self, done=None, total=None, detail=""):
        self.done = self.done + 1 if done is None else int(done)
        if total is not None: self.total = max(int(total), 1)
        self._show(detail)
    def _show(self, detail=""):
        fraction = min(self.done / self.total, 1.); filled = round(30*fraction)
        elapsed = time.perf_counter()-self.phase_started
        eta = elapsed/max(self.done,1)*max(self.total-self.done,0)
        print(f"\r34UH [{'#'*filled}{'-'*(30-filled)}] {self.done}/{self.total} "
              f"{100*fraction:5.1f}% ETA {eta/60:6.1f}m {(self.label+' '+detail)[:42]}",
              end="\n" if self.done >= self.total else "", flush=True)


def initialize(config):
    output = Path(config["results_directory"]); output.mkdir(parents=True, exist_ok=True)
    if (output / "_COMPLETED.json").exists(): return output, True
    path = output / "experiment_config.json"
    protected = ["smoke_test", "M_x", "M_y_values", "T", "stageA_network_ids",
                 "stageB_network_ids", "jacobian_workers_max", "workers", "inner_threads",
                 "run_all_A_rhs_benchmark"]
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8")); changed = [key for key in protected
            if old.get(key, config.get(key) if key == "workers" else None) != config.get(key)]
        if changed: raise ValueError(f"34UH checkpoint configuration mismatch: {changed}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Non-empty 34UH directory has no compatible configuration: {output}")
    atomic_json(config, path)
    for name in ("plots", "checkpoints"): (output/name).mkdir(exist_ok=True)
    return output, False


def validate_source_config(config):
    if config["smoke_test"]: return None
    required = ["experiment_config.json", "validation_edge_manifest.csv", "network_manifest.csv",
                "legacy_stageb_reference.csv", "full_implicit_results.csv", "gmres_diagnostics.csv", "_COMPLETED.json"]
    missing = [name for name in required if not (SOURCE_34UG/name).exists()]
    if missing: raise FileNotFoundError(f"Required completed 34UG artifacts are missing: {missing}")
    source = json.loads((SOURCE_34UG/"experiment_config.json").read_text(encoding="utf8"))
    expected = {"M_x":20,"M_y_values":[40,15],"T":1000,"na":2,"nb":3,
        "C_family":"gaussian_isotropic","B_update_mode":"free",
        "Q_update_mode":"diag_shrink_scalar","Q_shrinkage_rho":.25,
        "Q_update_damping":.5,"A_center":"A_VB","a0":1e-3,"b0":1e-3,
        "VB_MAX_ITER":75,"N_FFBS_SAMPLES":50,"PERTURBED_VB_MAX_ITER":30,
        "PERTURBED_RESCUE_MAX_ITER":50,"finite_difference_epsilon":1e-4}
    changed = [key for key,value in expected.items() if source.get(key) != value]
    if changed: raise ValueError(f"34UG source configuration differs from 34UH specification: {changed}")
    return source


def source_manifest(output):
    rows=[]
    for name in ("experiment_config.json","validation_edge_manifest.csv","network_manifest.csv",
                 "legacy_stageb_reference.csv","full_implicit_results.csv","runtime_per_group.csv",
                 "runtime_per_direction.csv","map_equivalence_checks.csv","jvp_validation.csv","gmres_diagnostics.csv"):
        path=SOURCE_34UG/name
        rows.append({"source_experiment":"34UG","artifact":name,"path":str(path),
                     "available":path.exists(),"bytes":path.stat().st_size if path.exists() else 0,
                     "reused_read_only":True})
    atomic_csv(pd.DataFrame(rows),str(output/"source_34ug_manifest.csv"))


def context_from_34ug(config, output, my):
    manifest=read_csv(SOURCE_34UG/"network_manifest.csv")
    network=config["stageA_network_ids"][str(my)]
    row=manifest[(manifest.M_y.eq(my))&manifest.true_network_id.eq(network)]
    if len(row)!=1: raise RuntimeError(f"Expected one paired 34UG My={my}, network={network}; found {len(row)}")
    meta=row.iloc[0].to_dict(); run_id=meta["run_id"]
    baseline=SOURCE_34UG/"checkpoints"/run_id/"baseline.pkl"
    if not baseline.exists(): raise FileNotFoundError(f"Missing 34UG baseline checkpoint: {baseline}")
    with baseline.open("rb") as handle: payload=pickle.load(handle)
    edges=read_csv(SOURCE_34UG/"validation_edge_manifest.csv"); edges=edges[edges.run_id.eq(run_id)].sort_values("benchmark_order")
    if len(edges)!=12: raise RuntimeError(f"Paired 34UG run {run_id} has {len(edges)} edges, expected 12.")
    legacy=read_csv(SOURCE_34UG/"legacy_stageb_reference.csv"); legacy=legacy[legacy.run_id.eq(run_id)]
    old=read_csv(SOURCE_34UG/"full_implicit_results.csv"); old=old[old.run_id.eq(run_id)]
    old_diag=read_csv(SOURCE_34UG/"gmres_diagnostics.csv"); old_diag=old_diag[old_diag.run_id.eq(run_id)]
    if len(legacy)!=12 or len(old)!=12 or len(old_diag)!=24: raise RuntimeError("Incomplete paired 34UG numerical reference.")
    append_rows(output/"validation_edge_manifest.csv",edges.to_dict("records"),["run_id","target","source"])
    return {"meta":meta,"payload":payload,"edges":edges,"legacy":legacy,"old":old,"old_diag":old_diag,
            "source":"34UG_reused","baseline_path":str(baseline)}


def smoke_context(config, output, my, progress=None):
    network=config["stageA_network_ids"][str(my)]
    meta=ug.metadata(config,my,network,"stageA"); meta["experiment_name"]=config["experiment"]
    payload=ug.get_baseline(config,output,meta)
    edges=ug.add_truth_coefficients(ug.get_edges(config,output,payload,meta),payload)
    append_rows(output/"validation_edge_manifest.csv",edges.to_dict("records"),["run_id","target","source"])
    legacy_rows=[]
    if progress is not None: progress.start(f"smoke My={my} legacy",len(edges),0)
    for edge in edges.itertuples(index=False):
        cov,row=ug.legacy_edge(config,output,payload,meta,int(edge.target),int(edge.source)); row.update(ug.covariance_columns(cov)); legacy_rows.append(row)
        if progress is not None: progress.update(detail=f"{int(edge.target)}<-{int(edge.source)}")
    return {"meta":meta,"payload":payload,"edges":edges,"legacy":pd.DataFrame(legacy_rows),
            "old":pd.DataFrame(),"old_diag":pd.DataFrame(),"source":"34UH_smoke_generated","baseline_path":str(ug.baseline_path(output,meta["run_id"]))}


def stageB_context(config, output, my, progress=None):
    network=config["stageB_network_ids"][str(my)]
    meta=ug.metadata(config,my,network,"stageB"); meta["experiment_name"]=config["experiment"]
    payload=ug.get_baseline(config,output,meta); edges=ug.add_truth_coefficients(ug.get_edges(config,output,payload,meta),payload)
    append_rows(output/"validation_edge_manifest.csv",edges.to_dict("records"),["run_id","target","source"])
    legacy_rows=[]
    if progress is not None: progress.start(f"stageB My={my} legacy",len(edges),0)
    for edge in edges.itertuples(index=False):
        cov,row=ug.legacy_edge(config,output,payload,meta,int(edge.target),int(edge.source)); row.update(ug.covariance_columns(cov)); legacy_rows.append(row)
        if progress is not None: progress.update(detail=f"{int(edge.target)}<-{int(edge.source)}")
    return {"meta":meta,"payload":payload,"edges":edges,"legacy":pd.DataFrame(legacy_rows),
            "old":pd.DataFrame(),"old_diag":pd.DataFrame(),"source":"34UH_new_confirmation",
            "baseline_path":str(ug.baseline_path(output,meta["run_id"]))}


def choose_workers(config, output, context, engine, parameterization):
    meta=context["meta"]; frame=read_csv(output/"jacobian_parallel_benchmark.csv")
    existing=frame[(frame.run_id.eq(meta["run_id"]))&frame.map_parameterization.eq(parameterization)] if len(frame) else pd.DataFrame()
    if len(existing): return int(existing.loc[existing.finite.astype(bool),"workers"].iloc[existing.loc[existing.finite.astype(bool),"seconds_per_column"].argmin()])
    maximum=min(config["jacobian_workers_max"],os.cpu_count() or 1); values=[1]+([2] if maximum>=2 else [])+([4] if maximum>=4 else [])
    n=len(engine.theta0); count=min(config["parallel_benchmark_columns"],n)
    indices=np.unique(np.linspace(0,n-1,count).round().astype(int))
    bench=benchmark_column_workers(engine.model,engine.y,engine.u,engine.reduced,indices,values,config["inner_threads"])
    for key,value in meta.items(): bench[key]=value
    bench["map_parameterization"]=parameterization
    append_rows(output/"jacobian_parallel_benchmark.csv",bench.to_dict("records"),["run_id","map_parameterization","workers"])
    valid=bench[bench.finite.astype(bool)]; return int(valid.loc[valid.seconds_per_column.idxmin(),"workers"])


def build_rhs(engine, indices, path, progress, label):
    path=Path(path); n,k=len(engine.theta0),len(indices)
    if path.exists():
        values=np.load(path,mmap_mode="r+")
        if values.shape!=(n,k): raise ValueError("RHS checkpoint shape mismatch.")
    else:
        values=np.lib.format.open_memmap(path,mode="w+",dtype=np.float64,shape=(n,k)); values[:]=np.nan; values.flush()
    done=np.all(np.isfinite(values),axis=0); progress.start(label,k,int(done.sum()))
    for position,index in enumerate(indices):
        if done[position]: continue
        values[:,position]=engine.perturbation_rhs(int(index)); values.flush(); done[position]=True
        progress.update(int(done.sum()),k,f"direction {index}")
    return np.asarray(values)


def checkpoint_directory(output, run_id, parameterization):
    path=output/"checkpoints"/run_id/parameterization; path.mkdir(parents=True,exist_ok=True); return path


def matrix_diagnostics(engine,J,meta,parameterization):
    rng=np.random.default_rng(340800+int(meta["M_y"])); errors=[]
    for _ in range(3):
        v=rng.normal(size=len(engine.theta0)); v/=np.linalg.norm(v)
        analytic=engine.jvp(v); errors.append(float(np.linalg.norm(J@v-analytic)/max(np.linalg.norm(analytic),EPS)))
    return {**meta,"map_parameterization":parameterization,"theta_dimension":len(engine.theta0),
        "estimated_J_memory_bytes":int(J.nbytes),"estimated_I_minus_J_memory_bytes":int(J.nbytes),
        "all_entries_finite":bool(np.all(np.isfinite(J))),"missing_columns":int(np.sum(~np.all(np.isfinite(J),axis=0))),
        "maximum_dense_vs_analytic_JVP_relative_error":max(errors),"median_dense_vs_analytic_JVP_relative_error":float(np.median(errors)),
        "dense_JVP_consistency_pass":bool(max(errors)<=1e-10)}


def factor_or_load(directory,J,parameterization):
    path=directory/"lu_factorization.npz"; response_path=directory/("I_minus_J_reduced.npy" if parameterization=="reduced" else "I_minus_J_full.npy")
    diagnostics_path=directory/"lu_factorization_diagnostics.json"
    if path.exists() and response_path.exists() and diagnostics_path.exists():
        item=np.load(path); return np.load(response_path),item["lu"],item["piv"].astype(int),json.loads(diagnostics_path.read_text(encoding="utf8"))
    response,lu,piv,diagnostics=factorize_response_matrix(J); atomic_array(response_path,response)
    temporary=Path(str(path)+".tmp")
    with temporary.open("wb") as handle: np.savez(handle,lu=lu,piv=piv)
    os.replace(temporary,path); atomic_json(diagnostics,diagnostics_path)
    return response,lu,piv,diagnostics


def process_network(config, output, context, parameterization, progress):
    meta,payload,edges=context["meta"],context["payload"],context["edges"]
    reduced=parameterization=="reduced"; engine=HybridFixedPointTangentMap(payload["model"],payload["data"]["y"],payload["data"]["u"],reduced=reduced)
    directory=checkpoint_directory(output,meta["run_id"],parameterization); workers=choose_workers(config,output,context,engine,parameterization)
    completed_path=directory/("J_reduced_completed.npy" if reduced else "J_full_completed.npy")
    completed=int(np.load(completed_path).sum()) if completed_path.exists() else 0
    progress.start(f"{meta['validation_stage']} My={meta['M_y']} {parameterization} J",len(engine.theta0),completed)
    J,runtime_path,segment_path=assemble_exact_jacobian(engine,directory,workers,config["inner_threads"],
        config["jacobian_checkpoint_columns"],lambda done,total,label:progress.update(done,total,label))
    runtime=read_csv(runtime_path); runtime["run_id"]=meta["run_id"]; runtime["M_y"]=meta["M_y"]; runtime["map_parameterization"]=parameterization
    append_rows(output/"jacobian_column_runtime.csv",runtime.to_dict("records"),["run_id","map_parameterization","column_index"])
    segments=read_csv(segment_path); assembly_seconds=float(segments.runtime_seconds.sum())
    diag=matrix_diagnostics(engine,J,meta,parameterization); append_rows(output/"dense_jacobian_diagnostics.csv",[diag],["run_id","map_parameterization"])
    response,lu,piv,lu_diag=factor_or_load(directory,J,parameterization); lu_diag={**meta,"map_parameterization":parameterization,**lu_diag}
    append_rows(output/"lu_factorization_diagnostics.csv",[lu_diag],["run_id","map_parameterization"])
    append_rows(output/"jacobian_assembly_summary.csv",[{**meta,"map_parameterization":parameterization,
        "theta_dimension":len(engine.theta0),"jacobian_workers":workers,"jacobian_assembly_seconds":assembly_seconds,
        "mean_column_seconds":runtime.runtime_seconds.mean(),"median_column_seconds":runtime.runtime_seconds.median(),
        "J_memory_bytes":J.nbytes,"I_minus_J_memory_bytes":response.nbytes}], ["run_id","map_parameterization"])
    indices=[global_index(int(row.target),lag,int(row.source),engine.layout.n_states,2) for row in edges.itertuples() for lag in range(2)]
    B=build_rhs(engine,indices,directory/"B_validation.npy",progress,f"My={meta['M_y']} {parameterization} RHS")
    X,solve_diag,residuals=solve_multiple_rhs(response,lu,piv,B); atomic_array(directory/"X_validation_direct.npy",X)
    append_rows(output/"multi_rhs_runtime.csv",[{**meta,"map_parameterization":parameterization,"rhs_set":"validation_24",**solve_diag}],
                ["run_id","map_parameterization","rhs_set"])
    residual_rows=[{**meta,"map_parameterization":parameterization,"direction_index":index,"relative_residual":float(residual),
                    "residual_pass":bool(residual<=config["direct_residual_target"])} for index,residual in zip(indices,residuals)]
    append_rows(output/"direct_solve_residuals.csv",residual_rows,["run_id","map_parameterization","direction_index"])
    result_rows=[]
    for edge_position,row in enumerate(edges.itertuples()):
        target,source=int(row.target),int(row.source); selected=[global_index(target,lag,source,engine.layout.n_states,2) for lag in range(2)]
        raw=X[np.ix_(selected,[2*edge_position,2*edge_position+1])]; covariance,projection=ug.ua.safe_covariance(raw)
        result_rows.append({**meta,"method":"D1_reduced_dense_LU" if reduced else "D0_full_dense_LU",
            "map_parameterization":parameterization,"target":target,"source":source,
            **ug.covariance_columns(covariance),**projection,
            "direct_residual_max":float(max(residuals[2*edge_position:2*edge_position+2]))})
    destination="reduced_direct_results.csv" if reduced else "direct_full_results.csv"
    append_rows(output/destination,result_rows,["run_id","method","target","source"])
    gmres_rows=[]; block_rows=[]
    if not reduced:
        block_preconditioner=None
        if config["run_block_jacobi"]:
            L=engine.layout; blocks=[("A",L.A_slice),("alpha",L.alpha_slice),("B",L.B_slice),("Q",L.Q_slice)]
            started=time.perf_counter(); block_preconditioner,_=block_jacobi_preconditioner(response,blocks); preconditioner_seconds=time.perf_counter()-started
        else: preconditioner_seconds=np.nan
        replay=np.zeros_like(X)
        progress.start(f"My={meta['M_y']} GMRES replay",len(indices),0)
        for position,(index,b) in enumerate(zip(indices,B.T)):
            direct=X[:,position]
            for pname,preconditioner in (("P0_identity",None),("P1_block_jacobi",block_preconditioner)):
                if pname.startswith("P1") and preconditioner is None: continue
                value,gdiag=explicit_gmres(response,b,config["gmres_rtol"],config["gmres_restart"],config["gmres_total_iterations"],preconditioner)
                error=float(np.linalg.norm(value-direct)/max(np.linalg.norm(direct),EPS))
                block_rows.append({**meta,"direction_index":index,"preconditioner":pname,
                    "preconditioner_setup_seconds":preconditioner_seconds if pname.startswith("P1") else 0.,
                    "response_relative_error_vs_direct":error,**{k:v for k,v in gdiag.items() if k!="residual_history"},
                    "residual_history":json.dumps(gdiag["residual_history"])})
                if pname=="P0_identity": replay[:,position]=value
            progress.update(position+1,len(indices),f"direction {index}")
        atomic_array(directory/"X_validation_gmres34ug_replay.npy",replay)
        append_rows(output/"block_jacobi_gmres.csv",block_rows,["run_id","direction_index","preconditioner"])
        old_diag=context["old_diag"]
        for position,index in enumerate(indices):
            oldrow=(old_diag[old_diag.direction_index.eq(index)]
                    if len(old_diag) and "direction_index" in old_diag else pd.DataFrame())
            gmres_rows.append({**meta,"direction_index":index,
                "response_relative_error":float(np.linalg.norm(replay[:,position]-X[:,position])/max(np.linalg.norm(X[:,position]),EPS)),
                "replayed_GMRES_residual":block_rows[2*position if block_preconditioner is not None else position]["relative_residual"],
                "old_34UG_GMRES_residual":float(oldrow.iloc[-1].relative_residual) if len(oldrow) else np.nan,
                "old_34UG_GMRES_iterations":int(oldrow.iloc[-1].iterations) if len(oldrow) else np.nan,
                "replay_semantics":"34UG settings with assembled exact dense operator"})
        append_rows(output/"gmres_vs_direct_response.csv",gmres_rows,["run_id","direction_index"])
    if not reduced and config["run_all_A_rhs_benchmark"]:
        all_indices=list(range(engine.layout.A_slice.stop)); B_all=build_rhs(engine,all_indices,directory/"B_all_A.npy",progress,f"My={meta['M_y']} all-A RHS")
        started=time.perf_counter(); X_all=lu_solve((lu,piv),B_all,check_finite=False); all_seconds=time.perf_counter()-started
        atomic_array(directory/"X_all_A_direct.npy",X_all)
        all_res=np.linalg.norm(response@X_all-B_all,axis=0)/np.maximum(np.linalg.norm(B_all,axis=0),EPS)
        append_rows(output/"multi_rhs_runtime.csv",[{**meta,"map_parameterization":parameterization,"rhs_set":"all_800_A",
            "n_rhs":len(all_indices),"batched_seconds":all_seconds,"seconds_per_rhs":all_seconds/len(all_indices),
            "maximum_relative_residual":float(all_res.max()),"memory_bytes_B_plus_X":int(B_all.nbytes+X_all.nbytes)}],
            ["run_id","map_parameterization","rhs_set"])
    append_rows(output/"runtime_summary.csv",[{**meta,"map_parameterization":parameterization,"baseline_fit_seconds":payload.get("baseline_fit_seconds",np.nan),
        "jacobian_assembly_seconds":assembly_seconds,"lu_factorization_seconds":lu_diag["factorization_seconds"],
        "validation_batched_RHS_seconds":solve_diag["batched_seconds"],"theta_dimension":len(engine.theta0)}],
        ["run_id","map_parameterization"])
    return {"engine":engine,"J":J,"response":response,"lu":lu,"piv":piv,"B":B,"X":X,
            "indices":indices,"results":pd.DataFrame(result_rows),"assembly_seconds":assembly_seconds,
            "factorization_seconds":lu_diag["factorization_seconds"],"solve_diag":solve_diag}


def covariance(row):
    return np.asarray([[row.cov_00,row.cov_01],[row.cov_10,row.cov_11]],float)


def fidelity_for(output,contexts,direct_file,method):
    direct=read_csv(output/direct_file); rows=[]; decomposition=[]
    if not len(direct) or not all(name in direct for name in ("run_id","method")):
        return pd.DataFrame(),pd.DataFrame()
    for context in contexts:
        meta,edges,legacy,old=context["meta"],context["edges"],context["legacy"],context["old"]
        current=direct[(direct.run_id.eq(meta["run_id"]))&direct.method.eq(method)]
        for item in current.itertuples(index=False):
            reference=legacy[(legacy.target.eq(item.target))&legacy.source.eq(item.source)]
            edge=edges[(edges.target.eq(item.target))&edges.source.eq(item.source)]
            if not len(reference) or not len(edge): continue
            L,_=ug.ua.safe_covariance(covariance(reference.iloc[-1])); D,_=ug.ua.safe_covariance(covariance(item))
            oldrow=(old[(old.target.eq(item.target))&old.source.eq(item.source)]
                    if len(old) and all(name in old for name in ("target","source")) else pd.DataFrame())
            G,_=ug.ua.safe_covariance(covariance(oldrow.iloc[-1])) if len(oldrow) else (np.full((2,2),np.nan),{})
            center=edge[["center_lag1","center_lag2"]].iloc[-1].to_numpy(float); truth=edge[["truth_lag1","truth_lag2"]].iloc[-1].to_numpy(float)
            def stats(C):
                delta=center-truth; d2=float(delta@np.linalg.pinv(C)@delta); return d2,bool(d2<=CHI2_95_DF2)
            dL,cL=stats(L); dD,cD=stats(D); lt=float(edge.iloc[-1].louis_cov_trace)
            signL,logL=np.linalg.slogdet(L); signD,logD=np.linalg.slogdet(D); eig=np.linalg.eigvalsh(D)
            row={**meta,"method":method,"target":item.target,"source":item.source,
                "relative_frobenius_error":float(np.linalg.norm(D-L)/max(np.linalg.norm(L),EPS)),
                "trace_relative_error":float(abs(np.trace(D)-np.trace(L))/max(abs(np.trace(L)),EPS)),
                "absolute_logdet_error":float(abs(logD-logL)) if signL>0 and signD>0 else np.nan,
                "ellipse_area_relative_error":float(abs(np.sqrt(np.linalg.det(D))-np.sqrt(np.linalg.det(L)))/max(np.sqrt(np.linalg.det(L)),EPS)),
                "legacy_trace":float(np.trace(L)),"direct_trace":float(np.trace(D)),
                "legacy_relative_inflation":float((np.trace(L)-lt)/max(lt,EPS)),
                "direct_relative_inflation":float((np.trace(D)-lt)/max(lt,EPS)),
                "stageB_relative_inflation_error":float(abs(np.trace(D)-np.trace(L))/max(lt,EPS)),
                "legacy_D2":dL,"direct_D2":dD,"Mahalanobis_D2_difference":abs(dD-dL),
                "legacy_covered":cL,"direct_covered":cD,"coverage_decision_agreement":bool(cL==cD),
                "finite_covariance":bool(np.all(np.isfinite(D))),"symmetric_covariance":bool(np.allclose(D,D.T)),
                "minimum_eigenvalue":float(eig.min()),"PSD":bool(eig.min()>=-1e-12),"covariance_condition_number":float(np.linalg.cond(D))}
            rows.append(row)
            if len(oldrow):
                decomposition.append({**meta,"target":item.target,"source":item.source,
                    "E_solver":float(np.linalg.norm(G-D)/max(np.linalg.norm(D),EPS)),
                    "E_implicit":row["relative_frobenius_error"],
                    "E_old":float(np.linalg.norm(G-L)/max(np.linalg.norm(L),EPS))})
    return pd.DataFrame(rows),pd.DataFrame(decomposition)


def summarize_fidelity(fidelity):
    rows=[]
    if not len(fidelity): return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for (method,my),g in fidelity.groupby(["method","M_y"]):
        spearman=spearmanr(g.legacy_relative_inflation,g.direct_relative_inflation).statistic if len(g)>1 else np.nan
        pearson=pearsonr(g.legacy_relative_inflation,g.direct_relative_inflation).statistic if len(g)>1 else np.nan
        row={"method":method,"M_y":my,"n_groups":len(g),"mean_relative_frobenius_error":g.relative_frobenius_error.mean(),
            "median_relative_frobenius_error":g.relative_frobenius_error.median(),"p90_relative_frobenius_error":g.relative_frobenius_error.quantile(.9),
            "maximum_relative_frobenius_error":g.relative_frobenius_error.max(),"mean_trace_relative_error":g.trace_relative_error.mean(),
            "median_trace_relative_error":g.trace_relative_error.median(),"p90_trace_relative_error":g.trace_relative_error.quantile(.9),
            "maximum_trace_relative_error":g.trace_relative_error.max(),"finite_fraction":g.finite_covariance.mean(),"PSD_fraction":g.PSD.mean(),
            "coverage_agreement":g.coverage_decision_agreement.mean(),"inflation_Spearman":spearman,"inflation_Pearson":pearson}
        row["fidelity_gate_pass"]=bool(row["finite_fraction"]==1 and row["PSD_fraction"]==1 and row["median_relative_frobenius_error"]<=.05 and row["p90_relative_frobenius_error"]<=.10 and row["median_trace_relative_error"]<=.05 and spearman>=.95 and row["coverage_agreement"]>=.95)
        rows.append(row)
    summary=pd.DataFrame(rows)
    inflation=summary[["method","M_y","n_groups","inflation_Pearson","inflation_Spearman"]].copy()
    coverage=summary[["method","M_y","n_groups","coverage_agreement"]].copy()
    return summary,inflation,coverage


def epsilon_diagnostic(config,output,context,direct,worst,progress):
    existing=read_csv(output/"legacy_epsilon_sensitivity.csv"); model=context["payload"]["model"]; data=context["payload"]["data"]
    rows=[]
    settings=[(eps,config["legacy_iteration_grid"][0]) for eps in config["legacy_epsilon_grid"]]
    progress.start(f"My={context['meta']['M_y']} epsilon diagnostic",len(worst)*len(settings),0)
    for edge in worst.itertuples(index=False):
        target,source=int(edge.target),int(edge.source); directrow=direct[(direct.run_id.eq(context["meta"]["run_id"]))&direct.target.eq(target)&direct.source.eq(source)].iloc[-1]; D=covariance(directrow)
        edge_rows=[]
        for epsilon,max_iter in settings:
            old=existing[(existing.run_id.eq(context["meta"]["run_id"]))&existing.target.eq(target)&existing.source.eq(source)&existing.epsilon.eq(epsilon)&existing.perturbed_max_iter.eq(max_iter)] if len(existing) else pd.DataFrame()
            if len(old): edge_rows.append(old.iloc[-1].to_dict()); progress.update(detail="resume"); continue
            columns=[]; fits=[]; started=time.perf_counter()
            for lag in range(2):
                index=global_index(target,lag,source,model.n_states,2)
                column,plus,minus=sensitivity_column(model,data["y"],data["u"],index,epsilon,max_iter,1e-4,reestimate_B=True,reestimate_Q=True)
                fits.extend([plus,minus])
                columns.append(column[[global_index(target,k,source,model.n_states,2) for k in range(2)]])
            C,_=ug.ua.safe_covariance(np.column_stack(columns)); row={**context["meta"],"target":target,"source":source,
                "epsilon":epsilon,"perturbed_max_iter":max_iter,"runtime_seconds":time.perf_counter()-started,
                "all_perturbed_fits_converged":bool(all(fit.converged for fit in fits)),
                "maximum_perturbed_iterations":int(max(fit.n_iter for fit in fits)),
                "relative_error_vs_direct":float(np.linalg.norm(C-D)/max(np.linalg.norm(D),EPS)),**ug.covariance_columns(C)}
            rows.append(row); edge_rows.append(row); append_rows(output/"legacy_epsilon_sensitivity.csv",[row],["run_id","target","source","epsilon","perturbed_max_iter"]); progress.update()
        values=pd.DataFrame(edge_rows).sort_values("epsilon")
        if len(values)>=2:
            first,last=covariance(values.iloc[0]),covariance(values.iloc[-1]); material=np.linalg.norm(first-last)/max(np.linalg.norm(D),EPS)>.05
            if material:
                for max_iter in config["legacy_iteration_grid"][1:]:
                    epsilon=1e-4
                    old=read_csv(output/"legacy_epsilon_sensitivity.csv")
                    done=old[(old.run_id.eq(context["meta"]["run_id"]))&old.target.eq(target)&old.source.eq(source)&old.epsilon.eq(epsilon)&old.perturbed_max_iter.eq(max_iter)] if len(old) else pd.DataFrame()
                    if len(done): continue
                    columns=[]; fits=[]; started=time.perf_counter()
                    for lag in range(2):
                        index=global_index(target,lag,source,model.n_states,2)
                        column,plus,minus=sensitivity_column(model,data["y"],data["u"],index,epsilon,max_iter,1e-4,reestimate_B=True,reestimate_Q=True)
                        fits.extend([plus,minus])
                        columns.append(column[[global_index(target,k,source,model.n_states,2) for k in range(2)]])
                    C,_=ug.ua.safe_covariance(np.column_stack(columns)); row={**context["meta"],"target":target,"source":source,
                        "epsilon":epsilon,"perturbed_max_iter":max_iter,"runtime_seconds":time.perf_counter()-started,
                        "all_perturbed_fits_converged":bool(all(fit.converged for fit in fits)),
                        "maximum_perturbed_iterations":int(max(fit.n_iter for fit in fits)),
                        "relative_error_vs_direct":float(np.linalg.norm(C-D)/max(np.linalg.norm(D),EPS)),**ug.covariance_columns(C)}
                    append_rows(output/"legacy_epsilon_sensitivity.csv",[row],["run_id","target","source","epsilon","perturbed_max_iter"])


def runtime_projection(config,output,contexts):
    runtime=read_csv(output/"runtime_summary.csv"); multi=read_csv(output/"multi_rhs_runtime.csv"); rows=[]
    source_runtime=read_csv(SOURCE_34UG/"legacy_stageb_reference.csv") if not config["smoke_test"] else pd.concat([c["legacy"] for c in contexts],ignore_index=True)
    for context in contexts:
        meta=context["meta"]; r=runtime[(runtime.run_id.eq(meta["run_id"]))&runtime.map_parameterization.eq("full")]
        m=multi[(multi.run_id.eq(meta["run_id"]))&multi.map_parameterization.eq("full")&multi.rhs_set.eq("validation_24")]
        legacy=source_runtime[source_runtime.run_id.eq(meta["run_id"])]
        if not len(r) or not len(m) or not len(legacy): continue
        setup=float(r.iloc[-1].jacobian_assembly_seconds+r.iloc[-1].lu_factorization_seconds); per_rhs=float(m.iloc[-1].batched_seconds)/int(m.iloc[-1].n_rhs); legacy_group=float(legacy.runtime_seconds.median())
        break_even=np.inf if legacy_group<=2*per_rhs else int(np.ceil(setup/(legacy_group-2*per_rhs)))
        for N in (1,10,12,40,63,90,380):
            direct=setup+2*N*per_rhs; old=N*legacy_group
            rows.append({**meta,"method":"D0_full_dense_LU","N_groups":N,"setup_seconds":setup,
                "solve_seconds":2*N*per_rhs,"projected_runtime_seconds":direct,"legacy_projected_runtime_seconds":old,
                "speedup":old/max(direct,EPS),"break_even_number_groups":break_even})
    atomic_csv(pd.DataFrame(rows),str(output/"amortized_runtime_projection.csv")); return pd.DataFrame(rows)


def make_plots(output,fidelity,decomposition,projection):
    plots=output/"plots"
    def save(name,draw):
        fig,ax=plt.subplots(); draw(ax); fig.tight_layout(); fig.savefig(plots/name); plt.close(fig)
    save("01_old_gmres_vs_direct_covariance_error.png",lambda ax:(decomposition.boxplot(column="E_solver",by="M_y",ax=ax) if len(decomposition) else None,ax.set_ylabel("GMRES vs direct error")))
    save("02_direct_vs_legacy_covariance_error.png",lambda ax:(fidelity.boxplot(column="relative_frobenius_error",by="M_y",ax=ax) if len(fidelity) else None,ax.set_ylabel("direct vs legacy error")))
    gm=read_csv(output/"gmres_vs_direct_response.csv")
    def solver_plot(ax):
        valid=(gm[np.isfinite(gm.old_34UG_GMRES_residual)&np.isfinite(gm.response_relative_error)&
                  (gm.old_34UG_GMRES_residual>0)&(gm.response_relative_error>0)] if len(gm) else pd.DataFrame())
        if len(valid):
            ax.scatter(valid.old_34UG_GMRES_residual,valid.response_relative_error); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel="GMRES residual",ylabel="response error")
    save("03_solver_error_vs_gmres_residual.png",solver_plot)
    save("04_My40_vs_My15_fidelity.png",lambda ax:(fidelity.boxplot(column="relative_frobenius_error",by="M_y",ax=ax) if len(fidelity) else None))
    save("05_stageB_inflation.png",lambda ax:(ax.scatter(fidelity.legacy_relative_inflation,fidelity.direct_relative_inflation,c=fidelity.M_y) if len(fidelity) else None,ax.set(xlabel="legacy inflation",ylabel="direct inflation")))
    save("06_covariance_trace.png",lambda ax:(ax.scatter(fidelity.legacy_trace,fidelity.direct_trace,c=fidelity.M_y) if len(fidelity) else None,ax.set(xlabel="legacy covariance trace",ylabel="direct covariance trace")))
    save("07_per_edge_frobenius_error.png",lambda ax:(ax.plot(np.arange(len(fidelity)),fidelity.relative_frobenius_error,"o") if len(fidelity) else None,ax.set_ylabel("relative error")))
    jc=read_csv(output/"jacobian_column_runtime.csv")
    save("08_jacobian_column_runtime.png",lambda ax:(jc.boxplot(column="runtime_seconds",by="M_y",ax=ax) if len(jc) else None,ax.set_ylabel("seconds/column")))
    save("09_setup_cost_vs_groups.png",lambda ax:([ax.plot(g.N_groups,g.projected_runtime_seconds,label=f"My={my}") for my,g in projection.groupby("M_y")] if len(projection) else None,ax.set(xlabel="groups",ylabel="seconds"),ax.legend() if len(projection) else None))
    save("10_amortized_speedup.png",lambda ax:([ax.plot(g.N_groups,g.speedup,marker="o",label=f"My={my}") for my,g in projection.groupby("M_y")] if len(projection) else None,ax.set(xlabel="groups",ylabel="speedup"),ax.legend() if len(projection) else None))
    bj=read_csv(output/"block_jacobi_gmres.csv")
    save("11_identity_vs_block_jacobi.png",lambda ax:(bj.groupby(["M_y","preconditioner"]).iterations.median().unstack().plot.bar(ax=ax) if len(bj) else None,ax.set_ylabel("median iterations")))
    eps=read_csv(output/"legacy_epsilon_sensitivity.csv")
    if len(eps): save("12_legacy_epsilon_sensitivity.png",lambda ax:([ax.plot(g.epsilon,g.relative_error_vs_direct,marker="o",label=f"My={my}") for my,g in eps.groupby("M_y")] ,ax.set(xlabel="epsilon",ylabel="error vs direct",xscale="log")))
    red=read_csv(output/"reduced_direct_results.csv")
    if len(red): save("13_full_vs_reduced_direct.png",lambda ax:(fidelity.boxplot(column="relative_frobenius_error",by="method",ax=ax),ax.set_ylabel("error")))


def finalize(config,output,contexts,stageB_run=False,stageB_pass=False):
    full_fidelity,decomposition=fidelity_for(output,contexts,"direct_full_results.csv","D0_full_dense_LU")
    reduced_fidelity,_=fidelity_for(output,contexts,"reduced_direct_results.csv","D1_reduced_dense_LU")
    fidelity=pd.concat([full_fidelity,reduced_fidelity],ignore_index=True)
    summary,inflation,coverage=summarize_fidelity(fidelity)
    atomic_csv(fidelity,str(output/"covariance_fidelity.csv")); atomic_csv(decomposition,str(output/"solver_error_decomposition.csv"))
    atomic_csv(summary,str(output/"method_summary.csv")); atomic_csv(inflation,str(output/"stageb_inflation_agreement.csv")); atomic_csv(coverage,str(output/"coverage_agreement.csv"))
    projection=runtime_projection(config,output,contexts); residuals=read_csv(output/"direct_solve_residuals.csv")
    assembly=read_csv(output/"jacobian_assembly_summary.csv"); lu=read_csv(output/"lu_factorization_diagnostics.csv"); multi=read_csv(output/"multi_rhs_runtime.csv")
    gm=read_csv(output/"gmres_vs_direct_response.csv"); eps=read_csv(output/"legacy_epsilon_sensitivity.csv")
    def s(my,key,default=np.nan):
        q=summary[(summary.method.eq("D0_full_dense_LU"))&summary.M_y.eq(my)]; return q.iloc[-1][key] if len(q) else default
    def reduced_s(my,key,default=np.nan):
        q=summary[(summary.method.eq("D1_reduced_dense_LU"))&summary.M_y.eq(my)]; return q.iloc[-1][key] if len(q) else default
    def rpass(my):
        q=residuals[(residuals.M_y.eq(my))&residuals.map_parameterization.eq("full")]; return bool(len(q) and q.residual_pass.astype(bool).all())
    def proj(N,key="speedup"):
        q=projection[projection.N_groups.eq(N)]; return q[key].mean() if len(q) else np.nan
    all_stageA=all(bool(s(my,"fidelity_gate_pass",False)) and rpass(my) for my in config["M_y_values"])
    stageA_full_residuals=(residuals[(residuals.validation_stage.eq("stageA"))&residuals.map_parameterization.eq("full")]
                           if len(residuals) and "validation_stage" in residuals else pd.DataFrame())
    dec15=(decomposition[decomposition.M_y.eq(15)]
           if len(decomposition) and "M_y" in decomposition else pd.DataFrame())
    my15_solver=float(dec15.E_solver.median()) if len(dec15) else np.nan
    my15_implicit=float(dec15.E_implicit.median()) if len(dec15) else np.nan
    block=read_csv(output/"block_jacobi_gmres.csv")
    block_improves=(bool(block.groupby("preconditioner").iterations.median().get("P1_block_jacobi",np.inf)<
                         block.groupby("preconditioner").iterations.median().get("P0_identity",np.inf)) if len(block) else np.nan)
    epsilon_converges=np.nan
    if len(eps):
        comparison=[]
        for _,group in eps[eps.perturbed_max_iter.eq(30)].groupby(["run_id","target","source"]):
            small=group.loc[group.epsilon.idxmin()].relative_error_vs_direct; large=group.loc[group.epsilon.idxmax()].relative_error_vs_direct
            comparison.append(float(small)<float(large))
        epsilon_converges=bool(np.mean(comparison)>=.5) if comparison else np.nan
    decision={"full_dense_jacobian_feasible":bool(len(assembly) and assembly.theta_dimension.max()<=1260),
        "My40_direct_residual_pass":rpass(40),"My15_direct_residual_pass":rpass(15),
        "My40_direct_vs_legacy_median_frob":s(40,"median_relative_frobenius_error"),"My40_direct_vs_legacy_p90_frob":s(40,"p90_relative_frobenius_error"),
        "My40_inflation_spearman":s(40,"inflation_Spearman"),"My40_coverage_agreement":s(40,"coverage_agreement"),
        "My15_direct_vs_legacy_median_frob":s(15,"median_relative_frobenius_error"),"My15_direct_vs_legacy_p90_frob":s(15,"p90_relative_frobenius_error"),
        "My15_inflation_spearman":s(15,"inflation_Spearman"),"My15_coverage_agreement":s(15,"coverage_agreement"),
        "My40_old_gmres_vs_direct_error":gm[gm.M_y.eq(40)].response_relative_error.median() if len(gm) else np.nan,
        "My15_old_gmres_vs_direct_error":my15_solver,
        "My15_failure_was_solver_dominated":bool(np.isfinite(my15_solver) and np.isfinite(my15_implicit) and my15_solver>my15_implicit),
        "My15_failure_was_implicit_legacy_mismatch":bool(np.isfinite(my15_implicit) and my15_implicit>.05 and (not np.isfinite(my15_solver) or my15_solver<=my15_implicit)),
        "jacobian_assembly_seconds":assembly[assembly.map_parameterization.eq("full")].jacobian_assembly_seconds.sum() if len(assembly) else np.nan,
        "lu_factorization_seconds":lu[lu.map_parameterization.eq("full")].factorization_seconds.sum() if len(lu) else np.nan,
        "batched_RHS_seconds":multi[(multi.map_parameterization.eq("full"))&multi.rhs_set.eq("validation_24")].batched_seconds.sum() if len(multi) else np.nan,
        "break_even_number_groups":projection.break_even_number_groups.max() if len(projection) else np.nan,
        **{f"speedup_{N}_groups":proj(N) for N in (12,40,63,90,380)},
        "block_jacobi_improves_gmres":block_improves,"reduced_path_tested":bool(len(reduced_fidelity)),
        "reduced_path_valid":bool(len(reduced_fidelity) and all(reduced_s(my,"fidelity_gate_pass",False) for my in config["M_y_values"])),
        "legacy_epsilon_sensitivity_needed":bool(not all_stageA and len(stageA_full_residuals) and
            stageA_full_residuals.relative_residual.max()<=config["direct_residual_target"]),
        "legacy_converges_toward_implicit":epsilon_converges,
        "stageB_confirmation_run":bool(stageB_run),"stageB_confirmation_pass":bool(stageB_pass),
        "preferred_method":"D0_full_dense_LU" if all_stageA and (not stageB_run or stageB_pass) else "legacy_stageB",
        "ready_to_replace_legacy_stageB":bool(all_stageA and stageB_run and stageB_pass),
        "recommended_next_experiment":"freeze full explicit factorized LRVB" if all_stageA and stageB_pass else ("study infinitesimal implicit versus finite legacy perturbation" if not all_stageA else "run/complete independent confirmation")}
    atomic_csv(pd.DataFrame([decision]),str(output/"decision_summary.csv"))
    for name in REQUIRED:
        if not (output/name).exists(): atomic_csv(pd.DataFrame(),str(output/name))
    make_plots(output,fidelity,decomposition,projection)
    atomic_json({"completed":True,"ready_to_replace_legacy_stageB":decision["ready_to_replace_legacy_stageB"],"completed_at":time.time()},output/"_COMPLETED.json")
    failed=output/"_FAILED.json"
    if failed.exists(): failed.unlink()


def main():
    args=parse_args(); config=configuration(args); output,completed=initialize(config); progress=Progress()
    if completed: progress.start("already complete",1,1); return
    try:
        validate_source_config(config)
        if not config["smoke_test"]: source_manifest(output)
        contexts=[]
        for my in config["M_y_values"]:
            contexts.append(smoke_context(config,output,my,progress) if config["smoke_test"] else context_from_34ug(config,output,my))
        manifests=[]
        for context in contexts:
            meta=context["meta"]; n=len(HybridFixedPointTangentMap(context["payload"]["model"],context["payload"]["data"]["y"],context["payload"]["data"]["u"]).theta0)
            expected_jvp=np.nan
            if not config["smoke_test"]:
                old_operator=read_csv(SOURCE_34UG/"operator_diagnostics.csv")
                q=old_operator[(old_operator.run_id.eq(meta["run_id"]))&old_operator.map_parameterization.eq("full")]
                if len(q): expected_jvp=float(q.iloc[-1].mean_time_per_jvp)
            manifests.append({**meta,"source":context["source"],"baseline_checkpoint":context["baseline_path"],"validation_groups":len(context["edges"]),
                "theta_full_dimension":n,"estimated_J_memory_bytes":8*n*n,"estimated_I_minus_J_memory_bytes":8*n*n,
                "measured_34UG_seconds_per_JVP":expected_jvp,
                "estimated_serial_jacobian_assembly_seconds":expected_jvp*n if np.isfinite(expected_jvp) else np.nan,
                "status":"ready"})
        atomic_csv(pd.DataFrame(manifests),str(output/"network_manifest.csv"))
        for context in contexts: process_network(config,output,context,"full",progress)
        fidelity,decomposition=fidelity_for(output,contexts,"direct_full_results.csv","D0_full_dense_LU")
        summary,_,_=summarize_fidelity(fidelity)
        residuals=read_csv(output/"direct_solve_residuals.csv")
        def fidelity_pass(my):
            q=summary[(summary.method.eq("D0_full_dense_LU"))&summary.M_y.eq(my)]
            return bool(len(q) and q.iloc[-1].fidelity_gate_pass)
        def residual_pass(my):
            r=residuals[(residuals.M_y.eq(my))&residuals.map_parameterization.eq("full")]
            return bool(len(r) and r.residual_pass.astype(bool).all())
        def passes(my): return fidelity_pass(my) and residual_pass(my)
        both=all(passes(my) for my in config["M_y_values"])
        if both:
            for context in contexts: process_network(config,output,context,"reduced",progress)
        else:
            direct=read_csv(output/"direct_full_results.csv")
            for context in contexts:
                my=context["meta"]["M_y"]
                # The epsilon diagnostic addresses implicit-versus-finite-
                # perturbation disagreement, never an inaccurate direct solve.
                if fidelity_pass(my) or not residual_pass(my): continue
                f=fidelity[fidelity.run_id.eq(context["meta"]["run_id"])].sort_values("relative_frobenius_error",ascending=False).head(3)
                epsilon_diagnostic(config,output,context,direct,f,progress)
        confirmation=[]; stageB_pass=False
        if both and config["run_stageB_after_gate"]:
            confirmation=[stageB_context(config,output,my,progress) for my in config["M_y_values"]]
            for context in confirmation: process_network(config,output,context,"full",progress)
            cf,_=fidelity_for(output,confirmation,"direct_full_results.csv","D0_full_dense_LU"); cs,_,_=summarize_fidelity(cf)
            confirmation_residuals=read_csv(output/"direct_solve_residuals.csv")
            stageB_pass=all(
                bool(cs[(cs.method.eq("D0_full_dense_LU"))&cs.M_y.eq(my)].iloc[-1].fidelity_gate_pass) and
                bool(len(confirmation_residuals[(confirmation_residuals.validation_stage.eq("stageB"))&
                    confirmation_residuals.M_y.eq(my)&confirmation_residuals.map_parameterization.eq("full")])) and
                bool(confirmation_residuals[(confirmation_residuals.validation_stage.eq("stageB"))&
                    confirmation_residuals.M_y.eq(my)&confirmation_residuals.map_parameterization.eq("full")].residual_pass.astype(bool).all())
                for my in config["M_y_values"])
        finalize(config,output,contexts+confirmation,bool(confirmation),stageB_pass)
        progress.start("complete",1,1)
    except Exception as error:
        atomic_json({"completed":False,"error_type":type(error).__name__,"error_message":str(error),
                     "traceback":traceback.format_exc(),"failed_at":time.time()},output/"_FAILED.json")
        raise


if __name__=="__main__": main()
