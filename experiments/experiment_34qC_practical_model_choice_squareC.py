"""Experiment 34Q-C: staged practical model choice for the square-C pipeline.

Stage 1 screens complete fitted pipelines without Stage-B. Stage 2 performs
selected-direction Stage-B only for eligible C modes. Stage 3 is one T=2000
confirmation of the best mode. All interval centers are A_VB.
"""
import os
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"): os.environ[_k]=os.environ.get("EXPERIMENT_34QC_INNER_THREADS","1")
import json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34n_lrvb_A_covariance as n
import experiments.experiment_34i_static_spectral_varx_gc as spi
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as qb
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,grouped_stats
from experiments.experiment_34b_hybrid_vb_ard_scaleup_fixed_B import network_scores
from src.stats.lrvb_stageB_smoother_feedback import global_index,project_selected_covariance,sensitivity_column
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics

RESULTS_DIR=os.environ.get("EXPERIMENT_34QC_RESULTS_DIR","results/experiment_34qC")
SMOKE_TEST=os.environ.get("EXPERIMENT_34QC_SMOKE","0")=="1"; N_WORKERS=int(os.environ.get("EXPERIMENT_34QC_WORKERS","2")); N_FFBS_SAMPLES=50
VB_MAX_ITER=75;VB_TOL=1e-4;PERTURBED_MAX_ITER=30;PERTURBED_RESCUE_MAX_ITER=50;PERTURBED_TOL=1e-4;MAX_STAGEB_SECONDS_PER_CMODE=8*3600
C_MODES=["identity","mild_mixing","strong_mixing"];N_FREQS=128;BANDS=spi.BANDS;BASE_SEED=5790000
METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_square_C_known_R"
FILES={"run":"run_summary_partial.csv","coeff":"_coeff_checkpoint.csv","group":"_group_checkpoint.csv","covdiag":"covariance_diagnostics_partial.csv","perturb":"stageB_perturbation_diagnostics_partial.csv","activity":"_activity_checkpoint.csv","gating":"_gating_checkpoint.csv","A":"_A_checkpoint.csv","B":"_B_checkpoint.csv","Q":"_Q_checkpoint.csv","latent":"_latent_checkpoint.csv","network":"network_recovery_summary_partial.csv","selected_scores":"_selected_scores_checkpoint.csv","spectral":"spectral_gc_summary_partial.csv","spectral_band":"_spectral_band_checkpoint.csv","deviance":"deviance_diagnostics_partial.csv","selected":"_selected_checkpoint.csv","groupsel":"_groupsel_checkpoint.csv","runtime":"_runtime_checkpoint.csv"}

class Progress:
 def __init__(self,total):self.total,self.done=total,0;self.start=time.perf_counter()
 def update(self,label):
  self.done+=1;f=self.done/max(self.total,1);eta=(time.perf_counter()-self.start)/max(self.done,1)*(self.total-self.done);k=round(30*f);print(f"\rExperiment 34Q-C [{'#'*k}{'-'*(30-k)}] {self.done}/{self.total} {100*f:5.1f}% ETA {eta/60:6.1f}m {label[:42]}",end="\n" if self.done==self.total else "",flush=True)

def meta_for(stage,mode,T,rep,M=20):return {"stage":stage,"config_label":f"{stage}_{mode}_M{M}_T{T}","M":M,"T":T,"C_MODE":mode,"true_network_id":0,"replicate_id":rep,"method":METHOD,"run_id":f"{stage}_{mode}_{T}_{rep}"}

def make_data(mode,T,rep,M=20):
 seed=BASE_SEED+M*100000+C_MODES.index(mode)*1000000+T*10+rep;A,B,_=j.fixed_network(M,BASE_SEED+M*100000);return qb.simulate(A,B,M,T,mode,seed),seed

def fit(data,seed):
 old=qb.VB_MAX_ITER;qb.VB_MAX_ITER=VB_MAX_ITER
 model=qb.fit_estBQ(data,seed+500);qb.VB_MAX_ITER=old;return model

def spectral(model,data,meta):
 oldM=spi.M;spi.M=model.n_states;omega=np.linspace(0,np.pi,N_FREQS)
 spectra,bands,_,_,_=spi.compute_spectral_tables(model.A_mean_matrices_,model.B_matrices,model.Q,omega,data,meta,"estimated_AQ")
 metrics,_=spi.metric_rows(bands);spi.M=oldM;return spectra,metrics

def deviance(model,data,meta):
 rows=[];proxy=data["y"]@np.linalg.pinv(data["C"]).T
 for name,signal in (("oracle_latent",data["x"]),("method_smoothed",model.smoothed_state_mean_),("method_filtered",model.filtered_state_mean_),("pseudo_inverse",proxy)):
  try:
   network,_,scores=network_scores(signal,name,data,meta,False)
   for part in scores:
    for score_type,g in part.groupby("score_type"):
     met,_=j.safe_curve(g.true_link,g.score);rows.append({**meta,"signal_type":name,"score_type":score_type,**met})
  except Exception as error:rows.append({**meta,"signal_type":name,"score_type":"unavailable","status":"skipped","reason":str(error)})
 return pd.DataFrame(rows)

def cheap_worker(task):
 mode,rep=task;meta=meta_for("stage1",mode,1000,rep);started=time.perf_counter();out={k:pd.DataFrame() for k in FILES}
 try:
  data,seed=make_data(mode,1000,rep);b=time.perf_counter();model=fit(data,seed);vbtime=time.perf_counter()-b;n.N_MC_SMOOTHER_SAMPLES=N_FFBS_SAMPLES;l=time.perf_counter();louis=n._louis_covariances(model,data,seed+700);ltime=time.perf_counter()-l
  sp,sm=spectral(model,data,meta);dv=deviance(model,data,meta);A=qb.a_recovery(model,data,meta);B=qa.B_recovery(model,data,meta);Q=qb.q_recovery(model,data,meta);L={**meta,**qb.latent_metrics(model,data)};net=qa.full_network(model,data,meta,louis)
  ordinary=qa.o.global_covariance_block(model.A_row_covariances_);lc=qa.o.global_covariance_block(louis);cd=[]
  for name,cov in (("ordinary_vb",ordinary),("stabilized_louis_eta_0p70_tau_0p90",lc)):cd.append({**meta,"covariance_estimator":name,"number_selected_directions":len(cov),"mean_variance":np.diag(cov).mean(),"median_variance":np.median(np.diag(cov)),"trace_selected":np.trace(cov),"louis_covariance_trace":np.trace(lc),"louis_covariance_diagonal_mean":np.diag(lc).mean(),"louis_covariance_diagonal_median":np.median(np.diag(lc)),"N_FFBS_SAMPLES":N_FFBS_SAMPLES})
  runtime={**meta,"baseline_VB_runtime_seconds":vbtime,"louis_runtime_seconds":ltime,"stageB_total_runtime_seconds":0.,"total_runtime_seconds":time.perf_counter()-started,"run_status":"success"};run={**meta,"run_status":"success",**{k:v for x in (A,B,Q,L) for k,v in x.items() if k not in meta},"total_runtime_seconds":runtime["total_runtime_seconds"]}
  out.update(A=pd.DataFrame([A]),B=pd.DataFrame([B]),Q=pd.DataFrame([Q]),latent=pd.DataFrame([L]),network=net,spectral=sm,spectral_band=sp,deviance=dv,covdiag=pd.DataFrame(cd),runtime=pd.DataFrame([runtime]),run=pd.DataFrame([run]))
 except Exception as error:
  row={**meta,"run_status":"failed","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc(),"total_runtime_seconds":time.perf_counter()-started};out["run"]=pd.DataFrame([row]);out["runtime"]=pd.DataFrame([row])
 return out

def selected_directions(model,data,louis,seed,limit=100):
 M=model.n_states;rng=np.random.default_rng(seed);rows=[];chosen=set()
 def add(t,s,kind):
  if (t,s) in chosen:return
  chosen.add((t,s))
  for lag in range(2):rows.append({"coefficient_global_index":global_index(t,lag,s,M,2),"target":t,"lag":lag+1,"source":s,"coefficient_type":"diagonal" if t==s else ("offdiag_nonzero" if data["mask"][t,s] else "offdiag_zero"),"selected_direction_type":kind})
 for t,s in zip(*np.where(data["mask"])):add(t,s,"true_offdiag_group")
 for t in rng.choice(M,min(8,M),replace=False):add(t,t,"diagonal_group")
 false=[(t,s) for t in range(M) for s in range(M) if t!=s and not data["mask"][t,s]];rng.shuffle(false)
 for t,s in false[:8]:add(t,s,"matched_false_group")
 scored=[]
 for t,s in false:
  cols=[s,M+s];m=np.asarray(model.A_mean_matrices_)[:,t,s];block=louis[t][np.ix_(cols,cols)];scored.append((np.sqrt(max(m@np.linalg.pinv(block)@m,0)),t,s))
 for _,t,s in sorted(scored,reverse=True)[:5]:add(t,s,"high_snr_false_group")
 for _,t,s in sorted(scored)[:5]:add(t,s,"low_snr_false_group")
 return pd.DataFrame(rows).head(limit)

def stageB_worker(task):
 stage,mode,T,rep=task;M=5 if stage=="stage0" else 20;meta=meta_for(stage,mode,T,rep,M);started=time.perf_counter();out={k:pd.DataFrame() for k in FILES}
 try:
  data,seed=make_data(mode,T,rep,M);b=time.perf_counter();model=fit(data,seed);vbtime=time.perf_counter()-b;n.N_MC_SMOOTHER_SAMPLES=N_FFBS_SAMPLES;l=time.perf_counter();louis=n._louis_covariances(model,data,seed+700);ltime=time.perf_counter()-l;sel=selected_directions(model,data,louis,seed+900,20 if stage=="stage0" else 100);indices=sel.coefficient_global_index.astype(int).tolist();total=2*model.n_states**2;columns=np.full((total,len(indices)),np.nan);diag=[];ptimes=[];clock=time.perf_counter()
  for pos,index in enumerate(indices):
   if time.perf_counter()-clock>MAX_STAGEB_SECONDS_PER_CMODE:break
   col,plus,minus=sensitivity_column(model,data["y"],data["u"],index,1e-4,10 if stage=="stage0" else PERTURBED_MAX_ITER,PERTURBED_TOL,reestimate_B=True,reestimate_Q=True);rescue=not(plus.converged and minus.converged)
   if rescue and stage!="stage0":col,plus,minus=sensitivity_column(model,data["y"],data["u"],index,1e-4,PERTURBED_RESCUE_MAX_ITER,PERTURBED_TOL,reestimate_B=True,reestimate_Q=True)
   columns[:,pos]=col;ptimes.extend([plus.runtime_seconds,minus.runtime_seconds]);diag.append({**meta,"perturbation_direction_index":index,"eps":1e-4,"plus_converged":plus.converged,"minus_converged":minus.converged,"plus_n_iter":plus.n_iter,"minus_n_iter":minus.n_iter,"plus_rescue_used":rescue,"minus_rescue_used":rescue,"plus_final_relative_A_change":plus.final_relative_A_change,"minus_final_relative_A_change":minus.final_relative_A_change,"plus_final_relative_B_change":plus.final_relative_B_change,"minus_final_relative_B_change":minus.final_relative_B_change,"plus_final_relative_Q_change":plus.final_relative_Q_change,"minus_final_relative_Q_change":minus.final_relative_Q_change,"plus_loglikelihood":plus.final_loglikelihood,"minus_loglikelihood":minus.final_loglikelihood,"variance_estimate_raw":col[index],"finite_difference_valid":np.all(np.isfinite(col)),"warning_flag":";".join(filter(None,[plus.warning_flag,minus.warning_flag]))})
  valid=np.flatnonzero(np.all(np.isfinite(columns),axis=0));valididx=[indices[x] for x in valid];raw,psd,projection=project_selected_covariance(columns[:,valid],valididx);sel=sel.set_index("coefficient_global_index").loc[valididx].reset_index();ordinary=qa.o.global_covariance_block(model.A_row_covariances_)[np.ix_(valididx,valididx)];lc=qa.o.global_covariance_block(louis)[np.ix_(valididx,valididx)];blocks={"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":lc,"lrvb_stageB_smoother_feedback_raw":raw,"lrvb_stageB_smoother_feedback_psd_projected":psd};records=qa.p.group_records(model,data,sel,{"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":lc,"lrvb_stageB_smoother_feedback":psd});variants=[v for v in qa.covariance_variants(blocks,records) if v["covariance_estimator"] in ("ordinary_vb","stabilized_louis_eta_0p70_tau_0p90","lrvb_stageB_smoother_feedback_raw","lrvb_stageB_smoother_feedback_psd_projected","soft_sparsity_aware_stageB_logistic_c4_tau2p5","hard_active_subspace_stageB_group_snr_louis_tau2p5","global_lambda_stageB_0p75")]
  for v in variants:
   out["coeff"]=pd.concat([out["coeff"],qa.coefficient_rows(model,data,sel,v,meta)],ignore_index=True);g,s=qa.group_rows(model,records,v,meta);out["group"]=pd.concat([out["group"],g],ignore_index=True);out["selected_scores"]=pd.concat([out["selected_scores"],s],ignore_index=True);out["gating"]=pd.concat([out["gating"],pd.DataFrame([qa.gating_row(meta,v,records)])],ignore_index=True);pr=v["projection"];out["covdiag"]=pd.concat([out["covdiag"],pd.DataFrame([{**meta,"covariance_estimator":v["covariance_estimator"],"number_selected_directions":len(sel),"number_negative_variances":pr.get("number_negative_variances",0),"fraction_negative_variances":pr.get("fraction_negative_variances",0.),"number_psd_projections":int(pr.get("psd_projection_used",False)),"fraction_psd_projected":float(pr.get("psd_projection_used",False)),**covariance_diagnostics(v["covariance"],ordinary,lc,sel.coefficient_type)}])],ignore_index=True)
  sp,sm=spectral(model,data,meta);dv=deviance(model,data,meta);A=qb.a_recovery(model,data,meta);B=qa.B_recovery(model,data,meta);Q=qb.q_recovery(model,data,meta);L={**meta,**qb.latent_metrics(model,data)};runtime={**meta,"baseline_VB_runtime_seconds":vbtime,"louis_runtime_seconds":ltime,"stageB_total_runtime_seconds":time.perf_counter()-clock,"average_perturbed_fit_runtime_seconds":np.mean(ptimes),"number_perturbed_fits":len(ptimes),"number_selected_directions":len(sel),"total_runtime_seconds":time.perf_counter()-started,"run_status":"success","stageB_truncated":len(valididx)<len(indices)}
  out.update(selected=pd.DataFrame([{**meta,**r} for r in sel.to_dict("records")]),groupsel=pd.DataFrame([{**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"]} for r in records]),perturb=pd.DataFrame(diag),activity=qa.activity_rows(meta,records,blocks),A=pd.DataFrame([A]),B=pd.DataFrame([B]),Q=pd.DataFrame([Q]),latent=pd.DataFrame([L]),network=qa.full_network(model,data,meta,louis),spectral=sm,spectral_band=sp,deviance=dv,runtime=pd.DataFrame([runtime]),run=pd.DataFrame([{**meta,"run_status":"success","total_runtime_seconds":runtime["total_runtime_seconds"]}]))
 except Exception as error:
  row={**meta,"run_status":"failed","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc(),"total_runtime_seconds":time.perf_counter()-started};out["run"]=pd.DataFrame([row]);out["runtime"]=pd.DataFrame([row])
 return out

def combine(t):return {k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in t.items()}
def cal_summary(f):
 keys=["stage","config_label","M","T","C_MODE","covariance_estimator","coefficient_type"]
 return f.groupby(keys,dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),median_signed_error=("signed_error","median"),mean_abs_error=("abs_error","mean"),median_abs_error=("abs_error","median"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),median_posterior_sd=("posterior_sd","median"),mean_interval_width_95=("interval_width_95","mean"),median_interval_width_95=("interval_width_95","median"),mean_standardized_error=("standardized_error","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),median_abs_standardized_error=("standardized_error",lambda x:x.abs().median()),n_coefficients=("standardized_error","size")).reset_index()
def group_summary(f):return f.groupby(["stage","config_label","M","T","C_MODE","covariance_estimator","edge_group_type"],dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("D2","size")).reset_index()
def model_choice(fr):
 rows=[]
 for (stage,mode,T),runs in fr["run"].groupby(["stage","C_MODE","T"]):
  def subset(name):
   f=fr[name]
   return f.loc[(f["stage"]==stage)&(f["C_MODE"]==mode)&(f["T"]==T)] if len(f) and {"stage","C_MODE","T"}.issubset(f.columns) else pd.DataFrame()
  net=subset("network");sp=subset("spectral");sp=sp.loc[sp.score_type=="transfer_spectral_gc_diagQ"] if len(sp) and "score_type" in sp else pd.DataFrame();best=sp.loc[sp.AUPRC.idxmax()] if len(sp) and sp.AUPRC.notna().any() else {};get=lambda name:net.loc[net.score_type==name,"AUPRC"].mean() if len(net) else np.nan;r=runs.mean(numeric_only=True)
  for name in ("A","B","Q","latent","runtime"):
   part=subset(name)
   if len(part):r=pd.concat([r,part.mean(numeric_only=True)]).groupby(level=0).last()
  cal=cal_summary(fr["coeff"]) if len(fr["coeff"]) else pd.DataFrame();soft=cal.loc[(cal.stage==stage)&(cal.C_MODE==mode)&(cal["T"]==T)&(cal.covariance_estimator=="soft_sparsity_aware_stageB_logistic_c4_tau2p5")] if len(cal) else pd.DataFrame();cov=lambda typ:soft.loc[soft.coefficient_type==typ,"empirical_coverage_95"].mean() if len(soft) else np.nan;close=lambda x,target,scale:max(0,1-abs(x-target)/scale) if np.isfinite(x) else np.nan;vals=[best.get("AUPRC",np.nan),best.get("TPR_at_FPR_0p05",np.nan),get("stabilized_louis_group_snr"),get("model_A_group_norm"),r.get("smoothed_state_correlation",np.nan),close(cov("offdiag_nonzero"),.95,.95),close(cov("offdiag_zero"),.95,.95),r.get("B_correlation_hat_true",np.nan),close(r.get("Q_trace_ratio_hat_to_true",np.nan),1,1)];w=np.array([.2,.15,.15,.1,.1,.1,.1,.05,.05]);ok=np.isfinite(vals);score=np.dot(w[ok]/w[ok].sum(),np.asarray(vals)[ok])-.05*r.get("total_runtime_seconds",0)/max(fr["run"].total_runtime_seconds.max(),1);rows.append({"stage":stage,"C_MODE":mode,"T":T,"spectral_best_band":best.get("band_name",np.nan),"spectral_best_band_AUPRC":vals[0],"spectral_best_band_TPR_at_FPR_0p05":vals[1],"full_network_AUPRC_stabilized_louis_group_snr":vals[2],"full_network_AUPRC_model_A_group_norm":vals[3],"active_coverage":cov("offdiag_nonzero"),"zero_coverage":cov("offdiag_zero"),"model_choice_score":score,**r.to_dict()})
 return pd.DataFrame(rows)
def save(tables):
 fr=combine(tables)
 for k,v in FILES.items():atomic_csv(fr[k],os.path.join(RESULTS_DIR,v))
 if len(fr["coeff"]):atomic_csv(cal_summary(fr["coeff"]),os.path.join(RESULTS_DIR,"calibration_summary_partial.csv"));atomic_csv(group_summary(fr["group"]),os.path.join(RESULTS_DIR,"group_calibration_summary_partial.csv"))
 atomic_csv(model_choice(fr),os.path.join(RESULTS_DIR,"model_choice_summary_partial.csv"))
 return fr
def execute(tasks,worker,tables):
 existing=combine(tables)["run"]
 if len(existing):
  done=set(zip(existing.stage,existing.C_MODE,existing["T"].astype(int),existing.replicate_id.astype(int)))
  tasks=[x for x in tasks if ((x[0],x[1],x[2],x[3]) if len(x)==4 else ("stage1",x[0],1000,x[1])) not in done]
 if not tasks:return
 progress=Progress(len(tasks))
 with ProcessPoolExecutor(max_workers=min(N_WORKERS,len(tasks))) as pool:
  for future in as_completed([pool.submit(worker,x) for x in tasks]):
   result=future.result()
   for k,v in result.items():
    if len(v):tables[k].append(v)
   save(tables);progress.update(result["run"].iloc[-1].get("C_MODE","done"))
def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34Q-C","M":20,"T_primary":1000,"C_MODES":C_MODES,"N_WORKERS":N_WORKERS,"N_FFBS_SAMPLES":50,"VB_MAX_ITER":75,"VB_CONVERGENCE_TOL":1e-4,"PERTURBED_VB_MAX_ITER":30,"PERTURBED_RESCUE_MAX_ITER":50,"MAX_STAGEB_SECONDS_PER_CMODE":MAX_STAGEB_SECONDS_PER_CMODE,"A_center":"A_VB","smoke_test":SMOKE_TEST};json.dump(config,open(os.path.join(RESULTS_DIR,"experiment_config.json"),"w"),indent=2);tables={k:[] for k in FILES}
 for k,name in FILES.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:f=pd.read_csv(path)
   except pd.errors.EmptyDataError:f=pd.DataFrame()
   if len(f):tables[k].append(f)
 if SMOKE_TEST:execute([("stage0","identity",500,0)],stageB_worker,tables)
 else:
  execute([(m,r) for m in C_MODES for r in range(2)],cheap_worker,tables);fr=combine(tables);choice=model_choice(fr);eligible=["identity"]
  for m in C_MODES[1:]:
   row=choice.loc[(choice.stage=="stage1")&(choice.C_MODE==m)].iloc[0];criteria=[row.full_network_AUPRC_model_A_group_norm>=.5,row.full_network_AUPRC_stabilized_louis_group_snr>=.55,row.spectral_best_band_AUPRC>=.55,row.smoothed_state_correlation>=.85]
   if sum(bool(x) for x in criteria)>=2:eligible.append(m)
  execute([("stage2",m,1000,r) for m in eligible for r in range(2)],stageB_worker,tables);rank=model_choice(combine(tables));rank=rank.loc[rank.stage=="stage2"] if (rank.stage=="stage2").any() else rank;best=rank.sort_values("model_choice_score").iloc[-1].C_MODE;execute([("stage3",best,2000,0)],stageB_worker,tables)
 fr=save(tables);cal=cal_summary(fr["coeff"]) if len(fr["coeff"]) else pd.DataFrame();groups=group_summary(fr["group"]) if len(fr["group"]) else pd.DataFrame();choice=model_choice(fr);outputs={"run_summary.csv":fr["run"],"model_choice_summary.csv":choice,"selected_groups.csv":fr["groupsel"],"selected_coefficients.csv":fr["selected"],"coefficient_calibration_results.csv":fr["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":fr["group"],"group_calibration_summary.csv":groups,"covariance_diagnostics.csv":fr["covdiag"],"stageB_perturbation_diagnostics.csv":fr["perturb"],"activity_score_diagnostics.csv":fr["activity"],"gating_diagnostics.csv":fr["gating"],"A_recovery_summary.csv":fr["A"],"B_recovery_summary.csv":fr["B"],"Q_recovery_summary.csv":fr["Q"],"latent_state_recovery_summary.csv":fr["latent"],"network_recovery_summary.csv":fr["network"],"selected_edge_recovery_summary.csv":fr["selected_scores"],"spectral_gc_summary.csv":fr["spectral"],"spectral_band_recovery_summary.csv":fr["spectral"],"deviance_diagnostics.csv":fr["deviance"],"runtime_summary.csv":fr["runtime"],"decision_summary.csv":choice}
 for name,f in outputs.items():atomic_csv(f,os.path.join(RESULTS_DIR,name))
if __name__=="__main__":main()
