"""Experiment 34R Stage 2: selected Stage-B LRVB for Gaussian rectangular C."""
import os
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[_k]=os.environ.get("EXPERIMENT_34R_STAGE2_INNER_THREADS","1")
import json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiments.experiment_34i_static_spectral_varx_gc as spi
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as qb
import experiments.experiment_34qC_practical_model_choice_squareC as qc
import experiments.experiment_34r_stage1_rectangular_C_breakpoint as r1
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations
from src.stats.lrvb_stageB_smoother_feedback import global_index,project_selected_covariance,sensitivity_column
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics

RESULTS_DIR=os.environ.get("EXPERIMENT_34R_STAGE2_RESULTS_DIR","results/experiment_34r_stage2");WORKERS=int(os.environ.get("EXPERIMENT_34R_STAGE2_WORKERS","2"));SMOKE=os.environ.get("EXPERIMENT_34R_STAGE2_SMOKE","0")=="1";RUN_CONFIRM=os.environ.get("EXPERIMENT_34R_STAGE2_CONFIRM","0")=="1"
CONFIGS=[{"label":"gaussian_positive_control_Mx20_My40_T1000","M_x":20,"M_y":40,"T":1000,"replicates":2},{"label":"gaussian_transition_Mx20_My15_T1000","M_x":20,"M_y":15,"T":1000,"replicates":2}]
if RUN_CONFIRM:CONFIGS.append({"label":"gaussian_positive_control_Mx20_My40_T2000_confirm","M_x":20,"M_y":40,"T":2000,"replicates":1})
if SMOKE:CONFIGS=[{"label":"smoke_Mx10_My8_T300","M_x":10,"M_y":8,"T":300,"replicates":1}]
METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_gaussian_rectangular_C_known_R";EPS=1e-4;PERTURB=5 if SMOKE else 30;RESCUE=5 if SMOKE else 50;CAP=10*3600
TABLES=("run","C","selected","groupsel","coeff","group","covdiag","perturb","activity","observability","latent","A","B","Q","network","spectral","spectra","deviance","runtime")
FILES={k:f"_{k}_checkpoint.csv" for k in TABLES};FILES["run"]="run_summary_partial.csv";WORK_KEYS=["config_label","M_x","M_y","T","true_network_id","replicate_id"]
PARTIAL={"covdiag":"covariance_diagnostics_partial.csv","perturb":"stageB_perturbation_diagnostics_partial.csv","latent":"latent_state_recovery_summary_partial.csv","A":"A_recovery_summary_partial.csv","B":"B_recovery_summary_partial.csv","Q":"Q_recovery_summary_partial.csv","network":"network_recovery_summary_partial.csv","spectra":"spectral_gc_summary_partial.csv","deviance":"deviance_diagnostics_partial.csv","runtime":"runtime_summary_partial.csv"}

class Progress:
 def __init__(self,n,label):self.n,self.i,self.label,self.start=n,0,label,time.perf_counter()
 def update(self):self.i+=1;f=self.i/max(self.n,1);eta=(time.perf_counter()-self.start)/self.i*(self.n-self.i);k=round(28*f);print(f"\r{self.label} [{'#'*k}{'-'*(28-k)}] {self.i}/{self.n} {100*f:5.1f}% ETA {eta/60:6.1f}m",end="\n" if self.i==self.n else "",flush=True)

def metadata(cfg,rep):return {"config_label":cfg["label"],"M_x":cfg["M_x"],"M_y":cfg["M_y"],"observation_ratio":cfg["M_y"]/cfg["M_x"],"C_family":"gaussian_isotropic","C_MODE":"gaussian_isotropic","T":cfg["T"],"true_network_id":0,"replicate_id":rep,"method":METHOD,"run_id":f"{cfg['label']}_{rep}"}
def data_for(cfg,rep):return r1.simulate(cfg["M_x"],cfg["M_y"],cfg["T"],"gaussian_isotropic",0,rep)

def select(model,data,L,meta,seed,limit):
 M=model.n_states;rng=np.random.default_rng(seed);norm=np.linalg.norm(data["C"],axis=0);groups=[];seen=set()
 def add(t,s,kind):
  if (t,s) not in seen:seen.add((t,s));groups.append((t,s,kind))
 for t,s in zip(*np.where(data["mask"])):add(int(t),int(s),"true_offdiag_all")
 order=np.argsort(norm);diags=np.unique(np.r_[order[:3],order[len(order)//2-1:len(order)//2+2],order[-3:]])[:8]
 for t in diags:add(int(t),int(t),"diagonal_observability_spread")
 false=[(t,s) for t in range(M) for s in range(M) if t!=s and not data["mask"][t,s]];rng.shuffle(false)
 for t,s in false[:8]:add(t,s,"matched_false")
 scored=[]
 for t,s in false:
  cols=[s,M+s];m=np.asarray(model.A_mean_matrices_)[:,t,s];scored.append((np.sqrt(max(m@np.linalg.pinv(L[t][np.ix_(cols,cols)])@m,0)),t,s))
 for _,t,s in sorted(scored,reverse=True)[:5]:add(t,s,"high_snr_false")
 if 2*(len(groups)+5)<=limit:
  for _,t,s in sorted(scored)[:5]:add(t,s,"low_snr_false")
 rows=[];grows=[]
 for t,s,kind in groups:
  cols=[s,M+s];m=np.asarray(model.A_mean_matrices_)[:,t,s];snr=np.sqrt(max(m@np.linalg.pinv(L[t][np.ix_(cols,cols)])@m,0));grows.append({**meta,"target":t,"source":s,"true_edge":bool(data["mask"][t,s]),"coefficient_type":"diagonal" if t==s else ("offdiag_nonzero" if data["mask"][t,s] else "offdiag_zero"),"selected_direction_type":kind,"selected_reason":kind,"source_observability_score":norm[s],"target_observability_score":norm[t],"edge_observability_score":norm[s]*norm[t],"group_snr_louis":snr,"estimated_group_norm":np.linalg.norm(m)})
  for lag in range(2):rows.append({**grows[-1],"coefficient_global_index":global_index(t,lag,s,M,2),"lag":lag+1})
 return pd.DataFrame(rows).head(limit),pd.DataFrame(grows)

def perturb_one(args):
 model,y,u,index=args;start=time.perf_counter();col,plus,minus=sensitivity_column(model,y,u,index,EPS,PERTURB,1e-4,reestimate_B=True,reestimate_Q=True);rescue=not(plus.converged and minus.converged)
 if rescue and not SMOKE:col,plus,minus=sensitivity_column(model,y,u,index,EPS,RESCUE,1e-4,reestimate_B=True,reestimate_Q=True)
 row={"perturbation_direction_index":index,"eps":EPS,"plus_converged":plus.converged,"minus_converged":minus.converged,"plus_n_iter":plus.n_iter,"minus_n_iter":minus.n_iter,"plus_rescue_used":rescue,"minus_rescue_used":rescue,"plus_final_relative_A_change":plus.final_relative_A_change,"minus_final_relative_A_change":minus.final_relative_A_change,"plus_final_relative_B_change":plus.final_relative_B_change,"minus_final_relative_B_change":minus.final_relative_B_change,"plus_final_relative_Q_change":plus.final_relative_Q_change,"minus_final_relative_Q_change":minus.final_relative_Q_change,"plus_loglikelihood":plus.final_loglikelihood,"minus_loglikelihood":minus.final_loglikelihood,"variance_estimate_raw":col[index],"finite_difference_valid":np.all(np.isfinite(col)),"warning_flag":";".join(filter(None,[plus.warning_flag,minus.warning_flag])),"runtime_seconds":time.perf_counter()-start}
 return index,col,row

def load_tables():
 out={k:[] for k in TABLES}
 for k,n in FILES.items():
  p=os.path.join(RESULTS_DIR,n)
  if os.path.exists(p):
   try:f=pd.read_csv(p)
   except pd.errors.EmptyDataError:f=pd.DataFrame()
   if len(f):out[k]=[f]
 return out
def frames(t):return {k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in t.items()}
def save_raw(t):
 f=frames(t)
 for k,n in FILES.items():atomic_csv(f[k],os.path.join(RESULTS_DIR,n))
 for k,n in PARTIAL.items():atomic_csv(f[k],os.path.join(RESULTS_DIR,n))
 return f
def column_path(meta,index):
 d=os.path.join(RESULTS_DIR,"stageB_column_checkpoints",meta["config_label"],f"replicate_{meta['replicate_id']}");os.makedirs(d,exist_ok=True);return os.path.join(d,f"direction_{int(index)}.npy")
def atomic_column(path,column):
 tmp=path+".tmp"
 with open(tmp,"wb") as h:np.save(h,np.asarray(column,float))
 os.replace(tmp,path)

def run_rep(cfg,rep,tables):
 meta=metadata(cfg,rep);data,seed=data_for(cfg,rep);start=time.perf_counter();old=qb.VB_MAX_ITER;qb.VB_MAX_ITER=10 if SMOKE else 75;b=time.perf_counter();model=qb.fit_estBQ(data,seed+500);base=time.perf_counter()-b;qb.VB_MAX_ITER=old;l=time.perf_counter();L=r1.louis(model,data,seed+700);ltime=time.perf_counter()-l;sel,gsel=select(model,data,L,meta,seed+900,20 if SMOKE else 90);indices=sel.coefficient_global_index.astype(int).tolist();columns={i:np.load(column_path(meta,i)) for i in indices if os.path.exists(column_path(meta,i))};diags=[];pending=[i for i in indices if i not in columns];p=Progress(len(pending),f"{cfg['label']} rep={rep+1}");clock=time.perf_counter()
 with ProcessPoolExecutor(max_workers=min(WORKERS,len(indices))) as pool:
  futures={pool.submit(perturb_one,(model,data["y"],data["u"],idx)):idx for idx in pending}
  for future in as_completed(futures):
   idx,col,row=future.result();columns[idx]=col;atomic_column(column_path(meta,idx),col);diags.append({**meta,**row});prior=frames(tables)["perturb"];checkpoint=pd.concat([prior,pd.DataFrame(diags)],ignore_index=True);atomic_csv(checkpoint,os.path.join(RESULTS_DIR,FILES["perturb"]));atomic_csv(checkpoint,os.path.join(RESULTS_DIR,PARTIAL["perturb"]));p.update()
   if time.perf_counter()-clock>CAP:
    for pending in futures:pending.cancel()
    break
 valid=[i for i in indices if i in columns and np.all(np.isfinite(columns[i]))];matrix=np.column_stack([columns[i] for i in valid]);raw,psd,projection=project_selected_covariance(matrix,valid);sel=sel.set_index("coefficient_global_index").loc[valid].reset_index();ordinary=qa.o.global_covariance_block(model.A_row_covariances_)[np.ix_(valid,valid)];lc=qa.o.global_covariance_block(L)[np.ix_(valid,valid)];blocks={"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":lc,"lrvb_stageB_smoother_feedback_raw":raw,"lrvb_stageB_smoother_feedback_psd_projected":psd};records=qa.p.group_records(model,data,sel,{"ordinary_vb":ordinary,"stabilized_louis_eta_0p70_tau_0p90":lc,"lrvb_stageB_smoother_feedback":psd});wanted={"ordinary_vb","stabilized_louis_eta_0p70_tau_0p90","lrvb_stageB_smoother_feedback_raw","lrvb_stageB_smoother_feedback_psd_projected","soft_sparsity_aware_stageB_logistic_c4_tau2p5","hard_active_subspace_stageB_group_snr_louis_tau2p5","global_lambda_stageB_0p75"};variants=[v for v in qa.covariance_variants(blocks,records) if v["covariance_estimator"] in wanted];norm=np.linalg.norm(data["C"],axis=0)
 for v in variants:
  c=qa.coefficient_rows(model,data,sel,v,meta);c["source_observability_score"]=[norm[int(x)] for x in c.source];c["target_observability_score"]=[norm[int(x)] for x in c.target];c["edge_observability_score"]=c.source_observability_score*c.target_observability_score;g,s=qa.group_rows(model,records,v,meta);g["source_observability_score"]=[norm[int(x)] for x in g.group_source];g["target_observability_score"]=[norm[int(x)] for x in g.group_target];g["edge_observability_score"]=g.source_observability_score*g.target_observability_score;tables["coeff"].append(c);tables["group"].append(g);pr=v["projection"];tables["covdiag"].append(pd.DataFrame([{**meta,"covariance_estimator":v["covariance_estimator"],"number_selected_directions":len(valid),"number_negative_variances":pr.get("number_negative_variances",0),"fraction_negative_variances":pr.get("fraction_negative_variances",0.),"number_psd_projections":int(pr.get("psd_projection_used",False)),"fraction_psd_projected":float(pr.get("psd_projection_used",False)),**covariance_diagnostics(v["covariance"],ordinary,lc,sel.coefficient_type)}]))
 tables["activity"].append(qa.activity_rows(meta,records,blocks))
 tables["selected"].append(sel);tables["groupsel"].append(gsel);tables["perturb"].append(pd.DataFrame(diags));A=qb.a_recovery(model,data,meta);B=qa.B_recovery(model,data,meta);Q=qb.q_recovery(model,data,meta);proxy=r1.ridge_proxy(data);lat={**meta,**qb.latent_metrics(model,{**data,"y":proxy@data["C"].T})};net=qa.full_network(model,data,meta,L);oldM=spi.M;spi.M=cfg["M_x"];spectra,bands,_,_,_=spi.compute_spectral_tables(model.A_mean_matrices_,model.B_matrices,model.Q,np.linspace(0,np.pi,128),data,meta,"estimated_AQ");spec,_=spi.metric_rows(bands);spi.M=oldM
 for k,v in meta.items():spec[k]=v
 dev=qc.deviance(model,data,meta);runtime={**meta,"baseline_VB_runtime_seconds":base,"louis_runtime_seconds":ltime,"stageB_total_runtime_seconds":time.perf_counter()-clock,"average_perturbed_fit_runtime_seconds":np.mean([x["runtime_seconds"] for x in diags]),"number_perturbed_fits":2*len(diags),"number_selected_directions":len(valid),"number_workers":WORKERS,"stageB_truncated":len(valid)<len(indices),"total_runtime_seconds":time.perf_counter()-start,"run_status":"success"};cm={**meta,**r1.c_metrics(data["C"])}
 for k,v in (("C",pd.DataFrame([cm])),("latent",pd.DataFrame([lat])),("A",pd.DataFrame([A])),("B",pd.DataFrame([B])),("Q",pd.DataFrame([Q])),("network",net),("spectral",spec),("spectra",bands),("deviance",dev),("runtime",pd.DataFrame([runtime])),("run",pd.DataFrame([{**meta,**runtime}]))):tables[k].append(v)
 save_raw(tables)

def calibration(f):
 keys=["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","coefficient_type"]
 return f.groupby(keys,dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),median_signed_error=("signed_error","median"),mean_abs_error=("abs_error","mean"),median_abs_error=("abs_error","median"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),median_posterior_sd=("posterior_sd","median"),mean_interval_width_95=("interval_width_95","mean"),median_interval_width_95=("interval_width_95","median"),mean_standardized_error=("standardized_error","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),median_abs_standardized_error=("standardized_error",lambda x:x.abs().median()),n_coefficients=("posterior_sd","size")).reset_index()
def group_summary(f):return f.groupby(["config_label","M_x","M_y","observation_ratio","T","covariance_estimator","edge_group_type"],dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("D2","size")).reset_index()
def decisions(f):
 cal=calibration(f["coeff"]);gs=group_summary(f["group"]);rows=[]
 for (label,mx,my,ratio,T,est),g in cal.groupby(["config_label","M_x","M_y","observation_ratio","T","covariance_estimator"]):
  get=lambda typ,col:g.loc[g.coefficient_type==typ,col].mean();run=f["run"].loc[f["run"].config_label==label].mean(numeric_only=True);net=f["network"].loc[f["network"].config_label==label];sp=f["spectral"].loc[(f["spectral"].config_label==label)&(f["spectral"].score_type=="transfer_spectral_gc_diagQ")];best=sp.loc[sp.AUPRC.idxmax()] if len(sp) else {};dv=f["deviance"].loc[f["deviance"].config_label==label];raw=dv.loc[dv.score_type=="raw_deviance"];deb=dv.loc[dv.score_type=="debiased_deviance"];grp=gs.loc[(gs.config_label==label)&(gs.covariance_estimator==est)];rows.append({"config_label":label,"M_x":mx,"M_y":my,"observation_ratio":ratio,"T":T,"covariance_estimator":est,"smoothed_state_correlation":run.get("smoothed_state_correlation",np.nan),"smoothed_state_MSE":run.get("smoothed_state_MSE",np.nan),"A_support_AUPRC_model_A_group_norm":net.loc[net.score_type=="model_A_group_norm","AUPRC"].mean(),"A_support_AUPRC_stabilized_louis_group_snr":net.loc[net.score_type=="stabilized_louis_group_snr","AUPRC"].mean(),"spectral_best_band":best.get("band_name",np.nan),"spectral_best_band_AUPRC":best.get("AUPRC",np.nan),"spectral_best_band_ROC_AUC":best.get("ROC_AUC",np.nan),"spectral_best_band_TPR_at_FPR_0p05":best.get("TPR_at_FPR_0p05",np.nan),"deviance_best_signal_type":raw.loc[raw.AUPRC.idxmax(),"signal_type"] if len(raw) else np.nan,"deviance_best_AUPRC":raw.AUPRC.max() if len(raw) else np.nan,"debiased_deviance_best_signal_type":deb.loc[deb.AUPRC.idxmax(),"signal_type"] if len(deb) else np.nan,"debiased_deviance_best_AUPRC":deb.AUPRC.max() if len(deb) else np.nan,"B_relative_frobenius_error":run.get("B_relative_frobenius_error",np.nan),"B_correlation_hat_true":run.get("B_correlation_hat_true",np.nan),"Q_relative_frobenius_error":run.get("Q_relative_frobenius_error",np.nan),"Q_trace_ratio_hat_to_true":run.get("Q_trace_ratio_hat_to_true",np.nan),"diag_coverage":get("diagonal","empirical_coverage_95"),"offdiag_nonzero_coverage":get("offdiag_nonzero","empirical_coverage_95"),"offdiag_zero_coverage":get("offdiag_zero","empirical_coverage_95"),"diag_std_z":get("diagonal","std_standardized_error"),"offdiag_nonzero_std_z":get("offdiag_nonzero","std_standardized_error"),"offdiag_zero_std_z":get("offdiag_zero","std_standardized_error"),"true_edge_group_chi2_95_coverage":grp.loc[grp.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"false_edge_group_chi2_95_coverage":grp.loc[grp.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"fraction_negative_variances":f["covdiag"].loc[(f["covdiag"].config_label==label)&(f["covdiag"].covariance_estimator==est),"fraction_negative_variances"].mean(),"stageB_total_runtime_seconds":run.get("stageB_total_runtime_seconds",np.nan),"total_runtime_seconds":run.get("total_runtime_seconds",np.nan),"stageB_truncated":bool(run.get("stageB_truncated",False)),"recommended_for_rectangular_C_pipeline":bool(my==40 and best.get("AUPRC",0)>=.5),"recommended_for_uncertainty":bool(.85<=get("offdiag_nonzero","empirical_coverage_95")<=.98),"notes":"A_center=A_VB; selected-direction Stage-B; B and Q re-estimated."})
 return pd.DataFrame(rows)
def finish(tables):
 f=frames(tables);cal=calibration(f["coeff"]);gs=group_summary(f["group"]);dec=decisions(f);obs=cal.merge(f["selected"][["config_label","coefficient_global_index","edge_observability_score"]],on="config_label",how="left") if len(cal) else pd.DataFrame();outputs={"run_summary.csv":f["run"],"C_diagnostics.csv":f["C"],"selected_groups.csv":f["groupsel"],"selected_coefficients.csv":f["selected"],"coefficient_calibration_results.csv":f["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":f["group"],"group_calibration_summary.csv":gs,"covariance_diagnostics.csv":f["covdiag"],"stageB_perturbation_diagnostics.csv":f["perturb"],"activity_score_diagnostics.csv":f["activity"],"observability_stageB_summary.csv":obs,"latent_state_recovery_summary.csv":f["latent"],"A_recovery_summary.csv":f["A"],"B_recovery_summary.csv":f["B"],"Q_recovery_summary.csv":f["Q"],"network_recovery_summary.csv":f["network"],"spectral_gc_summary.csv":f["spectra"],"spectral_band_recovery_summary.csv":f["spectral"],"deviance_diagnostics.csv":f["deviance"],"runtime_summary.csv":f["runtime"],"decision_summary.csv":dec}
 for name,x in outputs.items():atomic_csv(x,os.path.join(RESULTS_DIR,name))
 for name,x in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",gs),("observability_stageB_summary_partial.csv",obs),("decision_summary_partial.csv",dec)):atomic_csv(x,os.path.join(RESULTS_DIR,name))
 return f,cal,dec
def plots(f,cal,dec):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 def bar(frame,x,y,name):s=frame.groupby(x,dropna=False)[y].mean();fig,ax=plt.subplots();s.plot.bar(ax=ax);ax.set_ylabel(y);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 bar(cal,["M_y","covariance_estimator","coefficient_type"],"empirical_coverage_95","coverage.png");bar(cal,["M_y","covariance_estimator","coefficient_type"],"std_standardized_error","standardized_sd.png");bar(cal,["M_y","covariance_estimator"],"mean_interval_width_95","interval_width.png");bar(f["spectral"],["M_y","band_name"],"AUPRC","spectral_AUPRC.png");bar(f["network"],["M_y","score_type"],"AUPRC","A_support_AUPRC.png");bar(f["deviance"],["M_y","signal_type","score_type"],"AUPRC","deviance_AUPRC.png");bar(f["latent"],"M_y","smoothed_state_correlation","state_correlation.png");bar(f["B"],"M_y","B_relative_frobenius_error","B_recovery.png");bar(f["Q"],"M_y","Q_relative_frobenius_error","Q_recovery.png");bar(f["runtime"],"M_y","total_runtime_seconds","runtime.png");bar(f["runtime"],"number_selected_directions","stageB_total_runtime_seconds","runtime_directions.png");bar(f["covdiag"],["M_y","covariance_estimator"],"trace_over_louis_selected","stageB_louis_ratio.png")
def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34R_stage2_gaussian_rectangular_C_sparsity_aware_StageB","CONFIG_LIST":CONFIGS,"N_WORKERS":WORKERS,"N_FFBS_SAMPLES":50,"FINITE_DIFF_EPS_PRIMARY":EPS,"PERTURBED_VB_MAX_ITER":PERTURB,"PERTURBED_RESCUE_MAX_ITER":RESCUE,"MAX_STAGEB_SECONDS_PER_CONFIG_REPLICATE":CAP,"A_center":"A_VB","smoke_test":SMOKE};json.dump(config,open(os.path.join(RESULTS_DIR,"experiment_config.json"),"w"),indent=2);tables=load_tables();done=set(zip(frames(tables)["run"].config_label,frames(tables)["run"].replicate_id.astype(int))) if len(frames(tables)["run"]) else set()
 for cfg in CONFIGS:
  for rep in range(cfg["replicates"]):
   if (cfg["label"],rep) not in done:
    try:run_rep(cfg,rep,tables)
    except Exception as error:
     meta=metadata(cfg,rep);row={**meta,"run_status":"failed","error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()};tables["run"].append(pd.DataFrame([row]));tables["runtime"].append(pd.DataFrame([row]));save_raw(tables)
 f,cal,dec=finish(tables)
 if not SMOKE:plots(f,cal,dec)
if __name__=="__main__":main()
