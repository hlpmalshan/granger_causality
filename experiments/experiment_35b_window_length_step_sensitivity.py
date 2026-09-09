"""Experiment 35B: window-length and step-size sensitivity.

The estimator remains static inside each window.  Adaptation is solely through
chronological refitting of a piecewise-stationary latent VARX process.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]=os.environ.get("EXPERIMENT_35B_BLAS_THREADS","1")
import json,time,traceback,warnings
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats,make_network,relative
from experiments.experiment_34a_hybrid_vb_ard_A_fixed_B import recovery,safe_curve,vb_details
from experiments.experiment_34c_hybrid_vb_ard_with_estimated_B import B_metrics
from experiments.experiment_34f_vb_ard_Q_estimation_shrinkage import Q_metrics,diagnostics
from src.gc.spectral_varx_gc import compute_A_frequency_matrix,compute_A_transfer_function,compute_B_frequency_matrix,integrate_spectral_score
from src.ssm.em_varx_p_known_c_l1_mstep import stabilize_A_matrices_if_needed
from src.ssm.ssm_varx_p_simulator import var_companion_spectral_radius
from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls
from src.varx.varx_generator import generate_colored_input
from experiments.experiment_34g_vb_ard_C_mixing_robustness import make_C

M,na,nb,BURN_IN=20,2,3,300
C_MODE_LIST=["identity","mild_mixing"];SCENARIO_LIST=["support_onset","strength_shift"]
N_OUTER_RUNS=2;T_TOTAL=4000;CHANGE_POINT=2000;N_FREQS=128
WINDOW_CONFIGS=[{"label":"short_fine","WINDOW_LENGTH":750,"WINDOW_STEP":250},{"label":"baseline_fine_step","WINDOW_LENGTH":1000,"WINDOW_STEP":250},{"label":"baseline_35a","WINDOW_LENGTH":1000,"WINDOW_STEP":500},{"label":"long_stable","WINDOW_LENGTH":1500,"WINDOW_STEP":500}]
Q_VARIANT="estimate_Q_diag_shrink_scalar_rho_0p25";METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25"
BANDS={"low_band":(0,.2*np.pi),"mid_band":(.2*np.pi,.5*np.pi),"high_band":(.5*np.pi,np.pi),"full_band":(0,np.pi)}
BASE_SEED=5200000;SMOKE_TEST=os.environ.get("EXPERIMENT_35B_SMOKE","0")=="1"
if SMOKE_TEST:C_MODE_LIST=["identity"];SCENARIO_LIST=["support_onset"];N_OUTER_RUNS=1;WINDOW_CONFIGS=[{"label":"short_fine","WINDOW_LENGTH":750,"WINDOW_STEP":250},{"label":"baseline_35a","WINDOW_LENGTH":1000,"WINDOW_STEP":500}]
if os.environ.get("EXPERIMENT_35B_EXTENDED","0")=="1":WINDOW_CONFIGS=WINDOW_CONFIGS+[{"label":"very_long_stable","WINDOW_LENGTH":2000,"WINDOW_STEP":500}]
override=os.environ.get("EXPERIMENT_35B_N_OUTER_RUNS")
if override is not None:N_OUTER_RUNS=int(override)
RESULTS_DIR=os.environ.get("EXPERIMENT_35B_RESULTS_DIR","results/experiment_35b")

class ProgressBar:
 def __init__(self,total,width=30):self.total=max(int(total),0);self.width=width;self.done=0;print("Experiment 35B: all checkpointed windows are complete.") if not self.total else None
 def update(self,label):
  self.done+=1;f=min(self.done/max(self.total,1),1);n=round(self.width*f);print(f"\rExperiment 35B [{'#'*n}{'-'*(self.width-n)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:58]}",end="\n" if self.done>=self.total else "",flush=True)

def scenario_A(scenario,seed):
 A,mask=make_network(M,seed);links=np.argwhere(mask);rng=np.random.default_rng(seed+71);rng.shuffle(links)
 if scenario=="support_onset":
  split=min(10,max(1,len(links)//2));permanent=links[:split];changing=links[split:19];A0=np.asarray(A).copy();
  for target,source in changing:A0[:,target,source]=0
  A1=np.asarray(A).copy();unchanged=np.empty((0,2),int)
 else:
  permanent=np.empty((0,2),int);split=min(9,max(1,len(links)//2));changing=links[:split];unchanged=links[split:19];A0=np.asarray(A).copy();A1=A0.copy()
  for target,source in changing:A1[:,target,source]*=1.75
  A1=np.asarray(stabilize_A_matrices_if_needed(list(A1),target_radius=.82)[0])
 return A0,A1,permanent,changing,unchanged

def simulate(scenario,mode,seed):
 A0,A1,permanent,changing,unchanged=scenario_A(scenario,seed);rng=np.random.default_rng(seed+100)
 B=np.asarray([rng.normal(0,s,(M,1)) for s in (.45,.25,.15)]);Q=.5*np.eye(M);R=.6*np.eye(M);C=np.eye(M) if mode=="identity" else make_C(M,mode,seed)
 total=T_TOTAL+BURN_IN;u=np.asarray(generate_colored_input(total,.95,1.,seed+200));u=u[:,None] if u.ndim==1 else u
 x=np.zeros((total,M));noise=rng.multivariate_normal(np.zeros(M),Q,total)
 for t in range(2,total):
  active=A0 if t-BURN_IN<CHANGE_POINT else A1;x[t]=sum(active[k]@x[t-k-1] for k in range(na))+sum(B[k]@u[t-k] for k in range(nb))+noise[t]
 y=x@C.T+rng.multivariate_normal(np.zeros(M),R,total)
 return {"A_segments":[A0,A1],"B":B,"Q":Q,"R":R,"C":C,"x":x[BURN_IN:],"y":y[BURN_IN:],"u":u[BURN_IN:],"permanent":permanent,"changing":changing,"unchanged":unchanged}

def window_meta(start,wid,data,base,config):
 length=config["WINDOW_LENGTH"];end=start+length;center=(start+end)//2;f0=max(0,min(end,CHANGE_POINT)-start)/length;mixed=bool(f0>0 and f0<1)
 return {**base,"window_label":config["label"],"WINDOW_LENGTH":length,"WINDOW_STEP":config["WINDOW_STEP"],"window_id":wid,"window_start":start,"window_end":end,"window_center":center,"mixed_window":mixed,"pure_pre_window":end<=CHANGE_POINT,"pure_post_window":start>=CHANGE_POINT,"transition_window":mixed,"segment_label_by_center":0 if center<CHANGE_POINT else 1,"fraction_segment_0":f0,"fraction_segment_1":1-f0,"T_TOTAL":T_TOTAL}

def fit_window(data,meta,previous,seed):
 warm=previous is not None
 model=HybridVBARDVARXSSMKnownCWithBQControls(na,nb,data["C"],data["Q"],data["R"],B_update_mode="free",B_ridge_lambda=0.,
  initial_B_matrices=None if not warm else previous.B_matrices,estimate_Q=True,Q_update_mode="diag_shrink_scalar",Q_shrinkage_rho=.25,Q_update_damping=.5,
  include_A_posterior_uncertainty_in_Q=True,initial_Q=None if not warm else previous.Q,initial_alpha_mean=None if not warm else previous.alpha_mean_,
  init_A_matrices=None if not warm else previous.A_mean_matrices_,max_iter=100,tol_objective=1e-6,tol_A_change=1e-6,tol_B_change=1e-6,tol_Q_change=1e-6,tol_alpha_change=1e-6,
  a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed)
 start,end=meta["window_start"],meta["window_end"];model.fit(data["y"][start:end],data["u"][start:end]);return model

def state_path(run_id,window_label,window_id):return os.path.join(RESULTS_DIR,f"_window_state_{run_id}_{window_label}_{window_id:03d}.npz")
def save_state(model,run_id,window_label,window_id):
 path=state_path(run_id,window_label,window_id);temporary=path+".tmp.npz";np.savez_compressed(temporary,A=np.asarray(model.A_mean_matrices_),B=np.asarray(model.B_matrices),Q=model.Q,alpha=model.alpha_mean_);os.replace(temporary,path)
def load_state(run_id,window_label,window_id):
 path=state_path(run_id,window_label,window_id)
 if not os.path.exists(path):return None
 with np.load(path) as state:return SimpleNamespace(A_mean_matrices_=[x.copy() for x in state["A"]],B_matrices=[x.copy() for x in state["B"]],Q=state["Q"].copy(),alpha_mean_=state["alpha"].copy())

def spectral_curves(A,B,Q,omega):
 edges=[(i,j) for i in range(M) for j in range(M) if i!=j];curves=np.zeros((len(edges),len(omega)));exog=np.zeros((M,len(omega)));conditions=[];bad=clips=inversions=0
 for f,w in enumerate(omega):
  matrix=compute_A_frequency_matrix(A,w);condition=np.linalg.cond(matrix);conditions.append(condition);bad+=condition>1e10
  try:H=compute_A_transfer_function(A,w)
  except np.linalg.LinAlgError:H=compute_A_transfer_function(A,w,jitter=1e-8);inversions+=1
  spectrum=H@Q@H.conj().T;total=np.maximum(np.real(np.diag(spectrum)),1e-12);den=total[:,None]-np.abs(H)**2*np.diag(Q)[None,:];clips+=np.sum(den<1e-12);score=np.maximum(np.real(np.log(total[:,None]/np.maximum(den,1e-12))),0)
  for e,(i,j) in enumerate(edges):curves[e,f]=score[i,j]
  Hu=H@compute_B_frequency_matrix(B,w);exog[:,f]=np.sum(np.abs(Hu)**2,axis=1)
 return edges,curves,exog,{"max_condition_number_Aomega":max(conditions),"median_condition_number_Aomega":np.median(conditions),"spectral_inversion_warning_count":inversions,"denominator_clipping_count":int(clips),"number_of_frequency_points_with_bad_condition":int(bad)}

def edge_class(i,j,meta,data):
 pair=(i,j);changing={tuple(x) for x in data["changing"]};permanent={tuple(x) for x in data["permanent"]};unchanged={tuple(x) for x in data["unchanged"]}
 if pair in changing:return "emerging_edge" if meta["scenario"]=="support_onset" else "changed_strength_edge"
 if pair in permanent:return "permanent_edge"
 if pair in unchanged:return "unchanged_true_edge"
 active=data["A_segments"][meta["segment_label_by_center"]];return "active_edge" if np.any(active[:,i,j]!=0) else "false_edge"

def spectral_frames(A,B,Q,omega,data,meta,source):
 edges,curves,exog,numerical=spectral_curves(A,B,Q,omega);bands=[];spectra=[];active=data["A_segments"][meta["segment_label_by_center"]];norm=np.sqrt(np.sum(active**2,axis=0))
 for e,(i,j) in enumerate(edges):
  cls=edge_class(i,j,meta,data);truth=bool(norm[i,j]>0)
  for name,band in BANDS.items():
   v=integrate_spectral_score(omega,curves[e],band);bands.append({**meta,"parameter_source":source,"score_type":"transfer_spectral_gc_diagQ","source":j,"target":i,"band_name":name,"integrated_score":v["integral"],"average_score":v["average"],"peak_score":v["peak"],"peak_omega_over_pi":v["peak_frequency"]/np.pi,"true_direct_link":truth,"edge_class":cls,"true_A_group_norm":norm[i,j]})
  if cls!="false_edge":
   for f,w in enumerate(omega):spectra.append({**meta,"parameter_source":source,"source":j,"target":i,"omega_rad":w,"omega_over_pi":w/np.pi,"spectral_gc_value":curves[e,f],"true_direct_link":truth,"edge_class":cls})
 exrows=[]
 for i in range(M):
  for name,band in BANDS.items():
   v=integrate_spectral_score(omega,exog[i],band);exrows.append({**meta,"parameter_source":source,"source_index":i,"input_index":0,"band_name":name,"integrated_response_power":v["integral"],"average_response_power":v["average"],"peak_response_power":v["peak"],"peak_omega_over_pi":v["peak_frequency"]/np.pi})
 return pd.DataFrame(spectra),pd.DataFrame(bands),pd.DataFrame(exrows),numerical,curves,edges

def selected_extra_spectra(result,selected_pairs,omega,data,meta,source):
 curves,edges=result[4],result[5];lookup={pair:k for k,pair in enumerate(edges)};rows=[]
 for i,j in selected_pairs:
  cls=edge_class(i,j,meta,data)
  if cls!="false_edge":continue  # already retained by spectral_frames
  for f,w in enumerate(omega):rows.append({**meta,"parameter_source":source,"source":j,"target":i,"omega_rad":w,"omega_over_pi":w/np.pi,"spectral_gc_value":curves[lookup[(i,j)],f],"true_direct_link":False,"edge_class":cls})
 return pd.DataFrame(rows)

def recovery_rows(bands):
 rows=[];fixed=[];groups=["run_id","C_MODE","scenario","outer_run","T_TOTAL","window_label","WINDOW_LENGTH","WINDOW_STEP","window_id","window_center","mixed_window","pure_pre_window","pure_post_window","transition_window","parameter_source","score_type","band_name"]
 for keys,g in bands.groupby(groups,dropna=False):
  meta=dict(zip(groups,keys));metrics,curve=safe_curve(g.true_direct_link,g.integrated_score);rows.append({**meta,**metrics})
  for level in (.01,.03,.05,.1):
   eligible=[p for p in curve if np.isfinite(p[1]["fpr"]) and p[1]["fpr"]<=level];threshold,item=max(eligible,key=lambda p:(np.nan_to_num(p[1]["tpr"],nan=-1),-p[0])) if eligible else curve[0]
   fixed.append({**meta,"target_fpr_level":level,"threshold":threshold,"actual_fpr":item["fpr"],**{k:item[k] for k in ("tpr","precision","f1","tp","fp","tn","fn")}})
 return pd.DataFrame(rows),pd.DataFrame(fixed)

def similarity(oracle,estimated):
 keys=["run_id","C_MODE","scenario","outer_run","window_label","WINDOW_LENGTH","WINDOW_STEP","window_id","window_center","source","target","edge_class"];m=oracle.merge(estimated,on=keys+["omega_rad"],suffixes=("_oracle","_estimated"));rows=[]
 for vals,g in m.groupby(keys):
  x,y=g.spectral_gc_value_oracle.values,g.spectral_gc_value_estimated.values;po,pe=np.argmax(x),np.argmax(y);active=bool(g.true_direct_link_oracle.iloc[0]);rows.append({**dict(zip(keys,vals)),"active_edge":active,"activity_class":"active_edge" if active else "inactive_edge","spectral_curve_correlation":correlations(x,y),"spectral_curve_rmse":np.sqrt(np.mean((x-y)**2)),"spectral_curve_mae":np.mean(abs(x-y)),"integrated_score_absolute_error":abs(np.trapezoid(x,g.omega_rad)-np.trapezoid(y,g.omega_rad)),"integrated_score_relative_error":abs(np.trapezoid(x,g.omega_rad)-np.trapezoid(y,g.omega_rad))/max(abs(np.trapezoid(x,g.omega_rad)),1e-12),"peak_frequency_absolute_error":abs(g.omega_over_pi_oracle.iloc[po]-g.omega_over_pi_oracle.iloc[pe]),"peak_score_absolute_error":abs(x[po]-y[pe])})
 return pd.DataFrame(rows)

def uncertainty_summary(coeff):
 coeff=coeff.copy();coeff["signed_error"]=coeff.posterior_mean-coeff.true_value;coeff["standardized_error"]=coeff.signed_error/coeff.posterior_std.replace(0,np.nan)
 return coeff.groupby(["run_id","C_MODE","scenario","window_label","WINDOW_LENGTH","WINDOW_STEP","window_id","window_center","coefficient_type"],dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_posterior_std=("posterior_std","mean"),mean_signed_error=("signed_error","mean"),std_standardized_error=("standardized_error","std")).reset_index()

def score_comparison(bands,vb):
 rows=[]
 geometry=["run_id","C_MODE","scenario","window_label","WINDOW_LENGTH","WINDOW_STEP","window_id","window_center"]
 for keys,g in bands.loc[bands.parameter_source=="estimated_window_AQ"].groupby(geometry+["band_name"]):
  metrics,_=safe_curve(g.true_direct_link,g.integrated_score);rows.append({**dict(zip(geometry,keys[:-1])),"score_family":f"transfer_spectral_gc_{keys[-1]}",**metrics})
 for keys,g in vb.groupby(geometry):
  for family,column in (("vb_group_snr","group_snr_score"),("model_A_group_norm","posterior_mean_group_norm")):
   metrics,_=safe_curve(g.true_link,g[column]);rows.append({**dict(zip(geometry,keys)),"score_family":family,**metrics})
 return pd.DataFrame(rows)

def change_detection(bands):
 est=bands.loc[(bands.parameter_source=="estimated_window_AQ")&bands.edge_class.isin(["emerging_edge","changed_strength_edge"])] ;rows=[]
 geometry=["run_id","C_MODE","scenario","window_label","WINDOW_LENGTH","WINDOW_STEP"]
 for keys,g in est.groupby(geometry+["source","target","band_name","edge_class"]):
  pure=g.loc[~g.mixed_window];pre=pure.loc[pure.window_center<CHANGE_POINT].integrated_score;post=pure.loc[pure.window_center>=CHANGE_POINT].integrated_score
  source,target,band_name,edge_cls=keys[-4:];oracle=bands.loc[(bands.run_id==keys[0])&(bands.parameter_source=="oracle_window_AQ")&(bands.source==source)&(bands.target==target)&(bands.band_name==band_name)&(~bands.mixed_window)];opre=oracle.loc[oracle.window_center<CHANGE_POINT].integrated_score;opost=oracle.loc[oracle.window_center>=CHANGE_POINT].integrated_score;estimated_ratio=post.mean()/max(pre.mean(),1e-12);oracle_ratio=opost.mean()/max(opre.mean(),1e-12);aligned=pure.merge(oracle[["window_id","integrated_score"]],on="window_id",suffixes=("_estimated","_oracle"));across=correlations(aligned.integrated_score_estimated,aligned.integrated_score_oracle) if len(aligned)>1 else np.nan
  for threshold_type,q in (("threshold_95",.95),("threshold_99",.99)):
   false=bands.loc[(bands.run_id==keys[0])&(bands.parameter_source=="estimated_window_AQ")&(bands.band_name==band_name)&(bands.edge_class=="false_edge")&(~bands.mixed_window)&(bands.window_center<CHANGE_POINT),"integrated_score"]
   threshold=false.quantile(q) if len(false) else np.nan;detected=pure.loc[(pure.window_center>=CHANGE_POINT)&(pure.integrated_score>threshold),"window_center"];first=detected.min() if len(detected) else np.nan
   rows.append({**dict(zip(geometry,keys[:len(geometry)])),"source":source,"target":target,"band_name":band_name,"edge_class":edge_cls,"threshold_type":threshold_type,"score_pre_change_mean":pre.mean(),"score_pre_change_std":pre.std(),"score_post_change_mean":post.mean(),"score_change_ratio":estimated_ratio,"score_change_difference":post.mean()-pre.mean(),"oracle_pre_mean":opre.mean(),"oracle_post_mean":opost.mean(),"estimated_change_ratio":estimated_ratio,"oracle_change_ratio":oracle_ratio,"oracle_estimated_change_ratio_error":abs(estimated_ratio-oracle_ratio),"correlation_across_windows_between_estimated_and_oracle_scores":across,"first_detected_window_center":first,"detection_delay":first-CHANGE_POINT if np.isfinite(first) else np.nan,"detected_by_end":np.isfinite(first)})
 return pd.DataFrame(rows)

def save_plots(bands,recovery,similarity_frame,changes,comparison,windows,calibration):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True);est=bands.loc[bands.parameter_source=="estimated_window_AQ"]
 def plot_lines(frame,index,columns,value,name,ylabel):
  fig,ax=plt.subplots();p=frame.pivot_table(index=index,columns=columns,values=value,aggfunc="mean") if len(frame) else pd.DataFrame();p.plot(ax=ax,marker="o") if len(p) else None;ax.set(xlabel=index,ylabel=ylabel);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 onset=changes.loc[(changes.scenario=="support_onset")&(changes.threshold_type=="threshold_95")];plot_lines(onset,"WINDOW_LENGTH","window_label","detected_by_end","01_detection_rate.png","detection rate");plot_lines(onset,"WINDOW_LENGTH","window_label","detection_delay","02_detection_delay.png","mean detection delay")
 plot_lines(recovery.loc[(recovery.parameter_source=="estimated_window_AQ")&(recovery.band_name=="full_band")],"window_center","window_label","AUPRC","03_window_auprc.png","AUPRC")
 for n,b in enumerate(("low_band","mid_band","full_band"),4):plot_lines(est.loc[(est.edge_class=="emerging_edge")&(est.band_name==b)],"window_center","window_label","integrated_score",f"{n:02d}_{b}_emerging_course.png","integrated GC")
 strength=changes.loc[(changes.scenario=="strength_shift")&(changes.threshold_type=="threshold_95")];plot_lines(strength,"WINDOW_LENGTH","window_label","estimated_change_ratio","07_strength_ratio.png","estimated/oracle change ratio")
 plot_lines(similarity_frame,"WINDOW_LENGTH","window_label","spectral_curve_correlation","08_time_frequency_correlation.png","correlation")
 delay=onset.groupby("window_label",as_index=False).detection_delay.mean();quality=recovery.loc[(recovery.parameter_source=="estimated_window_AQ")&(recovery.band_name=="full_band")].groupby("window_label",as_index=False).AUPRC.mean();trade=delay.merge(quality,on="window_label");fig,ax=plt.subplots();ax.scatter(trade.detection_delay,trade.AUPRC);[ax.annotate(r.window_label,(r.detection_delay,r.AUPRC)) for _,r in trade.iterrows()];ax.set(xlabel="detection delay",ylabel="AUPRC");fig.tight_layout();fig.savefig(os.path.join(path,"09_auprc_delay_tradeoff.png"));plt.close(fig)
 plot_lines(windows,"WINDOW_LENGTH","window_label","runtime_fit_seconds","10_runtime.png","fit seconds");plot_lines(windows,"WINDOW_LENGTH","window_label","mixed_window","11_mixed_windows.png","mixed-window fraction")
 plot_lines(comparison,"WINDOW_LENGTH","score_family","AUPRC","12_score_comparison.png","AUPRC");plot_lines(windows,"WINDOW_LENGTH","window_label","practically_converged","13_practical_convergence.png","rate")
 coverage=calibration.loc[calibration.coefficient_type.astype(str).str.contains("offdiag_nonzero",case=False,na=False)] if len(calibration) else calibration;plot_lines(coverage,"WINDOW_LENGTH","window_label","empirical_coverage_95","14_active_coefficient_coverage.png","coverage")

def geometry_summary(windows):
 keys=["C_MODE","scenario","window_label","WINDOW_LENGTH","WINDOW_STEP"];rows=[]
 for vals,g in windows.groupby(keys,dropna=False):
  successful=g.loc[g.run_status=="success"].drop_duplicates(["run_id","window_id"]);post=successful.loc[successful.pure_post_window]
  earliest=post.window_center.min() if len(post) else np.nan
  rows.append({**dict(zip(keys,vals)),"number_of_windows":successful.groupby("run_id").window_id.nunique().mean(),"number_of_pure_pre_windows":successful.groupby("run_id").pure_pre_window.sum().mean(),"number_of_mixed_windows":successful.groupby("run_id").mixed_window.sum().mean(),"number_of_pure_post_windows":successful.groupby("run_id").pure_post_window.sum().mean(),"earliest_possible_pure_post_center":earliest,"theoretical_min_detection_delay_pure_window":earliest-CHANGE_POINT if np.isfinite(earliest) else np.nan,"temporal_sampling_interval":vals[4],"change_smearing_width":vals[3]})
 return pd.DataFrame(rows)

def time_frequency_similarity(spectra):
 keys=["run_id","C_MODE","scenario","outer_run","window_label","WINDOW_LENGTH","WINDOW_STEP","source","target","edge_class"]
 oracle=spectra.loc[spectra.parameter_source=="oracle_window_AQ"];estimated=spectra.loc[spectra.parameter_source=="estimated_window_AQ"];merged=oracle.merge(estimated,on=keys+["window_id","window_center","omega_rad"],suffixes=("_oracle","_estimated"));rows=[]
 for vals,g in merged.groupby(keys,dropna=False):
  x=g.spectral_gc_value_oracle.to_numpy();y=g.spectral_gc_value_estimated.to_numpy();po=int(np.argmax(x));pe=int(np.argmax(y));rows.append({**dict(zip(keys,vals)),"time_frequency_correlation":correlations(x,y),"time_frequency_rmse":np.sqrt(np.mean((x-y)**2)),"time_frequency_mae":np.mean(np.abs(x-y)),"peak_time_error":abs(g.window_center.iloc[po]-g.window_center.iloc[pe]),"peak_frequency_error":abs(g.omega_rad.iloc[po]-g.omega_rad.iloc[pe])/np.pi,"peak_score_error":abs(x[po]-y[pe])})
 return pd.DataFrame(rows)

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);omega=np.linspace(0,np.pi,N_FREQS)
 config={"experiment":"35B","SMOKE_TEST":SMOKE_TEST,"C_MODE_LIST":C_MODE_LIST,"SCENARIO_LIST":SCENARIO_LIST,"N_OUTER_RUNS":N_OUTER_RUNS,"T_TOTAL":T_TOTAL,"CHANGE_POINT":CHANGE_POINT,"WINDOW_CONFIGS":WINDOW_CONFIGS,"N_FREQS":N_FREQS,"METHOD":METHOD,"warm_start_from_previous_window":True,"Q_VARIANT":Q_VARIANT,"Q_shrinkage_rho":.25,"Q_update_damping":.5,"include_A_posterior_uncertainty_in_Q":True,"a0":1e-3,"b0":1e-3,"VB_MAX_ITER":100,"C_known":True,"R_fixed_true":True,"TODO":["35C strong square-C with selected geometry","35D rectangular C after estimator improvements","35E sliding-window versus recursive adaptation"]}
 cp=os.path.join(RESULTS_DIR,"experiment_config.json")
 if os.path.exists(cp):
  with open(cp,encoding="utf8") as h:old=json.load(h)
  if old!=config and os.path.exists(os.path.join(RESULTS_DIR,"window_fit_summary_partial.csv")):raise ValueError("Existing 35B checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as h:json.dump(config,h,indent=2)
 names={"window":"window_fit_summary_partial.csv","bands":"band_integrated_adaptive_spectral_gc_partial.csv","spectra":"_spectra_partial.csv","vb":"_vb_partial.csv","uncertainty":"_uncertainty_partial.csv","similarity":"spectral_oracle_similarity_by_window_partial.csv","exog":"_exog_partial.csv","numerical":"_numerical_partial.csv"};tables={k:[] for k in names}
 for k,n in names.items():
  p=os.path.join(RESULTS_DIR,n)
  if os.path.exists(p):
   try:d=pd.read_csv(p)
   except pd.errors.EmptyDataError:d=pd.DataFrame()
   if len(d):tables[k]=[d]
 oldw=pd.concat(tables["window"],ignore_index=True) if tables["window"] else pd.DataFrame()
 completed=set(zip(oldw.run_id,oldw.window_id.astype(int))) if len(oldw) else set()
 total=sum(len(range(0,T_TOTAL-c["WINDOW_LENGTH"]+1,c["WINDOW_STEP"])) for c in WINDOW_CONFIGS)*len(C_MODE_LIST)*len(SCENARIO_LIST)*N_OUTER_RUNS-len(completed);progress=ProgressBar(total)
 for mode in C_MODE_LIST:
  for scenario in SCENARIO_LIST:
   for outer in range(N_OUTER_RUNS):
    seed=BASE_SEED+C_MODE_LIST.index(mode)*1000000+SCENARIO_LIST.index(scenario)*100000+outer;data=simulate(scenario,mode,seed)
    for cfg in WINDOW_CONFIGS:
     starts=list(range(0,T_TOTAL-cfg["WINDOW_LENGTH"]+1,cfg["WINDOW_STEP"]));run_id=f"{mode}_{scenario}_{cfg['label']}_run{outer:02d}";previous=None;oracle_cache={}
     for wid,start in enumerate(starts):
      if (run_id,wid) in completed:previous=load_state(run_id,cfg["label"],wid);continue
      base={"run_id":run_id,"C_MODE":mode,"scenario":scenario,"outer_run":outer,"method":METHOD};meta=window_meta(start,wid,data,base,cfg);active=data["A_segments"][meta["segment_label_by_center"]];began=time.perf_counter();phase="fit";previous_status="success" if previous is not None else ("not_applicable" if wid==0 else "unavailable")
      try:
       model=fit_window(data,meta,previous,seed+wid+1000*WINDOW_CONFIGS.index(cfg));fit_runtime=time.perf_counter()-began;phase="spectral";spectral_start=time.perf_counter()
       cache_key=meta["segment_label_by_center"]
       if cache_key not in oracle_cache:oracle_cache[cache_key]=spectral_frames(active,data["B"],data["Q"],omega,data,meta,"oracle_window_AQ")
       oracle=oracle_cache[cache_key];oracle_frames=(oracle[0].assign(**{k:v for k,v in meta.items()}),oracle[1].assign(**{k:v for k,v in meta.items()}),oracle[2].assign(**{k:v for k,v in meta.items()}),oracle[3],oracle[4],oracle[5])
       estimated=spectral_frames(model.A_mean_matrices_,model.B_matrices,model.Q,omega,data,meta,"estimated_window_AQ");spectral_runtime=time.perf_counter()-spectral_start;diag=diagnostics(model);start0,end0=meta["window_start"],meta["window_end"];window_data={"A":active,"B":data["B"],"Q":data["Q"],"x":data["x"][start0:end0],"mask":np.any(active!=0,axis=0)}
       row,_=recovery(np.asarray(model.A_mean_matrices_),window_data,model.filtered_state_mean_,model.smoothed_state_mean_,meta,fit_runtime,diag["n_iter"],diag["converged_all"]);row.update(diag);row.update(B_metrics(model.B_matrices,window_data,model.B_initialization_mode_,diag["final_B_change_norm"]));row.update(Q_metrics(model.Q,window_data,model))
       proxy=data["y"][start0:end0]@np.linalg.pinv(data["C"]).T;row.update({"pinv_proxy_mse":np.mean((proxy-window_data["x"])**2),"pinv_proxy_correlation":np.nanmean([correlations(proxy[:,i],window_data["x"][:,i]) for i in range(M)]),"runtime_fit_seconds":fit_runtime,"runtime_spectral_seconds":spectral_runtime,"total_runtime_seconds":time.perf_counter()-began,"warm_start_used":previous is not None,"warm_start_source_window_id":wid-1 if previous is not None else np.nan,"cold_start_first_window":wid==0,"previous_window_fit_status":previous_status,"spectral_radius_A_true_active_segment":var_companion_spectral_radius(active),"spectral_radius_A_hat":var_companion_spectral_radius(model.A_mean_matrices_),"run_status":"success",**estimated[3]})
       full_est=estimated[1].loc[estimated[1].band_name=="full_band"].nlargest(50,"integrated_score");top_pairs={(int(r.target),int(r.source)) for _,r in full_est.iterrows()};false_pairs=[pair for pair in estimated[5] if edge_class(*pair,meta,data)=="false_edge"];selection_rng=np.random.default_rng(seed+wid+900);sampled={false_pairs[k] for k in selection_rng.choice(len(false_pairs),min(50,len(false_pairs)),replace=False)} if false_pairs else set();selected=top_pairs|sampled
       oracle_saved=pd.concat([oracle_frames[0],selected_extra_spectra(oracle_frames,selected,omega,data,meta,"oracle_window_AQ")],ignore_index=True);estimated_saved=pd.concat([estimated[0],selected_extra_spectra(estimated,selected,omega,data,meta,"estimated_window_AQ")],ignore_index=True)
       coeff,edges,_=vb_details(model,window_data,meta);sim=similarity(oracle_saved,estimated_saved);tables["bands"].extend([oracle_frames[1],estimated[1]]);tables["spectra"].extend([oracle_saved,estimated_saved]);tables["exog"].extend([oracle_frames[2],estimated[2]]);tables["vb"].append(edges);tables["uncertainty"].append(coeff);tables["similarity"].append(sim);tables["numerical"].append(pd.DataFrame([{**meta,"parameter_source":"estimated_window_AQ",**estimated[3]}]));status="success";previous=model;save_state(model,run_id,cfg["label"],wid)
      except Exception as error:
       fit_runtime=time.perf_counter()-began;spectral_runtime=time.perf_counter()-locals().get("spectral_start",time.perf_counter()) if phase=="spectral" else np.nan;status="failed_"+phase;row={**meta,"runtime_fit_seconds":fit_runtime,"runtime_spectral_seconds":spectral_runtime,"total_runtime_seconds":time.perf_counter()-began,"warm_start_used":previous is not None,"warm_start_source_window_id":wid-1 if previous is not None else np.nan,"cold_start_first_window":wid==0,"previous_window_fit_status":previous_status,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()};previous=None
      tables["window"].append(pd.DataFrame([row]));frames={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in tables.items()}
      for k,n in names.items():atomic_csv(frames[k],os.path.join(RESULTS_DIR,n))
      if len(frames["bands"]):
       rec,fixed=recovery_rows(frames["bands"]);changes=change_detection(frames["bands"]);comp=score_comparison(frames["bands"],frames["vb"]);geom=geometry_summary(frames["window"])
       partials={"window_geometry_summary_partial.csv":geom,"window_level_link_recovery_summary_partial.csv":rec,"window_level_fixed_fpr_summary_partial.csv":fixed,"change_detection_summary_partial.csv":changes,"spectral_oracle_similarity_by_window_partial.csv":frames["similarity"],"score_comparison_summary_partial.csv":comp,"runtime_summary_partial.csv":frames["window"],"decision_summary_partial.csv":pd.DataFrame(),"window_geometry_tradeoff_summary_partial.csv":pd.DataFrame()}
       for n,d in partials.items():atomic_csv(d,os.path.join(RESULTS_DIR,n))
      console={k:row.get(k,np.nan) for k in ("C_MODE","scenario","outer_run","window_label","WINDOW_LENGTH","WINDOW_STEP","window_id","window_start","window_end","window_center","mixed_window","warm_start_used","runtime_fit_seconds","runtime_spectral_seconds","run_status","n_iter","practically_converged","spectral_radius_A_hat","A_support_AUPRC","B_relative_frobenius_error","Q_relative_frobenius_error","smoothed_signal_correlation","max_condition_number_Aomega","spectral_inversion_warning_count")}
      if status=="success":
       current=comp.loc[(comp.run_id==run_id)&(comp.window_id==wid)].set_index("score_family");
       for family,label in (("vb_group_snr","VB_group_SNR_AUPRC"),("transfer_spectral_gc_low_band","low_band_spectral_GC_AUPRC"),("transfer_spectral_gc_mid_band","mid_band_spectral_GC_AUPRC"),("transfer_spectral_gc_full_band","full_band_spectral_GC_AUPRC")):console[label]=current.loc[family,"AUPRC"] if family in current.index else np.nan
       current_fixed=fixed.loc[(fixed.run_id==run_id)&(fixed.window_id==wid)&(fixed.parameter_source=="estimated_window_AQ")&(fixed.target_fpr_level==.05)];
       for band in ("low_band","mid_band","full_band"):console[f"{band}_TPR_at_FPR_0p05"]=current_fixed.loc[current_fixed.band_name==band,"tpr"].mean()
      print(console,flush=True);progress.update(f"{mode} {scenario} {cfg['label']} run={outer+1} window={wid+1}/{len(starts)} {status}")
 frames={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in tables.items()};rec,fixed=recovery_rows(frames["bands"]);changes=change_detection(frames["bands"]);comp=score_comparison(frames["bands"],frames["vb"]);cal=uncertainty_summary(frames["uncertainty"]);geom=geometry_summary(frames["window"]);tf=time_frequency_similarity(frames["spectra"])
 group=["C_MODE","scenario","window_label","WINDOW_LENGTH","WINDOW_STEP"];window_group=group+["window_id","window_center","mixed_window","pure_pre_window","pure_post_window","parameter_source","score_type","band_name"];metrics=["ROC_AUC","AUPRC","best_youden_J","best_F1","TPR_at_FPR_0p01","TPR_at_FPR_0p03","TPR_at_FPR_0p05","TPR_at_FPR_0p10","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p01","F1_at_FPR_0p05"];rec_summary=rec.groupby(window_group,dropna=False)[metrics].agg(["mean","median","std"]).reset_index()
 changes=changes.merge(geom[group+["theoretical_min_detection_delay_pure_window"]],on=group,how="left");change_summary=changes.groupby(group+["band_name","edge_class","threshold_type"],dropna=False).agg(detection_rate=("detected_by_end","mean"),mean_detection_delay=("detection_delay","mean"),median_detection_delay=("detection_delay","median"),std_detection_delay=("detection_delay","std"),theoretical_min_detection_delay_pure_window=("theoretical_min_detection_delay_pure_window","first"),mean_pre_change_score=("score_pre_change_mean","mean"),mean_post_change_score=("score_post_change_mean","mean"),mean_score_change_ratio=("score_change_ratio","mean"),mean_score_change_difference=("score_change_difference","mean"),mean_oracle_estimated_change_ratio_error=("oracle_estimated_change_ratio_error","mean")).reset_index();change_summary["excess_delay_over_theoretical_min"]=change_summary.mean_detection_delay-change_summary.theoretical_min_detection_delay_pure_window
 sim_summary=tf.groupby(group+["edge_class"])[["time_frequency_correlation","time_frequency_rmse","time_frequency_mae","peak_time_error","peak_frequency_error","peak_score_error"]].mean().reset_index()
 comp_summary=comp.groupby(group+["window_id","score_family"])[["ROC_AUC","AUPRC","TPR_at_FPR_0p01","TPR_at_FPR_0p05","precision_at_FPR_0p01","precision_at_FPR_0p05","F1_at_FPR_0p05"]].mean().reset_index();success=frames["window"].loc[frames["window"].run_status=="success"]
 base_rec=rec.loc[(rec.parameter_source=="estimated_window_AQ")&rec.band_name.isin(["low_band","mid_band","full_band"])];trade=base_rec.groupby(group+["band_name"]).agg(mean_window_AUPRC_all_windows=("AUPRC","mean"),mean_TPR_at_FPR_0p05_all_windows=("TPR_at_FPR_0p05","mean")).reset_index()
 for label,mask in (("pure",base_rec.pure_pre_window|base_rec.pure_post_window),("transition",base_rec.mixed_window)):
  part=base_rec.loc[mask].groupby(group+["band_name"]).agg(**{f"mean_window_AUPRC_{label}_windows":("AUPRC","mean"),f"mean_TPR_at_FPR_0p05_{label}_windows":("TPR_at_FPR_0p05","mean")}).reset_index();trade=trade.merge(part,on=group+["band_name"],how="left")
 runtime=frames["window"].groupby(group).agg(mean_runtime_fit_seconds=("runtime_fit_seconds","mean"),total_runtime_seconds=("total_runtime_seconds","sum"),practical_convergence_rate=("practically_converged","mean"),failure_rate=("run_status",lambda x:(x!="success").mean())).reset_index();trade=trade.merge(runtime,on=group,how="left")
 for threshold,suffix in (("threshold_95","95"),("threshold_99","99")):
  cd=change_summary.loc[change_summary.threshold_type==threshold,group+["band_name","detection_rate","mean_detection_delay","excess_delay_over_theoretical_min"]].rename(columns={"detection_rate":f"detection_rate_{suffix}","mean_detection_delay":f"mean_detection_delay_{suffix}","excess_delay_over_theoretical_min":f"excess_delay_over_theoretical_min_{suffix}"});trade=trade.merge(cd,on=group+["band_name"],how="left")
 simtrade=frames["similarity"].groupby(group).spectral_curve_correlation.mean().reset_index(name="mean_spectral_curve_correlation");trade=trade.merge(simtrade,on=group,how="left");tftrade=tf.groupby(group).time_frequency_correlation.mean().reset_index(name="mean_time_frequency_correlation");trade=trade.merge(tftrade,on=group,how="left")
 outputs={"run_summary.csv":frames["window"].groupby(["run_id"]+group+["outer_run"]).agg(n_windows=("window_id","count"),n_success=("run_status",lambda x:(x=="success").sum()),total_runtime_seconds=("total_runtime_seconds","sum")).reset_index(),"window_geometry_summary.csv":geom,"window_fit_summary.csv":frames["window"],"runtime_summary.csv":runtime,"parameter_recovery_by_window.csv":success,"parameter_recovery_summary.csv":grouped_stats(success,group+["window_id","window_center"],["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diagonal_relative_frobenius_error","A_support_AUPRC","A_support_TPR_at_FPR_0p01","A_support_TPR_at_FPR_0p05"]),"B_recovery_by_window.csv":success[[c for c in success if c.startswith("B_") or c in ["run_id"]+group+["window_id","window_center"]]],"Q_recovery_by_window.csv":success[[c for c in success if c.startswith("Q_") or c in ["run_id"]+group+["window_id","window_center"]]],"latent_recovery_by_window.csv":success[[c for c in success if "signal_" in c or c.startswith("pinv_") or c in ["run_id"]+group+["window_id","window_center"]]],"band_integrated_adaptive_spectral_gc.csv":frames["bands"],"window_level_link_recovery_summary.csv":rec_summary,"window_level_fixed_fpr_summary.csv":fixed,"change_detection_summary.csv":change_summary,"window_geometry_tradeoff_summary.csv":trade,"time_frequency_gc_maps.csv":frames["spectra"],"time_frequency_similarity_summary.csv":sim_summary,"spectral_oracle_similarity_by_window.csv":frames["similarity"],"score_comparison_by_window.csv":comp,"score_comparison_summary.csv":comp_summary,"exogenous_response_by_window.csv":frames["exog"],"exogenous_response_summary.csv":frames["exog"].groupby(group+["window_id","parameter_source","source_index","band_name"])[["integrated_response_power","average_response_power","peak_response_power","peak_omega_over_pi"]].mean().reset_index(),"vb_edge_score_by_window.csv":frames["vb"],"vb_uncertainty_calibration_by_window.csv":cal,"spectral_numerical_diagnostics.csv":frames["numerical"],"decision_summary.csv":trade.sort_values(["mean_window_AUPRC_all_windows","mean_runtime_fit_seconds"],ascending=[False,True])}
 for n,d in outputs.items():atomic_csv(d,os.path.join(RESULTS_DIR,n))
 save_plots(frames["bands"],rec,frames["similarity"],changes,comp,frames["window"],cal)
 for name in os.listdir(RESULTS_DIR):
  if name.startswith("_window_state_") and name.endswith(".npz"):os.remove(os.path.join(RESULTS_DIR,name))
 print(f"Experiment 35B complete. Outputs saved under {RESULTS_DIR}/");print("Interpretation guide: compare detection-rate/delay, W=750 reliability, step 250 versus 500, long-window smearing, scenario and C-mode sensitivity, robust bands, spectral GC versus VB group-SNR, and select the default geometry. TODO: 35C strong square-C; 35D rectangular C; 35E recursive adaptation.")

if __name__=="__main__":main()
