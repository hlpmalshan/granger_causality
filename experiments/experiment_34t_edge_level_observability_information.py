"""Experiment 34T: post-hoc edge observability and information analysis.

No estimator, Stage-B perturbation, bootstrap significance test, or adaptive
window is run. Bootstrap intervals resample completed run_id clusters only.
"""
import os,json,time
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34r_stage1_rectangular_C_breakpoint as r1
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv
from src.ssm.ssm_varx_p_simulator import build_var_companion_matrix

S_DIR=os.environ.get("EXPERIMENT_34T_34S_DIR","results/experiment_34s")
R_DIR=os.environ.get("EXPERIMENT_34T_STAGE2C_DIR","results/experiment_34r_stage2c_validate_tau1")
FALLBACK_DIRS=["results/experiment_34r_stage2b_posthoc_tau_sensitivity","results/experiment_34r_stage2"]
OUT=os.environ.get("EXPERIMENT_34T_RESULTS_DIR","results/experiment_34t");BOOT=int(os.environ.get("EXPERIMENT_34T_BOOTSTRAP_RUN_RESAMPLES","500"));EPS=1e-12;EPS_INFO=1e-10
S_FILES=["experiment_config.json","decision_summary.csv","support_regime_summary.csv","run_summary.csv","support_mask_summary.csv","C_diagnostics.csv","latent_state_recovery_summary.csv","A_recovery_summary.csv","B_recovery_summary.csv","Q_recovery_summary.csv","network_recovery_summary.csv","spectral_band_recovery_summary.csv","calibration_summary.csv","group_calibration_summary.csv","observability_summary.csv","observability_edge_table.csv","observability_quartile_summary.csv","observability_correlation_summary.csv","runtime_summary.csv","A_matrix_estimates.csv"]
R_FILES=["selected_groups.csv","selected_coefficients.csv","coefficient_calibration_results.csv","calibration_summary.csv","group_calibration_results.csv","group_calibration_summary.csv","covariance_diagnostics.csv","observability_stageB_summary.csv","observability_quartile_summary.csv","balance_score_summary.csv","decision_summary.csv","runtime_summary.csv"]
METRICS={
 "edge_o_inst_mean":("instantaneous_C_R","deployable_from_C_R_only",1),"edge_o_inst_min":("instantaneous_C_R","deployable_from_C_R_only",1),"edge_o_inst_product":("instantaneous_C_R","deployable_from_C_R_only",1),
 "edge_distinguishability":("C_column_ambiguity","deployable_from_C_R_only",1),"edge_column_abs_corr":("C_column_ambiguity","deployable_from_C_R_only",-1),
 "edge_o_dyn_estA_mean_K5":("dynamic_observability","deployable_from_fitted_model",1),"edge_o_dyn_estA_mean_K10":("dynamic_observability","deployable_from_fitted_model",1),"edge_o_dyn_estA_mean_K20":("dynamic_observability","deployable_from_fitted_model",1),"edge_o_dyn_trueA_mean_K10":("dynamic_observability","diagnostic_uses_truth",1),
 "parameter_info_trace":("parameter_information","deployable_from_fitted_model",1),"parameter_info_min_eig":("parameter_information","deployable_from_fitted_model",1),"parameter_info_logdet":("parameter_information","deployable_from_fitted_model",1),"Louis_group_cov_trace":("parameter_information","deployable_from_fitted_model",-1),"VB_group_cov_trace":("parameter_information","deployable_from_fitted_model",-1)}

def load(directory,name):
 path=os.path.join(directory,name)
 if not os.path.exists(path):return pd.DataFrame()
 try:return pd.read_csv(path)
 except pd.errors.EmptyDataError:return pd.DataFrame()
def schema_audit():
 report={"sources":{},"files_found":[],"files_missing":[],"edge_level_34S_analysis_possible":False,"stageB_selected_edge_analysis_possible":False,"lightweight_reconstruction_needed":False,"metric_limitations":[]}
 for label,directory,names in (("experiment_34s",S_DIR,S_FILES),("experiment_34r_stage2c",R_DIR,R_FILES)):
  report["sources"][label]={"directory":directory,"files":{}}
  for name in names:
   path=os.path.join(directory,name)
   if not os.path.exists(path):report["files_missing"].append(path);continue
   report["files_found"].append(path)
   if name.endswith(".json"):
    with open(path,encoding="utf8") as h:value=json.load(h)
    report["sources"][label]["files"][name]={"type":"json","top_level_keys":list(value)}
   else:
    value=load(directory,name);report["sources"][label]["files"][name]={"rows":len(value),"columns":list(value.columns)}
 report["edge_level_34S_analysis_possible"]=all(os.path.exists(os.path.join(S_DIR,x)) for x in ("observability_edge_table.csv","A_matrix_estimates.csv"))
 report["stageB_selected_edge_analysis_possible"]=all(os.path.exists(os.path.join(R_DIR,x)) for x in ("group_calibration_results.csv","activity_score_diagnostics.csv"))
 report["lightweight_reconstruction_needed"]=not report["edge_level_34S_analysis_possible"]
 if report["edge_level_34S_analysis_possible"]:report["metric_limitations"].append("No refit needed: C and dynamic K=5/10/20 metrics are deterministically reconstructed from saved seeds and A matrices.")
 report["metric_limitations"] += ["Per-edge ordinary-VB covariance blocks were not saved by 34S; VB_group_cov_trace and ordinary group D2 are unavailable.","34S did not save coefficient-level covariance rows; per-lag ordinary/Louis coefficient coverage is unavailable.","Stage 2C saved group covariance traces and D2, but not full 2x2 blocks; eigenvalue-based Stage-B information metrics are unavailable.","Edge-level deviance scores were not saved; detected_by_deviance_best_at_FPR_0p05 is unavailable."]
 return report

def normalize(x):x=np.asarray(x,float);return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x)+EPS)
def C_for(mx,my,T,network,replicate):
 seed=r1.BASE_SEED+mx*100000+network*1000+my*10000+T+replicate;return r1.make_C(my,mx,"gaussian_isotropic",seed)
def dynamic(A,C,R,K):
 F=build_var_companion_matrix(A);Caug=np.hstack([C,np.zeros_like(C)]);base=Caug.T@np.linalg.solve(R,Caug);power=np.eye(len(F));W=np.zeros_like(F)
 for _ in range(K+1):W+=power.T@base@power;power=power@F
 return normalize(np.diag(W)[:C.shape[1]])
def edge_triplet(values,target,source,prefix,row):
 row[f"{prefix}_mean"]=(values[target]+values[source])/2;row[f"{prefix}_min"]=min(values[target],values[source]);row[f"{prefix}_product"]=values[target]*values[source]

def reconstruct_34s_edge_table():
 edge=load(S_DIR,"observability_edge_table.csv");amat=load(S_DIR,"A_matrix_estimates.csv")
 if not len(edge) or not len(amat):return edge,pd.DataFrame()
 rebuilt=[];sources=[]
 for run_id,g in edge.groupby("run_id",sort=False):
  first=g.iloc[0];mx,my,T=int(first.M_x),int(first.M_y),int(first["T"]);net,rep=int(first.true_network_id),int(first.replicate_id);C=C_for(mx,my,T,net,rep);R=.6*np.eye(my);J=C.T@np.linalg.solve(R,C);inst=normalize(np.diag(J));Arows=amat[amat.run_id==run_id];Ahat=np.zeros((2,mx,mx));Atrue=np.zeros_like(Ahat)
  for row in Arows.itertuples():Ahat[int(row.lag)-1,int(row.target),int(row.source)]=row.A_hat;Atrue[int(row.lag)-1,int(row.target),int(row.source)]=row.A_true
  dyn={};
  for K in (5,10,20):dyn[("true",K)]=dynamic(Atrue,C,R,K);dyn[("est",K)]=dynamic(Ahat,C,R,K)
  corr=np.corrcoef(C.T);base={"run_id":run_id,"M_x":mx,"M_y":my,"observation_ratio":my/mx,"support_regime":first.support_regime,"support_density":first.support_density,"true_network_id":net,"replicate_id":rep}
  for source in range(mx):sources.append({**base,"source_j":source,"o_inst":J[source,source],"o_inst_norm":inst[source],"source_column_norm":np.linalg.norm(C[:,source]),"source_information_norm":np.sqrt(max(J[source,source],0)),**{f"o_dyn_trueA_K{K}":dyn[("true",K)][source] for K in (5,10,20)},**{f"o_dyn_estA_K{K}":dyn[("est",K)][source] for K in (5,10,20)}})
  for row in g.to_dict("records"):
   target,source=int(row["target_i"]),int(row["source_j"]);edge_triplet(inst,target,source,"edge_o_inst",row);row["edge_column_abs_corr"]=abs(corr[target,source]);row["edge_distinguishability"]=1-row["edge_column_abs_corr"]
   for K in (5,10,20):edge_triplet(dyn[("true",K)],target,source,f"edge_o_dyn_trueA_K{K}",row);edge_triplet(dyn[("est",K)],target,source,f"edge_o_dyn_estA_K{K}",row)
   row["edge_o_dyn_trueA_mean_K10"]=row["edge_o_dyn_trueA_K10_mean"];row["edge_o_dyn_estA_mean_K5"]=row["edge_o_dyn_estA_K5_mean"];row["edge_o_dyn_estA_mean_K10"]=row["edge_o_dyn_estA_K10_mean"];row["edge_o_dyn_estA_mean_K20"]=row["edge_o_dyn_estA_K20_mean"]
   row["parameter_info_trace"]=row.get("parameter_information_trace",np.nan);row["parameter_info_min_eig"]=row.get("parameter_information_min_eigenvalue",np.nan);row["parameter_info_logdet"]=row.get("parameter_information_logdet",np.nan);minimum=row["parameter_info_min_eig"];row["parameter_info_condition"]=(row["parameter_info_trace"]-minimum)/minimum if np.isfinite(minimum) and minimum>0 else np.nan
   row["Louis_group_cov_logdet"]=-row["parameter_info_logdet"] if np.isfinite(row["parameter_info_logdet"]) else np.nan;row["Louis_group_cov_max_eig"]=1/minimum if np.isfinite(minimum) and minimum>0 else np.nan;maximum_info=row["parameter_info_trace"]-minimum;row["Louis_group_cov_min_eig"]=1/maximum_info if np.isfinite(maximum_info) and maximum_info>0 else np.nan;row["VB_group_cov_trace"]=np.nan;row["VB_parameter_info_trace"]=np.nan
   row["group_estimation_error_norm"]=row.pop("group_estimation_error",np.nan);row["group_squared_error"]=row["group_estimation_error_norm"]**2;row["active_or_zero"]="active" if row["true_edge"] else "zero";row["ordinary_vb_group_covered_95"]=row.get("ordinary_vb_group_coverage",np.nan);row["stabilized_louis_group_covered_95"]=row.get("stabilized_louis_group_coverage",np.nan);row["stabilized_louis_group_cov_trace"]=row.get("Louis_group_cov_trace",np.nan)
   row.update({"coefficient_abs_error_lag1":float(abs(Ahat[0,target,source]-Atrue[0,target,source])),"coefficient_abs_error_lag2":float(abs(Ahat[1,target,source]-Atrue[1,target,source])),"signed_error_lag1":float(Ahat[0,target,source]-Atrue[0,target,source]),"signed_error_lag2":float(Ahat[1,target,source]-Atrue[1,target,source]),"detected_by_deviance_best_at_FPR_0p05":np.nan,"ordinary_vb_group_D2":np.nan,"stabilized_louis_group_D2":np.nan,"ordinary_vb_group_cov_trace":np.nan,"ordinary_vb_group_std_z":np.nan,"stabilized_louis_group_std_z":np.nan,"StageB_available":False,"full_StageB_group_covered_95":np.nan,"soft_tau1_group_covered_95":np.nan,"hard_tau1_group_covered_95":np.nan,"global_lambda_0p75_group_covered_95":np.nan,"StageB_over_Louis_trace_ratio":np.nan,"StageB_rescue_indicator":np.nan,"StageB_needed_indicator":np.nan,"soft_tau1_weight":np.nan,"soft_tau2p5_weight":np.nan,"suppressed_by_tau1":np.nan,"suppressed_by_tau2p5":np.nan,"inversion_regularized":False})
   for metric,(family,mtype,_) in METRICS.items():row[f"metric_type__{metric}"]=mtype
   rebuilt.append(row)
 return pd.DataFrame(rebuilt),pd.DataFrame(sources)

def stageB_table():
 selected=load(R_DIR,"selected_groups.csv");group=load(R_DIR,"group_calibration_results.csv");activity=load(R_DIR,"activity_score_diagnostics.csv");coeff=load(R_DIR,"coefficient_calibration_results.csv")
 if not len(selected) or not len(group):return pd.DataFrame()
 keys=["config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","run_id","target","source"]
 base=selected.drop_duplicates(keys).copy();g=group.rename(columns={"group_target":"target","group_source":"source"});gkeys=keys+["covariance_estimator"]
 wide=g.pivot_table(index=keys,columns="covariance_estimator",values=["D2","D2_below_chi2_95_df2","group_cov_trace","group_error_norm"],aggfunc="first");wide.columns=[f"{a}__{b}" for a,b in wide.columns];base=base.merge(wide.reset_index(),on=keys,how="left")
 akeys=["config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","run_id","group_target","group_source"]
 activity=activity.rename(columns={"group_target":"target","group_source":"source"});akeys=[x.replace("group_target","target").replace("group_source","source") for x in akeys];base=base.merge(activity[akeys+[c for c in ["soft_tau1_weight","soft_tau2p5_weight","suppressed_by_soft_tau1","suppressed_by_original_tau2p5","stageB_over_louis_trace_ratio","group_snr_louis"] if c in activity]],on=akeys,how="left")
 c=coeff[coeff.covariance_estimator.isin(["ordinary_vb","stabilized_louis_eta_0p70_tau_0p90"])].copy();cp=c.pivot_table(index=keys,columns=["covariance_estimator","lag"],values=["ci95_contains_true","signed_error","abs_error"],aggfunc="first");cp.columns=[f"{value}__{est}__lag{int(lag)}" for value,est,lag in cp.columns];base=base.merge(cp.reset_index(),on=keys,how="left")
 rows=[]
 for row in base.to_dict("records"):
  mx,my,T,net,rep=int(row["M_x"]),int(row["M_y"]),int(row["T"]),int(row["true_network_id"]),int(row["replicate_id"]);target,source=int(row["target"]),int(row["source"]);C=C_for(mx,my,T,net,rep);R=.6*np.eye(my);J=C.T@np.linalg.solve(R,C);inst=normalize(np.diag(J));Atrue,_,_=j.fixed_network(mx,r1.BASE_SEED+mx*100000+net*1000);dyn_true=dynamic(Atrue,C,R,10);corr=np.corrcoef(C.T);out={**row,"target_i":target,"source_j":source,"support_regime":"full_candidate_network","support_density":1.,"true_edge":bool(row["true_edge"]),"active_or_zero":"active" if row["true_edge"] else "zero","edge_column_abs_corr":abs(corr[target,source]),"edge_distinguishability":1-abs(corr[target,source]),"StageB_available":True,"estimated_group_norm":row.get("estimated_group_norm",np.nan),"true_group_norm":np.nan,"group_estimation_error_norm":row.get("group_error_norm__ordinary_vb",np.nan),"group_squared_error":row.get("group_error_norm__ordinary_vb",np.nan)**2,"ordinary_vb_group_covered_95":row.get("D2_below_chi2_95_df2__ordinary_vb",np.nan),"stabilized_louis_group_covered_95":row.get("D2_below_chi2_95_df2__stabilized_louis_eta_0p70_tau_0p90",np.nan),"full_StageB_group_covered_95":row.get("D2_below_chi2_95_df2__lrvb_stageB_smoother_feedback_psd_projected",np.nan),"soft_tau1_group_covered_95":row.get("D2_below_chi2_95_df2__soft_stageB_logistic_c4_tau1p0",np.nan),"hard_tau1_group_covered_95":row.get("D2_below_chi2_95_df2__hard_stageB_group_snr_tau1p0",np.nan),"global_lambda_0p75_group_covered_95":row.get("D2_below_chi2_95_df2__global_lambda_stageB_0p75",np.nan),"ordinary_vb_group_D2":row.get("D2__ordinary_vb",np.nan),"stabilized_louis_group_D2":row.get("D2__stabilized_louis_eta_0p70_tau_0p90",np.nan),"ordinary_vb_group_cov_trace":row.get("group_cov_trace__ordinary_vb",np.nan),"stabilized_louis_group_cov_trace":row.get("group_cov_trace__stabilized_louis_eta_0p70_tau_0p90",np.nan),"StageB_group_cov_trace":row.get("group_cov_trace__lrvb_stageB_smoother_feedback_psd_projected",np.nan),"VB_group_cov_trace":row.get("group_cov_trace__ordinary_vb",np.nan),"Louis_group_cov_trace":row.get("group_cov_trace__stabilized_louis_eta_0p70_tau_0p90",np.nan),"ordinary_vb_group_std_z":np.sqrt(max(row.get("D2__ordinary_vb",np.nan),0)/2),"stabilized_louis_group_std_z":np.sqrt(max(row.get("D2__stabilized_louis_eta_0p70_tau_0p90",np.nan),0)/2),"soft_tau1_weight":row.get("soft_tau1_weight",np.nan),"soft_tau2p5_weight":row.get("soft_tau2p5_weight",np.nan),"suppressed_by_tau1":row.get("suppressed_by_soft_tau1",np.nan),"suppressed_by_tau2p5":row.get("suppressed_by_original_tau2p5",np.nan),"detected_by_model_A_at_FPR_0p05":np.nan,"detected_by_Louis_SNR_at_FPR_0p05":np.nan,"detected_by_spectral_best_band_at_FPR_0p05":np.nan,"detected_by_deviance_best_at_FPR_0p05":np.nan,"missed_true_edge_indicator":np.nan,"false_positive_indicator":np.nan,"parameter_info_trace":np.nan,"parameter_info_min_eig":np.nan,"parameter_info_logdet":np.nan,"VB_parameter_info_trace":np.nan,"inversion_regularized":False}
  edge_triplet(inst,target,source,"edge_o_inst",out);edge_triplet(dyn_true,target,source,"edge_o_dyn_trueA_K10",out);out["edge_o_dyn_trueA_mean_K10"]=out["edge_o_dyn_trueA_K10_mean"]
  for K in (5,10,20):out[f"edge_o_dyn_estA_mean_K{K}"]=np.nan
  for lag in (1,2):out[f"coefficient_abs_error_lag{lag}"]=row.get(f"abs_error__ordinary_vb__lag{lag}",np.nan);out[f"signed_error_lag{lag}"]=row.get(f"signed_error__ordinary_vb__lag{lag}",np.nan);out[f"ordinary_vb_coefficient_coverage_95_lag{lag}"]=row.get(f"ci95_contains_true__ordinary_vb__lag{lag}",np.nan);out[f"stabilized_louis_coefficient_coverage_95_lag{lag}"]=row.get(f"ci95_contains_true__stabilized_louis_eta_0p70_tau_0p90__lag{lag}",np.nan)
  out["StageB_over_Louis_trace_ratio"]=out["StageB_group_cov_trace"]/(out["Louis_group_cov_trace"]+EPS_INFO);out["StageB_increment_trace"]=out["StageB_group_cov_trace"]-out["Louis_group_cov_trace"];out["StageB_rescue_indicator"]=bool(out["full_StageB_group_covered_95"] and not out["stabilized_louis_group_covered_95"])
  for metric,(family,mtype,_) in METRICS.items():out[f"metric_type__{metric}"]=mtype
  rows.append(out)
 result=pd.DataFrame(rows);median_by_my=result[result.true_edge].groupby("M_y").StageB_over_Louis_trace_ratio.median();median=result.M_y.map(median_by_my);result["StageB_needed_indicator"]=(result.StageB_over_Louis_trace_ratio>median)|(~result.stabilized_louis_group_covered_95.fillna(False).astype(bool))
 return result

def spearman(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 return float(np.corrcoef(rankdata(x[ok]),rankdata(y[ok]))[0,1]) if ok.sum()>2 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan
def pearson(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 return float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>2 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan
def curve_metric(y,score,name="ROC_AUC"):
 y=np.asarray(y);score=np.asarray(score,float);ok=pd.notna(y)&np.isfinite(score)
 if ok.sum()<2 or len(np.unique(y[ok].astype(bool)))<2:return np.nan
 metrics,_=j.safe_curve(y[ok].astype(bool),score[ok]);return metrics.get(name,np.nan)
def run_bootstrap(g,stat,seed):
 values=[]
 for _,part in g.groupby("run_id"):
  value=stat(part)
  if np.isfinite(value):values.append(value)
 values=np.asarray(values,float)
 if not len(values):return np.nan,np.nan,np.nan,0
 rng=np.random.default_rng(seed);samples=np.asarray([np.mean(rng.choice(values,len(values),replace=True)) for _ in range(BOOT)]) if len(values)>1 else values
 return float(values.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975)),len(values)
def usable_metrics(frame):return [metric for metric in METRICS if metric in frame and frame[metric].notna().any()]

def error_analysis(edge):
 rows=[];keys=["M_y","support_regime","support_density"]
 for values,g in edge.groupby(keys,dropna=False):
  for edge_class,part in (("true_edge",g[g.true_edge.astype(bool)]),("false_edge",g[~g.true_edge.astype(bool)])):
   for number,metric in enumerate(usable_metrics(part)):
    orientation=METRICS[metric][2];p=run_bootstrap(part,lambda x:pearson(orientation*x[metric],x.group_estimation_error_norm),1000+number);s=run_bootstrap(part,lambda x:spearman(orientation*x[metric],x.group_estimation_error_norm),2000+number)
    rows.append({**dict(zip(keys,values)),"edge_class":edge_class,"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1],"pearson_correlation":p[0],"pearson_bootstrap_ci95_lower":p[1],"pearson_bootstrap_ci95_upper":p[2],"spearman_correlation":s[0],"spearman_bootstrap_ci95_lower":s[1],"spearman_bootstrap_ci95_upper":s[2],"n_runs":s[3],"n_edges":len(part),"p_value":np.nan})
 return pd.DataFrame(rows)
def detection_analysis(edge):
 rows=[];outcomes=["detected_by_model_A_at_FPR_0p05","detected_by_spectral_best_band_at_FPR_0p05"]
 true=edge[edge.true_edge.astype(bool)]
 for values,g in true.groupby(["M_y","support_regime","support_density"],dropna=False):
  for outcome in outcomes:
   for number,metric in enumerate(usable_metrics(g)):
    orientation=METRICS[metric][2];auc=run_bootstrap(g,lambda x:curve_metric(x[outcome],orientation*x[metric],"ROC_AUC"),3000+number);auprc=run_bootstrap(g,lambda x:curve_metric(x[outcome],orientation*x[metric],"AUPRC"),4000+number);sp=run_bootstrap(g,lambda x:spearman(orientation*x[metric],x[outcome].astype(float)),5000+number);det=g[g[outcome]==True][metric];miss=g[g[outcome]==False][metric]
    rows.append({**dict(zip(["M_y","support_regime","support_density"],values)),"outcome":outcome,"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1],"mean_observability_detected":det.mean(),"mean_observability_missed":miss.mean(),"difference_detected_minus_missed":det.mean()-miss.mean(),"ROC_AUC":auc[0],"ROC_AUC_ci95_lower":auc[1],"ROC_AUC_ci95_upper":auc[2],"AUPRC":auprc[0],"spearman_detection":sp[0],"n_runs":max(auc[3],sp[3]),"n_true_edges":len(g)})
 return pd.DataFrame(rows)
def coverage_analysis(edge):
 rows=[];true=edge[edge.true_edge.astype(bool)];methods=[("ordinary_vb","ordinary_vb_group_covered_95"),("stabilized_louis","stabilized_louis_group_covered_95")]
 for values,g in true.groupby(["M_y","support_regime","support_density"],dropna=False):
  for method,outcome in methods:
   for number,metric in enumerate(usable_metrics(g)):
    orientation=METRICS[metric][2];failure=~g[outcome].astype(bool);auc=run_bootstrap(g.assign(_failure=failure),lambda x:curve_metric(x._failure,-orientation*x[metric],"ROC_AUC"),6000+number);pr=run_bootstrap(g.assign(_failure=failure),lambda x:curve_metric(x._failure,-orientation*x[metric],"AUPRC"),7000+number)
    covered=g[g[outcome]==True][metric];uncovered=g[g[outcome]==False][metric]
    rows.append({**dict(zip(["M_y","support_regime","support_density"],values)),"covariance_method":method,"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1],"coverage_failure_ROC_AUC":auc[0],"ROC_AUC_ci95_lower":auc[1],"ROC_AUC_ci95_upper":auc[2],"coverage_failure_AUPRC":pr[0],"mean_observability_covered":covered.mean(),"mean_observability_uncovered":uncovered.mean(),"quartile_coverage_gradient":np.nan,"n_runs":auc[3],"n_true_edges":len(g)})
 return pd.DataFrame(rows)
def stageB_analysis(stage):
 rows=[];true=stage[stage.true_edge.astype(bool)].copy()
 for my,g in true.groupby("M_y"):
  for number,metric in enumerate(usable_metrics(g)):
   orientation=METRICS[metric][2];ratio=run_bootstrap(g,lambda x:spearman(orientation*x[metric],x.StageB_over_Louis_trace_ratio),8000+number);rescue=run_bootstrap(g,lambda x:curve_metric(x.StageB_rescue_indicator,-orientation*x[metric]),9000+number);tau1=run_bootstrap(g,lambda x:curve_metric(x.suppressed_by_tau1,-orientation*x[metric]),10000+number);tau25=run_bootstrap(g,lambda x:curve_metric(x.suppressed_by_tau2p5,-orientation*x[metric]),11000+number)
   rows.append({"M_y":my,"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1],"spearman_StageB_over_Louis_trace_ratio":ratio[0],"StageB_rescue_ROC_AUC":rescue[0],"StageB_rescue_ROC_AUC_ci95_lower":rescue[1],"StageB_rescue_ROC_AUC_ci95_upper":rescue[2],"tau1_suppression_ROC_AUC":tau1[0],"tau2p5_suppression_ROC_AUC":tau25[0],"mean_observability_rescued":g[g.StageB_rescue_indicator==True][metric].mean(),"mean_observability_not_rescued":g[g.StageB_rescue_indicator==False][metric].mean(),"mean_observability_tau1_suppressed":g[g.suppressed_by_tau1==True][metric].mean(),"mean_observability_tau1_not_suppressed":g[g.suppressed_by_tau1==False][metric].mean(),"n_runs":max(ratio[3],rescue[3]),"n_selected_true_edges":len(g)})
 return pd.DataFrame(rows)
def quartile_analysis(edge,stage):
 combined=pd.concat([edge,stage],ignore_index=True,sort=False);true=combined[combined.true_edge.astype(bool)].copy();rows=[]
 primary=[m for m in METRICS if m in true and true[m].notna().any()]
 for metric in primary:
  part=true[true[metric].notna()].copy();keys=["M_y","support_regime","true_network_id","replicate_id"]
  part["observability_quartile"]=part.groupby(keys)[metric].transform(lambda s:pd.qcut(s.rank(method="first"),4,labels=["Q1_low","Q2","Q3","Q4_high"]))
  for values,g in part.groupby(["M_y","support_regime","support_density","observability_quartile"],observed=False,dropna=False):
   rows.append({**dict(zip(["M_y","support_regime","support_density","observability_quartile"],values)),"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1],"n_true_edges":len(g),"mean_group_estimation_error":g.group_estimation_error_norm.mean(),"median_group_estimation_error":g.group_estimation_error_norm.median(),"model_A_detection_rate_at_FPR_0p05":g.detected_by_model_A_at_FPR_0p05.mean(),"spectral_detection_rate_at_FPR_0p05":g.detected_by_spectral_best_band_at_FPR_0p05.mean(),"Louis_group_coverage_95":g.stabilized_louis_group_covered_95.mean(),"ordinary_vb_group_coverage_95":g.ordinary_vb_group_covered_95.mean(),"mean_Louis_group_cov_trace":g.Louis_group_cov_trace.mean(),"mean_parameter_info_trace":g.parameter_info_trace.mean(),"mean_StageB_over_Louis_trace_ratio":g.StageB_over_Louis_trace_ratio.mean(),"StageB_rescue_rate":g.StageB_rescue_indicator.mean(),"tau1_suppression_rate":g.suppressed_by_tau1.mean(),"tau2p5_suppression_rate":g.suppressed_by_tau2p5.mean()})
 return pd.DataFrame(rows)

def metric_comparison(error,detection,coverage,stage_need):
 rows=[]
 for metric,(family,mtype,_) in METRICS.items():
  e=error[(error.metric_name==metric)&error.edge_class.eq("true_edge")];d=detection[(detection.metric_name==metric)&detection.outcome.eq("detected_by_model_A_at_FPR_0p05")];c=coverage[(coverage.metric_name==metric)&coverage.covariance_method.eq("stabilized_louis")];s=stage_need[stage_need.metric_name==metric] if len(stage_need) else pd.DataFrame()
  error_score=max(0,-e.spearman_correlation.mean()) if len(e) else np.nan;detection_score=d.ROC_AUC.mean() if len(d) else np.nan;coverage_score=c.coverage_failure_ROC_AUC.mean() if len(c) else np.nan;stage_score=s.StageB_rescue_ROC_AUC.mean() if len(s) else np.nan
  rows.append({"metric_name":metric,"metric_family":family,"metric_type":mtype,"uses_truth":mtype=="diagnostic_uses_truth","deployable":mtype!="diagnostic_uses_truth","error_score":error_score,"detection_score":detection_score,"coverage_failure_score":coverage_score,"stageB_need_score":stage_score})
 out=pd.DataFrame(rows);local=[]
 for metric in out.metric_name:
  e=error[(error.metric_name==metric)&error.edge_class.eq("true_edge")]
  for values,g in e.groupby(["M_y","support_regime"]):local.append({"metric_name":metric,"M_y":values[0],"support_regime":values[1],"local_score":max(0,-g.spearman_correlation.mean())})
 local=pd.DataFrame(local)
 if len(local):
  local["rank"]=local.groupby(["M_y","support_regime"]).local_score.rank(pct=True);stability=local.groupby("metric_name")["rank"].std().map(lambda x:np.clip(1-x/.5,0,1)).rename("stability_score");out=out.merge(stability,on="metric_name",how="left")
 else:out["stability_score"]=np.nan
 available=out.stageB_need_score.notna();out["overall_metric_score"]=np.where(available,.25*out.error_score+.25*out.detection_score+.25*out.coverage_failure_score+.15*out.stageB_need_score+.10*out.stability_score,.35*out.error_score+.30*out.detection_score+.25*out.coverage_failure_score+.10*out.stability_score)
 return out
def regime_comparison(error,detection,coverage,stage_need):
 rows=[]
 for metric in METRICS:
  row={"metric_name":metric,"metric_family":METRICS[metric][0],"metric_type":METRICS[metric][1]}
  for my in (40,15):
   row[f"Spearman_error_corr_My{my}"]=error[(error.metric_name==metric)&error.edge_class.eq("true_edge")&(error.M_y==my)].spearman_correlation.mean();row[f"detection_ROC_AUC_My{my}"]=detection[(detection.metric_name==metric)&detection.outcome.eq("detected_by_model_A_at_FPR_0p05")&(detection.M_y==my)].ROC_AUC.mean();row[f"coverage_failure_ROC_AUC_My{my}"]=coverage[(coverage.metric_name==metric)&coverage.covariance_method.eq("stabilized_louis")&(coverage.M_y==my)].coverage_failure_ROC_AUC.mean();row[f"stageB_need_ROC_AUC_My{my}"]=stage_need[(stage_need.metric_name==metric)&(stage_need.M_y==my)].StageB_rescue_ROC_AUC.mean() if len(stage_need) else np.nan
  row["error_correlation_difference_My15_minus_My40"]=row["Spearman_error_corr_My15"]-row["Spearman_error_corr_My40"];row["useful_only_underdetermined"]=bool(np.isfinite(row["detection_ROC_AUC_My15"]) and row["detection_ROC_AUC_My15"]>=.65 and (not np.isfinite(row["detection_ROC_AUC_My40"]) or row["detection_ROC_AUC_My40"]<.6));row["stable_across_regimes"]=bool(np.isfinite(row["Spearman_error_corr_My15"]) and np.isfinite(row["Spearman_error_corr_My40"]) and abs(row["error_correlation_difference_My15_minus_My40"])<=.15);rows.append(row)
 return pd.DataFrame(rows)
def support_comparison(error,detection,coverage):
 e=error[error.edge_class.eq("true_edge")].groupby(["M_y","support_regime","support_density","metric_name"],dropna=False).spearman_correlation.mean().reset_index();d=detection[detection.outcome.eq("detected_by_model_A_at_FPR_0p05")].groupby(["M_y","support_regime","support_density","metric_name"],dropna=False).ROC_AUC.mean().reset_index();c=coverage[coverage.covariance_method.eq("stabilized_louis")].groupby(["M_y","support_regime","support_density","metric_name"],dropna=False).coverage_failure_ROC_AUC.mean().reset_index();out=e.merge(d,on=["M_y","support_regime","support_density","metric_name"],how="outer").merge(c,on=["M_y","support_regime","support_density","metric_name"],how="outer");out["observability_predictiveness_score"]=np.nanmean(np.c_[np.maximum(0,-out.spearman_correlation),out.ROC_AUC,out.coverage_failure_ROC_AUC],axis=1);out["diagnostic_structural_prior_only"]=~out.support_regime.eq("full_candidate_network");return out
def parameter_summary(edge,stage):
 allrows=pd.concat([edge.assign(data_source="34S"),stage.assign(data_source="34R_stage2C")],ignore_index=True,sort=False);cols=[c for c in ["parameter_info_trace","parameter_info_min_eig","parameter_info_logdet","parameter_info_condition","Louis_group_cov_trace","Louis_group_cov_logdet","Louis_group_cov_min_eig","Louis_group_cov_max_eig","VB_group_cov_trace","VB_parameter_info_trace","StageB_group_cov_trace","StageB_over_Louis_trace_ratio"] if c in allrows]
 return allrows.groupby(["data_source","M_y","support_regime","true_edge"],dropna=False)[cols].agg(["mean","median","std","count"]).reset_index().set_axis(["_".join(str(x) for x in col if x) if isinstance(col,tuple) else col for col in allrows.groupby(["data_source","M_y","support_regime","true_edge"],dropna=False)[cols].agg(["mean","median","std","count"]).reset_index().columns],axis=1)
def decision_table(compare,mycomp,quart,stage_need):
 out=compare.merge(mycomp,on=["metric_name","metric_family","metric_type"],how="left");rows=[]
 for row in out.to_dict("records"):
  metric=row["metric_name"];q=quart[quart.metric_name==metric];mon=[]
  for _,g in q.groupby(["M_y","support_regime"]):
   g=g.sort_values("observability_quartile");mon.append(spearman(np.arange(len(g)),-g.mean_group_estimation_error))
  row["quartile_monotonicity_score"]=np.nanmean(mon) if mon else np.nan;row["M_y_values_supported"]=";".join(map(str,sorted(q.M_y.unique()))) if len(q) else "";row["support_regimes_supported"]=";".join(sorted(q.support_regime.unique())) if len(q) else ""
  for my in (40,15):row[f"tau1_suppression_ROC_AUC_My{my}"]=stage_need[(stage_need.metric_name==metric)&(stage_need.M_y==my)].tau1_suppression_ROC_AUC.mean() if len(stage_need) else np.nan
  rows.append(row)
 result=pd.DataFrame(rows);result["recommended_primary_observability_metric"]=False;result["recommended_secondary_observability_metric"]=False
 deploy=result[result.deployable&result.overall_metric_score.notna()];diagnostic=result[result.overall_metric_score.notna()]
 if len(deploy):result.loc[result.metric_name.eq(deploy.loc[deploy.overall_metric_score.idxmax(),"metric_name"]),"recommended_primary_observability_metric"]=True
 if len(diagnostic):
  remaining=diagnostic[~diagnostic.metric_name.eq(deploy.loc[deploy.overall_metric_score.idxmax(),"metric_name"])] if len(deploy) else diagnostic
  if len(remaining):result.loc[result.metric_name.eq(remaining.loc[remaining.overall_metric_score.idxmax(),"metric_name"]),"recommended_secondary_observability_metric"]=True
 best=deploy.loc[deploy.overall_metric_score.idxmax()] if len(deploy) else None
 strong=bool(best is not None and best.overall_metric_score>=.65 and best.error_score>=.15 and best.detection_score>=.6 and best.coverage_failure_score>=.6);result["recommended_next_step"]="observability-adaptive Stage-B covariance" if strong else "stronger sparse posterior geometry or empirical calibration";result["notes"]="Observability is an estimability diagnostic, never edge-existence evidence. Truth-based dynamic metrics are diagnostic only."
 return result

def roc_points(y,score):
 y=np.asarray(y);score=np.asarray(score,float);ok=pd.notna(y)&np.isfinite(score);y=y[ok].astype(bool);score=score[ok]
 if len(np.unique(y))<2:return np.array([]),np.array([])
 order=np.argsort(-score);y=y[order];tp=np.r_[0,np.cumsum(y)];fp=np.r_[0,np.cumsum(~y)];return fp/max((~y).sum(),1),tp/max(y.sum(),1)
def savefig(name):plt.tight_layout();plt.savefig(os.path.join(OUT,"plots",name),dpi=160);plt.close()
def scatter_plot(frame,x,name):
 g=frame[frame.true_edge.astype(bool)&frame[x].notna()];plt.scatter(g[x],g.group_estimation_error_norm,s=5,alpha=.25);plt.xlabel(x);plt.ylabel("group estimation error");savefig(name)
def plots(edge,stage,quart,compare,mycomp,support,coverage,stage_need):
 os.makedirs(os.path.join(OUT,"plots"),exist_ok=True);scatter_plot(edge,"edge_o_inst_mean","01_error_instantaneous.png");scatter_plot(edge,"edge_o_dyn_estA_mean_K10","02_error_dynamic.png");scatter_plot(edge,"parameter_info_trace","03_error_parameter_information.png")
 primary="edge_o_inst_mean";q=quart[quart.metric_name.eq(primary)];order=["Q1_low","Q2","Q3","Q4_high"]
 for column,name in (("model_A_detection_rate_at_FPR_0p05","04_detection_by_quartile.png"),("spectral_detection_rate_at_FPR_0p05","05_spectral_detection_by_quartile.png"),("Louis_group_coverage_95","06_Louis_coverage_by_quartile.png"),("tau1_suppression_rate","08_tau1_suppression_by_quartile.png"),("tau2p5_suppression_rate","09_tau2p5_suppression_by_quartile.png")):
  for my,g in q.groupby("M_y"):
   z=g.groupby("observability_quartile",observed=False)[column].mean().reindex(order);plt.plot(range(4),z,marker="o",label=f"M_y={my}")
  plt.xticks(range(4),order);plt.ylabel(column);plt.legend();savefig(name)
 if len(stage):plt.scatter(stage.edge_o_inst_mean,stage.StageB_over_Louis_trace_ratio,s=8,alpha=.4);plt.xlabel("instantaneous observability");plt.ylabel("StageB/Louis trace ratio")
 else:plt.text(.5,.5,"Stage-B data unavailable",ha="center");plt.axis("off")
 savefig("07_stageB_ratio_vs_observability.png")
 compare.sort_values("overall_metric_score").plot.barh(x="metric_name",y="overall_metric_score",legend=False);savefig("10_metric_comparison.png")
 best=compare.loc[compare.overall_metric_score.idxmax(),"metric_name"] if compare.overall_metric_score.notna().any() else primary;b=mycomp[mycomp.metric_name.eq(best)]
 if len(b):plt.bar(["M_y=40","M_y=15"],[b.Spearman_error_corr_My40.iloc[0],b.Spearman_error_corr_My15.iloc[0]])
 plt.ylabel("Spearman error correlation");plt.title(best);savefig("11_My_comparison_best_metric.png")
 s=support[support.metric_name.eq(best)]
 for my,g in s.groupby("M_y"):g=g.sort_values("support_density");plt.plot(g.support_density,g.observability_predictiveness_score,marker="o",label=f"M_y={my}")
 plt.xlabel("support density");plt.ylabel("predictiveness");plt.legend();savefig("12_support_regime_best_metric.png")
 family=compare.groupby("metric_family").overall_metric_score.mean().sort_values();family.plot.barh();plt.xlabel("overall metric score");savefig("13_metric_family_comparison.png")
 top=compare.nlargest(3,"overall_metric_score").metric_name
 true=edge[edge.true_edge.astype(bool)]
 for metric in top:
  orientation=METRICS[metric][2];fpr,tpr=roc_points(~true.stabilized_louis_group_covered_95.astype(bool),-orientation*true[metric]);plt.plot(fpr,tpr,label=metric)
 plt.xlabel("FPR");plt.ylabel("TPR");plt.legend(fontsize=7);savefig("14_coverage_failure_ROC_top3.png")
 if len(stage):
  true=stage[stage.true_edge.astype(bool)]
  for metric in top:
   if metric in true and true[metric].notna().any():fpr,tpr=roc_points(true.StageB_rescue_indicator,-METRICS[metric][2]*true[metric]);plt.plot(fpr,tpr,label=metric)
  plt.xlabel("FPR");plt.ylabel("TPR");plt.legend(fontsize=7)
 else:plt.text(.5,.5,"Stage-B data unavailable",ha="center");plt.axis("off")
 savefig("15_stageB_need_ROC_top3.png")

def atomic_json(value,path):
 temporary=path+".tmp"
 with open(temporary,"w",encoding="utf8") as h:json.dump(value,h,indent=2,allow_nan=True)
 os.replace(temporary,path)
def missing_report(audit,edge,stage):
 rows=[]
 for text in audit["metric_limitations"]:rows.append({"metric_or_analysis":"schema limitation","available":False,"reason":text,"fallback_action":"saved NaN; no fabrication"})
 for metric in METRICS:
  rows.append({"metric_or_analysis":metric,"available_34S":metric in edge and edge[metric].notna().any(),"available_StageB":metric in stage and stage[metric].notna().any() if len(stage) else False,"reason":"available where indicated","fallback_action":"none"})
 return pd.DataFrame(rows)
def final_summary(decision,error,detection,coverage,stage_need):
 deploy=decision[decision.deployable&decision.overall_metric_score.notna()];diag=decision[decision.metric_type.eq("diagnostic_uses_truth")&decision.overall_metric_score.notna()];best_d=deploy.loc[deploy.overall_metric_score.idxmax()] if len(deploy) else None;best_diag=diag.loc[diag.overall_metric_score.idxmax()] if len(diag) else None
 def family(name):
  q=decision[(decision.metric_family==name)&decision.overall_metric_score.notna()];return q.loc[q.overall_metric_score.idxmax(),"metric_name"] if len(q) else "unavailable"
 print(f"\nBest deployable metric: {best_d.metric_name if best_d is not None else 'unavailable'}\nBest diagnostic metric: {best_diag.metric_name if best_diag is not None else 'unavailable'}\nBest C/R-only metric: {family('instantaneous_C_R')}\nBest dynamic metric: {family('dynamic_observability')}\nBest parameter-information metric: {family('parameter_information')}")
 if best_d is not None:
  for my in (40,15):print(f"M_y={my}: metric={best_d.metric_name}, error_corr={best_d.get(f'Spearman_error_corr_My{my}',np.nan):.3f}, detection_AUC={best_d.get(f'detection_ROC_AUC_My{my}',np.nan):.3f}, coverage_failure_AUC={best_d.get(f'coverage_failure_ROC_AUC_My{my}',np.nan):.3f}, StageB_need_AUC={best_d.get(f'stageB_need_ROC_AUC_My{my}',np.nan):.3f}")
 strong=bool(best_d is not None and best_d.overall_metric_score>=.65 and best_d.error_score>=.15 and best_d.detection_score>=.6 and best_d.coverage_failure_score>=.6);stress_only=bool(best_d is not None and best_d.get("detection_ROC_AUC_My15",0)>=.65 and best_d.get("detection_ROC_AUC_My40",0)<.6)
 error_predictive=bool(best_d is not None and best_d.error_score>=.15);detection_predictive=bool(best_d is not None and best_d.detection_score>=.60);coverage_predictive=bool(best_d is not None and best_d.coverage_failure_score>=.60);stage_predictive=bool(best_d is not None and np.isfinite(best_d.stageB_need_score) and best_d.stageB_need_score>=.60);tau_auc=np.nanmax([best_d.get("tau1_suppression_ROC_AUC_My40",np.nan),best_d.get("tau1_suppression_ROC_AUC_My15",np.nan)]) if best_d is not None else np.nan;tau_explained=bool(np.isfinite(tau_auc) and tau_auc>=.60);next_step="observability-adaptive Stage-B covariance" if strong else "stronger sparse posterior geometry"
 print(f"Primary conclusions:\n1. best deployable observability metric: {best_d.metric_name if best_d is not None else 'unavailable'}\n2. best truth-based diagnostic metric: {best_diag.metric_name if best_diag is not None else 'unavailable'}\n3. predicts group estimation error: {error_predictive}\n4. predicts true-edge detection failure: {detection_predictive}\n5. predicts Louis coverage failure: {coverage_predictive}\n6. predicts Stage-B correction need: {stage_predictive}\n7. low observability explains tau=1 suppression: {tau_explained}\n8. recommended next experiment: {next_step}")
 print(f"Final recommendation:\n- proceed_to_observability_adaptive_stageB: {strong}\n- proceed_to_sparse_prior_redesign: {not strong}\n- keep_My15_as_stress_only: {stress_only}\n- use_My40_as_positive_control: True")

def main():
 started=time.perf_counter();os.makedirs(OUT,exist_ok=True);audit=schema_audit();atomic_json(audit,os.path.join(OUT,"input_schema_report.json"));print(f"Schema audit: found={len(audit['files_found'])}, missing={len(audit['files_missing'])}, 34S_edges={audit['edge_level_34S_analysis_possible']}, StageB_edges={audit['stageB_selected_edge_analysis_possible']}, reconstruction_needed={audit['lightweight_reconstruction_needed']}")
 if not audit["edge_level_34S_analysis_possible"]:raise RuntimeError("34S edge table/A matrices are missing. Lightweight unperturbed reconstruction is required; no Stage-B fallback will be run.")
 edge,source=reconstruct_34s_edge_table();stage=stageB_table() if audit["stageB_selected_edge_analysis_possible"] else pd.DataFrame();print(f"Tables: runs={edge.run_id.nunique()}, edges={len(edge)}, true_edges={edge.true_edge.sum()}, selected_StageB_edges={len(stage)}")
 error=error_analysis(edge);print(f"Analysis 1/8 error correlations complete: {len(error)} rows")
 detection=detection_analysis(edge);print(f"Analysis 2/8 detection prediction complete: {len(detection)} rows")
 coverage=coverage_analysis(edge);print(f"Analysis 3/8 coverage prediction complete: {len(coverage)} rows")
 stage_need=stageB_analysis(stage) if len(stage) else pd.DataFrame();print(f"Analysis 4/8 Stage-B need complete: {len(stage_need)} rows")
 quart=quartile_analysis(edge,stage);print(f"Analysis 5/8 quartiles complete: {len(quart)} rows")
 compare=metric_comparison(error,detection,coverage,stage_need);print("Top metrics:",", ".join(compare.nlargest(3,"overall_metric_score").metric_name))
 mycomp=regime_comparison(error,detection,coverage,stage_need);support=support_comparison(error,detection,coverage);parameter=parameter_summary(edge,stage);decision=decision_table(compare,mycomp,quart,stage_need)
 runtime=pd.DataFrame([{"experiment":"34T_edge_level_observability_information_analysis","runtime_seconds":time.perf_counter()-started,"n_34S_runs":edge.run_id.nunique(),"n_34S_edges":len(edge),"n_stageB_selected_edges":len(stage),"bootstrap_run_resamples":BOOT,"expensive_refits_run":False,"stageB_perturbations_run":False}]);missing=missing_report(audit,edge,stage)
 config={"experiment":"34T_edge_level_observability_information_analysis","source_34S":S_DIR,"source_34R_stage2C":R_DIR,"fallback_sources":FALLBACK_DIRS,"K_obs_list":[5,10,20],"primary_dynamic_metric":"edge_o_dyn_estA_mean_K10","primary_diagnostic_dynamic_metric":"edge_o_dyn_trueA_mean_K10","eps":EPS,"eps_info":EPS_INFO,"bootstrap_unit":"run_id","bootstrap_resamples":BOOT,"A_center":"A_VB","refits":False,"StageB_perturbations":False}
 outputs={"experiment_config.json":config,"edge_observability_information_table.csv":edge,"stageB_selected_edge_information_table.csv":stage,"source_observability_table.csv":source,"error_correlation_summary.csv":error,"detection_prediction_summary.csv":detection,"coverage_failure_prediction_summary.csv":coverage,"stageB_need_prediction_summary.csv":stage_need,"observability_quartile_summary.csv":quart,"observability_metric_comparison_summary.csv":compare,"My_regime_comparison_summary.csv":mycomp,"support_regime_observability_summary.csv":support,"parameter_information_summary.csv":parameter,"decision_summary.csv":decision,"runtime_summary.csv":runtime,"missing_metric_report.csv":missing}
 for name,value in outputs.items():atomic_json(value,os.path.join(OUT,name)) if name.endswith(".json") else atomic_csv(value,os.path.join(OUT,name))
 plots(edge,stage,quart,compare,mycomp,support,coverage,stage_need);runtime.loc[0,"runtime_seconds"]=time.perf_counter()-started;atomic_csv(runtime,os.path.join(OUT,"runtime_summary.csv"));final_summary(decision,error,detection,coverage,stage_need)
if __name__=="__main__":main()
