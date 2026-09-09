"""Experiment 34P: oracle and sparsity-aware Stage-B LRVB covariance."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = os.environ.get("EXPERIMENT_34P_BLAS_THREADS", "1")

import json
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34n_lrvb_A_covariance as n
import experiments.experiment_34o_lrvb_stageB_smoother_feedback as o
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv, correlations, grouped_stats
from src.stats.lrvb_stageB_smoother_feedback import flatten_A, global_index
from src.stats.sparsity_aware_lrvb_covariance import (
    covariance_diagnostics, gated_covariance, globally_damped_covariance,
    group_activity_scores, hill_weight, logistic_weight, project_psd,
)

CONFIG_LIST = [
    {"label":"small_controlled_M5","M":5,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":2,"N_REPLICATES_PER_NETWORK":5,"covariance_mode":"selected"},
    {"label":"diagnostic_M20_subset","M":20,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":3,"covariance_mode":"selected"},
]
TAU_SNR_GRID=[1.,1.5,2.,2.5,3.,4.];TAU_QUANTILES=[.70,.80,.85,.90,.95];R_GRID=[2,4];C_GRID=[1.,2.,4.];TAU_SOFT_GRID=[1.5,2.,2.5,3.];LAMBDA_GLOBAL_GRID=[.25,.50,.75]
BASE_SEED=o.BASE_SEED;SMOKE_TEST=os.environ.get("EXPERIMENT_34P_SMOKE","0")=="1"
if SMOKE_TEST:CONFIG_LIST=[{"label":"small_controlled_M5","M":5,"T_VALUES":[1000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":1,"covariance_mode":"selected"}];TAU_SNR_GRID=[2.];TAU_QUANTILES=[.8];R_GRID=[2];C_GRID=[2.];TAU_SOFT_GRID=[2.];LAMBDA_GLOBAL_GRID=[.5]
RESULTS_DIR=os.environ.get("EXPERIMENT_34P_RESULTS_DIR","results/experiment_34p");SOURCE_34O_DIR=os.environ.get("EXPERIMENT_34P_SOURCE_34O_DIR","results/experiment_34o")

class ProgressBar:
 def __init__(self,total,width=30):self.total,self.width,self.done=max(int(total),0),width,0
 def update(self,label):
  self.done+=1;fraction=min(self.done/max(self.total,1),1.);filled=round(self.width*fraction);print(f"\rExperiment 34P [{'#'*filled}{'-'*(self.width-filled)}] {self.done}/{self.total} ({100*fraction:5.1f}%) {label[:58]}",end="\n" if self.done>=self.total else "",flush=True)

def matching(frame,meta):
 mask=np.ones(len(frame),bool)
 for key in ("config_label","M","T","true_network_id","replicate_id"):
  if key in frame:mask&=frame[key].astype(str)==str(meta[key])
 return frame.loc[mask]

def load_stage_b_cache(meta,expected_indices):
 covpath=os.path.join(SOURCE_34O_DIR,"_covstore_partial.csv");selpath=os.path.join(SOURCE_34O_DIR,"selected_coefficients.csv")
 if not(os.path.exists(covpath) and os.path.exists(selpath)):return None
 try:cov=matching(pd.read_csv(covpath),meta);selected=matching(pd.read_csv(selpath),meta)
 except Exception:return None
 selected=selected.sort_values("coefficient_global_index");expected=np.asarray(expected_indices,int)
 if not np.array_equal(selected.coefficient_global_index.astype(int).to_numpy(),expected):return None
 row=cov.loc[cov.covariance_estimator=="lrvb_stageB_smoother_feedback_psd_projected"]
 if row.empty:return None
 matrix=np.asarray(json.loads(row.iloc[-1].covariance_json),float)
 return matrix if matrix.shape==(len(expected),len(expected)) else None

def group_records(model,data,selected,blocks):
 M=model.n_states;positions={int(index):p for p,index in enumerate(selected.coefficient_global_index)};mean=np.asarray(model.A_mean_matrices_);records=[]
 for target in range(M):
  for source in range(M):
   indices=[global_index(target,lag,source,M,j.na) for lag in range(j.na)]
   available=[index for index in indices if index in positions]
   if not available:continue
   pos=[positions[index] for index in available];local_lags=[(index-target*(j.na*M))//M for index in available]
   current=blocks["ordinary_vb"][np.ix_(pos,pos)];louis=blocks["stabilized_louis_eta_0p70_tau_0p90"][np.ix_(pos,pos)];stage=blocks["lrvb_stageB_smoother_feedback"][np.ix_(pos,pos)];m=mean[local_lags,target,source];truth=np.asarray(data["A"])[local_lags,target,source]
   alpha=1e-15 if target==source else model.alpha_mean_[target,source];scores=group_activity_scores(m,current,louis,max(alpha,1e-15));records.append({"target":target,"source":source,"indices":indices,"available_indices":available,"positions":pos,"lags":local_lags,"mean":m,"truth":truth,"true_edge":bool(target!=source and data["mask"][target,source]),"diagonal":target==source,"complete_group":len(available)==j.na,"stageB_over_louis_trace_ratio":np.trace(stage)/max(np.trace(louis),1e-15),"needed_variance_ratio_if_truth_available":np.linalg.norm(m-truth)**2/max(np.trace(louis),1e-15),**scores})
 return records

def weights_for_records(records,kind,score_type=None,threshold=np.nan,family=None,parameter=None):
 weights={}
 for record in records:
  if record["diagonal"]:weight=1.
  elif kind=="oracle":weight=float(record["true_edge"])
  elif kind=="hard":weight=float(record[score_type]>=threshold)
  elif kind=="soft" and family=="hill":weight=hill_weight(record[score_type],threshold,parameter)
  elif kind=="soft" and family=="logistic":weight=logistic_weight(record[score_type],threshold,parameter)
  else:weight=0.
  for position in record["positions"]:weights[position]=weight
 return np.asarray([weights.get(position,0.) for position in range(sum(len(r["positions"]) for r in records))])

def variants(blocks,records,selected):
 louis=blocks["stabilized_louis_eta_0p70_tau_0p90"];stage=blocks["lrvb_stageB_smoother_feedback"];result=[]
 def add(name,cov,gating,score="none",threshold=np.nan,family="none",weights=None,projection=None):result.append({"covariance_estimator":name,"covariance":cov,"gating_type":gating,"gate_score_type":score,"gate_threshold":threshold,"gate_weight_family":family,"weights":weights,"projection":projection or {"psd_projection_used":False,"fraction_negative_variances":0.}})
 add("ordinary_vb",blocks["ordinary_vb"],"baseline");add("stabilized_louis_eta_0p70_tau_0p90",louis,"baseline");add("lrvb_stageA_alpha_feedback",blocks["lrvb_stageA_alpha_feedback"],"baseline");add("lrvb_stageB_smoother_feedback",stage,"full_stageB")
 oracle=weights_for_records(records,"oracle");g=gated_covariance(louis,stage,oracle);add("oracle_active_subspace_stageB",g.covariance,"oracle","oracle_support",np.nan,"hard",oracle,g.diagnostics)
 offdiag=[record for record in records if not record["diagonal"]]
 for threshold in TAU_SNR_GRID:
  w=weights_for_records(records,"hard","group_snr_louis",threshold);g=gated_covariance(louis,stage,w);add(f"hard_active_subspace_stageB_group_snr_louis_tau_{threshold:.4g}",g.covariance,"hard","group_snr_louis",threshold,"hard",w,g.diagnostics)
 for score in ("group_norm","inverse_alpha_score"):
  for quantile in TAU_QUANTILES:
   cutoff=float(np.quantile([r[score] for r in offdiag],quantile));w=weights_for_records(records,"hard",score,cutoff);g=gated_covariance(louis,stage,w);add(f"hard_active_subspace_stageB_{score}_quantile_{quantile:.2f}",g.covariance,"hard",score,quantile,"empirical_quantile",w,g.diagnostics)
 for family,parameters in (("hill",R_GRID),("logistic",C_GRID)):
  for parameter in parameters:
   for threshold in TAU_SOFT_GRID:
    w=weights_for_records(records,"soft","group_snr_louis",threshold,family,parameter);g=gated_covariance(louis,stage,w);label=f"soft_sparsity_aware_stageB_{family}_p{parameter:g}_tau_{threshold:g}";add(label,g.covariance,"soft","group_snr_louis",threshold,f"{family}_{parameter:g}",w,g.diagnostics);con=gated_covariance(louis,stage,w,True);add(label+"_conservative",con.covariance,"soft_conservative","group_snr_louis",threshold,f"{family}_{parameter:g}",w,con.diagnostics)
 for damping in LAMBDA_GLOBAL_GRID:
  g=globally_damped_covariance(louis,stage,damping);add(f"global_lambda_stageB_{damping:.2f}",g.covariance,"global_lambda","none",damping,"global",np.full(len(selected),np.sqrt(damping)),g.diagnostics)
 return result

def gate_diagnostic(meta,variant,records):
 off=[r for r in records if not r["diagonal"]];group_weights=[]
 for r in off:group_weights.append(float(np.mean(variant["weights"][r["positions"]])) if variant["weights"] is not None else (1. if variant["gating_type"]=="full_stageB" else 0.))
 truth=np.asarray([r["true_edge"] for r in off],bool);weights=np.asarray(group_weights);selected=weights>=.5;tp=np.sum(selected&truth);fp=np.sum(selected&~truth)
 return {**meta,"covariance_estimator":variant["covariance_estimator"],"gating_type":variant["gating_type"],"gate_score_type":variant["gate_score_type"],"gate_threshold":variant["gate_threshold"],"gate_weight_family":variant["gate_weight_family"],"number_groups_selected":int(selected.sum()),"fraction_groups_selected":selected.mean() if len(selected) else np.nan,"true_edge_selection_rate":selected[truth].mean() if truth.any() else np.nan,"false_edge_selection_rate":selected[~truth].mean() if (~truth).any() else np.nan,"selected_true_edge_precision":tp/max(tp+fp,1),"selected_true_edge_recall":tp/max(truth.sum(),1),"mean_soft_weight_true_edges":weights[truth].mean() if truth.any() else np.nan,"mean_soft_weight_false_edges":weights[~truth].mean() if (~truth).any() else np.nan,"median_soft_weight_true_edges":np.median(weights[truth]) if truth.any() else np.nan,"median_soft_weight_false_edges":np.median(weights[~truth]) if (~truth).any() else np.nan}

def coefficient_rows(model,data,selected,variant,meta):
 theta=flatten_A(model.A_mean_matrices_);truth=flatten_A(data["A"]);rows=[];diag=np.diag(variant["covariance"])
 for position,item in selected.reset_index(drop=True).iterrows():
  index=int(item.coefficient_global_index);sd=np.sqrt(max(diag[position],0));error=theta[index]-truth[index];weight=variant["weights"][position] if variant["weights"] is not None else np.nan
  rows.append({**meta,"covariance_estimator":variant["covariance_estimator"],"gating_type":variant["gating_type"],"gate_score_type":variant["gate_score_type"],"gate_threshold":variant["gate_threshold"],"gate_weight_family":variant["gate_weight_family"],"gate_weight_value":weight,"coefficient_global_index":index,"target":item.target,"lag":item.lag,"source":item.source,"coefficient_type":item.coefficient_type,"selected_direction_type":item.selected_direction_type,"posterior_center":theta[index],"posterior_variance":diag[position],"posterior_sd":sd,"true_value":truth[index],"signed_error":error,"abs_error":abs(error),"squared_error":error**2,"interval_width_95":3.92*sd,"standardized_error":error/max(sd,1e-15),"ci95_lower":theta[index]-1.96*sd,"ci95_upper":theta[index]+1.96*sd,"ci95_contains_true":abs(error)<=1.96*sd})
 return pd.DataFrame(rows)

def group_rows(model,records,variant,meta):
 rows=[];scores=[]
 for record in records:
  if record["diagonal"] or not record["complete_group"]:continue
  block=variant["covariance"][np.ix_(record["positions"],record["positions"])];difference=record["mean"]-record["truth"];D2=difference@np.linalg.pinv(block)@difference;snr=np.sqrt(max(record["mean"]@np.linalg.pinv(block)@record["mean"],0));common={**meta,"covariance_estimator":variant["covariance_estimator"],"gating_type":variant["gating_type"],"gate_score_type":variant["gate_score_type"],"gate_threshold":variant["gate_threshold"],"gate_weight_family":variant["gate_weight_family"],"target":record["target"],"source":record["source"],"edge_group_type":"true_edge_group" if record["true_edge"] else "false_edge_group","true_link":record["true_edge"]};rows.append({**common,"mahalanobis_D2":D2,"D2_below_chi2_95_df2":D2<=5.991464547,"D2_below_chi2_99_df2":D2<=9.210340372,"group_cov_trace":np.trace(block)});scores.append({**common,"score":snr,"score_type":"group_snr"})
 return pd.DataFrame(rows),pd.DataFrame(scores)

def model_group_norm_scores(records,meta):
 rows=[]
 for record in records:
  if record["diagonal"] or not record["complete_group"]:continue
  rows.append({**meta,"covariance_estimator":"not_applicable","gating_type":"baseline","gate_score_type":"none","gate_threshold":np.nan,"gate_weight_family":"none","target":record["target"],"source":record["source"],"edge_group_type":"true_edge_group" if record["true_edge"] else "false_edge_group","true_link":record["true_edge"],"score":np.linalg.norm(record["mean"]),"score_type":"model_A_group_norm"})
 return pd.DataFrame(rows)

def calibration_summary(frame):
 keys=["config_label","M","T","covariance_estimator","gating_type","gate_score_type","gate_threshold","gate_weight_family","coefficient_type"];rows=[]
 for values,g in frame.groupby(keys,dropna=False):rows.append({**dict(zip(keys,values)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_interval_width_95":g.interval_width_95.mean(),"median_interval_width_95":g.interval_width_95.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"n_coefficients":len(g)})
 return pd.DataFrame(rows)

def group_summary(frame):
 keys=["config_label","M","T","covariance_estimator","gating_type","gate_score_type","gate_threshold","gate_weight_family","edge_group_type"]
 return frame.groupby(keys,dropna=False).agg(mean_mahalanobis_D2=("mahalanobis_D2","mean"),median_mahalanobis_D2=("mahalanobis_D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("mahalanobis_D2","size")).reset_index()

def edge_summary(frame):
 keys=["config_label","M","T","covariance_estimator","gating_type","gate_score_type","gate_threshold","gate_weight_family"];rows=[]
 for values,g in frame.groupby(keys,dropna=False):metrics,_=j.safe_curve(g.true_link,g.score);rows.append({**dict(zip(keys,values)),**metrics})
 return pd.DataFrame(rows)

def empirical_variance(frame):
 keys=["config_label","true_network_id","M","T","coefficient_global_index","covariance_estimator","coefficient_type"];rows=[]
 for values,g in frame.groupby(keys,dropna=False):emp=g.signed_error.var(ddof=1) if len(g)>1 else g.squared_error.mean();post=g.posterior_variance.mean();ratio=post/max(emp,1e-15);rows.append({**dict(zip(keys,values)),"posterior_variance":post,"empirical_error_variance":emp,"variance_ratio_posterior_to_empirical":ratio,"posterior_sd":np.sqrt(max(post,0)),"empirical_sd":np.sqrt(max(emp,0)),"sd_ratio_posterior_to_empirical":np.sqrt(max(post,0))/max(np.sqrt(max(emp,0)),1e-15),"variance_ratio_between_0p5_and_2":.5<=ratio<=2,"variance_ratio_above_10":ratio>10,"variance_ratio_above_100":ratio>100,"n_replicates":len(g)})
 return pd.DataFrame(rows)

def covariance_row(meta,variant,blocks,selected):
 projection=variant["projection"];row={**meta,"covariance_estimator":variant["covariance_estimator"],"gating_type":variant["gating_type"],"gate_score_type":variant["gate_score_type"],"gate_threshold":variant["gate_threshold"],"gate_weight_family":variant["gate_weight_family"],"number_selected_directions":len(selected),"number_negative_variances":projection.get("number_negative_variances",0),"fraction_negative_variances":projection.get("fraction_negative_variances",0.),"number_psd_projections":int(projection.get("psd_projection_used",False)),"fraction_psd_projected":float(projection.get("psd_projection_used",False)),**covariance_diagnostics(variant["covariance"],blocks["ordinary_vb"],blocks["stabilized_louis_eta_0p70_tau_0p90"],selected.coefficient_type)};return row

def activity_rows(meta,records):
 rows=[]
 for r in records:
  if r["diagonal"]:continue
  rows.append({**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"],"coefficient_type_group":"true_edge_group" if r["true_edge"] else "false_edge_group","group_norm":r["group_norm"],"group_snr_current":r["group_snr_current"],"group_snr_louis":r["group_snr_louis"],"E_alpha":r["E_alpha"],"inverse_alpha_score":r["inverse_alpha_score"],"negative_log_alpha":r["negative_log_alpha"],"stageB_over_louis_trace_ratio":r["stageB_over_louis_trace_ratio"],"needed_variance_ratio_if_truth_available":r["needed_variance_ratio_if_truth_available"],"hard_gate_selected":r["group_snr_louis"]>=2.,"soft_gate_weight":hill_weight(r["group_snr_louis"],2.,4)})
 frame=pd.DataFrame(rows)
 if len(frame):
  for label,mask in (("all",np.ones(len(frame),bool)),("true",frame.true_edge_group.to_numpy(bool)),("false",~frame.true_edge_group.to_numpy(bool))):
   for outcome in ("stageB_over_louis_trace_ratio","needed_variance_ratio_if_truth_available"):
    for score in ("group_snr_louis","inverse_alpha_score"):frame[f"corr_{outcome}_with_{score}_{label}"]=correlations(frame.loc[mask,outcome],frame.loc[mask,score]) if mask.sum()>1 else np.nan
 return frame

def decision_summary(cal,groups,edges,emp,gating,covdiag,runtime):
 rows=[];keys=["config_label","M","T","covariance_estimator","gating_type","gate_score_type","gate_threshold","gate_weight_family"]
 for values,g in cal.groupby(keys,dropna=False):
  by={kind:part for kind,part in g.groupby("coefficient_type")};v=lambda kind,col:getattr(by[kind],col).mean() if kind in by else np.nan;gs=groups
  mask=(gs.M==values[1])&(gs["T"]==values[2])&(gs.covariance_estimator==values[3]);group=gs.loc[mask];edge=edges.loc[(edges.M==values[1])&(edges["T"]==values[2])&(edges.covariance_estimator==values[3])];ev=emp.loc[(emp.M==values[1])&(emp["T"]==values[2])&(emp.covariance_estimator==values[3])];gate=gating.loc[(gating.M==values[1])&(gating["T"]==values[2])&(gating.covariance_estimator==values[3])];cov=covdiag.loc[(covdiag.M==values[1])&(covdiag["T"]==values[2])&(covdiag.covariance_estimator==values[3])];active=np.nanmean([v("diagonal","empirical_coverage_95"),v("offdiag_nonzero","empirical_coverage_95")]);rows.append({**dict(zip(keys,values)),"diag_coverage":v("diagonal","empirical_coverage_95"),"offdiag_nonzero_coverage":v("offdiag_nonzero","empirical_coverage_95"),"offdiag_zero_coverage":v("offdiag_zero","empirical_coverage_95"),"diag_std_z":v("diagonal","std_standardized_error"),"offdiag_nonzero_std_z":v("offdiag_nonzero","std_standardized_error"),"offdiag_zero_std_z":v("offdiag_zero","std_standardized_error"),"diag_mean_interval_width_95":v("diagonal","mean_interval_width_95"),"offdiag_nonzero_mean_interval_width_95":v("offdiag_nonzero","mean_interval_width_95"),"offdiag_zero_mean_interval_width_95":v("offdiag_zero","mean_interval_width_95"),"true_edge_group_chi2_95_coverage":group.loc[group.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"false_edge_group_chi2_95_coverage":group.loc[group.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"selected_edge_AUC":edge.ROC_AUC.mean(),"selected_edge_AUPRC":edge.AUPRC.mean(),"selected_edge_TPR_at_FPR_0p05":edge.TPR_at_FPR_0p05.mean(),"selected_edge_precision_at_FPR_0p05":edge.precision_at_FPR_0p05.mean(),"variance_ratio_to_empirical_diag":ev.loc[ev.coefficient_type=="diagonal","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_nonzero":ev.loc[ev.coefficient_type=="offdiag_nonzero","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_zero":ev.loc[ev.coefficient_type=="offdiag_zero","variance_ratio_posterior_to_empirical"].mean(),"number_groups_selected":gate.number_groups_selected.mean(),"fraction_groups_selected":gate.fraction_groups_selected.mean(),"true_edge_selection_rate":gate.true_edge_selection_rate.mean(),"false_edge_selection_rate":gate.false_edge_selection_rate.mean(),"selected_true_edge_precision":gate.selected_true_edge_precision.mean(),"selected_true_edge_recall":gate.selected_true_edge_recall.mean(),"fraction_negative_variances":cov.fraction_negative_variances.mean(),"fraction_psd_projected":cov.fraction_psd_projected.mean(),"total_runtime_seconds":runtime.loc[(runtime.M==values[1])&(runtime["T"]==values[2])].filter(like="total_runtime_seconds_mean").mean(axis=1).mean(),"recommended_for_uncertainty":bool(.88<=active<=.97 and .90<=v("offdiag_zero","empirical_coverage_95")<=.99),"recommended_for_ranking":bool(edge.AUPRC.mean()>=edges.loc[(edges.M==values[1])&(edges["T"]==values[2]),"AUPRC"].max()-.05),"notes":"All intervals retain A_VB center; gating modifies only Stage-B minus stabilized-Louis covariance."})
 return pd.DataFrame(rows)

def summaries(frames):
 cal=calibration_summary(frames["coeff"]);groups=group_summary(frames["group"]);edges=edge_summary(frames["scores"]);emp=empirical_variance(frames["coeff"]);runtime=grouped_stats(frames["runtime"],["config_label","M","T"],["baseline_runtime_seconds","stageB_runtime_seconds","gating_runtime_seconds","total_runtime_seconds","stageB_cache_hit"]);decision=decision_summary(cal,groups,edges,emp,frames["gating"],frames["covdiag"],runtime);return cal,groups,edges,emp,runtime,decision

def save_plots(cal,activity,gating,emp,covdiag,edges,runtime):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 def bar(frame,x,y,name):
  s=frame.groupby(x,dropna=False)[y].mean();fig,ax=plt.subplots();s.plot.bar(ax=ax);ax.set_ylabel(y);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 bar(cal,"gating_type","empirical_coverage_95","coverage.png");bar(cal,"gating_type","std_standardized_error","standardized_sd.png");bar(cal,"gating_type","mean_interval_width_95","interval_width.png");bar(cal.loc[cal.coefficient_type=="offdiag_zero"],"gating_type","empirical_coverage_95","zero_coverage_width.png");bar(cal,"coefficient_type","empirical_coverage_95","active_zero_coverage.png");bar(covdiag,"gating_type","trace_over_louis_selected","stageB_louis_ratio.png");bar(cal.loc[cal.gating_type.isin(["oracle","full_stageB","baseline"])],"gating_type","empirical_coverage_95","oracle_comparison.png");bar(cal.loc[cal.gating_type=="hard"],"gate_threshold","empirical_coverage_95","hard_sweep.png");bar(cal.loc[cal.gating_type.str.startswith("soft")],"gate_threshold","empirical_coverage_95","soft_sweep.png");bar(activity,"true_edge_group","group_snr_louis","snr_distribution.png");bar(activity,"true_edge_group","E_alpha","alpha_distribution.png");fig,ax=plt.subplots();ax.scatter(activity.group_snr_louis,activity.stageB_over_louis_trace_ratio,s=3);fig.tight_layout();fig.savefig(os.path.join(path,"trace_ratio_snr.png"));plt.close(fig);fig,ax=plt.subplots();ax.scatter(activity.group_snr_louis,activity.needed_variance_ratio_if_truth_available,s=3);fig.tight_layout();fig.savefig(os.path.join(path,"needed_ratio_snr.png"));plt.close(fig);bar(gating.loc[gating.gating_type=="hard"],"gate_threshold","selected_true_edge_precision","hard_precision_recall.png");bar(gating.loc[gating.gating_type.str.startswith("soft")],"gate_threshold","mean_soft_weight_true_edges","soft_weights.png");bar(runtime,"M","total_runtime_seconds_mean","runtime.png")

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34P","CONFIG_LIST":CONFIG_LIST,"TAU_SNR_GRID":TAU_SNR_GRID,"TAU_QUANTILES":TAU_QUANTILES,"R_GRID":R_GRID,"C_GRID":C_GRID,"TAU_SOFT_GRID":TAU_SOFT_GRID,"LAMBDA_GLOBAL_GRID":LAMBDA_GLOBAL_GRID,"source_34O_dir":SOURCE_34O_DIR,"center_estimator":"A_VB","smoke_test":SMOKE_TEST};cp=os.path.join(RESULTS_DIR,"experiment_config.json");names={"activity":"activity_score_diagnostics_partial.csv","gating":"gating_diagnostics_partial.csv","coeff":"_coeff_partial.csv","group":"_group_partial.csv","scores":"_scores_partial.csv","covdiag":"covariance_diagnostics_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={key:[] for key in names}
 if os.path.exists(cp) and os.path.exists(os.path.join(RESULTS_DIR,"run_summary_partial.csv")):
  with open(cp,encoding="utf8") as handle:old=json.load(handle)
  for key in ("CONFIG_LIST","TAU_SNR_GRID","TAU_QUANTILES","R_GRID","C_GRID","TAU_SOFT_GRID","LAMBDA_GLOBAL_GRID"):
   if old.get(key)!=config.get(key):raise ValueError("Existing 34P checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as handle:json.dump(config,handle,indent=2)
 for key,name in names.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:frame=pd.read_csv(path)
   except pd.errors.EmptyDataError:frame=pd.DataFrame()
   if len(frame):tables[key]=[frame]
 previous=pd.concat(tables["run"],ignore_index=True) if tables["run"] else pd.DataFrame();completed=set(zip(previous.config_label,previous["T"].astype(int),previous.true_network_id.astype(int),previous.replicate_id.astype(int))) if len(previous) else set();total=sum((cfg["label"],T,net,rep) not in completed for cfg in CONFIG_LIST for T in cfg["T_VALUES"] for net in range(cfg["N_TRUE_NETWORKS"]) for rep in range(cfg["N_REPLICATES_PER_NETWORK"]));progress=ProgressBar(total)
 for cfg in CONFIG_LIST:
  M=cfg["M"]
  for network_id in range(cfg["N_TRUE_NETWORKS"]):
   network_seed=BASE_SEED+M*100000+network_id*1000;A,B,mask=j.fixed_network(M,network_seed)
   for T in cfg["T_VALUES"]:
    for replicate in range(cfg["N_REPLICATES_PER_NETWORK"]):
     key=(cfg["label"],T,network_id,replicate)
     if key in completed:continue
     meta={"config_label":cfg["label"],"M":M,"T":T,"true_network_id":network_id,"replicate_id":replicate};seed=network_seed+T*10+replicate;started=time.perf_counter();status="success"
     try:
      data=j.simulate(A,B,M,T,seed);base_start=time.perf_counter();model=j.fit_model(data,seed+500);louis=n._louis_covariances(model,data,seed+700);stageA,_,_=n._lrvb_covariances(model,data,meta);base_runtime=time.perf_counter()-base_start;selected=o.select_directions(data,cfg["covariance_mode"],seed+900,smoke_test=SMOKE_TEST).sort_values("coefficient_global_index").reset_index(drop=True);expected=selected.coefficient_global_index.astype(int).to_numpy();stageB=load_stage_b_cache(meta,expected);cache_hit=stageB is not None;stage_start=time.perf_counter()
      if stageB is None:
       valid,raw,stageB,_,_,_,_=o.run_stage_b(model,data,selected,meta);selected=selected.set_index("coefficient_global_index").loc[valid].reset_index();expected=np.asarray(valid,int)
      stage_runtime=time.perf_counter()-stage_start;global_sets={"ordinary_vb":o.global_covariance_block(model.A_row_covariances_),"stabilized_louis_eta_0p70_tau_0p90":o.global_covariance_block(louis),"lrvb_stageA_alpha_feedback":o.global_covariance_block(stageA)};blocks={name:value[np.ix_(expected,expected)] for name,value in global_sets.items()};blocks["lrvb_stageB_smoother_feedback"]=stageB;records=group_records(model,data,selected,blocks);tables["activity"].append(activity_rows(meta,records));gate_start=time.perf_counter();all_variants=variants(blocks,records,selected)
      for variant in all_variants:
       tables["gating"].append(gate_diagnostic(meta,variant,records));tables["coeff"].append(coefficient_rows(model,data,selected,variant,meta));groups,scores=group_rows(model,records,variant,meta);tables["group"].append(groups);tables["scores"].append(scores);tables["covdiag"].append(covariance_row(meta,variant,blocks,selected))
      tables["scores"].append(model_group_norm_scores(records,meta))
      gating_runtime=time.perf_counter()-gate_start;tables["runtime"].append({**meta,"baseline_runtime_seconds":base_runtime,"stageB_runtime_seconds":stage_runtime,"gating_runtime_seconds":gating_runtime,"total_runtime_seconds":time.perf_counter()-started,"stageB_cache_hit":cache_hit,"run_status":"success"})
     except Exception as error:status="failed";tables["runtime"].append({**meta,"total_runtime_seconds":time.perf_counter()-started,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()})
     tables["run"].append({**meta,"fit_status":status,"total_runtime_seconds":time.perf_counter()-started});frames={name:(pd.concat(items,ignore_index=True) if items and isinstance(items[0],pd.DataFrame) else pd.DataFrame(items)) for name,items in tables.items()}
     for name,file in names.items():atomic_csv(frames[name],os.path.join(RESULTS_DIR,file))
     if len(frames["coeff"]):
      cal,groups,edges,emp,runtime,decision=summaries(frames)
      for file,frame in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",groups),("empirical_variance_comparison_partial.csv",emp),("runtime_summary_partial.csv",runtime),("decision_summary_partial.csv",decision)):atomic_csv(frame,os.path.join(RESULTS_DIR,file))
     progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
 frames={name:(pd.concat(items,ignore_index=True) if items and isinstance(items[0],pd.DataFrame) else pd.DataFrame(items)) for name,items in tables.items()};cal,groups,edges,emp,runtime,decision=summaries(frames);oracle=decision.loc[decision.gating_type=="oracle"];globaldiag=decision.loc[decision.gating_type=="global_lambda"];outputs={"run_summary.csv":frames["run"],"activity_score_diagnostics.csv":frames["activity"],"gating_diagnostics.csv":frames["gating"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":groups,"selected_edge_recovery_summary.csv":edges,"network_recovery_summary.csv":edges,"empirical_variance_comparison.csv":emp,"covariance_diagnostics.csv":frames["covdiag"],"oracle_active_subspace_diagnostic.csv":oracle,"global_lambda_diagnostic.csv":globaldiag,"runtime_summary.csv":runtime,"decision_summary.csv":decision}
 for file,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,file))
 save_plots(cal,frames["activity"],frames["gating"],emp,frames["covdiag"],edges,runtime)

if __name__=="__main__":main()
