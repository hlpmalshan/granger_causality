"""Experiment 34Q-A: sparsity-aware Stage-B LRVB with estimated B."""
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
 os.environ[_name]=os.environ.get("EXPERIMENT_34QA_BLAS_THREADS","1")
import json,time,traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34n_lrvb_A_covariance as n
import experiments.experiment_34o_lrvb_stageB_smoother_feedback as o
import experiments.experiment_34p_sparsity_aware_stageB_lrvb as p
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats,relative
from src.ssm.vb_ard_varx_ssm_b_controls import HybridVBARDVARXSSMKnownCWithBControls
from src.stats.lrvb_stageB_smoother_feedback import flatten_A,global_index,project_selected_covariance,sensitivity_column
from src.stats.sparsity_aware_lrvb_covariance import covariance_diagnostics,gated_covariance,globally_damped_covariance,logistic_weight

CONFIG_LIST=[{"label":"small_freeB_M5","M":5,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":2,"N_REPLICATES_PER_NETWORK":5,"covariance_mode":"selected"},{"label":"main_freeB_M20_subset","M":20,"T_VALUES":[1000,2000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":3,"covariance_mode":"selected"}]
FINITE_DIFF_EPS_GRID=[3e-5,1e-4,3e-4];FINITE_DIFF_EPS_PRIMARY=1e-4;PERTURBED_VB_MAX_ITER=50;PERTURBED_CONVERGENCE_TOL=1e-4;VB_MAX_ITER=100;BASE_SEED=5650000
SMOKE_TEST=os.environ.get("EXPERIMENT_34QA_SMOKE","0")=="1"
if SMOKE_TEST:CONFIG_LIST=[{"label":"small_freeB_M5","M":5,"T_VALUES":[1000],"N_TRUE_NETWORKS":1,"N_REPLICATES_PER_NETWORK":1,"covariance_mode":"selected"}];FINITE_DIFF_EPS_GRID=[1e-4];PERTURBED_VB_MAX_ITER=30
RESULTS_DIR=os.environ.get("EXPERIMENT_34QA_RESULTS_DIR","results/experiment_34qA")

class ProgressBar:
 def __init__(self,total,width=30):self.total,self.width,self.done=max(int(total),0),width,0
 def update(self,label):
  self.done+=1;f=min(self.done/max(self.total,1),1.);k=round(self.width*f);print(f"\rExperiment 34Q-A [{'#'*k}{'-'*(self.width-k)}] {self.done}/{self.total} ({100*f:5.1f}%) {label[:55]}",end="\n" if self.done>=self.total else "",flush=True)

def fit_free_B(data,seed,max_iter=None):
 if max_iter is None:max_iter=VB_MAX_ITER
 return HybridVBARDVARXSSMKnownCWithBControls(j.na,j.nb,data["C"],data["Q"],data["R"],B_update_mode="free",B_ridge_lambda=0.,base_B_ridge=0.,max_iter=max_iter,tol_objective=1e-6,tol_A_change=1e-6,tol_B_change=1e-6,tol_alpha_change=1e-6,a0=1e-3,b0=1e-3,diagonal_prior_precision=1e-4,posterior_jitter=1e-8,random_state=seed).fit(data["y"],data["u"])

def select_directions(model,data,louis,seed):
 selected=o.select_directions(data,"selected",seed,smoke_test=False);M=model.n_states;existing=set(selected.coefficient_global_index.astype(int));candidates=[]
 for target in range(M):
  for source in range(M):
   if target==source or data["mask"][target,source]:continue
   cols=[lag*M+source for lag in range(j.na)];mean=np.asarray(model.A_mean_matrices_)[:,target,source];block=louis[target][np.ix_(cols,cols)];score=np.sqrt(max(mean@np.linalg.pinv(block)@mean,0));candidates.append((score,target,source))
 count=10 if M>=20 else min(3,len(candidates));extras=sorted(candidates,reverse=True)[:count]+sorted(candidates)[:count]
 rows=selected.to_dict("records")
 for rank,(_,target,source) in enumerate(extras):
  kind="high_snr_false_group" if rank<count else "low_snr_false_group"
  for lag in range(j.na):
   index=global_index(target,lag,source,M,j.na)
   if index not in existing:rows.append({"coefficient_global_index":index,"target":target,"lag":lag+1,"source":source,"coefficient_type":"offdiag_zero","selected_direction_type":kind});existing.add(index)
 return pd.DataFrame(rows).sort_values("coefficient_global_index").reset_index(drop=True)

def run_stageB_free(model,data,selected,meta,reestimate_B=True,reestimate_Q=False):
 indices=selected.coefficient_global_index.astype(int).tolist();total=j.na*model.n_states**2;columns=np.full((total,len(indices)),np.nan);rows=[];times=[];epsilon_directions=set()
 for kind in ("diagonal","offdiag_nonzero","offdiag_zero"):
  values=selected.loc[selected.coefficient_type==kind,"coefficient_global_index"].astype(int).tolist()
  if values:epsilon_directions.add(values[0])
 for position,index in enumerate(indices):
  eps_values=FINITE_DIFF_EPS_GRID if index in epsilon_directions else [FINITE_DIFF_EPS_PRIMARY]
  for epsilon in eps_values:
   try:
    column,plus,minus=sensitivity_column(model,data["y"],data["u"],index,epsilon,PERTURBED_VB_MAX_ITER,PERTURBED_CONVERGENCE_TOL,reestimate_B=reestimate_B,reestimate_Q=reestimate_Q)
    if epsilon==FINITE_DIFF_EPS_PRIMARY:columns[:,position]=column
    times.extend([plus.runtime_seconds,minus.runtime_seconds]);rows.append({**meta,"perturbation_direction_index":index,"eps":epsilon,"epsilon_stability_direction":index in epsilon_directions,"plus_converged":plus.converged,"minus_converged":minus.converged,"plus_n_iter":plus.n_iter,"minus_n_iter":minus.n_iter,"plus_final_relative_A_change":plus.final_relative_A_change,"minus_final_relative_A_change":minus.final_relative_A_change,"plus_final_relative_B_change":plus.final_relative_B_change,"minus_final_relative_B_change":minus.final_relative_B_change,"plus_final_relative_Q_change":plus.final_relative_Q_change,"minus_final_relative_Q_change":minus.final_relative_Q_change,"plus_loglikelihood":plus.final_loglikelihood,"minus_loglikelihood":minus.final_loglikelihood,"variance_estimate_raw":column[index],"finite_difference_valid":np.all(np.isfinite(column)),"warning_flag":";".join(filter(None,[plus.warning_flag,minus.warning_flag])),"B_reestimated_in_perturbed_fits":reestimate_B,"Q_reestimated_in_perturbed_fits":reestimate_Q})
   except Exception as error:rows.append({**meta,"perturbation_direction_index":index,"eps":epsilon,"epsilon_stability_direction":index in epsilon_directions,"finite_difference_valid":False,"warning_flag":f"{type(error).__name__}: {error}","B_reestimated_in_perturbed_fits":reestimate_B,"Q_reestimated_in_perturbed_fits":reestimate_Q})
 valid=np.flatnonzero(np.all(np.isfinite(columns),axis=0));valid_indices=[indices[k] for k in valid]
 if not valid_indices:raise FloatingPointError("No valid free-B Stage-B directions.")
 raw,psd,diag=project_selected_covariance(columns[:,valid],valid_indices);frame=pd.DataFrame(rows);mapping=dict(zip(valid_indices,np.diag(psd)));frame["variance_estimate_psd"]=[mapping.get(int(index),np.nan) if np.isclose(eps,FINITE_DIFF_EPS_PRIMARY) else np.nan for index,eps in zip(frame.perturbation_direction_index,frame.eps)]
 return valid_indices,raw,psd,frame,diag,times

def covariance_variants(blocks,records):
 L=blocks["stabilized_louis_eta_0p70_tau_0p90"];B=blocks["lrvb_stageB_smoother_feedback_psd_projected"];variants=[]
 def add(name,cov,weights=None,diag=None):variants.append({"covariance_estimator":name,"covariance":cov,"weights":weights,"projection":diag or {"psd_projection_used":False,"fraction_negative_variances":0.}})
 add("ordinary_vb",blocks["ordinary_vb"]);add("stabilized_louis_eta_0p70_tau_0p90",L);add("lrvb_stageB_smoother_feedback_raw",blocks["lrvb_stageB_smoother_feedback_raw"]);add("lrvb_stageB_smoother_feedback_psd_projected",B)
 def weight(mode,tau=None):
  values={}
  for r in records:
   w=1. if r["diagonal"] else (float(r["true_edge"]) if mode=="oracle" else (logistic_weight(r["group_snr_louis"],2.5,4.) if mode=="soft" else float(r["group_snr_louis"]>=tau)))
   for pos in r["positions"]:values[pos]=w
  return np.asarray([values.get(k,0.) for k in range(L.shape[0])])
 for name,w in (("oracle_active_subspace_stageB",weight("oracle")),("soft_sparsity_aware_stageB_logistic_c4_tau2p5",weight("soft")),("hard_active_subspace_stageB_group_snr_louis_tau2p5",weight("hard",2.5)),("hard_active_subspace_stageB_group_snr_louis_tau3p0",weight("hard",3.))):
  g=gated_covariance(L,B,w);add(name,g.covariance,w,g.diagnostics)
 w=weight("soft");g=gated_covariance(L,B,w,True);add("soft_sparsity_aware_stageB_logistic_c4_tau2p5_conservative",g.covariance,w,g.diagnostics);g=globally_damped_covariance(L,B,.75);add("global_lambda_stageB_0p75",g.covariance,np.full(len(w),np.sqrt(.75)),g.diagnostics)
 return variants

def coefficient_rows(model,data,selected,variant,meta):
 theta=flatten_A(model.A_mean_matrices_);truth=flatten_A(data["A"]);diag=np.diag(variant["covariance"]);rows=[]
 for position,item in selected.reset_index(drop=True).iterrows():
  index=int(item.coefficient_global_index);sd=np.sqrt(max(diag[position],0));error=theta[index]-truth[index];rows.append({**meta,"covariance_estimator":variant["covariance_estimator"],"coefficient_global_index":index,"lag":item.lag,"target":item.target,"source":item.source,"coefficient_type":item.coefficient_type,"selected_direction_type":item.selected_direction_type,"posterior_center":theta[index],"true_value":truth[index],"signed_error":error,"abs_error":abs(error),"squared_error":error**2,"posterior_variance":diag[position],"posterior_sd":sd,"ci95_lower":theta[index]-1.96*sd,"ci95_upper":theta[index]+1.96*sd,"ci95_contains_true":abs(error)<=1.96*sd,"standardized_error":error/max(sd,1e-15),"interval_width_95":3.92*sd})
 return pd.DataFrame(rows)

def group_rows(model,records,variant,meta):
 rows=[];scores=[]
 for r in records:
  if r["diagonal"] or not r["complete_group"]:continue
  block=variant["covariance"][np.ix_(r["positions"],r["positions"])];difference=r["mean"]-r["truth"];D2=difference@np.linalg.pinv(block)@difference;snr=np.sqrt(max(r["mean"]@np.linalg.pinv(block)@r["mean"],0));common={**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"],"edge_group_type":"true_edge_group" if r["true_edge"] else "false_edge_group","covariance_estimator":variant["covariance_estimator"]};rows.append({**common,"D2":D2,"D2_below_chi2_95_df2":D2<=5.991464547,"D2_below_chi2_99_df2":D2<=9.210340372,"group_cov_trace":np.trace(block),"group_error_norm":np.linalg.norm(difference),"group_snr":snr});scores.append({**common,"true_link":r["true_edge"],"score":snr})
 return pd.DataFrame(rows),pd.DataFrame(scores)

def calibration_summary(frame):
 keys=["config_label","M","T","covariance_estimator","coefficient_type"];rows=[]
 for values,g in frame.groupby(keys,dropna=False):rows.append({**dict(zip(keys,values)),"empirical_coverage_95":g.ci95_contains_true.mean(),"mean_signed_error":g.signed_error.mean(),"median_signed_error":g.signed_error.median(),"mean_abs_error":g.abs_error.mean(),"median_abs_error":g.abs_error.median(),"rmse":np.sqrt(g.squared_error.mean()),"mean_posterior_sd":g.posterior_sd.mean(),"median_posterior_sd":g.posterior_sd.median(),"mean_interval_width_95":g.interval_width_95.mean(),"median_interval_width_95":g.interval_width_95.median(),"mean_standardized_error":g.standardized_error.mean(),"std_standardized_error":g.standardized_error.std(ddof=0),"median_abs_standardized_error":g.standardized_error.abs().median(),"n_coefficients":len(g)})
 return pd.DataFrame(rows)

def group_summary(frame):return frame.groupby(["config_label","M","T","covariance_estimator","edge_group_type"],dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("D2","size")).reset_index()

def edge_summary(frame):
 rows=[]
 for values,g in frame.groupby(["config_label","M","T","covariance_estimator"],dropna=False):metrics,_=j.safe_curve(g.true_link,g.score);rows.append({"config_label":values[0],"M":values[1],"T":values[2],"covariance_estimator":values[3],**metrics})
 return pd.DataFrame(rows)

def full_network(model,data,meta,louis):
 rows=[];truth=[];scores={"model_A_group_norm":[],"ordinary_vb_group_snr":[],"stabilized_louis_group_snr":[]};mean=np.asarray(model.A_mean_matrices_);M=model.n_states
 for target in range(M):
  for source in range(M):
   if target==source:continue
   cols=[lag*M+source for lag in range(j.na)];m=mean[:,target,source];truth.append(bool(data["mask"][target,source]));scores["model_A_group_norm"].append(np.linalg.norm(m));scores["ordinary_vb_group_snr"].append(np.sqrt(max(m@np.linalg.pinv(model.A_row_covariances_[target][np.ix_(cols,cols)])@m,0)));scores["stabilized_louis_group_snr"].append(np.sqrt(max(m@np.linalg.pinv(louis[target][np.ix_(cols,cols)])@m,0)))
 for name,score in scores.items():metrics,_=j.safe_curve(truth,score);rows.append({**meta,"score_type":name,**metrics})
 return pd.DataFrame(rows)

def B_recovery(model,data,meta):
 true,hat=np.asarray(data["B"]),np.asarray(model.B_matrices);return {**meta,"B_relative_frobenius_error":relative(hat,true),"B_lag0_relative_error":relative(hat[0],true[0]),"B_lag1_relative_error":relative(hat[1],true[1]),"B_lag2_relative_error":relative(hat[2],true[2]),"B_energy_ratio_hat_to_true":np.linalg.norm(hat)**2/max(np.linalg.norm(true)**2,1e-15),"B_correlation_hat_true":correlations(hat.ravel(),true.ravel())}

def A_recovery(model,data,meta):
 row,_=j.recovery(np.asarray(model.A_mean_matrices_),data,model.filtered_state_mean_,model.smoothed_state_mean_,meta,0.,model.n_iter_,model.converged_);return {**meta,"A_relative_frobenius_error":row["A_relative_frobenius_error"],"A_offdiag_relative_frobenius_error":row["A_offdiag_relative_frobenius_error"],"A_diag_relative_frobenius_error":row["A_diagonal_relative_frobenius_error"],"active_group_norm_correlation":row["A_group_norm_pearson_correlation"],"support_AUPRC_model_A_group_norm":row["A_support_AUPRC"]}

def empirical(frame):
 rows=[]
 for values,g in frame.groupby(["config_label","true_network_id","M","T","coefficient_global_index","covariance_estimator","coefficient_type"],dropna=False):emp=g.signed_error.var(ddof=1) if len(g)>1 else g.squared_error.mean();post=g.posterior_variance.mean();rows.append({"config_label":values[0],"true_network_id":values[1],"M":values[2],"T":values[3],"coefficient_global_index":values[4],"covariance_estimator":values[5],"coefficient_type":values[6],"posterior_variance":post,"empirical_error_variance":emp,"variance_ratio_posterior_to_empirical":post/max(emp,1e-15),"posterior_sd":np.sqrt(max(post,0)),"empirical_sd":np.sqrt(max(emp,0)),"sd_ratio_posterior_to_empirical":np.sqrt(max(post,0))/max(np.sqrt(max(emp,0)),1e-15),"n_replicates":len(g)})
 return pd.DataFrame(rows)

def gating_row(meta,variant,records):
 off=[r for r in records if not r["diagonal"]];weights=[]
 for r in off:weights.append(np.mean(variant["weights"][r["positions"]]) if variant["weights"] is not None else (1. if "stageB" in variant["covariance_estimator"] else 0.))
 weights=np.asarray(weights);truth=np.asarray([r["true_edge"] for r in off]);sel=weights>=.5;tp=np.sum(sel&truth);fp=np.sum(sel&~truth)
 return {**meta,"covariance_estimator":variant["covariance_estimator"],"number_groups_selected":sel.sum(),"fraction_groups_selected":sel.mean(),"true_edge_selection_rate":sel[truth].mean() if truth.any() else np.nan,"false_edge_selection_rate":sel[~truth].mean() if (~truth).any() else np.nan,"selected_true_edge_precision":tp/max(tp+fp,1),"selected_true_edge_recall":tp/max(truth.sum(),1),"mean_soft_weight_true_edges":weights[truth].mean() if truth.any() else np.nan,"mean_soft_weight_false_edges":weights[~truth].mean() if (~truth).any() else np.nan,"median_soft_weight_true_edges":np.median(weights[truth]) if truth.any() else np.nan,"median_soft_weight_false_edges":np.median(weights[~truth]) if (~truth).any() else np.nan}

def activity_rows(meta,records,blocks):
 rows=[]
 for r in records:
  if r["diagonal"]:continue
  stage=blocks["lrvb_stageB_smoother_feedback_psd_projected"][np.ix_(r["positions"],r["positions"])];rows.append({**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"],"group_norm":r["group_norm"],"group_snr_current":r["group_snr_current"],"group_snr_louis":r["group_snr_louis"],"group_snr_stageB":np.sqrt(max(r["mean"]@np.linalg.pinv(stage)@r["mean"],0)),"E_alpha":r["E_alpha"],"inverse_alpha_score":r["inverse_alpha_score"],"soft_logistic_weight_c4_tau2p5":logistic_weight(r["group_snr_louis"],2.5,4),"hard_tau2p5_selected":r["group_snr_louis"]>=2.5,"hard_tau3p0_selected":r["group_snr_louis"]>=3.,"stageB_over_louis_trace_ratio":r["stageB_over_louis_trace_ratio"],"needed_ratio_if_truth_available":r["needed_variance_ratio_if_truth_available"]})
 return pd.DataFrame(rows)

def summaries(frames):
 cal=calibration_summary(frames["coeff"]);groups=group_summary(frames["group"]);edges=edge_summary(frames["scores"]);emp=empirical(frames["coeff"]);Bsum=grouped_stats(frames["B"],["config_label","M","T"],["B_relative_frobenius_error","B_lag0_relative_error","B_lag1_relative_error","B_lag2_relative_error","B_energy_ratio_hat_to_true","B_correlation_hat_true"]);soft_name="soft_sparsity_aware_stageB_logistic_c4_tau2p5";repkeys=["config_label","M","T","true_network_id","replicate_id"];soft=frames["coeff"].loc[frames["coeff"].covariance_estimator==soft_name].groupby(repkeys+["coefficient_type"]).agg(coverage=("ci95_contains_true","mean"),std_z=("standardized_error",lambda x:x.std(ddof=0))).reset_index();active=soft.loc[soft.coefficient_type=="offdiag_nonzero",repkeys+["coverage","std_z"]];score_rows=[]
 for values,g in frames["scores"].loc[frames["scores"].covariance_estimator==soft_name].groupby(repkeys):metrics,_=j.safe_curve(g.true_link,g.score);score_rows.append({**dict(zip(repkeys,values)),"selected_edge_AUPRC":metrics["AUPRC"]})
 merged=frames["B"].merge(active,on=repkeys,how="left").merge(pd.DataFrame(score_rows),on=repkeys,how="left")
 for values,g in merged.groupby(["config_label","M","T"]):
  mask=(Bsum.config_label==values[0])&(Bsum.M==values[1])&(Bsum["T"]==values[2]);Bsum.loc[mask,"correlation_B_error_active_coverage"]=correlations(g.B_relative_frobenius_error,g.coverage);Bsum.loc[mask,"correlation_B_error_offdiag_nonzero_std_z"]=correlations(g.B_relative_frobenius_error,g.std_z);Bsum.loc[mask,"correlation_B_error_selected_edge_AUPRC"]=correlations(g.B_relative_frobenius_error,g.selected_edge_AUPRC)
 Asum=grouped_stats(frames["A"],["config_label","M","T"],["A_relative_frobenius_error","A_offdiag_relative_frobenius_error","A_diag_relative_frobenius_error","active_group_norm_correlation","support_AUPRC_model_A_group_norm"]);runtime=grouped_stats(frames["runtime"],["config_label","M","T"],["baseline_VB_runtime_seconds","louis_runtime_seconds","stageB_total_runtime_seconds","average_perturbed_fit_runtime_seconds","number_perturbed_fits","number_selected_directions","total_runtime_seconds"]);return cal,groups,edges,emp,Bsum,Asum,runtime

def decision(cal,groups,edges,network,emp,Braw,Araw,covdiag,runtime):
 rows=[]
 for keys,g in cal.groupby(["config_label","M","T","covariance_estimator"],dropna=False):
  by={k:v for k,v in g.groupby("coefficient_type")};val=lambda k,c:getattr(by[k],c).mean() if k in by else np.nan;gr=groups.loc[(groups.M==keys[1])&(groups["T"]==keys[2])&(groups.covariance_estimator==keys[3])];ed=edges.loc[(edges.M==keys[1])&(edges["T"]==keys[2])&(edges.covariance_estimator==keys[3])];net=network.loc[(network.M==keys[1])&(network["T"]==keys[2])];ev=emp.loc[(emp.M==keys[1])&(emp["T"]==keys[2])&(emp.covariance_estimator==keys[3])];br=Braw.loc[(Braw.M==keys[1])&(Braw["T"]==keys[2])];ar=Araw.loc[(Araw.M==keys[1])&(Araw["T"]==keys[2])];cd=covdiag.loc[(covdiag.M==keys[1])&(covdiag["T"]==keys[2])&(covdiag.covariance_estimator==keys[3])];active=np.nanmean([val("diagonal","empirical_coverage_95"),val("offdiag_nonzero","empirical_coverage_95")]);rows.append({"config_label":keys[0],"M":keys[1],"T":keys[2],"covariance_estimator":keys[3],"diag_coverage":val("diagonal","empirical_coverage_95"),"offdiag_nonzero_coverage":val("offdiag_nonzero","empirical_coverage_95"),"offdiag_zero_coverage":val("offdiag_zero","empirical_coverage_95"),"diag_std_z":val("diagonal","std_standardized_error"),"offdiag_nonzero_std_z":val("offdiag_nonzero","std_standardized_error"),"offdiag_zero_std_z":val("offdiag_zero","std_standardized_error"),"diag_mean_interval_width_95":val("diagonal","mean_interval_width_95"),"offdiag_nonzero_mean_interval_width_95":val("offdiag_nonzero","mean_interval_width_95"),"offdiag_zero_mean_interval_width_95":val("offdiag_zero","mean_interval_width_95"),"true_edge_group_chi2_95_coverage":gr.loc[gr.edge_group_type=="true_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"false_edge_group_chi2_95_coverage":gr.loc[gr.edge_group_type=="false_edge_group","fraction_D2_below_chi2_95_df2"].mean(),"selected_edge_AUC":ed.ROC_AUC.mean(),"selected_edge_AUPRC":ed.AUPRC.mean(),"selected_edge_TPR_at_FPR_0p05":ed.TPR_at_FPR_0p05.mean(),"selected_edge_precision_at_FPR_0p05":ed.precision_at_FPR_0p05.mean(),"full_network_AUPRC_model_A_group_norm":net.loc[net.score_type=="model_A_group_norm","AUPRC"].mean(),"full_network_AUPRC_stabilized_louis_group_snr":net.loc[net.score_type=="stabilized_louis_group_snr","AUPRC"].mean(),"B_relative_frobenius_error":br.B_relative_frobenius_error.mean(),"B_energy_ratio_hat_to_true":br.B_energy_ratio_hat_to_true.mean(),"A_relative_frobenius_error":ar.A_relative_frobenius_error.mean(),"A_offdiag_relative_frobenius_error":ar.A_offdiag_relative_frobenius_error.mean(),"variance_ratio_to_empirical_diag":ev.loc[ev.coefficient_type=="diagonal","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_nonzero":ev.loc[ev.coefficient_type=="offdiag_nonzero","variance_ratio_posterior_to_empirical"].mean(),"variance_ratio_to_empirical_offdiag_zero":ev.loc[ev.coefficient_type=="offdiag_zero","variance_ratio_posterior_to_empirical"].mean(),"fraction_negative_variances":cd.fraction_negative_variances.mean(),"fraction_psd_projected":cd.fraction_psd_projected.mean(),"total_runtime_seconds":runtime.loc[(runtime.M==keys[1])&(runtime["T"]==keys[2])].filter(like="total_runtime_seconds_mean").mean(axis=1).mean(),"recommended_for_uncertainty":bool(.88<=active<=.97 and .90<=val("offdiag_zero","empirical_coverage_95")<=.99),"recommended_for_ranking":bool(ed.AUPRC.mean()>=edges.loc[(edges.M==keys[1])&(edges["T"]==keys[2]),"AUPRC"].max()-.05),"notes":"Free-B Hybrid VB; perturbed Stage-B fits re-estimate B; all centers remain A_VB."})
 return pd.DataFrame(rows)

def save_plots(cal,coeff,activity,Braw,edges,covdiag,runtime):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 def bar(frame,x,y,name):
  s=frame.groupby(x,dropna=False)[y].mean();fig,ax=plt.subplots();s.plot.bar(ax=ax);ax.set_ylabel(y);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(os.path.join(path,name));plt.close(fig)
 bar(cal,"covariance_estimator","empirical_coverage_95","coverage.png");bar(cal,"covariance_estimator","std_standardized_error","standardized_sd.png");bar(cal,"covariance_estimator","mean_interval_width_95","interval_width.png");bar(cal,"covariance_estimator","mean_posterior_sd","posterior_sd.png");bar(cal,"coefficient_type","empirical_coverage_95","active_zero_coverage.png");bar(activity,"true_edge_group","soft_logistic_weight_c4_tau2p5","soft_weights.png");bar(activity,"true_edge_group","group_snr_louis","snr_distribution.png");merged=Braw.merge(cal.loc[cal.coefficient_type=="offdiag_nonzero"],on=["config_label","M","T"]);fig,ax=plt.subplots();ax.scatter(merged.B_relative_frobenius_error,merged.empirical_coverage_95);fig.tight_layout();fig.savefig(os.path.join(path,"B_error_coverage.png"));plt.close(fig);fig,ax=plt.subplots();ax.scatter(merged.B_relative_frobenius_error,merged.std_standardized_error);fig.tight_layout();fig.savefig(os.path.join(path,"B_error_stdz.png"));plt.close(fig);bar(edges,"covariance_estimator","AUPRC","selected_AUPRC.png");bar(covdiag,"covariance_estimator","trace_over_louis_selected","variance_ratio.png");bar(runtime,"M","total_runtime_seconds_mean","runtime.png")

def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34Q-A","CONFIG_LIST":CONFIG_LIST,"FINITE_DIFF_EPS_GRID":FINITE_DIFF_EPS_GRID,"FINITE_DIFF_EPS_PRIMARY":FINITE_DIFF_EPS_PRIMARY,"PERTURBED_VB_MAX_ITER":PERTURBED_VB_MAX_ITER,"PERTURBED_CONVERGENCE_TOL":PERTURBED_CONVERGENCE_TOL,"VB_MAX_ITER":VB_MAX_ITER,"B_REESTIMATED_IN_PERTURBED_FITS":True,"C_MODE":"identity","Q_MODE":"fixed_Q_true","R_MODE":"fixed_R_true","center_estimator":"A_VB","smoke_test":SMOKE_TEST};cp=os.path.join(RESULTS_DIR,"experiment_config.json");names={"selected":"_selected_partial.csv","groupsel":"_groupsel_partial.csv","coeff":"_coeff_partial.csv","group":"_group_partial.csv","scores":"_scores_partial.csv","network":"_network_partial.csv","B":"_B_partial.csv","A":"_A_partial.csv","covdiag":"covariance_diagnostics_partial.csv","perturb":"stageB_perturbation_diagnostics_partial.csv","activity":"_activity_partial.csv","gating":"_gating_partial.csv","runtime":"_runtime_partial.csv","run":"run_summary_partial.csv"};tables={k:[] for k in names}
 if os.path.exists(cp) and os.path.exists(os.path.join(RESULTS_DIR,"run_summary_partial.csv")):
  with open(cp,encoding="utf8") as h:old=json.load(h)
  for key in ("CONFIG_LIST","FINITE_DIFF_EPS_GRID","FINITE_DIFF_EPS_PRIMARY","PERTURBED_VB_MAX_ITER","PERTURBED_CONVERGENCE_TOL","VB_MAX_ITER"):
   if old.get(key)!=config.get(key):raise ValueError("Existing 34Q-A checkpoint configuration differs.")
 with open(cp,"w",encoding="utf8") as h:json.dump(config,h,indent=2)
 for key,name in names.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:f=pd.read_csv(path)
   except pd.errors.EmptyDataError:f=pd.DataFrame()
   if len(f):tables[key]=[f]
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
      data=j.simulate(A,B,M,T,seed);base=time.perf_counter();model=fit_free_B(data,seed+500);base_runtime=time.perf_counter()-base;lstart=time.perf_counter();louis=n._louis_covariances(model,data,seed+700);louis_runtime=time.perf_counter()-lstart;selected=select_directions(model,data,louis,seed+900);sstart=time.perf_counter();valid,raw,psd,perturb,projection,times=run_stageB_free(model,data,selected,meta,True);stage_runtime=time.perf_counter()-sstart;selected=selected.set_index("coefficient_global_index").loc[valid].reset_index();tables["selected"].append(pd.DataFrame([{**meta,**row} for row in selected.to_dict("records")]));global_sets={"ordinary_vb":o.global_covariance_block(model.A_row_covariances_),"stabilized_louis_eta_0p70_tau_0p90":o.global_covariance_block(louis)};blocks={name:value[np.ix_(valid,valid)] for name,value in global_sets.items()};blocks["lrvb_stageB_smoother_feedback_raw"]=raw;blocks["lrvb_stageB_smoother_feedback_psd_projected"]=psd;records=p.group_records(model,data,selected,{"ordinary_vb":blocks["ordinary_vb"],"stabilized_louis_eta_0p70_tau_0p90":blocks["stabilized_louis_eta_0p70_tau_0p90"],"lrvb_stageB_smoother_feedback":psd});tables["groupsel"].append(pd.DataFrame([{**meta,"group_target":r["target"],"group_source":r["source"],"true_edge_group":r["true_edge"],"complete_group":r["complete_group"]} for r in records]));all_variants=covariance_variants(blocks,records)
      for variant in all_variants:
       tables["coeff"].append(coefficient_rows(model,data,selected,variant,meta));g,s=group_rows(model,records,variant,meta);tables["group"].append(g);tables["scores"].append(s);tables["gating"].append(gating_row(meta,variant,records));proj=variant["projection"];tables["covdiag"].append({**meta,"covariance_estimator":variant["covariance_estimator"],"number_selected_directions":len(selected),"number_negative_variances":proj.get("number_negative_variances",0),"fraction_negative_variances":proj.get("fraction_negative_variances",0.),"number_psd_projections":int(proj.get("psd_projection_used",False)),"fraction_psd_projected":float(proj.get("psd_projection_used",False)),**covariance_diagnostics(variant["covariance"],blocks["ordinary_vb"],blocks["stabilized_louis_eta_0p70_tau_0p90"],selected.coefficient_type)})
      tables["network"].append(full_network(model,data,meta,louis));tables["B"].append(B_recovery(model,data,meta));tables["A"].append(A_recovery(model,data,meta));tables["perturb"].append(perturb);tables["activity"].append(activity_rows(meta,records,blocks));tables["runtime"].append({**meta,"baseline_VB_runtime_seconds":base_runtime,"louis_runtime_seconds":louis_runtime,"stageB_total_runtime_seconds":stage_runtime,"average_perturbed_fit_runtime_seconds":np.mean(times),"number_perturbed_fits":len(times),"number_selected_directions":len(selected),"B_estimation_enabled":True,"total_runtime_seconds":time.perf_counter()-started,"run_status":"success"})
     except Exception as error:status="failed";tables["runtime"].append({**meta,"total_runtime_seconds":time.perf_counter()-started,"run_status":status,"error_type":type(error).__name__,"error_message":str(error),"traceback":traceback.format_exc()})
     tables["run"].append({**meta,"fit_status":status,"total_runtime_seconds":time.perf_counter()-started});frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()}
     for k,name in names.items():atomic_csv(frames[k],os.path.join(RESULTS_DIR,name))
     if len(frames["coeff"]):
      cal,groups,edges,emp,Bsum,Asum,runtime=summaries(frames);dec=decision(cal,groups,edges,frames["network"],emp,frames["B"],frames["A"],frames["covdiag"],runtime)
      for name,f in (("calibration_summary_partial.csv",cal),("group_calibration_summary_partial.csv",groups),("B_recovery_summary_partial.csv",Bsum),("A_recovery_summary_partial.csv",Asum),("runtime_summary_partial.csv",runtime),("decision_summary_partial.csv",dec)):atomic_csv(f,os.path.join(RESULTS_DIR,name))
     progress.update(f"{cfg['label']} T={T} net={network_id+1} rep={replicate+1} {status}")
 frames={k:(pd.concat(v,ignore_index=True) if v and isinstance(v[0],pd.DataFrame) else pd.DataFrame(v)) for k,v in tables.items()};cal,groups,edges,emp,Bsum,Asum,runtime=summaries(frames);dec=decision(cal,groups,edges,frames["network"],emp,frames["B"],frames["A"],frames["covdiag"],runtime);outputs={"run_summary.csv":frames["run"],"selected_coefficients.csv":frames["selected"],"selected_groups.csv":frames["groupsel"],"coefficient_calibration_results.csv":frames["coeff"],"calibration_summary.csv":cal,"group_calibration_results.csv":frames["group"],"group_calibration_summary.csv":groups,"network_recovery_summary.csv":frames["network"],"selected_edge_recovery_summary.csv":edges,"B_recovery_summary.csv":Bsum,"A_recovery_summary.csv":Asum,"covariance_diagnostics.csv":frames["covdiag"],"stageB_perturbation_diagnostics.csv":frames["perturb"],"activity_score_diagnostics.csv":frames["activity"],"gating_diagnostics.csv":frames["gating"],"empirical_variance_comparison.csv":emp,"deviance_diagnostics.csv":pd.DataFrame([{"status":"not_computed_secondary_runtime_priority","reason":"34Q-A prioritizes free-B covariance calibration"}]),"runtime_summary.csv":runtime,"decision_summary.csv":dec}
 for name,f in outputs.items():atomic_csv(f,os.path.join(RESULTS_DIR,name))
 if not SMOKE_TEST:save_plots(cal,frames["coeff"],frames["activity"],frames["B"],edges,frames["covdiag"],runtime)

if __name__=="__main__":main()
