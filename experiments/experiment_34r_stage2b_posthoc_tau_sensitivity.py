"""Experiment 34R Stage 2B: post-hoc gate-threshold sensitivity.

No model fitting, smoothing, Louis sampling, or Stage-B perturbation occurs.
Coefficient variances are recomputed exactly from saved Louis/Stage-B diagonals.
The Stage-2 CSV schema lacks 2x2 off-diagonal covariance entries, so group
diagnostics use an explicitly labelled diagonal-block reconstruction.
"""
import json,os
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations

INPUT_DIR=os.environ.get("EXPERIMENT_34R_STAGE2B_INPUT_DIR","results/experiment_34r_stage2")
OUTPUT_DIR=os.environ.get("EXPERIMENT_34R_STAGE2B_RESULTS_DIR","results/experiment_34r_stage2b_posthoc_tau_sensitivity")
TAUS=[1.,1.5,2.,2.5];SLOPE=4.;EPS_VAR=1e-12
REQUIRED=["experiment_config.json","selected_coefficients.csv","selected_groups.csv","calibration_summary.csv","group_calibration_summary.csv","covariance_diagnostics.csv","activity_score_diagnostics.csv","observability_stageB_summary.csv","decision_summary.csv","runtime_summary.csv"]
OPTIONAL=["coefficient_calibration_results.csv","group_calibration_results.csv","stageB_perturbation_diagnostics.csv","C_diagnostics.csv","network_recovery_summary.csv"]
KEYS=["config_label","M_x","M_y","observation_ratio","C_family","T","true_network_id","replicate_id","coefficient_global_index"]

def label(gate,tau):return f"{gate}_stageB_group_snr_tau{str(tau).replace('.','p')}" if gate=="hard" else f"soft_stageB_logistic_c4_tau{str(tau).replace('.','p')}"
def inspect_inputs():
 report={"input_directory":INPUT_DIR,"files_found":[],"files_missing":[],"files":{},"searched_quantity_locations":{}}
 for name in REQUIRED+OPTIONAL:
  path=os.path.join(INPUT_DIR,name)
  if not os.path.exists(path):report["files_missing"].append(name);continue
  report["files_found"].append(name)
  if name.endswith(".json"):
   with open(path,encoding="utf8") as h:data=json.load(h)
   report["files"][name]={"type":"json","top_level_keys":list(data)}
  else:
   try:f=pd.read_csv(path)
   except pd.errors.EmptyDataError:f=pd.DataFrame()
   report["files"][name]={"rows":len(f),"columns":list(f.columns)}
 search_terms=["posterior_variance","posterior_sd","covariance_estimator","lrvb_stageB_smoother_feedback_raw","lrvb_stageB_smoother_feedback_psd_projected","stabilized_louis_eta_0p70_tau_0p90","ordinary_vb","soft_sparsity_aware_stageB_logistic_c4_tau2p5","hard_active_subspace_stageB_group_snr_louis_tau2p5","group_snr_louis","true_value","posterior_center","coefficient_type","selected_direction_type","target","source","lag"]
 for term in search_terms:
  report["searched_quantity_locations"][term]=[name for name,meta in report["files"].items() if term in meta.get("columns",[])]
 coeff=report["files"].get("coefficient_calibration_results.csv",{}).get("columns",[]);activity=report["files"].get("activity_score_diagnostics.csv",{}).get("columns",[])
 needed={"posterior_variance","covariance_estimator","posterior_center","true_value","coefficient_type","target","source","lag"};report["inferred_covariance_columns"]=[x for x in coeff if "variance" in x or "covariance" in x or "posterior_sd" in x];report["inferred_coefficient_level_columns"]=coeff;report["inferred_group_level_columns"]=report["files"].get("group_calibration_results.csv",{}).get("columns",[]);report["exact_coefficient_recomputation_possible"]=needed.issubset(coeff) and "group_snr_louis" in activity;report["exact_group_block_recomputation_possible"]=False;report["group_block_limitation"]="Stage 2 CSVs save group trace/D2 but not the 2x2 Louis and Stage-B covariance block elements. Group results therefore use a diagonal-only reconstruction and are labelled as such.";report["posthoc_tau_recomputation_possible"]=report["exact_coefficient_recomputation_possible"]
 return report

def load(name,required=True):
 path=os.path.join(INPUT_DIR,name)
 if not os.path.exists(path):
  if required:raise FileNotFoundError(path)
  return pd.DataFrame()
 return pd.read_csv(path)

def build():
 coeff=load("coefficient_calibration_results.csv");activity=load("activity_score_diagnostics.csv");groups=load("selected_groups.csv")
 base_keys=[x for x in KEYS if x in coeff];keep=["posterior_center","true_value","signed_error","abs_error","squared_error","coefficient_type","selected_direction_type","target","source","lag","source_observability_score","target_observability_score","edge_observability_score"]
 center=coeff.drop_duplicates(base_keys)[base_keys+[x for x in keep if x not in base_keys]]
 wide=coeff.loc[coeff.covariance_estimator.isin(["stabilized_louis_eta_0p70_tau_0p90","lrvb_stageB_smoother_feedback_psd_projected"])].pivot_table(index=base_keys,columns="covariance_estimator",values="posterior_variance",aggfunc="first").reset_index();wide=wide.rename(columns={"stabilized_louis_eta_0p70_tau_0p90":"variance_louis","lrvb_stageB_smoother_feedback_psd_projected":"variance_stageB"});data=center.merge(wide,on=base_keys,how="inner")
 akeys=[x for x in ["config_label","M_x","M_y","observation_ratio","C_family","T","true_network_id","replicate_id","group_target","group_source"] if x in activity];amap=activity.drop_duplicates(akeys)[akeys+["group_snr_louis"]].rename(columns={"group_target":"target","group_source":"source"});merge_keys=[x for x in ["config_label","M_x","M_y","observation_ratio","C_family","T","true_network_id","replicate_id","target","source"] if x in data and x in amap];data=data.merge(amap,on=merge_keys,how="left",suffixes=("","_activity"));rows=[]
 for gate in ("soft","hard"):
  for tau in TAUS:
   w=np.where(data.target==data.source,1.,1/(1+np.exp(-SLOPE*(data.group_snr_louis-tau))) if gate=="soft" else (data.group_snr_louis>=tau).astype(float));var=data.variance_louis+w*w*(data.variance_stageB-data.variance_louis);var=np.maximum(var,EPS_VAR);sd=np.sqrt(var);part=data.copy();part["gate_type"],part["tau"],part["gate_weight"],part["covariance_estimator"]=gate,tau,w,label(gate,tau);part["posterior_variance"],part["posterior_sd"]=var,sd;part["ci95_lower"]=part.posterior_center-1.96*sd;part["ci95_upper"]=part.posterior_center+1.96*sd;part["ci95_contains_true"]=(part.true_value>=part.ci95_lower)&(part.true_value<=part.ci95_upper);part["standardized_error"]=(part.posterior_center-part.true_value)/sd;part["interval_width_95"]=3.92*sd;rows.append(part)
 post=pd.concat(rows,ignore_index=True)
 baseline=coeff.loc[coeff.covariance_estimator.isin(["ordinary_vb","stabilized_louis_eta_0p70_tau_0p90","lrvb_stageB_smoother_feedback_raw","lrvb_stageB_smoother_feedback_psd_projected","global_lambda_stageB_0p75","soft_sparsity_aware_stageB_logistic_c4_tau2p5"])].copy();baseline["gate_type"]="baseline";baseline["tau"]=np.nan;baseline["gate_weight"]=np.nan
 return post,baseline,groups

def calibration(f):
 keys=["gate_type","tau","covariance_estimator","config_label","M_x","M_y","observation_ratio","T","coefficient_type"]
 return f.groupby(keys,dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),mean_signed_error=("signed_error","mean"),median_signed_error=("signed_error","median"),mean_abs_error=("abs_error","mean"),median_abs_error=("abs_error","median"),rmse=("squared_error",lambda x:np.sqrt(x.mean())),mean_posterior_sd=("posterior_sd","mean"),median_posterior_sd=("posterior_sd","median"),mean_interval_width_95=("interval_width_95","mean"),median_interval_width_95=("interval_width_95","median"),mean_standardized_error=("standardized_error","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),median_abs_standardized_error=("standardized_error",lambda x:x.abs().median()),n_coefficients=("posterior_sd","size")).reset_index()

def group_results(post):
 rows=[];gkeys=["gate_type","tau","covariance_estimator","config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","target","source"]
 for values,g in post.groupby(gkeys,dropna=False):
  if len(g.lag.unique())<2:continue
  g=g.sort_values("lag").drop_duplicates("lag");d=(g.posterior_center-g.true_value).to_numpy();v=g.posterior_variance.to_numpy();D2=np.sum(d*d/np.maximum(v,EPS_VAR));obs=g.edge_observability_score.mean();rows.append({**dict(zip(gkeys,values)),"true_edge_group":g.coefficient_type.eq("offdiag_nonzero").any(),"edge_group_type":"true_edge_group" if g.coefficient_type.eq("offdiag_nonzero").any() else "false_edge_group","D2":D2,"D2_below_chi2_95_df2":D2<=5.991464547,"D2_below_chi2_99_df2":D2<=9.210340372,"group_cov_trace":v.sum(),"group_error_norm":np.linalg.norm(d),"group_snr":np.sqrt(np.sum(g.posterior_center.to_numpy()**2/np.maximum(v,EPS_VAR))),"edge_observability_score":obs,"covariance_reconstruction":"diagonal_only_group_block"})
 result=pd.DataFrame(rows)
 if len(result):
  true=result.true_edge_group.astype(bool);median=result.loc[true,"edge_observability_score"].median();result["observability_group_type"]=np.where(~true,"false_edge_group",np.where(result.edge_observability_score<=median,"low_observability_true_edge_group","high_observability_true_edge_group"))
 return result
def group_summary(g):
 overall=g.loc[g.true_edge_group].copy();overall["observability_group_type"]="true_edge_group";x=pd.concat([g,overall],ignore_index=True)
 return x.groupby(["gate_type","tau","covariance_estimator","config_label","M_x","M_y","observation_ratio","T","observability_group_type"],dropna=False).agg(mean_D2=("D2","mean"),median_D2=("D2","median"),fraction_D2_below_chi2_95_df2=("D2_below_chi2_95_df2","mean"),fraction_D2_below_chi2_99_df2=("D2_below_chi2_99_df2","mean"),mean_group_cov_trace=("group_cov_trace","mean"),median_group_cov_trace=("group_cov_trace","median"),n_groups=("D2","size")).reset_index()

def _corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 return float(np.corrcoef(x[ok],y[ok])[0,1]) if ok.sum()>1 and np.std(x[ok])>0 and np.std(y[ok])>0 else np.nan

def gate_diagnostics(post):
 keys=["config_label","M_x","M_y","observation_ratio","T","true_network_id","replicate_id","target","source","gate_type","tau"]
 g=post.loc[post.target!=post.source].groupby(keys,dropna=False).agg(true_edge_group=("coefficient_type",lambda x:x.eq("offdiag_nonzero").any()),gate_weight=("gate_weight","first"),edge_observability_score=("edge_observability_score","mean"),group_snr_louis=("group_snr_louis","first")).reset_index();g["selected_by_gate"]=g.gate_weight>=.5;g["suppressed_true_edge_tau2p5"]=g.true_edge_group&(~g.selected_by_gate)&g.tau.eq(2.5)
 out=[]
 for vals,x in g.groupby(["config_label","M_x","M_y","observation_ratio","T","gate_type","tau"],dropna=False):
  tr=x.true_edge_group.astype(bool);sel=x.selected_by_gate.astype(bool);ts=x[tr&sel];tu=x[tr&~sel]
  out.append({**dict(zip(["config_label","M_x","M_y","observation_ratio","T","gate_type","tau"],vals)),"number_true_edges_selected_by_gate":int((tr&sel).sum()),"fraction_true_edges_selected_by_gate":float((tr&sel).sum()/max(tr.sum(),1)),"number_false_edges_selected_by_gate":int((~tr&sel).sum()),"fraction_false_edges_selected_by_gate":float((~tr&sel).sum()/max((~tr).sum(),1)),"precision_of_gate":float((tr&sel).sum()/max(sel.sum(),1)),"recall_of_true_edges_by_gate":float((tr&sel).sum()/max(tr.sum(),1)),"mean_observability_selected_true_edges":ts.edge_observability_score.mean(),"mean_observability_suppressed_true_edges":tu.edge_observability_score.mean(),"median_observability_selected_true_edges":ts.edge_observability_score.median(),"median_observability_suppressed_true_edges":tu.edge_observability_score.median(),"mean_group_snr_selected_true_edges":ts.group_snr_louis.mean(),"mean_group_snr_suppressed_true_edges":tu.group_snr_louis.mean(),"n_true_edges":int(tr.sum()),"n_false_edges":int((~tr).sum())})
 return pd.DataFrame(out),g

def observability_diagnostics(post,gate_rows):
 z=post.loc[post.target!=post.source].groupby(["config_label","M_y","observation_ratio","gate_type","tau","true_network_id","replicate_id","target","source"],dropna=False).agg(mean_abs_standardized_error=("standardized_error",lambda x:x.abs().mean())).reset_index();g=gate_rows.merge(z,on=["config_label","M_y","observation_ratio","gate_type","tau","true_network_id","replicate_id","target","source"],how="left")
 rows=[]
 for vals,x in g.groupby(["config_label","M_y","observation_ratio","gate_type","tau"],dropna=False):
  tr=x.true_edge_group.astype(bool);sel=x.selected_by_gate.astype(bool);a=x[tr&sel];b=x[tr&~sel]
  rows.append({**dict(zip(["config_label","M_y","observation_ratio","gate_type","tau"],vals)),"mean_edge_observability_selected_true_edges":a.edge_observability_score.mean(),"mean_edge_observability_suppressed_true_edges":b.edge_observability_score.mean(),"median_edge_observability_selected_true_edges":a.edge_observability_score.median(),"median_edge_observability_suppressed_true_edges":b.edge_observability_score.median(),"correlation_observability_group_snr":_corr(x.edge_observability_score,x.group_snr_louis),"correlation_observability_gate_weight":_corr(x.edge_observability_score,x.gate_weight),"correlation_observability_abs_standardized_error":_corr(x.edge_observability_score,x.mean_abs_standardized_error),"observability_predicts_suppression":bool(np.isfinite(a.edge_observability_score.mean()) and np.isfinite(b.edge_observability_score.mean()) and b.edge_observability_score.mean()<a.edge_observability_score.mean())})
 return pd.DataFrame(rows)

def observability_quartiles(all_coeff):
 x=all_coeff.loc[all_coeff.coefficient_type.eq("offdiag_nonzero")].copy();group_keys=["config_label","M_y","observation_ratio","true_network_id","replicate_id","target","source"]
 obs=x.groupby(group_keys,dropna=False).edge_observability_score.mean().reset_index()
 obs["observability_quartile"]=obs.groupby(["config_label","M_y"])["edge_observability_score"].transform(lambda s:pd.qcut(s.rank(method="first"),4,labels=["Q1_low","Q2","Q3","Q4_high"]))
 x=x.merge(obs[group_keys+["observability_quartile"]],on=group_keys,how="left")
 if "variance_stageB" not in x:x["variance_stageB"]=np.nan
 if "variance_louis" not in x:x["variance_louis"]=np.nan
 x["stageB_louis_variance_ratio"]=x.variance_stageB/np.maximum(x.variance_louis,EPS_VAR)
 return x.groupby(["config_label","M_y","observation_ratio","gate_type","tau","covariance_estimator","observability_quartile"],dropna=False).agg(true_edge_coverage_95=("ci95_contains_true","mean"),mean_posterior_variance=("posterior_variance","mean"),mean_StageB_Louis_variance_ratio=("stageB_louis_variance_ratio","mean"),mean_soft_weight=("gate_weight","mean"),n_coefficients=("posterior_variance","size")).reset_index()

def balance_summary(cal):
 x=cal.loc[cal.gate_type.isin(["soft","hard"])];rows=[]
 for vals,g in x.groupby(["gate_type","tau","M_y","observation_ratio"],dropna=False):
  d={r.coefficient_type:r for _,r in g.iterrows()};diag=d.get("diagonal");active=d.get("offdiag_nonzero");zero=d.get("offdiag_zero")
  def val(r,n):return getattr(r,n) if r is not None else np.nan
  ac,zc=val(active,"empirical_coverage_95"),val(zero,"empirical_coverage_95");az,zz=val(active,"std_standardized_error"),val(zero,"std_standardized_error")
  scores=[np.clip(1-abs(ac-.95)/.95,0,1),np.clip(1-abs(zc-.95)/.95,0,1),np.clip(1-abs(az-1),0,1),np.clip(1-abs(zz-1),0,1)]
  rows.append({"gate_type":vals[0],"tau":vals[1],"M_y":vals[2],"observation_ratio":vals[3],"diag_coverage":val(diag,"empirical_coverage_95"),"offdiag_nonzero_coverage":ac,"offdiag_zero_coverage":zc,"diag_std_z":val(diag,"std_standardized_error"),"offdiag_nonzero_std_z":az,"offdiag_zero_std_z":zz,"active_score":scores[0],"zero_score":scores[1],"active_z_score":scores[2],"zero_z_score":scores[3],"tau_balance_score":.35*scores[0]+.25*scores[1]+.25*scores[2]+.15*scores[3]})
 out=pd.DataFrame(rows);common=out.groupby(["gate_type","tau"]).tau_balance_score.agg(["mean",lambda s:s.max()-s.min()]).reset_index();common.columns=["gate_type","tau","mean_condition_score","condition_score_range"];common["common_tau_score"]=common.mean_condition_score-.25*common.condition_score_range
 return out.merge(common,on=["gate_type","tau"],how="left")

def decisions(balance,gsum,gates,obs,baseline_groups,cal):
 out=balance.copy();group=gsum.pivot_table(index=["gate_type","tau","M_y","observation_ratio"],columns="observability_group_type",values="fraction_D2_below_chi2_95_df2",aggfunc="mean").reset_index().rename(columns={"true_edge_group":"true_edge_group_chi2_95_coverage","false_edge_group":"false_edge_group_chi2_95_coverage"});out=out.merge(group,on=["gate_type","tau","M_y","observation_ratio"],how="left")
 gd=gates.groupby(["gate_type","tau","M_y","observation_ratio"],dropna=False).agg(true_edge_gate_recall=("recall_of_true_edges_by_gate","mean"),false_edge_gate_selection_rate=("fraction_false_edges_selected_by_gate","mean"),gate_precision=("precision_of_gate","mean"),mean_observability_selected_true_edges=("mean_observability_selected_true_edges","mean"),mean_observability_suppressed_true_edges=("mean_observability_suppressed_true_edges","mean")).reset_index();out=out.merge(gd,on=["gate_type","tau","M_y","observation_ratio"],how="left")
 od=obs.groupby(["gate_type","tau","M_y","observation_ratio"],dropna=False).observability_predicts_suppression.mean().reset_index();out=out.merge(od,on=["gate_type","tau","M_y","observation_ratio"],how="left")
 out["recommended_for_My40"]=False;out["recommended_for_My15"]=False;out["recommended_common_tau"]=False
 for gate in out.gate_type.unique():
  q=out[out.gate_type.eq(gate)];best_by={my:q[q.M_y.eq(my)].sort_values("tau_balance_score",ascending=False).iloc[0].tau for my in q.M_y.unique()};best_common=q.groupby("tau").common_tau_score.first().idxmax();out.loc[out.gate_type.eq(gate)&out.tau.eq(best_common),"recommended_common_tau"]=True
  for my,t in best_by.items():out.loc[out.gate_type.eq(gate)&out.M_y.eq(my)&out.tau.eq(t),f"recommended_for_My{int(my)}"]=True
  near=all(q[q.M_y.eq(my)&q.tau.eq(best_common)].tau_balance_score.iloc[0]>=q[q.M_y.eq(my)].tau_balance_score.max()-.05 for my in best_by)
  current=q[q.tau.eq(2.5)].set_index("M_y");candidate=q[q.tau.eq(best_common)].set_index("M_y");improved=all(candidate.loc[my,"offdiag_nonzero_coverage"]>=current.loc[my,"offdiag_nonzero_coverage"] for my in best_by);closer=all(abs(candidate.loc[my,"offdiag_nonzero_std_z"]-1)<=abs(current.loc[my,"offdiag_nonzero_std_z"]-1) for my in best_by)
  full_zero=cal[(cal.covariance_estimator.eq("lrvb_stageB_smoother_feedback_psd_projected"))&(cal.coefficient_type.eq("offdiag_zero"))].groupby("M_y").empirical_coverage_95.mean();zeros=all(candidate.loc[my,"offdiag_zero_coverage"]<=.98 or candidate.loc[my,"offdiag_zero_coverage"]<=full_zero.get(my,np.nan) for my in best_by)
  full_false=baseline_groups[(baseline_groups.covariance_estimator.eq("lrvb_stageB_smoother_feedback_psd_projected"))&(~baseline_groups.true_edge_group.astype(bool))].groupby("M_y").D2_below_chi2_95_df2.mean() if len(baseline_groups) else pd.Series(dtype=float);group_ok=all(candidate.loc[my,"false_edge_group_chi2_95_coverage"]<=full_false.get(my,1.)+.05 for my in best_by)
  success=near and improved and closer and zeros and group_ok
  out.loc[out.gate_type.eq(gate),"common_tau_success"]=success
 note=(f"common_tau_success={success}; " + ("use common lower tau" if success and best_common<2.5 else "retain tau=2.5" if success else "move to observability-aware gating"));out.loc[out.gate_type.eq(gate),"notes"]=note
 return out

def covariance_diagnostics(post):
 x=post.copy();x["stageB_louis_variance_ratio"]=x.variance_stageB/np.maximum(x.variance_louis,EPS_VAR)
 return x.groupby(["gate_type","tau","M_y","observation_ratio","coefficient_type"],dropna=False).agg(mean_gate_weight=("gate_weight","mean"),median_gate_weight=("gate_weight","median"),mean_variance=("posterior_variance","mean"),median_variance=("posterior_variance","median"),mean_StageB_Louis_variance_ratio=("stageB_louis_variance_ratio","mean"),fraction_variance_finite=("posterior_variance",lambda s:np.isfinite(s).mean()),fraction_variance_positive=("posterior_variance",lambda s:(s>0).mean()),n_coefficients=("posterior_variance","size")).reset_index()

def savefig(name):
 plt.tight_layout();plt.savefig(os.path.join(OUTPUT_DIR,"plots",name),dpi=160);plt.close()

def _line_by_my(df,y,name,ylabel,gate="soft"):
 q=df[df.gate_type.eq(gate)]
 for my,g in q.groupby("M_y"):
  z=g.groupby("tau")[y].mean().sort_index();plt.plot(z.index,z.values,marker="o",label=f"M_y={int(my)}")
 plt.xlabel("tau");plt.ylabel(ylabel);plt.legend();savefig(name)

def plots(decision,gate,obs,quart,group_rows):
 os.makedirs(os.path.join(OUTPUT_DIR,"plots"),exist_ok=True)
 _line_by_my(decision,"offdiag_nonzero_coverage","01_active_coverage_vs_tau.png","95% coverage")
 _line_by_my(decision,"offdiag_zero_coverage","02_zero_coverage_vs_tau.png","95% coverage")
 _line_by_my(decision,"offdiag_nonzero_std_z","03_active_std_z_vs_tau.png","std(z)")
 _line_by_my(decision,"offdiag_zero_std_z","04_zero_std_z_vs_tau.png","std(z)")
 _line_by_my(decision,"tau_balance_score","05_tau_balance_score_vs_tau.png","tau balance score")
 for gt,g in decision.groupby("gate_type"):
  z=g.groupby("tau").common_tau_score.first().sort_index();plt.plot(z.index,z.values,marker="o",label=gt)
 plt.xlabel("tau");plt.ylabel("common tau score");plt.legend();savefig("06_common_tau_score_vs_tau.png")
 _line_by_my(decision,"true_edge_group_chi2_95_coverage","07_true_group_coverage_vs_tau.png","chi-square 95% coverage")
 _line_by_my(decision,"false_edge_group_chi2_95_coverage","08_false_group_coverage_vs_tau.png","chi-square 95% coverage")
 _line_by_my(decision,"true_edge_gate_recall","09_gate_true_recall_vs_tau.png","true-edge recall")
 _line_by_my(decision,"false_edge_gate_selection_rate","10_gate_false_selection_vs_tau.png","false-edge selection rate")
 best=float(decision[decision.gate_type.eq("soft")].groupby("tau").common_tau_score.first().idxmax())
 for tau,num in [(2.5,11),(best,12)]:
  q=gate[(gate.gate_type.eq("soft"))&(gate.tau.eq(tau))&gate.true_edge_group]
  arrays=[q.loc[q.selected_by_gate,"edge_observability_score"].dropna(),q.loc[~q.selected_by_gate,"edge_observability_score"].dropna()]
  plt.boxplot(arrays,showmeans=True);plt.xticks([1,2],["selected","suppressed"]);plt.ylabel("edge observability");plt.title(f"soft gate, tau={tau:g}");savefig(f"{num:02d}_observability_selected_suppressed_tau{str(tau).replace('.','p')}.png")
 q=gate[(gate.gate_type.eq("soft"))&(gate.tau.eq(2.5))];status=np.where(~q.true_edge_group,"false",np.where(q.selected_by_gate,"true selected","true suppressed"))
 for s in np.unique(status):
  m=status==s;plt.scatter(q.loc[m,"edge_observability_score"],q.loc[m,"group_snr_louis"],s=18,alpha=.7,label=s)
 plt.xlabel("edge observability");plt.ylabel("Louis group SNR");plt.legend();savefig("13_group_snr_vs_observability.png")
 estimators=["lrvb_stageB_smoother_feedback_psd_projected",label("soft",2.5),label("soft",best)];q=quart[quart.covariance_estimator.isin(estimators)]
 for est,g in q.groupby("covariance_estimator"):
  z=g.groupby("observability_quartile",observed=False).true_edge_coverage_95.mean();plt.plot(range(len(z)),z.values,marker="o",label=est)
 if len(q):plt.xticks(range(4),["Q1 low","Q2","Q3","Q4 high"])
 plt.ylabel("true-edge coefficient coverage");plt.legend(fontsize=7);savefig("14_coverage_by_observability_quartile.png")

def _atomic_json(obj,path):
 tmp=path+".tmp"
 with open(tmp,"w",encoding="utf8") as h:json.dump(obj,h,indent=2,allow_nan=True)
 os.replace(tmp,path)

def main():
 os.makedirs(OUTPUT_DIR,exist_ok=True);report=inspect_inputs();_atomic_json(report,os.path.join(OUTPUT_DIR,"input_schema_report.json"))
 if not report["posthoc_tau_recomputation_possible"]:
  missing=sorted({"posterior_variance","covariance_estimator","posterior_center","true_value","coefficient_type","target","source","lag"}-set(report["inferred_coefficient_level_columns"]))
  failure=pd.DataFrame([{"status":"failed_schema_validation","missing_columns":";".join(missing),"notes":"Exact post-hoc recomputation is impossible. Future Stage-B runs must save coefficient-level Louis and Stage-B variances plus group_snr_louis; save full 2x2 blocks for exact group calibration."}]);atomic_csv(failure,os.path.join(OUTPUT_DIR,"posthoc_tau_decision_summary.csv"));raise RuntimeError(failure.iloc[0].notes)
 print("34R Stage 2B: loading saved Stage-2 outputs (no model fitting)")
 post,baseline,_=build();all_coeff=pd.concat([baseline,post],ignore_index=True,sort=False);cal=calibration(all_coeff);groups=group_results(post);gsum=group_summary(groups);gates,gate_rows=gate_diagnostics(post);obs=observability_diagnostics(post,gate_rows);quart=observability_quartiles(all_coeff);balance=balance_summary(cal);base_groups=load("group_calibration_results.csv",False);decision=decisions(balance,gsum,gates,obs,base_groups,cal);cov=covariance_diagnostics(post)
 outputs={"posthoc_tau_coefficient_results.csv":all_coeff,"posthoc_tau_calibration_summary.csv":cal,"posthoc_tau_group_results.csv":groups,"posthoc_tau_group_calibration_summary.csv":gsum,"posthoc_tau_gate_diagnostics.csv":gates,"posthoc_tau_observability_diagnostics.csv":obs,"posthoc_tau_observability_quartile_summary.csv":quart,"posthoc_tau_balance_summary.csv":balance,"posthoc_tau_decision_summary.csv":decision,"posthoc_covariance_diagnostics.csv":cov}
 for name,frame in outputs.items():atomic_csv(frame,os.path.join(OUTPUT_DIR,name))
 plots(decision,gate_rows,obs,quart,groups)
 show=["gate_type","tau","M_y","offdiag_nonzero_coverage","offdiag_zero_coverage","offdiag_nonzero_std_z","offdiag_zero_std_z","true_edge_group_chi2_95_coverage","false_edge_group_chi2_95_coverage","tau_balance_score","true_edge_gate_recall","false_edge_gate_selection_rate"]
 print(decision[show].sort_values(["M_y","gate_type","tau"]).to_string(index=False,float_format=lambda x:f"{x:.3f}"))
 soft=decision[decision.gate_type.eq("soft")];best={int(my):float(g.sort_values("tau_balance_score",ascending=False).iloc[0].tau) for my,g in soft.groupby("M_y")};common=float(soft.groupby("tau").common_tau_score.first().idxmax());stable=all(soft[soft.M_y.eq(my)&soft.tau.eq(common)].tau_balance_score.iloc[0]>=soft[soft.M_y.eq(my)].tau_balance_score.max()-.05 for my in best);o25=obs[(obs.gate_type.eq("soft"))&obs.tau.eq(2.5)].observability_predicts_suppression.mean()>.5;note=soft[soft.tau.eq(common)].notes.iloc[0];action="common lower tau" if "use common lower" in note else "observability-aware gating"
 print(f"\nBest tau M_y=40: {best.get(40,np.nan):g}\nBest tau M_y=15: {best.get(15,np.nan):g}\nBest common tau: {common:g}\nCommon tau stable: {stable}\nLow observability predicts tau=2.5 suppression: {o25}\nRecommendation: {action}\nCompleted: {OUTPUT_DIR}")

if __name__=="__main__":main()
