"""Experiment 34R Stage 1: rectangular-C observability break-point screen.

No Stage-B LRVB is performed. Independent fits are process-parallel and all
scientific work units are deterministic and atomically checkpointed.
"""
import os
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[_k]=os.environ.get("EXPERIMENT_34R_INNER_THREADS","1")
import json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import experiments.experiment_34j_louis_missing_information_A_uncertainty as j
import experiments.experiment_34i_static_spectral_varx_gc as spi
import experiments.experiment_34qA_sparsity_aware_stageB_free_B as qa
import experiments.experiment_34qB_sparsity_aware_stageB_estBQ_squareC as qb
import experiments.experiment_34qC_practical_model_choice_squareC as qc
from experiments.experiment_33c_source_count_scaling_lgc_metric import atomic_csv,correlations,grouped_stats
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.louis_missing_information import estimate_missing_information_numba,prior_precision_for_row,sample_companion_trajectories_ffbs,stabilized_louis_covariance
from src.varx.varx_generator import generate_colored_input

RESULTS_DIR=os.environ.get("EXPERIMENT_34R_RESULTS_DIR","results/experiment_34r_stage1");SMOKE=os.environ.get("EXPERIMENT_34R_SMOKE","0")=="1";WORKERS=int(os.environ.get("EXPERIMENT_34R_WORKERS","2"));RUN_M30=os.environ.get("EXPERIMENT_34R_RUN_M30","0")=="1"
MY_GRID=[40,30,25,20,18,16,15,14,12,10,8];FAMILIES=["gaussian_isotropic","leadfield_like_correlated"];N_FFBS=50;BASE_SEED=5880000;METHOD="hybrid_vb_ard_free_B_estimate_Q_shrink_scalar_rho_0p25_known_rectangular_C_known_R"
RAW={k:f"_{k}_checkpoint.csv" for k in ("run","C","latent","source","edge","A","B","Q","network","spectral","spectra","deviance","uncertainty","selected","runtime")};RAW["run"]="run_summary_partial.csv"

class Progress:
 def __init__(self,n):self.n,self.i=n,0;self.start=time.perf_counter()
 def update(self,label):self.i+=1;f=self.i/max(self.n,1);eta=(time.perf_counter()-self.start)/self.i*(self.n-self.i);k=round(30*f);print(f"\rExperiment 34R-1 [{'#'*k}{'-'*(30-k)}] {self.i}/{self.n} {100*f:5.1f}% ETA {eta/60:6.1f}m {label[:42]}",end="\n" if self.i==self.n else "",flush=True)

def make_C(my,mx,family,seed):
 rng=np.random.default_rng(seed+151)
 if family=="gaussian_isotropic":C=rng.normal(0,1/np.sqrt(my),(my,mx));C/=np.median(np.linalg.norm(C,axis=0))
 else:
  source=np.linspace(0,1,mx);sensor=np.linspace(-.05,1.05,my);sigma=.14;amp=np.exp(rng.normal(0,.22,mx));C=np.exp(-((sensor[:,None]-source[None,:])**2)/(2*sigma**2))*amp;C+=.04*rng.normal(size=C.shape);C/=np.sqrt(np.mean(np.sum(C*C,axis=1)))
 if np.linalg.matrix_rank(C)<min(my,mx):raise np.linalg.LinAlgError("C is rank deficient")
 return C

def c_metrics(C):
 s=np.linalg.svd(C,compute_uv=False);p=s*s/np.sum(s*s);er=np.exp(-np.sum(p*np.log(np.maximum(p,1e-300))));cn=np.linalg.norm(C,axis=0);rn=np.linalg.norm(C,axis=1)
 def offcorr(X):
  if len(X)<2:return np.nan,np.nan
  z=np.corrcoef(X);v=np.abs(z[~np.eye(len(z),dtype=bool)]);return np.nanmean(v),np.nanmax(v)
 cc=offcorr(C.T);rc=offcorr(C)
 return {"rank_C":np.linalg.matrix_rank(C),"effective_rank_C":er,"condition_number_C":s[0]/s[-1],"min_singular_value":s[-1],"median_singular_value":np.median(s),"max_singular_value":s[0],"singular_value_decay_ratio":s[-1]/s[0],"mean_column_norm":cn.mean(),"min_column_norm":cn.min(),"max_column_norm":cn.max(),"coefficient_of_variation_column_norms":cn.std()/cn.mean(),"mean_abs_column_correlation":cc[0],"max_abs_column_correlation":cc[1],"mean_abs_row_correlation":rc[0],"max_abs_row_correlation":rc[1],"observability_index":er/C.shape[1]}

def simulate(mx,my,T,family,network,rep):
 nseed=BASE_SEED+mx*100000+network*1000;seed=nseed+my*10000+FAMILIES.index(family)*1000000+T+rep;A,B,_=j.fixed_network(mx,nseed);C=make_C(my,mx,family,seed);Q=.5*np.eye(mx);R=.6*np.eye(my);u=generate_colored_input(T+j.BURN_IN,.95,1.,seed+200);z=generate_ssm_varx_p_data(A,B,u,Q,R,C=C,D=None,burn_in=j.BURN_IN,random_seed=seed+300,return_augmented=True);return {"A":A,"B":B,"C":C,"Q":Q,"R":R,"mask":np.any(np.asarray(A)!=0,axis=0)&~np.eye(mx,dtype=bool),"x":z["x"],"y":z["y"],"u":z["u"]},seed

def ridge_proxy(data):
 C=data["C"];lam=1e-3*np.trace(C@C.T)/C.shape[0];return data["y"]@np.linalg.solve(C@C.T+lam*np.eye(C.shape[0]),C).copy()

def louis(model,data,seed):
 samples=sample_companion_trajectories_ffbs(model.smooth_result_,model.F,N_FFBS,seed);beta=model._pack_A();M=model.n_states;pri=np.asarray([prior_precision_for_row(M,2,r,model.alpha_mean_,model.diagonal_prior_precision) for r in range(M)]);missing=estimate_missing_information_numba(samples,data["u"],beta,model.B_matrices,model.Q,2,3,pri);out=[]
 for current,item in zip(model.A_row_covariances_,missing):out.append(stabilized_louis_covariance(np.linalg.pinv(current),item["missing_information"],.7,.9)[0])
 return out

def select_uncertainty(model,data,L,seed):
 M=model.n_states;rng=np.random.default_rng(seed);groups=[];seen=set()
 def add(t,s,kind):
  if (t,s) not in seen:seen.add((t,s));groups.append((t,s,kind))
 for t,s in zip(*np.where(data["mask"])):add(t,s,"true_offdiag")
 for t in rng.choice(M,min(10,M),False):add(t,t,"diagonal")
 false=[(t,s) for t in range(M) for s in range(M) if t!=s and not data["mask"][t,s]];rng.shuffle(false)
 for t,s in false[:10]:add(t,s,"matched_false")
 scores=[]
 for t,s in false:
  cols=[s,M+s];m=np.asarray(model.A_mean_matrices_)[:,t,s];scores.append((np.sqrt(max(m@np.linalg.pinv(L[t][np.ix_(cols,cols)])@m,0)),t,s))
 for _,t,s in sorted(scores,reverse=True)[:5]:add(t,s,"high_snr_false")
 for _,t,s in sorted(scores)[:5]:add(t,s,"low_snr_false")
 rows=[]
 for t,s,k in groups:
  for lag in range(2):rows.append({"coefficient_global_index":qa.global_index(t,lag,s,M,2),"target":t,"lag":lag+1,"source":s,"coefficient_type":"diagonal" if t==s else ("offdiag_nonzero" if data["mask"][t,s] else "offdiag_zero"),"selected_direction_type":k})
 return pd.DataFrame(rows).head(80)

def uncertainty(model,data,L,meta,seed):
 sel=select_uncertainty(model,data,L,seed);idx=sel.coefficient_global_index.astype(int);blocks={"ordinary_vb":qa.o.global_covariance_block(model.A_row_covariances_)[np.ix_(idx,idx)],"stabilized_louis_eta_0p70_tau_0p90":qa.o.global_covariance_block(L)[np.ix_(idx,idx)]};rows=[]
 for name,cov in blocks.items():rows.append(qa.coefficient_rows(model,data,sel,{"covariance_estimator":name,"covariance":cov},meta))
 return sel,pd.concat(rows,ignore_index=True)

def work(task):
 stage,mx,my,T,fam,net,rep=task;ratio=my/mx;meta={"stage":stage,"M_x":mx,"M_y":my,"observation_ratio":ratio,"C_family":fam,"C_MODE":fam,"T":T,"true_network_id":net,"replicate_id":rep,"method":METHOD,"run_id":f"{stage}_{mx}_{my}_{fam}_{T}_{net}_{rep}"};out={k:pd.DataFrame() for k in RAW};start=time.perf_counter()
 try:
  data,seed=simulate(mx,my,T,fam,net,rep);old=qb.VB_MAX_ITER;qb.VB_MAX_ITER=10 if stage=="stage0" else 75;model=qb.fit_estBQ(data,seed+500);qb.VB_MAX_ITER=old;L=louis(model,data,seed+700);proxy=ridge_proxy(data);cm=c_metrics(data["C"]);A=qb.a_recovery(model,data,meta);B=qa.B_recovery(model,data,meta);Q=qb.q_recovery(model,data,meta);lat={**meta,**qb.latent_metrics(model,{**data,"y":proxy@data["C"].T})};lat.update({"pseudo_inverse_state_MSE":np.mean((proxy-data["x"])**2),"pseudo_inverse_state_correlation":np.nanmean([correlations(proxy[:,i],data["x"][:,i]) for i in range(mx)])});netf=qa.full_network(model,data,meta,L);oldM=spi.M;spi.M=mx;spectra,bands,_,_,_=spi.compute_spectral_tables(model.A_mean_matrices_,model.B_matrices,model.Q,np.linspace(0,np.pi,128),data,meta,"estimated_AQ");spec,_=spi.metric_rows(bands);spi.M=oldM
  for key,value in meta.items():spec[key]=value
  dev=qc.deviance(model,data,meta);sel,unc=uncertainty(model,data,L,meta,seed+900);cn=np.linalg.norm(data["C"],axis=0);source=[]
  for s in range(mx):source.append({**meta,"source_index":s,"column_observability_score_j":cn[s],"source_state_recovery_correlation_j":correlations(model.smoothed_state_mean_[:,s],data["x"][:,s]),"source_state_MSE_j":np.mean((model.smoothed_state_mean_[:,s]-data["x"][:,s])**2),"diagonal_A_error_j":np.linalg.norm(np.asarray(model.A_mean_matrices_)[:,s,s]-np.asarray(data["A"])[:,s,s])})
  groupnorm=np.sqrt(np.sum(np.asarray(model.A_mean_matrices_)**2,axis=0));truth_vec=[];score_vec=[];pairs=[]
  for t in range(mx):
   for s in range(mx):
    if t!=s:pairs.append((t,s));truth_vec.append(bool(data["mask"][t,s]));score_vec.append(groupnorm[t,s])
  _,curve=j.safe_curve(truth_vec,score_vec);eligible=[x for x in curve if np.isfinite(x[1]["fpr"]) and x[1]["fpr"]<=.05];threshold=max(eligible,key=lambda x:x[1]["tpr"])[0] if eligible else np.inf;ranks=pd.Series(score_vec).rank(ascending=False,method="average").to_numpy();low=bands.loc[(bands.score_type=="transfer_spectral_gc_diagQ")&(bands.band_name=="low_band")].set_index(["target","source"]).integrated_score
  edge=[]
  for t in range(mx):
   for s in range(mx):
    if t!=s:
     idx=pairs.index((t,s));detected=score_vec[idx]>=threshold;edge.append({**meta,"target":t,"source":s,"target_observability_score_i":cn[t],"source_observability_score_j":cn[s],"edge_observability_score_ij":cn[t]*cn[s],"true_edge":data["mask"][t,s],"estimated_group_norm":score_vec[idx],"spectral_low_band_score":low.get((t,s),np.nan),"deviance_score_smoothed":np.nan,"detection_rank_model_A":ranks[idx],"detection_rank_spectral":np.nan,"detection_rank_deviance":np.nan,"missed_true_edge_indicator":bool(data["mask"][t,s] and not detected),"false_positive_indicator":bool(not data["mask"][t,s] and detected)})
  practical=max(model.A_change_history_[-1],model.B_change_history_[-1],model.Q_change_history_[-1],model.alpha_change_history_[-1])<1e-4;run={**meta,**cm,**{k:v for x in (A,B,Q,lat) for k,v in x.items() if k not in meta},"practically_converged":practical,"failure_rate":0.,"runtime_seconds":time.perf_counter()-start,"run_status":"success"};out.update(run=pd.DataFrame([run]),C=pd.DataFrame([{**meta,**cm}]),latent=pd.DataFrame([lat]),source=pd.DataFrame(source),edge=pd.DataFrame(edge),A=pd.DataFrame([A]),B=pd.DataFrame([B]),Q=pd.DataFrame([Q]),network=netf,spectral=spec,spectra=bands,deviance=dev,uncertainty=unc,selected=pd.DataFrame([{**meta,**r} for r in sel.to_dict("records")]),runtime=pd.DataFrame([{**meta,"runtime_seconds":run["runtime_seconds"],"run_status":"success"}]))
 except Exception as e:
  row={**meta,"run_status":"failed","failure_rate":1.,"runtime_seconds":time.perf_counter()-start,"error_type":type(e).__name__,"error_message":str(e),"traceback":traceback.format_exc()};out["run"]=pd.DataFrame([row]);out["runtime"]=pd.DataFrame([row])
 return out

def combine(t):return {k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in t.items()}
def metric_best(frame,score_type=None):
 if not len(frame):return {}
 g=frame if score_type is None else frame.loc[frame.score_type==score_type]
 if not len(g):return {}
 return g.loc[g.AUPRC.idxmax()].to_dict()
def uncertainty_summary(f):
 if not len(f):return pd.DataFrame()
 return f.groupby(["stage","M_x","M_y","observation_ratio","C_family","T","covariance_estimator","coefficient_type"],dropna=False).agg(empirical_coverage_95=("ci95_contains_true","mean"),std_standardized_error=("standardized_error",lambda x:x.std(ddof=0)),mean_interval_width=("interval_width_95","mean"),median_interval_width=("interval_width_95","median"),n_coefficients=("posterior_sd","size")).reset_index()
def breakpoint(fr):
 rows=[];keys=["M_x","M_y","observation_ratio","C_family","T"]
 for values,runs in fr["run"].groupby(keys,dropna=False):
  mask=lambda f:np.logical_and.reduce([f[k]==v for k,v in zip(keys,values)]);net=fr["network"].loc[mask(fr["network"])] if len(fr["network"]) else pd.DataFrame();sp=fr["spectral"].loc[mask(fr["spectral"])] if len(fr["spectral"]) else pd.DataFrame();dv=fr["deviance"].loc[mask(fr["deviance"])] if len(fr["deviance"]) else pd.DataFrame();u=uncertainty_summary(fr["uncertainty"]);u=u.loc[mask(u)] if len(u) else u;mean=runs.mean(numeric_only=True);getnet=lambda s:net.loc[net.score_type==s,"AUPRC"].mean() if len(net) else np.nan;best=metric_best(sp,"transfer_spectral_gc_diagQ");dev=metric_best(dv,"raw_deviance");deb=metric_best(dv,"debiased_deviance");uc=lambda est,typ:u.loc[(u.covariance_estimator==est)&(u.coefficient_type==typ),"empirical_coverage_95"].mean() if len(u) else np.nan;state=mean.get("smoothed_state_correlation",np.nan);sa=best.get("AUPRC",np.nan);aa=getnet("model_A_group_norm");pc=mean.get("practically_converged",0);fail=mean.get("failure_rate",0)
  if fail>.2 or pc<.8:reg="numerically_unstable"
  elif state<.8 or sa<.5 or aa<.35:reg="broken"
  elif state>=.9 and sa>=.7 and aa>=.55:reg="safe"
  else:reg="transition"
  rows.append({**dict(zip(keys,values)),"mean_smoothed_state_correlation":state,"mean_smoothed_state_MSE":mean.get("smoothed_state_MSE",np.nan),"mean_A_support_AUPRC_model_A_group_norm":aa,"mean_A_support_AUPRC_stabilized_louis_group_snr":getnet("stabilized_louis_group_snr"),"mean_A_support_TPR_at_FPR_0p05":net.loc[net.score_type=="model_A_group_norm","TPR_at_FPR_0p05"].mean() if len(net) else np.nan,"spectral_best_band":best.get("band_name",np.nan),"mean_spectral_best_band_AUPRC":sa,"mean_spectral_best_band_ROC_AUC":best.get("ROC_AUC",np.nan),"mean_spectral_best_band_TPR_at_FPR_0p05":best.get("TPR_at_FPR_0p05",np.nan),"deviance_best_signal_type":dev.get("signal_type",np.nan),"mean_deviance_best_AUPRC":dev.get("AUPRC",np.nan),"debiased_deviance_best_signal_type":deb.get("signal_type",np.nan),"mean_debiased_deviance_best_AUPRC":deb.get("AUPRC",np.nan),"mean_B_relative_error":mean.get("B_relative_frobenius_error",np.nan),"mean_B_correlation_hat_true":mean.get("B_correlation_hat_true",np.nan),"mean_Q_relative_error":mean.get("Q_relative_frobenius_error",np.nan),"mean_Q_trace_ratio_hat_to_true":mean.get("Q_trace_ratio_hat_to_true",np.nan),"ordinary_vb_offdiag_nonzero_coverage":uc("ordinary_vb","offdiag_nonzero"),"stabilized_louis_offdiag_nonzero_coverage":uc("stabilized_louis_eta_0p70_tau_0p90","offdiag_nonzero"),"stabilized_louis_offdiag_zero_coverage":uc("stabilized_louis_eta_0p70_tau_0p90","offdiag_zero"),"effective_rank_C":mean.get("effective_rank_C",np.nan),"condition_number_C":mean.get("condition_number_C",np.nan),"mean_abs_column_correlation":mean.get("mean_abs_column_correlation",np.nan),"coefficient_of_variation_column_norms":mean.get("coefficient_of_variation_column_norms",np.nan),"observability_index":mean.get("observability_index",np.nan),"practical_convergence_rate":pc,"failure_rate":fail,"runtime_seconds":mean.get("runtime_seconds",np.nan),"regime_label":reg})
 return pd.DataFrame(rows)
def choices(bp):
 if not len(bp):return bp
 maxrun=max(bp.runtime_seconds.max(),1);rows=[]
 for _,r in bp.iterrows():
  vals=np.array([r.mean_spectral_best_band_AUPRC,r.mean_spectral_best_band_TPR_at_FPR_0p05,r.mean_A_support_AUPRC_stabilized_louis_group_snr,r.mean_A_support_AUPRC_model_A_group_norm,r.mean_smoothed_state_correlation,r.mean_B_correlation_hat_true,max(0,1-abs(r.mean_Q_trace_ratio_hat_to_true-1)),r.mean_deviance_best_AUPRC,r.mean_debiased_deviance_best_AUPRC]);w=np.array([.2,.15,.15,.1,.15,.05,.05,.05,.05]);ok=np.isfinite(vals);score=np.dot(w[ok]/w[ok].sum(),vals[ok])-.05*r.runtime_seconds/maxrun;rows.append({**r.to_dict(),"model_choice_score":score})
 out=pd.DataFrame(rows);out["recommended_for_stage2_LRVB"]=False
 for family,g in out.groupby("C_family"):
  for regime in ("safe","transition","broken"):
   q=g.loc[g.regime_label==regime].sort_values("M_y")
   if len(q):out.loc[q.index[-1] if regime=="safe" else q.index[0],"recommended_for_stage2_LRVB"]=True
 out["recommended_for_spectral_pipeline"]=out.mean_spectral_best_band_AUPRC>=.7;out["recommended_for_rectangular_C_pipeline"]=out.regime_label.isin(["safe","transition"]);out["recommended_for_rejection"]=out.regime_label.isin(["broken","numerically_unstable"]);out["reason"]=out.regime_label.map({"safe":"meets all safe thresholds","transition":"usable but below at least one safe threshold","broken":"at least one scientific metric crossed its failure threshold","numerically_unstable":"fit failure or practical convergence threshold violated"});return out
def save(t):
 fr=combine(t)
 for k,name in RAW.items():atomic_csv(fr[k],os.path.join(RESULTS_DIR,name))
 bp=breakpoint(fr);dec=choices(bp);atomic_csv(bp,os.path.join(RESULTS_DIR,"breakpoint_summary_partial.csv"));atomic_csv(dec,os.path.join(RESULTS_DIR,"decision_summary_partial.csv"));return fr,bp,dec
def execute(tasks,tables):
 old=combine(tables)["run"];done=set(zip(old.stage,old.M_x.astype(int),old.M_y.astype(int),old.C_family,old["T"].astype(int),old.true_network_id.astype(int),old.replicate_id.astype(int))) if len(old) else set();tasks=[x for x in tasks if x not in done]
 if not tasks:return
 p=Progress(len(tasks))
 with ProcessPoolExecutor(max_workers=min(WORKERS,len(tasks))) as pool:
  for future in as_completed([pool.submit(work,x) for x in tasks]):
   r=future.result()
   for k,v in r.items():
    if len(v):tables[k].append(v)
   save(tables);row=r["run"].iloc[-1];p.update(f"My={row.M_y} {row.C_family} {row.run_status}")
def plots(bp,spectral,edge):
 path=os.path.join(RESULTS_DIR,"plots");os.makedirs(path,exist_ok=True)
 metrics=[("mean_smoothed_state_correlation","state_correlation"),("mean_smoothed_state_MSE","state_MSE"),("mean_A_support_AUPRC_model_A_group_norm","A_AUPRC"),("mean_A_support_TPR_at_FPR_0p05","A_TPR05"),("mean_spectral_best_band_AUPRC","spectral_AUPRC"),("mean_deviance_best_AUPRC","deviance_AUPRC"),("mean_B_relative_error","B_error"),("mean_Q_relative_error","Q_error"),("condition_number_C","C_condition"),("mean_abs_column_correlation","column_correlation"),("model_choice_score","model_choice"),("runtime_seconds","runtime")]
 choice=choices(bp)
 for col,name in metrics:
  f=choice if col=="model_choice_score" else bp;fig,ax=plt.subplots()
  for family,g in f.groupby("C_family"):ax.plot(g.observation_ratio,g[col],marker="o",label=family)
  ax.set(xlabel="M_y / M_x",ylabel=col);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(os.path.join(path,name+".png"));plt.close(fig)
 if len(spectral):
  g=spectral.loc[spectral.score_type=="transfer_spectral_gc_diagQ"]
  fig,ax=plt.subplots()
  for (family,band),q in g.groupby(["C_family","band_name"]):ax.plot(q.observation_ratio,q.AUPRC,marker="o",label=f"{family}:{band}")
  ax.set(xlabel="M_y / M_x",ylabel="spectral AUPRC");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(os.path.join(path,"spectral_bands.png"));plt.close(fig)
 if len(edge):
  true=edge.loc[edge.true_edge.astype(bool)];fig,ax=plt.subplots();ax.boxplot([true.loc[~true.missed_true_edge_indicator.astype(bool),"edge_observability_score_ij"],true.loc[true.missed_true_edge_indicator.astype(bool),"edge_observability_score_ij"]],labels=["detected","missed"]);ax.set_ylabel("edge observability");fig.tight_layout();fig.savefig(os.path.join(path,"edge_observability_detected_missed.png"));plt.close(fig)
  q=true.copy();q["quartile"]=pd.qcut(q.edge_observability_score_ij,4,duplicates="drop");rate=q.groupby("quartile",observed=True).missed_true_edge_indicator.mean();fig,ax=plt.subplots();rate.plot.bar(ax=ax);ax.set_ylabel("missed true-edge rate");fig.tight_layout();fig.savefig(os.path.join(path,"missed_rate_observability_quartile.png"));plt.close(fig)
def main():
 os.makedirs(RESULTS_DIR,exist_ok=True);config={"experiment":"34R_stage1_rectangular_C_breakpoint","M_x":20,"M_Y_GRID":MY_GRID,"C_FAMILY_LIST":FAMILIES,"T":1000,"N_TRUE_NETWORKS":2,"N_REPLICATES_PER_NETWORK":2,"N_FFBS_SAMPLES":50,"N_FREQS":128,"N_WORKERS":WORKERS,"smoke_test":SMOKE,"run_optional_M30":RUN_M30,"StageB":False};json.dump(config,open(os.path.join(RESULTS_DIR,"experiment_config.json"),"w"),indent=2);tables={k:[] for k in RAW}
 for k,name in RAW.items():
  path=os.path.join(RESULTS_DIR,name)
  if os.path.exists(path):
   try:f=pd.read_csv(path)
   except pd.errors.EmptyDataError:f=pd.DataFrame()
   if len(f):tables[k].append(f)
 if SMOKE:execute([("stage0",10,8,300,"gaussian_isotropic",0,0)],tables)
 else:
  execute([("stage1A",20,my,1000,fam,net,rep) for my in MY_GRID for fam in FAMILIES for net in range(2) for rep in range(2)],tables);_,bp,_=save(tables);base=bp.loc[(bp.M_x==20)&(bp.T==1000)];confirm=[]
  for regime in ("safe","transition","broken"):
   q=base.loc[base.regime_label==regime].sort_values("M_y")
   if len(q):z=q.iloc[-1] if regime=="safe" else q.iloc[0];confirm.append(("stage1B",20,int(z.M_y),2000,z.C_family,0,0))
  execute(list(dict.fromkeys(confirm)),tables)
  if RUN_M30:execute([("stage1C",30,my,1000,fam,0,0) for my in [60,45,30,24,18,15] for fam in FAMILIES],tables)
 fr,bp,dec=save(tables);unc=uncertainty_summary(fr["uncertainty"]);keys=["stage","M_x","M_y","observation_ratio","C_family","T"]
 outputs={"run_summary.csv":fr["run"],"C_diagnostics.csv":fr["C"],"latent_state_recovery_summary.csv":fr["latent"],"source_observability_summary.csv":fr["source"],"edge_observability_summary.csv":fr["edge"],"A_recovery_summary.csv":fr["A"],"B_recovery_summary.csv":fr["B"],"Q_recovery_summary.csv":fr["Q"],"network_recovery_summary.csv":fr["network"],"spectral_gc_summary.csv":fr["spectra"],"spectral_band_recovery_summary.csv":fr["spectral"],"deviance_diagnostics.csv":fr["deviance"],"cheap_uncertainty_calibration_summary.csv":unc,"selected_coefficients_for_uncertainty.csv":fr["selected"],"breakpoint_summary.csv":bp,"model_choice_summary.csv":dec,"runtime_summary.csv":fr["runtime"],"decision_summary.csv":dec}
 for name,f in outputs.items():atomic_csv(f,os.path.join(RESULTS_DIR,name))
 if not SMOKE:plots(bp,fr["spectral"],fr["edge"])
if __name__=="__main__":main()
