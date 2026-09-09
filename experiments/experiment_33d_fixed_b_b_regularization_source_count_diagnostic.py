import os

BLAS_THREADS_PER_WORKER = int(os.environ.get("EXPERIMENT_33D_BLAS_THREADS", "1"))
os.environ["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["OMP_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(BLAS_THREADS_PER_WORKER)

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from experiments.experiment_33c_source_count_scaling_lgc_metric import (
    FPR_LEVELS, atomic_csv, correlations, curve_metrics, decisions,
    grouped_stats, make_network, relative, spearman,
)
from src.ssm.em_varx_p_known_c_group_lasso_b_controls import (
    EMVARXPSSMKnownCGroupLassoPosteriorMomentBBasis,
    EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge,
    EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB,
)
from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
)
from src.ssm.kalman_varx_p import extract_current_latent_state, kalman_smooth_varx_p_companion
from src.ssm.ssm_varx_p_simulator import generate_ssm_varx_p_data
from src.stats.lasso_gc_statistic import NUMBA_AVAILABLE, compute_lasso_gc_eq7_network
from src.stats.observed_likelihood_deviance import final_observed_log_likelihood
from src.stats.scalable_debiased_varx_network import compute_all_pair_debiased_varx_network
from src.varx.varx_generator import generate_colored_input


REDUCED_FOCUSED_RUN = True
INCLUDE_OPTIONAL_M10 = False
OPTIONAL_SOURCE_COUNT_VALUES = [10]

if REDUCED_FOCUSED_RUN:
    SOURCE_COUNT_VALUES = [20]
    if INCLUDE_OPTIONAL_M10:
        SOURCE_COUNT_VALUES += OPTIONAL_SOURCE_COUNT_VALUES
    T_VALUES = [2000]
    C_MODE_LIST = ["identity"]
    N_OUTER_RUNS = 10
    B_MODEL_VARIANTS = [
        "free_B", "fixed_B_true", "ridge_B_medium",
        "free_B_long", "basis_B_long",
    ]
    LAMBDA_GRIDS = {
        20: [0.003, 0.03],
        10: [0.003, 0.01],
    }
else:
    SOURCE_COUNT_VALUES = [10, 20]
    T_VALUES = [1000, 2000]
    C_MODE_LIST = ["identity", "mild_mixing"]
    N_OUTER_RUNS = 20
    B_MODEL_VARIANTS = [
        "free_B", "fixed_B_true", "ridge_B_low", "ridge_B_medium",
        "ridge_B_high", "basis_B_short", "free_B_long", "basis_B_long",
    ]
    LAMBDA_GRIDS = {
        10: [0.0, 0.003, 0.01, 0.03],
        20: [0.003, 0.01, 0.03, 0.10],
    }
# Runtime-controlled grid permitted by the experiment specification. The two
# values retain the OLS reference and a sparse LASSO operating point.
LGC_LAMBDA_GRID = [0.0, 0.003]
N_INPUTS, na, BURN_IN = 1, 2, 300
LINK_DENSITY, ALPHA, RIDGE_LAMBDA_DEBIAS = 0.05, 0.05, 1.0
MAX_ITER, TOL, BASE_SEED = 100, 1e-6, 4000000
CUSTOM_THRESHOLDS = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0)
RESULTS_DIR = os.environ.get("EXPERIMENT_33D_RESULTS_DIR", "results/experiment_33d")
N_WORKERS = int(os.environ.get("EXPERIMENT_33D_WORKERS", str(min(4, os.cpu_count() or 1))))


VARIANT_CONFIG = {
    "free_B": (3, 3, 0, 1.0, False),
    "fixed_B_true": (3, 3, 0, 1.0, True),
    "ridge_B_low": (3, 3, 0, 1.0, False),
    "ridge_B_medium": (3, 3, 0, 10.0, False),
    "ridge_B_high": (3, 3, 0, 100.0, False),
    "basis_B_short": (3, 3, 2, 1.0, False),
    "free_B_long": (10, 10, 0, 1.0, False),
    "basis_B_long": (10, 10, 4, 1.0, False),
}


def gaussian_basis(nb_model, n_basis):
    lags = np.arange(nb_model, dtype=float)
    centers = np.linspace(0, nb_model - 1, n_basis)
    width = max(1.0, (nb_model - 1) / max(n_basis - 1, 1))
    basis = np.exp(-0.5 * ((lags[:, None] - centers[None, :]) / width) ** 2)
    norms = np.linalg.norm(basis, axis=0)
    return basis / np.where(norms > 0, norms, 1.0)


def true_B(n, nb_true, seed):
    rng = np.random.default_rng(seed + 100)
    if nb_true == 3:
        return [rng.normal(0, scale, (n, 1)) for scale in (0.45, 0.25, 0.15)], "independent_short_scales"
    amplitudes = rng.normal(0, 0.45, (n, 1))
    tau = rng.uniform(2.5, 4.5, (n, 1))
    return [amplitudes * np.exp(-lag / tau) for lag in range(nb_true)], "source_specific_exponential_decay"


def simulate(n, T, c_mode, nb_true, seed):
    A, mask = make_network(n, seed)
    B, profile = true_B(n, nb_true, seed)
    C = np.eye(n)
    if c_mode == "mild_mixing":
        rng = np.random.default_rng(seed + 150)
        noise = rng.normal(0, 0.05 / np.sqrt(n), (n, n)); np.fill_diagonal(noise, 0.0); C += noise
    u = generate_colored_input(T + BURN_IN, 0.95, 1.0, seed + 200)
    Q, R = 0.50 * np.eye(n), 0.60 * np.eye(n)
    result = generate_ssm_varx_p_data(
        A_matrices=A, B_matrices=B, u=u, Q=Q, R=R, C=C, D=None,
        burn_in=BURN_IN, random_seed=seed + 300, return_augmented=True,
    )
    return dict(x=result["x"], y=result["y"], u=result["u"], A=A, B=B,
                C=C, Q=Q, R=R, mask=mask, B_profile_type=profile)


def model_kwargs(data, nb_model, lambda_fraction, ridge_multiplier, seed):
    return dict(
        na=na, nb=nb_model, C=data["C"], D=None, max_iter=MAX_ITER, tol=TOL,
        ridge_m_step=1e-4, covariance_floor=1e-6, R_init=data["R"], Q_init=data["Q"],
        estimate_Q=False, estimate_R=False, R_floor=.30, zero_constraints=[],
        initial_parameters=None, jitter_scale=0.0, random_seed=seed, verbose=False,
        alpha_Q=0.0, alpha_R=0.0, shrinkage_target_Q="spherical", shrinkage_target_R="spherical",
        lambda_A_group_fraction=lambda_fraction, ridge_A_offdiag=1e-4,
        ridge_A_diag=1e-4, ridge_B=1e-4 * ridge_multiplier,
        group_solver_max_iter=5000, group_solver_tol=1e-7,
        stabilize_A=True, target_radius=.98,
    )


def fit_variant(data, variant, lambda_fraction, seed):
    nb_true, nb_model, n_basis, ridge_multiplier, fixed = VARIANT_CONFIG[variant]
    kwargs = model_kwargs(data, nb_model, lambda_fraction, ridge_multiplier, seed)
    if fixed:
        model = EMVARXPSSMKnownCGroupLassoPosteriorMomentFixedB(
            fixed_B_matrices=data["B"], **kwargs
        )
    elif n_basis:
        basis = gaussian_basis(nb_model, n_basis)
        model = EMVARXPSSMKnownCGroupLassoPosteriorMomentBBasis(
            B_basis_matrix=basis, **kwargs
        )
    elif variant.startswith("ridge_B"):
        kwargs.pop("ridge_B")
        model = EMVARXPSSMKnownCGroupLassoPosteriorMomentBRidge(
            ridge_B_base=1e-4, ridge_B_multiplier=ridge_multiplier, **kwargs
        )
    else:
        model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(**kwargs)
    model.fit(data["y"], data["u"])
    posterior = kalman_smooth_varx_p_companion(
        data["y"], data["u"], model.F, model.G, model.Q_aug, model.R,
        model.C_aug, nb_model, model.D,
    )
    n = data["x"].shape[1]
    filtered = extract_current_latent_state(posterior["filter"]["x_filt"], n)
    smoothed = extract_current_latent_state(posterior["smoother"]["x_smooth"], n)
    return model, filtered, smoothed


def recovery(model, data, meta, filtered, smoothed):
    A, Ah, B, Bh = map(np.asarray, (data["A"], model.A_matrices, data["B"], model.B_matrices))
    n = A.shape[1]; off = np.broadcast_to(~np.eye(n, dtype=bool), A.shape); diag = ~off
    support_rows = []
    for target in range(n):
        for source in range(n):
            if source != target:
                support_rows.append({**meta, "target": target, "source": source,
                    "true_link": data["mask"][target, source],
                    "true_A_group_norm": np.linalg.norm(A[:, target, source]),
                    "estimated_A_group_norm": np.linalg.norm(Ah[:, target, source])})
    support = pd.DataFrame(support_rows)
    support_metrics, _ = curve_metrics(support.true_link, support.estimated_A_group_norm)
    true_mean = support.loc[support.true_link, "estimated_A_group_norm"].mean()
    false_mean = support.loc[~support.true_link, "estimated_A_group_norm"].mean()
    def latent(signal):
        return np.mean((signal-data["x"])**2), np.nanmean([correlations(signal[:, i], data["x"][:, i]) for i in range(n)])
    fmse, fcorr = latent(filtered); smse, scorr = latent(smoothed)
    B_energy_true, B_energy_hat = np.sum(B**2), np.sum(Bh**2)
    row = {**meta,
        "A_relative_frobenius_error": relative(Ah,A), "A_offdiag_relative_frobenius_error": relative(Ah[off],A[off]),
        "A_diagonal_relative_frobenius_error": relative(Ah[diag],A[diag]), "A_mean_absolute_error": np.mean(abs(Ah-A)),
        "A_offdiag_mean_absolute_error": np.mean(abs(Ah[off]-A[off])), "A_diagonal_mean_absolute_error": np.mean(abs(Ah[diag]-A[diag])),
        "A_pearson_correlation": correlations(A.ravel(),Ah.ravel()), "A_offdiag_pearson_correlation": correlations(A[off],Ah[off]),
        "A_offdiag_spearman_correlation": spearman(A[off],Ah[off]),
        "A_group_norm_pearson_correlation": correlations(support.true_A_group_norm,support.estimated_A_group_norm),
        "A_group_norm_spearman_correlation": spearman(support.true_A_group_norm,support.estimated_A_group_norm),
        "mean_estimated_group_norm_true_links": true_mean, "mean_estimated_group_norm_false_links": false_mean,
        "group_norm_true_false_ratio": true_mean/false_mean if false_mean>0 else np.inf,
        "B_relative_frobenius_error": relative(Bh,B), "B_mean_absolute_error": np.mean(abs(Bh-B)),
        "B_pearson_correlation": correlations(B.ravel(),Bh.ravel()), "B_spearman_correlation": spearman(B.ravel(),Bh.ravel()),
        "B_energy_true": B_energy_true, "B_energy_hat": B_energy_hat,
        "B_energy_ratio_hat_to_true": B_energy_hat/B_energy_true,
        "B_basis_coefficient_norm": np.linalg.norm(model.B_basis_coefficients) if hasattr(model,"B_basis_coefficients") and model.B_basis_coefficients is not None else np.nan,
        "B_reconstruction_error": relative(Bh,B),
        "B_basis_reconstruction_error": relative(Bh,B) if meta["B_basis_flag"] else np.nan,
        "B_basis_matrix_normalization": "unit_l2_columns" if meta["B_basis_flag"] else "not_applicable",
        "filtered_signal_mse": fmse, "smoothed_signal_mse": smse,
        "filtered_signal_correlation": fcorr, "smoothed_signal_correlation": scorr,
        "full_log_likelihood": final_observed_log_likelihood(model), "full_spectral_radius": model.spectral_radius(),
        "full_em_iterations": len(model.log_likelihoods), "Q_trace": np.trace(model.Q), "R_trace": np.trace(model.R),
        "estimate_Q": model.estimate_Q, "estimate_R": model.estimate_R, "C_condition_number": np.linalg.cond(data["C"]),
        "offdiag_A_group_count": model.count_nonzero_offdiag_A_groups(), "offdiag_A_coefficient_count": model.count_nonzero_offdiag_A_coefficients(),
        "mean_offdiag_A_group_norm": model.mean_offdiag_A_group_norm(), "median_offdiag_A_group_norm": model.median_offdiag_A_group_norm(),
        "max_offdiag_A_group_norm": model.max_offdiag_A_group_norm(),
        "group_solver_converged_last": model.group_solver_converged_history[-1],
        "group_solver_mean_iterations_last": model.group_solver_iterations_history[-1],
        "group_solver_objective_last": model.group_solver_objective_history[-1],
        "mean_lambda_A_group_effective": model.mean_lambda_A_group_effective,
        "median_lambda_A_group_effective": model.median_lambda_A_group_effective,
        "min_lambda_A_group_effective": model.min_lambda_A_group_effective,
        "max_lambda_A_group_effective": model.max_lambda_A_group_effective,
    }
    for lag in range(len(B)): row[f"B_lag{lag}_relative_frobenius_error"] = relative(Bh[lag],B[lag])
    for key,value in support_metrics.items(): row["A_support_"+key]=value
    return row,support


def readouts(signal, signal_type, data, meta, nb_model):
    network = compute_all_pair_debiased_varx_network(
        signal,data["u"],data["mask"],na,nb_model,RIDGE_LAMBDA_DEBIAS,ALPHA,CUSTOM_THRESHOLDS)
    for key,value in {**meta,"signal_type":signal_type}.items(): network[key]=value
    lgcs=[]
    for lgc_lambda in LGC_LAMBDA_GRID:
        frame=compute_lasso_gc_eq7_network(signal,data["u"],data["mask"],na,nb_model,lgc_lambda)
        for key,value in {**meta,"signal_type":signal_type}.items(): frame[key]=value
        lgcs.append(frame)
    return network,pd.concat(lgcs,ignore_index=True)


def work(args):
    data,variant,lambda_fraction,base,seed=args
    nb_true,nb_model,n_basis,ridge_multiplier,fixed=VARIANT_CONFIG[variant]
    meta={**base,"B_MODEL_VARIANT":variant,"nb_true":nb_true,"nb_model":nb_model,
          "n_B_basis":n_basis if n_basis else np.nan,"lambda_A_group_fraction":lambda_fraction,
          "ridge_B_multiplier":ridge_multiplier,"fixed_B_flag":fixed,"B_basis_flag":bool(n_basis),
          "B_profile_type":data["B_profile_type"]}
    model,filtered,smoothed=fit_variant(data,variant,lambda_fraction,seed)
    parameter,support=recovery(model,data,meta,filtered,smoothed)
    networks=[];lgcs=[]
    for name,signal in (("em_filtered",filtered),("em_smoothed",smoothed)):
        network,lgc=readouts(signal,name,data,meta,nb_model);networks.append(network);lgcs.append(lgc)
    return parameter,support,pd.concat(networks),pd.concat(lgcs)


def baseline_readout_cache(data, base, nb_models):
    """Compute invariant oracle/observed/proxy readouts once per lag order."""
    proxy = data["y"] @ np.linalg.pinv(data["C"]).T
    cache = {}
    for nb_model in sorted(set(nb_models)):
        networks, lgcs = [], []
        for name, signal in (
            ("oracle_latent", data["x"]),
            ("observed_y", data["y"]),
            ("pinv_proxy", proxy),
        ):
            network, lgc = readouts(signal, name, data, base, nb_model)
            networks.append(network); lgcs.append(lgc)
        cache[nb_model] = (pd.concat(networks), pd.concat(lgcs))
    return cache


def attach_variant_metadata(frame, parameter):
    frame = frame.copy()
    for column in (
        "B_MODEL_VARIANT", "nb_true", "nb_model", "n_B_basis",
        "lambda_A_group_fraction", "ridge_B_multiplier", "fixed_B_flag",
        "B_basis_flag", "B_profile_type",
    ):
        frame[column] = parameter[column]
    return frame


def roc_outputs(network,lgc,support):
    frames=[]
    for score in ("raw_deviance","debiased_deviance"):
        part=network.copy();part["score_type"]=score;part["score"]=part[score];part["lgc_lambda"]=np.nan;frames.append(part)
    for score in ("lgc_raw","lgc_clipped"):
        part=lgc.copy();part["score_type"]=score;part["score"]=part[score];frames.append(part)
    part=support.copy();part["signal_type"]="model_A";part["score_type"]="estimated_A_group_norm";part["score"]=part.estimated_A_group_norm;part["lgc_lambda"]=np.nan;frames.append(part)
    scores=pd.concat(frames,ignore_index=True)
    groups=["n_sources","T","C_MODE","B_MODEL_VARIANT","nb_true","nb_model","n_B_basis","lambda_A_group_fraction","signal_type","score_type","lgc_lambda"]
    summaries=[];points=[]
    for keys,group in scores.groupby(groups,dropna=False):
        summary,curve=curve_metrics(group.true_link,group.score);meta=dict(zip(groups,keys));summaries.append({**meta,"n_links":len(group),**summary})
        for threshold,metrics in curve:points.append({**meta,"threshold":threshold,**metrics,"recall":metrics["tpr"],"youden_j":metrics["tpr"]-metrics["fpr"]})
    return pd.DataFrame(summaries),pd.DataFrame(points)


def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    partial_names={
        "network":"network_results_partial.csv",
        "lgc":"lgc_eq7_results_partial.csv",
        "B":"B_recovery_summary_partial.csv",
        "support":"A_support_results_partial.csv",
        # Completion ledger last: its rows certify that all detailed tables
        # for those work units were promoted successfully first.
        "parameter":"parameter_summary_partial.csv",
    }
    frames={key:(pd.read_csv(os.path.join(RESULTS_DIR,name)) if os.path.exists(os.path.join(RESULTS_DIR,name)) else pd.DataFrame()) for key,name in partial_names.items()}
    keycols=["outer_run","n_sources","T","C_MODE","B_MODEL_VARIANT","lambda_A_group_fraction"]
    # Existing full-grid checkpoints are scientifically compatible for rows
    # that fall inside the focused grid. Reuse only that exact intersection.
    for name, frame in frames.items():
        if len(frame):
            allowed = (
                frame["outer_run"].astype(int).lt(N_OUTER_RUNS) &
                frame["n_sources"].astype(int).isin(SOURCE_COUNT_VALUES) &
                frame["T"].astype(int).isin(T_VALUES) &
                frame["C_MODE"].isin(C_MODE_LIST) &
                frame["B_MODEL_VARIANT"].isin(B_MODEL_VARIANTS)
            )
            lambda_allowed = np.zeros(len(frame), dtype=bool)
            for source_count, lambda_grid in LAMBDA_GRIDS.items():
                lambda_allowed |= (
                    frame["n_sources"].astype(int).eq(source_count) &
                    frame["lambda_A_group_fraction"].astype(float).isin(lambda_grid)
                ).to_numpy()
            frames[name] = frame.loc[allowed & lambda_allowed].reset_index(drop=True)
    completed=set(map(tuple,frames["parameter"][keycols].itertuples(index=False,name=None))) if len(frames["parameter"]) else set()
    detail_identity = {
        "parameter": keycols,
        "B": keycols,
        "support": keycols + ["target", "source"],
        "network": keycols + ["signal_type", "target", "source"],
        "lgc": keycols + ["signal_type", "lgc_lambda", "target", "source"],
    }
    for name, identity in detail_identity.items():
        if len(frames[name]):
            frames[name] = frames[name].drop_duplicates(identity, keep="last")
            keys = frames[name][keycols].apply(tuple, axis=1)
            frames[name] = frames[name].loc[keys.isin(completed)].reset_index(drop=True)
    config={"REDUCED_FOCUSED_RUN":REDUCED_FOCUSED_RUN,"INCLUDE_OPTIONAL_M10":INCLUDE_OPTIONAL_M10,"OPTIONAL_SOURCE_COUNT_VALUES":OPTIONAL_SOURCE_COUNT_VALUES,"SOURCE_COUNT_VALUES":SOURCE_COUNT_VALUES,"T_VALUES":T_VALUES,"C_MODE_LIST":C_MODE_LIST,"N_OUTER_RUNS":N_OUTER_RUNS,"B_MODEL_VARIANTS":B_MODEL_VARIANTS,"LAMBDA_GRIDS":LAMBDA_GRIDS,"LGC_LAMBDA_GRID":LGC_LAMBDA_GRID,"workers":N_WORKERS,"numba":NUMBA_AVAILABLE}
    with open(os.path.join(RESULTS_DIR,"experiment_config.json"),"w",encoding="utf-8") as handle:json.dump(config,handle,indent=2)
    run_rows=[]
    for outer in range(N_OUTER_RUNS):
        seed=BASE_SEED+outer
        for n in SOURCE_COUNT_VALUES:
            for T in T_VALUES:
                for c_mode in C_MODE_LIST:
                    datasets={nb:simulate(n,T,c_mode,nb,seed) for nb in (3,10)}
                    baseline_caches = {
                        nb_true: baseline_readout_cache(
                            dataset,
                            {"outer_run":outer,"random_seed":seed,"n_sources":n,"T":T,"C_MODE":c_mode},
                            [config[1] for config in VARIANT_CONFIG.values() if config[0] == nb_true],
                        )
                        for nb_true, dataset in datasets.items()
                    }
                    tasks=[]
                    for variant in B_MODEL_VARIANTS:
                        nb_true=VARIANT_CONFIG[variant][0]
                        for lam in LAMBDA_GRIDS[n]:
                            if (outer,n,T,c_mode,variant,float(lam)) not in completed:tasks.append((datasets[nb_true],variant,lam,{"outer_run":outer,"random_seed":seed,"n_sources":n,"T":T,"C_MODE":c_mode},seed+500))
                    with ThreadPoolExecutor(max_workers=min(N_WORKERS,max(1,len(tasks)))) as executor:
                        for parameter,support,network,lgc in executor.map(work,tasks):
                            baseline_network, baseline_lgc = baseline_caches[
                                int(parameter["nb_true"])
                            ][int(parameter["nb_model"])]
                            network = pd.concat([
                                attach_variant_metadata(baseline_network, parameter), network
                            ], ignore_index=True)
                            lgc = pd.concat([
                                attach_variant_metadata(baseline_lgc, parameter), lgc
                            ], ignore_index=True)
                            frames["parameter"]=pd.concat([frames["parameter"],pd.DataFrame([parameter])],ignore_index=True);frames["B"]=pd.concat([frames["B"],pd.DataFrame([parameter])],ignore_index=True)
                            frames["support"]=pd.concat([frames["support"],support],ignore_index=True);frames["network"]=pd.concat([frames["network"],network],ignore_index=True);frames["lgc"]=pd.concat([frames["lgc"],lgc],ignore_index=True)
                            completed.add(tuple(parameter[column] for column in keycols))
                            print(f"33D N={n} T={T} C={c_mode} B={parameter['B_MODEL_VARIANT']} nb={parameter['nb_true']}/{parameter['nb_model']} basis={parameter['n_B_basis']} lambda={parameter['lambda_A_group_fraction']:g} LL={parameter['full_log_likelihood']:.1f} iter={parameter['full_em_iterations']} radius={parameter['full_spectral_radius']:.3f} Q/R={parameter['Q_trace']:.1f}/{parameter['R_trace']:.1f} A/Aoff/B={parameter['A_relative_frobenius_error']:.3f}/{parameter['A_offdiag_relative_frobenius_error']:.3f}/{parameter['B_relative_frobenius_error']:.3f} Benergy={parameter['B_energy_ratio_hat_to_true']:.3f} AUPRC={parameter['A_support_AUPRC']:.3f} TPR@.05={parameter['A_support_TPR_at_FPR_leq_0p05']:.3f}")
                            for signal_type in ("em_filtered", "em_smoothed"):
                                selected = network[network.signal_type == signal_type]
                                deviance_metrics, _ = curve_metrics(
                                    selected.true_link, selected.debiased_deviance
                                )
                                lgc_candidates = []
                                for display_lgc_lambda in LGC_LAMBDA_GRID:
                                    selected_lgc = lgc[
                                        (lgc.signal_type == signal_type) &
                                        (lgc.lgc_lambda == display_lgc_lambda)
                                    ]
                                    lgc_metrics, _ = curve_metrics(
                                        selected_lgc.true_link,
                                        selected_lgc.lgc_clipped,
                                    )
                                    lgc_candidates.append((
                                        lgc_metrics["TPR_at_FPR_leq_0p05"],
                                        display_lgc_lambda,
                                        lgc_metrics,
                                    ))
                                _, best_lgc_lambda, best_lgc = max(
                                    lgc_candidates, key=lambda item: item[0]
                                )
                                print(
                                    f"  {signal_type}: debiased TPR/precision/F1@FPR.05="
                                    f"{deviance_metrics['TPR_at_FPR_leq_0p05']:.3f}/"
                                    f"{deviance_metrics['precision_at_FPR_leq_0p05']:.3f}/"
                                    f"{deviance_metrics['F1_at_FPR_leq_0p05']:.3f}; "
                                    f"best LGC(lambda={best_lgc_lambda:g}) TPR@FPR.05="
                                    f"{best_lgc['TPR_at_FPR_leq_0p05']:.3f}; "
                                    f"mean false/true deviance="
                                    f"{selected.loc[~selected.true_link, 'debiased_deviance'].mean():.3f}/"
                                    f"{selected.loc[selected.true_link, 'debiased_deviance'].mean():.3f}"
                                )
        for key,name in partial_names.items():atomic_csv(frames[key],os.path.join(RESULTS_DIR,name))
        partial_roc,_=roc_outputs(frames["network"],frames["lgc"],frames["support"])
        atomic_csv(partial_roc,os.path.join(RESULTS_DIR,"roc_summary_partial.csv"))
        atomic_csv(partial_roc,os.path.join(RESULTS_DIR,"fixed_fpr_operating_points_partial.csv"))
        run_rows.append({"outer_run":outer,"completed_work_units":len(completed),"timestamp":time.time()});atomic_csv(pd.DataFrame(run_rows),os.path.join(RESULTS_DIR,"run_summary.csv"))
    groups=["n_sources","T","C_MODE","B_MODEL_VARIANT","nb_true","nb_model","n_B_basis","lambda_A_group_fraction"]
    parameter_metrics=[c for c in frames["parameter"] if c.startswith(("A_","B_","filtered_","smoothed_","full_","offdiag_")) and pd.api.types.is_numeric_dtype(frames["parameter"][c])]
    parameter_summary=grouped_stats(frames["parameter"],groups,parameter_metrics)
    B_metrics=[c for c in frames["B"] if c.startswith("B_") and pd.api.types.is_numeric_dtype(frames["B"][c])]
    B_summary=grouped_stats(frames["B"],groups,B_metrics)
    link_summary=grouped_stats(frames["network"],["n_sources","T","C_MODE","B_MODEL_VARIANT","signal_type","lambda_A_group_fraction","true_link"],["raw_deviance","full_bias_term","reduced_bias_term","bias_correction","debiased_deviance","raw_p_value","debiased_p_value"])
    decision=decisions(frames["network"],[c for c in frames["network"] if c.endswith("_detected")],["n_sources","T","C_MODE","B_MODEL_VARIANT","signal_type","lambda_A_group_fraction"])
    lgc_decision=decisions(frames["lgc"],[c for c in frames["lgc"] if c.startswith("lgc_detected_threshold_")],["n_sources","T","C_MODE","B_MODEL_VARIANT","signal_type","lambda_A_group_fraction","lgc_lambda"])
    roc,points=roc_outputs(frames["network"],frames["lgc"],frames["support"])
    outputs={"network_results.csv":frames["network"],"lgc_eq7_results.csv":frames["lgc"],"parameter_summary.csv":parameter_summary,"B_recovery_summary.csv":B_summary,"A_support_results.csv":frames["support"],"link_strength_summary.csv":link_summary,"decision_summary.csv":decision,"lgc_decision_summary.csv":lgc_decision,"roc_summary.csv":roc,"roc_curve_points.csv":points,"fixed_fpr_operating_points.csv":roc}
    for name,frame in outputs.items():atomic_csv(frame,os.path.join(RESULTS_DIR,name))
    print("\nFocused-run interpretation guide:")
    print("1. free_B vs fixed_B_true: improvement indicates B-estimation leakage into A.")
    print("2. free_B vs ridge_B_medium: improvement without poor B recovery supports practical B stabilization.")
    print("3. free_B_long vs basis_B_long: improvement supports basis reduction for flexible long filters.")
    print("4. If none improve A recovery, high-dimensional latent A estimation is the main bottleneck.")
    print("Prioritize TPR, precision, and F1 at FPR <= 0.05; compare score inference with model_A support.")


if __name__=="__main__":main()

# TODO Experiment 33F: add density/coverage-style edge diagnostics if needed.
