"""Deployable diagnostic-subset design for legacy Stage-B LRVB.

Selectors in this module accept only pre-Stage-B features.  Truth labels and
Stage-B results are intentionally confined to the evaluation helpers.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, wasserstein_distance


EPS = 1e-12
FEATURES = [
    "estimated_group_norm",
    "vb_cov_trace",
    "louis_cov_trace",
    "log_louis_cov_determinant",
    "louis_group_snr",
    "log_ard_group_precision",
    "spectral_transfer_full_band_score",
    "source_observability",
    "target_observability",
    "log_edge_o_inst_product",
    "edge_o_dyn_estA_mean_K10",
]

CATEGORY_COLUMNS = [
    "active_coverage_conclusion",
    "zero_coverage_conclusion",
    "rescue_conclusion",
    "inflation_conclusion",
    "observability_dependence_conclusion",
    "snr_dependence_conclusion",
    "low_observability_conclusion",
    "numerical_stability_conclusion",
]

CONTINUOUS_COLUMNS = [
    "Delta_active",
    "Delta_zero",
    "rescue_rate",
    "median_relative_inflation",
    "observability_rho",
    "SNR_rho",
    "Q1_minus_Q4_active_coverage_gap",
    "failure_fraction",
]


def fit_robust_scaler(calibration: pd.DataFrame, features=FEATURES):
    rows = []
    for feature in features:
        values = pd.to_numeric(calibration[feature], errors="coerce").to_numpy(float)
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= EPS:
            scale = float(np.nanstd(values))
        rows.append({"feature": feature, "calibration_median": median,
                     "calibration_robust_scale": max(scale, EPS)})
    return pd.DataFrame(rows)


def apply_robust_scaler(frame: pd.DataFrame, scaler: pd.DataFrame, features=FEATURES):
    output = frame.copy()
    lookup = scaler.set_index("feature")
    for feature in features:
        median = float(lookup.loc[feature, "calibration_median"])
        scale = float(lookup.loc[feature, "calibration_robust_scale"])
        output[f"z__{feature}"] = (pd.to_numeric(output[feature], errors="coerce") - median) / scale
    return output


def _rank_bins(values, n_bins):
    values = np.asarray(values, float)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum(n_bins - 1, (n_bins * order) // max(len(values), 1))


def add_diagnostic_strata(frame: pd.DataFrame):
    """Add deployable observability-quartile x Louis-SNR-tertile cells."""
    pieces = []
    for _, group in frame.groupby("run_id", sort=False):
        group = group.copy()
        obs = _rank_bins(group["log_edge_o_inst_product"], 4)
        snr = _rank_bins(group["louis_group_snr"], 3)
        group["diagnostic_observability_quartile"] = obs
        group["diagnostic_snr_tertile"] = snr
        group["diagnostic_stratum"] = [f"Q{o + 1}_S{s + 1}" for o, s in zip(obs, snr)]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.copy()


def quota_cell_schedule(strata, max_k):
    """Return a nested, approximately proportional diagnostic-cell schedule.

    The first seats cover available low/high observability and low/high SNR
    corners.  Remaining nonempty cells then receive one seat, after which a
    largest-deficit (sequential largest-remainder) rule tracks population
    proportions.  Every prefix is therefore nested and its counts sum to k.
    """
    strata = np.asarray(strata, str)
    counts = Counter(strata)
    cells = sorted(counts)
    if not cells:
        return []
    capacity = dict(counts)
    allocated = Counter()
    schedule = []

    def add(cell):
        if cell in capacity and allocated[cell] < capacity[cell] and len(schedule) < max_k:
            schedule.append(cell)
            allocated[cell] += 1

    # Four deployable extremes; missing corners are simply ignored.
    for cell in ("Q1_S1", "Q1_S3", "Q4_S1", "Q4_S3"):
        add(cell)
    for cell in cells:
        if allocated[cell] == 0:
            add(cell)
    total = float(sum(counts.values()))
    while len(schedule) < min(int(max_k), int(total)):
        rank = len(schedule) + 1
        available = [cell for cell in cells if allocated[cell] < capacity[cell]]
        cell = max(available, key=lambda item: (
            rank * counts[item] / total - allocated[item], counts[item], item
        ))
        add(cell)
    return schedule


def quota_counts(schedule, k):
    return dict(Counter(schedule[: int(k)]))


def _eligible(remaining, strata, cell):
    return [index for index in remaining if strata[index] == cell]


def stratified_random_order(strata, max_k, seed):
    rng = np.random.default_rng(seed)
    schedule = quota_cell_schedule(strata, max_k)
    bins = defaultdict(list)
    for index, cell in enumerate(np.asarray(strata, str)):
        bins[cell].append(index)
    for values in bins.values():
        rng.shuffle(values)
    return [int(bins[cell].pop()) for cell in schedule]


def stratified_maximin_order(Z, strata, max_k, initialization="centroid"):
    Z = np.asarray(Z, float); strata = np.asarray(strata, str)
    schedule = quota_cell_schedule(strata, max_k)
    remaining = set(range(len(Z))); selected = []
    centroid = np.mean(Z, axis=0)
    uncertainty_column = FEATURES.index("louis_cov_trace")
    for cell in schedule:
        candidates = _eligible(remaining, strata, cell)
        if not selected:
            if initialization == "uncertainty":
                index = max(candidates, key=lambda i: (Z[i, uncertainty_column], -i))
            else:
                index = min(candidates, key=lambda i: (np.linalg.norm(Z[i] - centroid), i))
        else:
            distance = cdist(Z[candidates], Z[selected]).min(axis=1)
            index = candidates[int(np.argmax(distance))]
        selected.append(int(index)); remaining.remove(index)
    return selected


def stratified_facility_order(Z, strata, max_k, bandwidth, weight_mode="uniform"):
    Z = np.asarray(Z, float); strata = np.asarray(strata, str)
    distance2 = cdist(Z, Z, metric="sqeuclidean")
    similarity = np.exp(-distance2 / (2.0 * max(float(bandwidth), EPS) ** 2))
    if weight_mode == "louis_uncertainty":
        raw = Z[:, FEATURES.index("louis_cov_trace")]
        raw = (raw - np.min(raw)) / max(float(np.ptp(raw)), EPS)
        weights = 1.0 + raw
    else:
        weights = np.ones(len(Z))
    represented = np.zeros(len(Z)); remaining = set(range(len(Z))); selected = []
    for cell in quota_cell_schedule(strata, max_k):
        candidates = _eligible(remaining, strata, cell)
        gains = [np.sum(weights * (np.maximum(represented, similarity[:, i]) - represented)) for i in candidates]
        index = candidates[int(np.argmax(gains))]
        represented = np.maximum(represented, similarity[:, index])
        selected.append(int(index)); remaining.remove(index)
    return selected


def stratified_logdet_order(Z, strata, source, target, max_k, node_scale=0.0, delta=1e-3):
    Z = np.asarray(Z, float); source = np.asarray(source, int); target = np.asarray(target, int)
    dimensions = int(max(source.max(initial=0), target.max(initial=0)) + 1)
    if node_scale > 0:
        nodes = np.zeros((len(Z), 2 * dimensions))
        nodes[np.arange(len(Z)), source] = node_scale
        nodes[np.arange(len(Z)), dimensions + target] = node_scale
        phi = np.column_stack([np.ones(len(Z)), Z, nodes])
    else:
        phi = np.column_stack([np.ones(len(Z)), Z])
    inverse = np.eye(phi.shape[1]) / float(delta)
    strata = np.asarray(strata, str); remaining = set(range(len(Z))); selected = []
    for cell in quota_cell_schedule(strata, max_k):
        candidates = _eligible(remaining, strata, cell)
        gains = [np.log1p(max(float(phi[i] @ inverse @ phi[i]), 0.0)) for i in candidates]
        index = candidates[int(np.argmax(gains))]
        vector = phi[index]; iv = inverse @ vector
        inverse -= np.outer(iv, iv) / max(1.0 + float(vector @ iv), EPS)
        selected.append(int(index)); remaining.remove(index)
    return selected


def calibration_bandwidth(frame: pd.DataFrame):
    Z = frame[[f"z__{name}" for name in FEATURES]].to_numpy(float)
    distances = pdist(Z)
    positive = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def _category(value, lower, upper, below, middle, above):
    if not np.isfinite(value):
        return np.nan
    if value < lower:
        return below
    if value > upper:
        return above
    return middle


def _coverage_delta(frame, mask):
    rows = frame.loc[np.asarray(mask, bool)]
    if not len(rows):
        return np.nan
    return float(rows["stageb_group_covered_95"].mean() - rows["louis_group_covered_95"].mean())


def _safe_spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) <= EPS or np.ptp(y[valid]) <= EPS:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def conclusion_vector(frame: pd.DataFrame, perturbations: pd.DataFrame | None = None):
    """Calculate the pre-specified eight-part methodological conclusion."""
    frame = frame.copy()
    active = frame["true_edge"].astype(bool).to_numpy()
    delta_active = _coverage_delta(frame, active)
    delta_zero = _coverage_delta(frame, ~active)
    louis_fail = ~frame["louis_group_covered_95"].astype(bool)
    rescue_rate = float(frame.loc[louis_fail, "stageb_group_covered_95"].mean()) if louis_fail.any() else np.nan
    inflation = pd.to_numeric(frame["relative_stageb_inflation"], errors="coerce").to_numpy(float)
    obs_rho = _safe_spearman(frame["log_edge_o_inst_product"], inflation)
    snr_rho = _safe_spearman(frame["louis_group_snr"], inflation)

    q = frame["diagnostic_observability_quartile"] if "diagnostic_observability_quartile" in frame else _rank_bins(frame["log_edge_o_inst_product"], 4)
    q = np.asarray(q, int)
    q1 = active & (q == 0); q4 = active & (q == 3)
    q1_coverage = float(frame.loc[q1, "stageb_group_covered_95"].mean()) if q1.any() else np.nan
    q4_coverage = float(frame.loc[q4, "stageb_group_covered_95"].mean()) if q4.any() else np.nan
    gap = q1_coverage - q4_coverage if np.isfinite(q1_coverage) and np.isfinite(q4_coverage) else np.nan

    finite_edge = np.isfinite(frame[["Sigma_stageb_00", "Sigma_stageb_01", "Sigma_stageb_10", "Sigma_stageb_11"]]).all(axis=1)
    finite_fraction = float(finite_edge.mean()) if len(frame) else np.nan
    eig = pd.to_numeric(frame["stageb_min_eig_after"], errors="coerce").to_numpy(float)
    psd_fraction = float(np.mean(eig >= -1e-10)) if len(eig) else np.nan
    minimum_eig = float(np.nanmin(eig)) if np.isfinite(eig).any() else np.nan
    failure_fraction = 1.0 - finite_fraction if np.isfinite(finite_fraction) else np.nan
    convergence_fraction = np.nan
    if perturbations is not None and len(perturbations):
        convergence_fraction = float((perturbations["plus_converged"].astype(bool) & perturbations["minus_converged"].astype(bool)).mean())

    median_inflation = float(np.nanmedian(inflation)) if np.isfinite(inflation).any() else np.nan
    result = {
        "Delta_active": delta_active,
        "active_coverage_conclusion": _category(delta_active, -0.05, 0.05, "HARMFUL", "NEUTRAL", "BENEFICIAL"),
        "Delta_zero": delta_zero,
        "zero_coverage_conclusion": _category(delta_zero, -0.02, 0.02, "WORSENS", "NEUTRAL", "IMPROVES"),
        "rescue_rate": rescue_rate,
        "rescue_conclusion": _category(rescue_rate, 0.25, 0.50 - EPS, "LOW", "MODERATE", "HIGH"),
        "median_relative_inflation": median_inflation,
        "mean_relative_inflation": float(np.nanmean(inflation)),
        "fraction_relative_inflation_gt_0": float(np.nanmean(inflation > 0)),
        "fraction_relative_inflation_gt_0p25": float(np.nanmean(inflation > 0.25)),
        "inflation_conclusion": _category(median_inflation, -0.10, 0.10, "DEFLATION", "SMALL_NEUTRAL", "MATERIAL_INFLATION"),
        "observability_rho": obs_rho,
        "observability_dependence_conclusion": _category(obs_rho, -0.20, 0.20, "NEGATIVE", "WEAK_FLAT", "POSITIVE"),
        "SNR_rho": snr_rho,
        "snr_dependence_conclusion": _category(snr_rho, -0.20, 0.20, "NEGATIVE", "WEAK_FLAT", "POSITIVE"),
        "Q1_active_coverage": q1_coverage,
        "Q4_active_coverage": q4_coverage,
        "Q1_minus_Q4_active_coverage_gap": gap,
        "low_observability_conclusion": _category(gap, -0.10, 0.10, "Q1_WORSE", "SIMILAR", "Q1_BETTER"),
        "finite_result_fraction": finite_fraction,
        "perturbation_convergence_fraction": convergence_fraction,
        "PSD_fraction": psd_fraction,
        "minimum_covariance_eigenvalue": minimum_eig,
        "failure_fraction": failure_fraction,
        "numerical_stability_conclusion": "ACCEPTABLE" if failure_fraction <= 0.05 and finite_fraction >= 0.95 else "UNSTABLE",
        "n_edges": int(len(frame)),
        "n_active_edges": int(active.sum()),
        "n_zero_edges": int((~active).sum()),
    }
    return result


def conclusion_agreement(reference: dict | pd.Series, subset: dict | pd.Series):
    row = {}
    matches = []
    for column in CATEGORY_COLUMNS:
        ref, sub = reference.get(column, np.nan), subset.get(column, np.nan)
        defined = pd.notna(ref) and pd.notna(sub)
        match = bool(ref == sub) if defined else np.nan
        row[f"agreement__{column}"] = match
        if defined:
            matches.append(float(match))
    row["categorical_agreement"] = float(np.mean(matches)) if matches else np.nan
    for column in CONTINUOUS_COLUMNS:
        ref, sub = float(reference.get(column, np.nan)), float(subset.get(column, np.nan))
        row[f"abs_error_{column}"] = abs(sub - ref) if np.isfinite(ref) and np.isfinite(sub) else np.nan
    row["primary_usefulness_reversal"] = bool(
        {reference.get("active_coverage_conclusion"), subset.get("active_coverage_conclusion")} == {"BENEFICIAL", "HARMFUL"}
    )
    row["stability_agreement"] = bool(
        reference.get("numerical_stability_conclusion") == subset.get("numerical_stability_conclusion")
    )
    return row


def success_assessment(agreement: pd.DataFrame):
    """Apply the exact held-out success thresholds to per-network rows."""
    n = agreement["run_id"].nunique() if len(agreement) else 0
    required_networks = int(np.ceil(2 * n / 3))
    finite = lambda column: pd.to_numeric(agreement[column], errors="coerce")
    checks = {
        "no_primary_reversal": bool(not agreement["primary_usefulness_reversal"].fillna(True).any()) if len(agreement) else False,
        "mean_categorical_agreement_ge_0p85": bool(finite("categorical_agreement").mean() >= 0.85),
        "networks_agreement_ge_0p80": int((finite("categorical_agreement") >= 0.80).sum()),
        "required_networks_agreement_ge_0p80": required_networks,
        "Delta_active_error_le_0p10": bool(finite("abs_error_Delta_active").notna().all() and finite("abs_error_Delta_active").mean() <= 0.10),
        "rescue_rate_error_le_0p15": bool(finite("abs_error_rescue_rate").notna().all() and finite("abs_error_rescue_rate").mean() <= 0.15),
        "observability_rho_error_le_0p20": bool(finite("abs_error_observability_rho").notna().all() and finite("abs_error_observability_rho").mean() <= 0.20),
        "SNR_rho_error_le_0p20": bool(finite("abs_error_SNR_rho").notna().all() and finite("abs_error_SNR_rho").mean() <= 0.20),
        "stability_agrees_all": bool(agreement["stability_agreement"].fillna(False).all()) if len(agreement) else False,
    }
    checks["successful"] = bool(
        checks["no_primary_reversal"]
        and checks["mean_categorical_agreement_ge_0p85"]
        and checks["networks_agreement_ge_0p80"] >= required_networks
        and checks["Delta_active_error_le_0p10"]
        and checks["rescue_rate_error_le_0p15"]
        and checks["observability_rho_error_le_0p20"]
        and checks["SNR_rho_error_le_0p20"]
        and checks["stability_agrees_all"]
    )
    return checks


def representativeness(reference: pd.DataFrame, subset: pd.DataFrame, features=FEATURES):
    rows = []
    for feature in features:
        ref = pd.to_numeric(reference[feature], errors="coerce").dropna().to_numpy(float)
        sub = pd.to_numeric(subset[feature], errors="coerce").dropna().to_numpy(float)
        rows.append({"feature": feature, "reference_mean": np.mean(ref), "reference_median": np.median(ref),
                     "reference_q05": np.quantile(ref, .05), "reference_q25": np.quantile(ref, .25),
                     "reference_q75": np.quantile(ref, .75), "reference_q95": np.quantile(ref, .95),
                     "subset_mean": np.mean(sub), "subset_median": np.median(sub),
                     "subset_q05": np.quantile(sub, .05), "subset_q25": np.quantile(sub, .25),
                     "subset_q75": np.quantile(sub, .75), "subset_q95": np.quantile(sub, .95),
                     "wasserstein_distance": wasserstein_distance(ref, sub)})
    return rows


def node_coverage(subset: pd.DataFrame):
    source = subset.groupby("source").size(); target = subset.groupby("target").size()
    return {
        "unique_source_nodes": int(len(source)), "unique_target_nodes": int(len(target)),
        "minimum_edges_per_represented_source": int(source.min()),
        "minimum_edges_per_represented_target": int(target.min()),
        "maximum_source_repetition": int(source.max()), "maximum_target_repetition": int(target.max()),
    }


__all__ = [
    "CATEGORY_COLUMNS", "CONTINUOUS_COLUMNS", "FEATURES", "add_diagnostic_strata",
    "apply_robust_scaler", "calibration_bandwidth", "conclusion_agreement",
    "conclusion_vector", "fit_robust_scaler", "node_coverage", "quota_cell_schedule",
    "quota_counts", "representativeness", "stratified_facility_order",
    "stratified_logdet_order", "stratified_maximin_order", "stratified_random_order",
    "success_assessment",
]
