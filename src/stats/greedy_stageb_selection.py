"""Deployable greedy selectors for atomic Stage-B directed-edge groups.

All inputs to these routines are pre-Stage-B descriptors.  Reference Stage-B
quantities belong in experiment-level evaluation code and must never be passed
as selector features.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform


EPS = 1e-12


def rbf_similarity(features: np.ndarray, bandwidth: float) -> np.ndarray:
    features = np.asarray(features, float)
    distance2 = squareform(pdist(features, metric="sqeuclidean")) if len(features) > 1 else np.zeros((1, 1))
    return np.exp(-distance2 / (2.0 * max(float(bandwidth), EPS) ** 2))


def median_bandwidth(features: np.ndarray) -> float:
    distances = pdist(np.asarray(features, float))
    positive = distances[np.isfinite(distances) & (distances > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def purpose_order(utility: np.ndarray):
    utility = np.maximum(np.asarray(utility, float), 0.0)
    order = np.argsort(-utility, kind="stable")
    cumulative = 0.0
    rows = []
    for rank, index in enumerate(order, 1):
        gain = float(utility[index]); cumulative += gain
        rows.append({"index": int(index), "marginal_gain": gain,
                     "cumulative_objective": cumulative,
                     "normalized_marginal_gain": gain / max(float(utility.sum()), EPS)})
    return rows


def facility_objective(selected, similarity, utility, lam, weight_mode="purpose"):
    utility = np.maximum(np.asarray(utility, float), 0.0)
    normalized = utility / max(float(utility.max()), EPS)
    weights = np.ones(len(utility)) if weight_mode == "uniform" else EPS + normalized
    if not selected:
        representation = 0.0
    else:
        representation = float(np.sum(weights * np.max(similarity[:, list(selected)], axis=1)) / np.sum(weights))
    purpose = float(utility[list(selected)].sum() / max(float(utility.sum()), EPS)) if selected else 0.0
    return float(lam * purpose + (1.0 - lam) * representation)


def facility_order(similarity: np.ndarray, utility: np.ndarray, lam: float,
                   weight_mode="purpose"):
    n = len(utility); selected = []; remaining = set(range(n)); rows = []
    current = 0.0
    for _ in range(n):
        gains = {index: facility_objective(selected + [index], similarity, utility, lam, weight_mode) - current
                 for index in remaining}
        index = max(gains, key=lambda item: (gains[item], -item))
        gain = float(gains[index]); selected.append(index); remaining.remove(index); current += gain
        rows.append({"index": int(index), "marginal_gain": gain,
                     "cumulative_objective": current,
                     "normalized_marginal_gain": gain})
    return rows


def logdet_objective(selected, descriptors, utility, delta=1e-3):
    descriptors = np.asarray(descriptors, float)
    utility = np.maximum(np.asarray(utility, float), 0.0)
    q = EPS + utility / max(float(utility.max()), EPS)
    matrix = float(delta) * np.eye(descriptors.shape[1])
    for index in selected:
        phi = descriptors[index]
        matrix += q[index] * np.outer(phi, phi)
    sign, value = np.linalg.slogdet(matrix)
    baseline = descriptors.shape[1] * np.log(float(delta))
    return float(value - baseline) if sign > 0 else -np.inf


def logdet_order(descriptors: np.ndarray, utility: np.ndarray, delta=1e-3):
    descriptors = np.asarray(descriptors, float)
    utility = np.maximum(np.asarray(utility, float), 0.0)
    q = EPS + utility / max(float(utility.max()), EPS)
    information = float(delta) * np.eye(descriptors.shape[1])
    inverse = np.linalg.inv(information)
    remaining = set(range(len(descriptors))); rows = []; cumulative = 0.0
    for _ in range(len(descriptors)):
        gains = {index: float(np.log1p(q[index] * descriptors[index] @ inverse @ descriptors[index]))
                 for index in remaining}
        index = max(gains, key=lambda item: (gains[item], -item))
        gain = max(float(gains[index]), 0.0)
        vector = np.sqrt(q[index]) * descriptors[index]
        denominator = 1.0 + vector @ inverse @ vector
        inverse -= np.outer(inverse @ vector, inverse @ vector) / max(float(denominator), EPS)
        information += np.outer(vector, vector)
        cumulative += gain; remaining.remove(index)
        rows.append({"index": int(index), "marginal_gain": gain,
                     "cumulative_objective": cumulative,
                     "normalized_marginal_gain": gain / max(cumulative, EPS),
                     "minimum_information_eigenvalue": float(np.linalg.eigvalsh(information).min())})
    return rows


def pivoted_cholesky_order(kernel: np.ndarray):
    kernel = 0.5 * (np.asarray(kernel, float) + np.asarray(kernel, float).T)
    residual = kernel.copy(); original = max(float(np.trace(kernel)), EPS)
    remaining = set(range(len(kernel))); rows = []
    for _ in range(len(kernel)):
        diagonal = np.diag(residual)
        index = max(remaining, key=lambda item: (diagonal[item], -item))
        pivot = max(float(residual[index, index]), EPS)
        column = residual[:, index].copy()
        residual -= np.outer(column, column) / pivot
        residual = 0.5 * (residual + residual.T)
        remaining.remove(index)
        remaining_diagonal = np.maximum(np.diag(residual)[list(remaining)], 0.0) if remaining else np.asarray([0.0])
        ratio = max(float(np.trace(residual)), 0.0) / original
        rows.append({"index": int(index), "marginal_gain": 2.0 * pivot,
                     "cumulative_objective": 1.0 - ratio,
                     "normalized_marginal_gain": pivot / original,
                     "residual_trace_fraction": ratio,
                     "maximum_residual_block_trace": 2.0 * float(remaining_diagonal.max()),
                     "kernel_reconstruction_error": float(np.linalg.norm(residual, "fro") / max(np.linalg.norm(kernel, "fro"), EPS))})
    return rows


def stratified_random_order(snr: np.ndarray, observability: np.ndarray, seed: int):
    rng = np.random.default_rng(seed); n = len(snr)
    def quartiles(values):
        order = np.argsort(np.argsort(np.asarray(values), kind="stable"), kind="stable")
        return np.minimum(3, (4 * order) // max(n, 1))
    sq, oq = quartiles(snr), quartiles(observability)
    bins = {}
    for index in range(n): bins.setdefault((int(sq[index]), int(oq[index])), []).append(index)
    for values in bins.values(): rng.shuffle(values)
    order = []
    while any(bins.values()):
        for key in sorted(bins):
            if bins[key]: order.append(bins[key].pop())
    return [{"index": int(index), "marginal_gain": np.nan,
             "cumulative_objective": np.nan, "normalized_marginal_gain": np.nan}
            for index in order]


def randomized_submodularity_check(objective, n: int, trials=1000, seed=0):
    rng = np.random.default_rng(seed); violations = []; monotonicity = []
    universe = np.arange(n)
    for _ in range(int(trials)):
        permutation = rng.permutation(universe)
        a_size = int(rng.integers(0, max(n - 1, 1)))
        b_size = int(rng.integers(a_size, n))
        A = list(permutation[:a_size]); B = list(permutation[:b_size])
        outside = [index for index in universe if index not in B]
        if not outside: continue
        edge = int(rng.choice(outside))
        gain_A = objective(A + [edge]) - objective(A)
        gain_B = objective(B + [edge]) - objective(B)
        violations.append(gain_B - gain_A)
        monotonicity.extend([gain_A, gain_B])
    violations = np.asarray(violations, float); monotonicity = np.asarray(monotonicity, float)
    return {"n_trials": int(len(violations)),
            "diminishing_returns_violation_count": int(np.sum(violations > 1e-10)),
            "maximum_diminishing_returns_violation": float(max(0.0, violations.max(initial=0.0))),
            "negative_marginal_gain_count": int(np.sum(monotonicity < -1e-10)),
            "minimum_marginal_gain": float(monotonicity.min(initial=0.0))}


def exact_greedy_ratios(objective, greedy_order, n_small=12, budgets=(3, 4, 5)):
    rows = []
    for budget in budgets:
        exact = max(objective(list(items)) for items in combinations(range(n_small), int(budget)))
        greedy = objective([int(item["index"]) for item in greedy_order[:budget]])
        rows.append({"N_small": n_small, "k": int(budget), "greedy_objective": greedy,
                     "exact_objective": exact, "greedy_exact_ratio": greedy / max(exact, EPS)})
    return rows


__all__ = ["exact_greedy_ratios", "facility_objective", "facility_order",
           "logdet_objective", "logdet_order", "median_bandwidth",
           "pivoted_cholesky_order", "purpose_order", "randomized_submodularity_check",
           "rbf_similarity", "stratified_random_order"]
