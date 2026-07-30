import copy
import numpy as np
from scipy.stats import chi2

from src.stats.observed_likelihood_deviance import observed_likelihood_deviance, final_observed_log_likelihood

def _as_2d_input(u):
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    return u

# Parameterize a positive-definite covariance matrix using lower-triangular Cholesky parameters.
# Diagonal entries are stored in log form.
# Off-diagonal entries are stored directly.
def _cholesky_to_unconstrained_params(S):
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    jitter = 1e-10 * np.eye(n)

    L = np.linalg.cholesky(S + jitter)
    params = []
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                params.append(np.log(max(L[i, j], 1e-12)))
            else:
                params.append(L[i, j])

    return np.asarray(params, dtype=float)

# Convert unconstrained Cholesky parameters back to covariance matrix.
def _unconstrained_params_to_covariance(params, n):
    params = np.asarray(params, dtype=float)
    L = np.zeros((n, n), dtype=float)

    cursor = 0

    for i in range(n):
        for j in range(i + 1):
            value = params[cursor]
            if i == j:
                L[i, j] = np.exp(value)
            else:
                L[i, j] = value

            cursor += 1

    S = L @ L.T
    S = 0.5 * (S + S.T)

    return S

# Check whether A_lag[target, source] is constrained to zero.
# lag_index is zero-based.
def _is_A_entry_constrained(zero_constraints, lag_index, target, source):
    lag_number = lag_index + 1

    if zero_constraints is None:
        return False

    for constraint in zero_constraints:
        if constraint["target"] != target:
            continue

        if constraint["source"] != source:
            continue

        lags = constraint.get("lags", "all")

        if lags == "all":
            return True

        if lag_number in list(lags):
            return True

    return False

# Pack selected model parameters into a vector.
# The default includes:
#     A matrices
#     B matrices
#     Q covariance, Cholesky parameterization
#     R covariance, Cholesky parameterization
# C is not packed because C is treated as known in Experiment 31E.
# D is not packed because D is fixed to None/zero in our current setup.
# For reduced models, constrained A entries are excluded from the parameter vector.
def pack_model_parameters(model, parameter_blocks=("A", "B", "Q", "R")):
    parameter_blocks = tuple(parameter_blocks)
    zero_constraints = getattr(model, "zero_constraints", [])

    theta_parts = []
    layout = []

    n_latent = model.C.shape[1]

    if "A" in parameter_blocks:
        for lag_index, A in enumerate(model.A_matrices):
            A = np.asarray(A, dtype=float)
            for target in range(A.shape[0]):
                for source in range(A.shape[1]):
                    if _is_A_entry_constrained(
                            zero_constraints=zero_constraints,
                            lag_index=lag_index,
                            target=target,
                            source=source
                    ):
                        continue

                    theta_parts.append(A[target, source])

                    layout.append({
                        "block": "A",
                        "lag_index": lag_index,
                        "target": target,
                        "source": source
                    })

    if "B" in parameter_blocks:
        for lag_index, B in enumerate(model.B_matrices):
            B = np.asarray(B, dtype=float)
            for target in range(B.shape[0]):
                for input_index in range(B.shape[1]):
                    theta_parts.append(B[target, input_index])
                    layout.append({
                        "block": "B",
                        "lag_index": lag_index,
                        "target": target,
                        "input_index": input_index
                    })

    if "Q" in parameter_blocks:
        q_params = _cholesky_to_unconstrained_params(model.Q)
        start = len(theta_parts)
        theta_parts.extend(q_params.tolist())
        layout.append({
            "block": "Q",
            "start": start,
            "stop": start + len(q_params),
            "dimension": model.Q.shape[0]
        })

    if "R" in parameter_blocks:
        r_params = _cholesky_to_unconstrained_params(model.R)
        start = len(theta_parts)
        theta_parts.extend(r_params.tolist())
        layout.append({
            "block": "R",
            "start": start,
            "stop": start + len(r_params),
            "dimension": model.R.shape[0]
        })

    theta = np.asarray(theta_parts, dtype=float)
    return theta, layout

# Write packed parameter vector into a model copy.
def unpack_model_parameters(model, theta, layout):
    theta = np.asarray(theta, dtype=float)
    for item in layout:
        block = item["block"]
        if block == "A":
            lag_index = item["lag_index"]
            target = item["target"]
            source = item["source"]

            model.A_matrices[lag_index][target, source] = theta[layout.index(item)]

        elif block == "B":
            lag_index = item["lag_index"]
            target = item["target"]
            input_index = item["input_index"]

            model.B_matrices[lag_index][target, input_index] = theta[layout.index(item)]

        elif block == "Q":
            start = item["start"]
            stop = item["stop"]
            dimension = item["dimension"]

            model.Q = _unconstrained_params_to_covariance(theta[start:stop], dimension)

        elif block == "R":
            start = item["start"]
            stop = item["stop"]
            dimension = item["dimension"]

            model.R = _unconstrained_params_to_covariance(theta[start:stop], dimension)

        else:
            raise ValueError(f"Unknown parameter block: {block}")

    if hasattr(model, "_apply_zero_constraints_to_A_matrices"):
        model._apply_zero_constraints_to_A_matrices()

    model._refresh_companion_matrices()

    return model

# Evaluate observed-data log-likelihood after replacing model parameters.
# This uses the model's E-step / Kalman smoothing routine.
def observed_log_likelihood_at_theta(base_model, theta, layout, y, u):
    working_model = copy.deepcopy(base_model)

    working_model = unpack_model_parameters(
        model=working_model,
        theta=theta,
        layout=layout
    )

    if not hasattr(working_model, "_e_step"):
        raise RuntimeError(
            "The EM model does not expose _e_step(y, u). "
            "Add an observed log-likelihood evaluator using your Kalman filter."
        )

    result = working_model._e_step(y=y, u=u)

    if "log_likelihood" not in result:
        raise RuntimeError("The E-step result does not contain 'log_likelihood'.")

    return float(result["log_likelihood"])

# Numerically compute:
# score s = grad log p(y | theta)
# observed information I = - Hessian log p(y | theta)
# around the fitted model parameters.
# The covariance matrices Q and R are parameterized through their Cholesky factors so that finite-difference perturbations preserve positive definiteness.
def finite_difference_score_information(
        model,
        y,
        u,
        parameter_blocks=("A", "B", "Q", "R"),
        fd_epsilon=3e-4,
        relative_info_floor=1e-8,
        verbose=False
):
    y = np.asarray(y, dtype=float)
    u = _as_2d_input(u)
    theta0, layout = pack_model_parameters(model=model, parameter_blocks=parameter_blocks)

    p = len(theta0)
    if p == 0:
        raise ValueError("No parameters were packed. Check parameter_blocks.")

    step = fd_epsilon * np.maximum(1.0, np.abs(theta0))

    def f(theta):
        return observed_log_likelihood_at_theta(
            base_model=model,
            theta=theta,
            layout=layout,
            y=y,
            u=u
        )

    ll0 = f(theta0)
    score = np.zeros(p, dtype=float)

    hessian = np.zeros((p, p), dtype=float)
    f_plus = np.zeros(p, dtype=float)
    f_minus = np.zeros(p,dtype=float)
    for i in range(p):
        theta_plus = theta0.copy()
        theta_minus = theta0.copy()

        theta_plus[i] += step[i]
        theta_minus[i] -= step[i]

        f_plus[i] = f(theta_plus)
        f_minus[i] = f(theta_minus)

        score[i] = (f_plus[i] - f_minus[i]) / (2.0 * step[i])

        hessian[i, i] = (f_plus[i] - 2.0 * ll0 + f_minus[i]) / (step[i] ** 2)

        if verbose:
            print(f"    finite-difference diagonal {i + 1}/{p}")

    for i in range(p):
        for j in range(i + 1, p):
            theta_pp = theta0.copy()
            theta_pm = theta0.copy()
            theta_mp = theta0.copy()
            theta_mm = theta0.copy()

            theta_pp[i] += step[i]
            theta_pp[j] += step[j]

            theta_pm[i] += step[i]
            theta_pm[j] -= step[j]

            theta_mp[i] -= step[i]
            theta_mp[j] += step[j]

            theta_mm[i] -= step[i]
            theta_mm[j] -= step[j]

            f_pp = f(theta_pp)
            f_pm = f(theta_pm)
            f_mp = f(theta_mp)
            f_mm = f(theta_mm)

            h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])

            hessian[i, j] = h_ij
            hessian[j, i] = h_ij

        if verbose:
            print(f"    finite-difference row {i + 1}/{p}")

    hessian = 0.5 * (hessian + hessian.T)
    
    observed_information = -hessian
    observed_information = 0.5 * (observed_information + observed_information.T)

    eigenvalues, eigenvectors = np.linalg.eigh(observed_information)

    positive_eigenvalues = eigenvalues[eigenvalues > 0]

    if len(positive_eigenvalues) > 0:
        scale = float(np.median(positive_eigenvalues))
    else:
        scale = 1.0

    floor = relative_info_floor * max(scale, 1.0)

    eigenvalues_regularized = np.maximum(eigenvalues, floor)
    information_regularized = (eigenvectors @ np.diag(eigenvalues_regularized) @ eigenvectors.T)

    information_regularized = 0.5 * (information_regularized + information_regularized.T)

    score_norm = float(np.linalg.norm(score))

    min_info_eigenvalue = float(np.min(eigenvalues))
    max_info_eigenvalue = float(np.max(eigenvalues))

    regularized_condition_number = float(np.max(eigenvalues_regularized) / np.min(eigenvalues_regularized))

    return {
        "theta": theta0,
        "layout": layout,
        "log_likelihood": ll0,
        "score": score,
        "observed_information": observed_information,
        "observed_information_regularized": information_regularized,
        "score_norm": score_norm,
        "min_info_eigenvalue": min_info_eigenvalue,
        "max_info_eigenvalue": max_info_eigenvalue,
        "regularized_condition_number": regularized_condition_number,
        "n_parameters": p,
        "fd_epsilon": fd_epsilon,
        "relative_info_floor": relative_info_floor
    }

# Compute:
# B = s^T I^{-1} s
# using finite-difference score and observed information.
def observed_likelihood_bias_term(
        model,
        y,
        u,
        parameter_blocks=("A", "B", "Q", "R"),
        fd_epsilon=3e-4,
        relative_info_floor=1e-8,
        verbose=False
):
    info_result = finite_difference_score_information(
        model=model,
        y=y,
        u=u,
        parameter_blocks=parameter_blocks,
        fd_epsilon=fd_epsilon,
        relative_info_floor=relative_info_floor,
        verbose=verbose
    )

    s = info_result["score"]
    I_reg = info_result["observed_information_regularized"]

    try:
        solved = np.linalg.solve(I_reg, s)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(I_reg) @ s

    bias = float(s.T @ solved)

    if bias < 0 and abs(bias) < 1e-8:
        bias = 0.0

    info_result["bias_term"] = bias

    return info_result

# Observed-data de-biased likelihood deviance:
# D_raw = 2(ll_full - ll_reduced)
# D_db = D_raw - B_reduced + B_full
# where: B_m = s_m^T I_m^{-1} s_m.
def observed_debiased_likelihood_deviance(
        full_model,
        reduced_model,
        y,
        u,
        df_removed,
        parameter_blocks=("A", "B", "Q", "R"),
        fd_epsilon=3e-4,
        relative_info_floor=1e-8,
        clip_negative=True,
        verbose=False
):
    raw_result = observed_likelihood_deviance(
        full_model=full_model,
        reduced_model=reduced_model,
        df_removed=df_removed,
        clip_negative=clip_negative
    )

    if verbose:
        print("  computing full-model observed bias term...")

    full_bias = observed_likelihood_bias_term(
        model=full_model,
        y=y,
        u=u,
        parameter_blocks=parameter_blocks,
        fd_epsilon=fd_epsilon,
        relative_info_floor=relative_info_floor,
        verbose=verbose
    )

    if verbose:
        print("  computing reduced-model observed bias term...")

    reduced_bias = observed_likelihood_bias_term(
        model=reduced_model,
        y=y,
        u=u,
        parameter_blocks=parameter_blocks,
        fd_epsilon=fd_epsilon,
        relative_info_floor=relative_info_floor,
        verbose=verbose
    )

    D_raw = raw_result["observed_likelihood_deviance"]

    B_full = full_bias["bias_term"]
    B_reduced = reduced_bias["bias_term"]

    D_debiased = (D_raw - B_reduced + B_full)

    if clip_negative and D_debiased < 0:
        D_debiased_for_p = 0.0
    else:
        D_debiased_for_p = D_debiased

    p_value_debiased = float(chi2.sf(D_debiased_for_p, df=df_removed))

    return {
        "ll_full": raw_result["ll_full"],
        "ll_reduced": raw_result["ll_reduced"],

        "observed_raw_deviance": float(D_raw),
        "observed_raw_deviance_for_p": float(raw_result["deviance_for_p"]),
        "observed_raw_p_value": float(raw_result["p_value"]),

        "bias_full": float(B_full),
        "bias_reduced": float(B_reduced),
        "bias_correction": float(-B_reduced + B_full),

        "observed_debiased_deviance": float(D_debiased),
        "observed_debiased_deviance_for_p": float(D_debiased_for_p),
        "observed_debiased_p_value": p_value_debiased,

        "df": df_removed,

        "full_score_norm": full_bias["score_norm"],
        "reduced_score_norm": reduced_bias["score_norm"],

        "full_n_parameters": full_bias["n_parameters"],
        "reduced_n_parameters": reduced_bias["n_parameters"],

        "full_min_info_eigenvalue": full_bias["min_info_eigenvalue"],
        "reduced_min_info_eigenvalue": reduced_bias["min_info_eigenvalue"],

        "full_info_condition_number": full_bias["regularized_condition_number"],
        "reduced_info_condition_number": reduced_bias["regularized_condition_number"],

        "fd_epsilon": fd_epsilon,
        "relative_info_floor": relative_info_floor,
        "parameter_blocks": ",".join(parameter_blocks)
    }

def _symmetrize(M):
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)

def _positive_scale_from_eigenvalues(eigenvalues):
    positive = eigenvalues[eigenvalues > 0]

    if len(positive) == 0:
        return 1.0

    return float(max(np.median(positive), 1.0))

# Compute B = s^T I^{-1} s using several robust inverse strategies.
# Method: "solve" "pinv" "pinv_psd" "eig_clip" "ridge" "shrinkage"
def information_inverse_quadratic_form(
        score,
        information,
        method="pinv",
        pinv_rcond=1e-6,
        eigen_floor_rel=1e-6,
        ridge_rel=1e-6,
        shrinkage_alpha=0.10
):
    s = np.asarray(score, dtype=float)
    I = _symmetrize(information)
    p = I.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(I)
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))

    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))

    scale = _positive_scale_from_eigenvalues(eigenvalues)
    original_positive = eigenvalues[eigenvalues > 0]

    if len(original_positive) > 0:
        original_condition_number = float(np.max(original_positive) / np.min(original_positive))
    else:
        original_condition_number = np.inf

    effective_rank = int(np.sum(eigenvalues > pinv_rcond * max(max_abs_eigenvalue, 1.0)))

    # Method 1: direct solve.
    if method == "solve":
        try:
            x = np.linalg.solve(I, s)
            I_used = I
            used_solver = "solve"
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(I, rcond=pinv_rcond) @ s
            I_used = I
            used_solver = "pinv_fallback"

    # Method 2: Moore-Penrose pseudo-inverse.
    elif method == "pinv":
        x = np.linalg.pinv(I, rcond=pinv_rcond) @ s
        I_used = I
        used_solver = "pinv"

    # Method 3: PSD pseudo-inverse.
    # Negative eigenvalues are discarded.
    # Very small positive eigenvalues are discarded.
    elif method == "pinv_psd":
        cutoff = pinv_rcond * max(max_eigenvalue, 1.0)
        inv_eigenvalues = np.zeros_like(eigenvalues)
        keep = eigenvalues > cutoff
        inv_eigenvalues[keep] = 1.0 / eigenvalues[keep]

        x = (eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T @ s)

        I_used = (eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T)
        used_solver = "pinv_psd"

    # Method 4: eigenvalue clipping.
    # Negative and tiny eigenvalues are lifted to a floor.
    elif method == "eig_clip":
        floor = eigen_floor_rel * scale
        clipped_eigenvalues = np.maximum(eigenvalues, floor)

        x = (eigenvectors @ np.diag(1.0 / clipped_eigenvalues) @ eigenvectors.T @ s)

        I_used = (eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T)
        used_solver = "eig_clip"

    # Method 5: ridge stabilization.
    # I_ridge = I + lambda I.
    # If still non-positive, use eigen clipping afterward.
    elif method == "ridge":
        ridge_value = ridge_rel * scale
        I_ridge = I + ridge_value * np.eye(p)

        ridge_eigenvalues, ridge_eigenvectors = np.linalg.eigh(_symmetrize(I_ridge))
        floor = eigen_floor_rel * scale

        ridge_eigenvalues_clipped = np.maximum(ridge_eigenvalues, floor)

        x = (ridge_eigenvectors @ np.diag(1.0 / ridge_eigenvalues_clipped) @ ridge_eigenvectors.T @ s)

        I_used = (ridge_eigenvectors@ np.diag(ridge_eigenvalues_clipped) @ ridge_eigenvectors.T)
        used_solver = "ridge_plus_clip"

    # Method 6: shrinkage stabilization.
    # I_shrink = (1-alpha) I + alpha tau I_p
    # This uses the covariance-shrinkage idea: move an unstable empirical matrix toward a spherical target.
    elif method == "shrinkage":
        tau = float(np.trace(I) / p)
        if not np.isfinite(tau) or tau <= 0:
            tau = scale

        I_target = tau * np.eye(p)
        I_shrink = ((1.0 - shrinkage_alpha) * I + shrinkage_alpha * I_target)

        shrink_eigenvalues, shrink_eigenvectors = np.linalg.eigh(_symmetrize(I_shrink))

        floor = eigen_floor_rel * max(scale, tau, 1.0)

        shrink_eigenvalues_clipped = np.maximum(shrink_eigenvalues, floor)

        x = (shrink_eigenvectors @ np.diag(1.0 / shrink_eigenvalues_clipped) @ shrink_eigenvectors.T @ s)

        I_used = (shrink_eigenvectors @ np.diag(shrink_eigenvalues_clipped) @ shrink_eigenvectors.T)
        used_solver = "shrinkage_plus_clip"

    else:
        raise ValueError(f"Unknown inverse method: {method}")

    bias = float(s.T @ x)
    if bias < 0 and abs(bias) < 1e-8:
        bias = 0.0

    I_used = _symmetrize(I_used)
    used_eigenvalues = np.linalg.eigvalsh(I_used)
    positive_used = used_eigenvalues[used_eigenvalues > 0]

    if len(positive_used) > 0:
        used_condition_number = float(np.max(positive_used) / np.min(positive_used))
    else:
        used_condition_number = np.inf

    return {
        "bias": bias,
        "method": method,
        "used_solver": used_solver,
        "pinv_rcond": pinv_rcond,
        "eigen_floor_rel": eigen_floor_rel,
        "ridge_rel": ridge_rel,
        "shrinkage_alpha": shrinkage_alpha,
        "n_parameters": p,
        "effective_rank": effective_rank,
        "score_norm": float(np.linalg.norm(s)),
        "min_info_eigenvalue": min_eigenvalue,
        "max_info_eigenvalue": max_eigenvalue,
        "original_condition_number": original_condition_number,
        "used_condition_number": used_condition_number,
        "bias_is_finite": bool(np.isfinite(bias))
    }

# Compute observed de-biased deviance using already-computed score/Hessian information.
# This avoids recomputing finite-difference Hessians for each inverse method.
def observed_debiased_deviance_from_precomputed_information(
        raw_result,
        full_info,
        reduced_info,
        df_removed,
        inverse_method_spec,
        clip_negative=True
):
    method = inverse_method_spec["method"]
    kwargs = inverse_method_spec.get("kwargs", {})

    full_bias_result = information_inverse_quadratic_form(
        score=full_info["score"],
        information=full_info["observed_information"],
        method=method,
        **kwargs
    )

    reduced_bias_result = information_inverse_quadratic_form(
        score=reduced_info["score"],
        information=reduced_info["observed_information"],
        method=method,
        **kwargs
    )

    D_raw = float(raw_result["observed_likelihood_deviance"])
    B_full = float(full_bias_result["bias"])
    B_reduced = float(reduced_bias_result["bias"])

    D_debiased = (D_raw - B_reduced + B_full)

    if clip_negative and D_debiased < 0:
        D_for_p = 0.0
    else:
        D_for_p = D_debiased

    p_value = float(chi2.sf(D_for_p, df=df_removed))

    return {
        "observed_raw_deviance": D_raw,
        "observed_raw_p_value": float(raw_result["p_value"]),

        "bias_full": B_full,
        "bias_reduced": B_reduced,
        "bias_correction": float(-B_reduced + B_full),

        "observed_debiased_deviance": float(D_debiased),
        "observed_debiased_deviance_for_p": float(D_for_p),
        "observed_debiased_p_value": p_value,

        "full_bias_method": full_bias_result["method"],
        "reduced_bias_method": reduced_bias_result["method"],

        "full_used_solver": full_bias_result["used_solver"],
        "reduced_used_solver": reduced_bias_result["used_solver"],

        "full_effective_rank": full_bias_result["effective_rank"],
        "reduced_effective_rank": reduced_bias_result["effective_rank"],

        "full_score_norm": full_bias_result["score_norm"],
        "reduced_score_norm": reduced_bias_result["score_norm"],

        "full_min_info_eigenvalue": full_bias_result["min_info_eigenvalue"],
        "reduced_min_info_eigenvalue": reduced_bias_result["min_info_eigenvalue"],

        "full_original_condition_number": full_bias_result["original_condition_number"],
        "reduced_original_condition_number": reduced_bias_result["original_condition_number"],

        "full_used_condition_number": full_bias_result["used_condition_number"],
        "reduced_used_condition_number": reduced_bias_result["used_condition_number"],

        "full_bias_is_finite": full_bias_result["bias_is_finite"],
        "reduced_bias_is_finite": reduced_bias_result["bias_is_finite"],

        "df": df_removed
    }