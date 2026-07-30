import numpy as np

from src.ssm.ssm_varx_p_simulator import make_exogenous_regressor

def _symmetrize(P):
    return 0.5 * (P + P.T)

def _safe_logdet_spd(S, jitter=1e-10, max_attempts=8):
    S = np.asarray(S, dtype=float)
    S = _symmetrize(S)

    n = S.shape[0]
    scale = np.trace(S) / max(n, 1)

    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    for attempt in range(max_attempts):
        added = jitter * (10 ** attempt) * scale
        candidate = S + added * np.eye(n)

        sign, logdet = np.linalg.slogdet(candidate)

        if sign > 0 and np.isfinite(logdet):
            return candidate, float(logdet)
        
    raise np.linalg.LinAlgError("Could not stabilize innovation covariance.")

# Kalman filter for companion-form latent VARX(p).
# State model: s_t = F s_{t-1} + G phi_t + eta_t
# Observation model: y_t = C_aug s_t + D u_t + v_t
# where: phi_t = [u_t, u_{t-1}, ..., u_{t-nb+1}].
def kalman_filter_varx_p_companion(y, u, F, G, Q_aug, R, C_aug, nb, D=None, s0_mean=None, s0_cov=None):
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    if u.ndim == 1:
        u = u.reshape(-1, 1)

    T, n_obs = y.shape
    n_aug = F.shape[0]
    n_inputs = u.shape[1]

    if D is None:
        D = np.zeros((n_obs, n_inputs))
    else:
        D = np.asarray(D, dtype=float)

    if s0_mean is None:
        s0_mean = np.zeros(n_aug)
    else:
        s0_mean = np.asarray(s0_mean, dtype=float)

    if s0_cov is None:
        s0_cov = 10.0 * np.eye(n_aug)
    else:
        s0_cov = np.asarray(s0_cov, dtype=float)

    x_pred = np.zeros((T, n_aug))
    P_pred = np.zeros((T, n_aug, n_aug))

    x_filt = np.zeros((T, n_aug))
    P_filt = np.zeros((T, n_aug, n_aug))

    innovations = np.zeros((T, n_obs))
    innovation_covariances = np.zeros((T, n_obs, n_obs))
    kalman_gains = np.zeros((T, n_aug, n_obs))

    log_likelihood = 0.0

    I = np.eye(n_aug)

    previous_mean = s0_mean.copy()
    previous_cov = s0_cov.copy()

    for t in range(T):
        phi_t = make_exogenous_regressor(u=u, t=t, nb=nb)
        if t == 0:
            mean_pred = previous_mean + G @ phi_t
            cov_pred = previous_cov + Q_aug
        else:
            mean_pred = F @ previous_mean + G @ phi_t
            cov_pred = F @ previous_cov @ F.T + Q_aug

        cov_pred = _symmetrize(cov_pred)

        y_offset = D @ u[t]
        innovation = y[t] - y_offset - C_aug @ mean_pred

        S = C_aug @ cov_pred @ C_aug.T + R
        S, logdet_S = _safe_logdet_spd(S)

        K = cov_pred @ C_aug.T @ np.linalg.pinv(S)
        
        mean_filt = mean_pred + K @ innovation

        # Joseph form for numerical stability
        cov_filt = (I - K @ C_aug) @ cov_pred @ (I - K @ C_aug).T + K @ R @ K.T
        cov_filt = _symmetrize(cov_filt)

        quad = float(innovation.T @ np.linalg.pinv(S) @ innovation)
        log_likelihood += -0.5 * (n_obs * np.log(2.0 * np.pi) + logdet_S + quad)

        x_pred[t] = mean_pred
        P_pred[t] = cov_pred

        x_filt[t] = mean_filt
        P_filt[t] = cov_filt

        innovations[t] = innovation
        innovation_covariances[t] = S
        kalman_gains[t] = K

        previous_mean = mean_filt
        previous_cov = cov_filt

    return {
        "x_pred": x_pred,
        "P_pred": P_pred,
        "x_filt": x_filt,
        "P_filt": P_filt,
        "innovations": innovations,
        "innovation_covariances": innovation_covariances,
        "kalman_gains": kalman_gains,
        "log_likelihood": log_likelihood
    }

# Rauch-Tung-Striebel smoother for companion-form state.
# Also returns an approximate lag-one smoothed covariance:
# P_lag_one[t] ≈ Cov(s_t, s_{t-1} | y_{1:T})
# This is needed for EM sufficient statistics.
def rts_smoother_varx_p_companion(filter_result, F):
    x_pred = filter_result["x_pred"]
    P_pred = filter_result["P_pred"]
    x_filt = filter_result["x_filt"]
    P_filt = filter_result["P_filt"]

    T, n_aug = x_filt.shape

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    smoother_gains = np.zeros((T - 1, n_aug, n_aug))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T-2, -1, -1):
        J = P_filt[t] @ F.T @ np.linalg.pinv(P_pred[t+1])
        smoother_gains[t] = J

        x_smooth[t] = x_filt[t] + J @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t+1] - P_pred[t+1]) @ J.T

        P_smooth[t] = _symmetrize(P_smooth[t])

    P_lag_one = np.zeros_like(P_smooth)
    for t in range(1, T):
        P_lag_one[t] = P_smooth[t] @ smoother_gains[t-1].T

    return {
        "x_smooth": x_smooth,
        "P_smooth": P_smooth,
        "P_lag_one": P_lag_one,
        "smoother_gains": smoother_gains
    }

# Convenience wrapper: Kalman filter + RTS smoother
def kalman_smooth_varx_p_companion(y, u, F, G, Q_aug, R, C_aug, nb, D=None, s0_mean=None, s0_cov=None):
    filter_result = kalman_filter_varx_p_companion(y=y, u=u, F=F, G=G, Q_aug=Q_aug, R=R, C_aug=C_aug, nb=nb, D=D, s0_mean=s0_mean, s0_cov=s0_cov)

    smoother_result = rts_smoother_varx_p_companion(filter_result=filter_result, F=F)

    return {
        "filter": filter_result,
        "smoother": smoother_result,
        "log_likelihood": filter_result["log_likelihood"]
    }

# Extract x_t from companion state s_t.
# s_t = [x_t, x_{t-1}, ..., x_{t-p+1}]
def extract_current_latent_state(companion_states, n_latent):
    return companion_states[:, :n_latent]