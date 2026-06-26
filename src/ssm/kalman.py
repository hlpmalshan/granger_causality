import numpy as np

# Kalman filter for:
# x_t = A x_{t-1} + B u_t + w_t
# y_t = C x_t + D u_t + v_t
# Returns prediction and filtering results needed for RTS smoothing.
def kalman_filter_varx(y, u, A, B, Q, R, C=None, D=None, x0_mean=None, x0_cov=None):
    y = np.asarray(y)
    u = np.asarray(u)

    n_samples, n_obs = y.shape
    n_states = A.shape[0]
    n_inputs = u.shape[1]

    if C is None: C = np.eye(n_states)
    if D is None: D = np.zeros((n_obs, n_inputs))
    if x0_mean is None: x0_mean = np.zeros(n_states)
    if x0_cov is None: x0_cov = 100 * np.eye(n_states)

    I = np.eye(n_states)

    x_pred = np.zeros((n_samples, n_states))
    P_pred = np.zeros((n_samples, n_states, n_states))

    x_filt = np.zeros((n_samples, n_states))
    P_filt = np.zeros((n_samples, n_states, n_states))

    log_likelihood = 0.0
    for t in range(n_samples):
        if t == 0:
            x_pred[t] = x0_mean
            P_pred[t] = x0_cov

        else:
            x_pred[t] = A @ x_filt[t-1] + B @ u[t]
            P_pred[t] = A @ P_filt[t-1] @ A.T + Q

        innovation = y[t] - C @ x_pred[t] - D @ u[t]
        S = C @ P_pred[t] @ C.T + R
        K = np.linalg.solve(S.T, (P_pred[t] @ C.T).T)
        x_filt[t] = x_pred[t] + K @ innovation

        # Joseph form for numerical stability
        KC = K @ C
        P_filt[t] = (I - KC) @ P_pred[t] @ (I - KC).T + K @ R @ K.T

        sign, logdet = np.linalg.slogdet(S)
        if sign > 0:
            log_likelihood += -0.5 * (n_obs * np.log(2*np.pi) + logdet + innovation.T @ np.linalg.solve(S, innovation))

    return {
        "x_pred": x_pred,
        "P_pred": P_pred,
        "x_filt": x_filt,
        "P_filt": P_filt,
        "log_likelihood": log_likelihood
    }

# Rauch-Tung-Striebel smoother
def rts_smoother(filter_result, A):
    x_pred = filter_result["x_pred"]
    P_pred = filter_result["P_pred"]
    x_filt = filter_result["x_filt"]
    P_filt = filter_result["P_filt"]

    n_samples, n_states = x_filt.shape

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)

    smoother_gains = np.zeros((n_samples-1, n_states, n_states))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(n_samples-2, -1, -1):
        J = P_filt[t] @ A.T @ np.linalg.pinv(P_pred[t + 1])
        smoother_gains[t] = J
        x_smooth[t] = x_filt[t] + J @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    P_lag_one = np.zeros_like(P_smooth)
    for t in range(1, n_samples):
        P_lag_one[t] = P_smooth[t] @ smoother_gains[t-1].T

    return {
        "x_smooth": x_smooth,
        "P_smooth": P_smooth,
        "P_lag_one": P_lag_one,
        "smoother_gains": smoother_gains
    }