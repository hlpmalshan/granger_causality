import numpy as np

from src.ssm.ssm_varx_p_simulator import build_var_companion_matrix, build_varx_companion_input_matrix, build_companion_process_covariance, build_companion_observation_matrix, make_exogenous_regressor, var_companion_spectral_radius
from src.ssm.kalman_varx_p import kalman_smooth_varx_p_companion, extract_current_latent_state
from src.varx.varx_order_selection import fit_varx_multitarget

def _symmetrize(M):
    return 0.5 * (M + M.T)

def _regularize_covariance(M, epsilon=1e-6):
    M =_symmetrize(M)
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)

    if min_eig < epsilon:
        M = M + (epsilon - min_eig) * np.eye(M.shape[0])

    return _symmetrize(M)

# EM estimator for latent VARX(p) state-space model with known C.
# Latent dynamics: 
# x_t = A1 x_{t-1} + ... + Ap x_{t-p} + B0 u_t + ... + Bq u_{t-q} + w_t where w_t ~ N(0, Q)
# Observation:
# y_t = C x_t + D u_t + v_t where v_t ~ N(0, R)
# The state-space implementation uses the companion state:
# s_t = [x_t, x_{t-1}, ..., x_{t-p+1}].
class EMVARXPSSMKnownC:
    def __init__(self, na, nb, C, D=None, max_iter=50, tol=1e-4, ridge_m_step=1e-6, covariance_floor=1e-6, R_init=None, Q_init=None, estimate_R=True, R_floor=1e-3, verbose=False):
        self.na = na
        self.nb = nb
        self.C = np.asarray(C, dtype=float)
        self.D = D
        self.max_iter = max_iter
        self.tol = tol
        self.ridge_m_step = ridge_m_step
        self.covariance_floor = covariance_floor

        self.R_init = R_init
        self.Q_init = Q_init
        self.estimate_R = estimate_R
        self.R_floor = R_floor

        self.verbose = verbose

        self.A_matrices = None
        self.B_matrices = None
        self.Q = None
        self.R = None

        self.F = None
        self.G = None
        self.Q_aug = None
        self.C_aug = None

        self.log_likelihoods = []
        self.smooth_result = None

    # Initialize latent states using pseudo-inverse of C, then fit a directly observed VARX(p) to the proxy states.
    def _initialize_from_proxy(self, y, u):
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.ndim == 1:
            u = u.reshape(-1, 1)

        T, n_obs = y.shape
        n_latent = self.C.shape[1]
        n_inputs = u.shape[1]

        if self.D is None:
            D = np.zeros((n_obs, n_inputs))
        else:
            D = np.asarray(self.D, dtype=float)

        self.D = D

        y_minus_du = y - u @ D.T
        
        C_pinv = np.linalg.pinv(self.C)
        x_proxy = y_minus_du @ C_pinv.T

        fit = fit_varx_multitarget(
            y=x_proxy, u=u, na=self.na, nb=self.nb, gamma=0.0, penalty="diag", include_intercept=False, start_lag=max(self.na, self.nb-1)
        )
        beta = fit["beta"]

        A_matrices = []
        B_matrices = []

        cursor = 0

        for _ in range(self.na):
            block = beta[cursor:cursor + n_latent, :]
            A_matrices.append(block.T)
            cursor += n_latent

        for _ in range(self.nb):
            block = beta[cursor:cursor + n_inputs, :]
            B_matrices.append(block.T)
            cursor += n_inputs

        Q = fit["residual_covariance"]
        '''
        residual_obs = y_minus_du - x_proxy @ self.C.T
        R = (residual_obs.T @ residual_obs) / T
        '''

        if self.Q_init is not None:
            Q = np.asarray(self.Q_init, dtype=float)

        if self.R_init is not None:
            R = np.asarray(self.R_init, dtype=float)
        else:
            # Important:
            # Do not initialize R from y - C C^dagger y,
            # because that is nearly zero when C is square/invertible.
            y_cov = np.cov(y.T, bias=True)
            R = 0.25 * np.diag(np.diag(y_cov))

        self.Q = _regularize_covariance(Q, epsilon=self.covariance_floor)
        self.R = _regularize_covariance(R, epsilon=self.R_floor)

        self.A_matrices = A_matrices
        self.B_matrices = B_matrices
        
        '''
        self.Q = _regularize_covariance(Q, epsilon=self.covariance_floor)
        self.R = _regularize_covariance(R, epsilon=self.covariance_floor)
        '''
        self._refresh_companion_matrices()

    def _refresh_companion_matrices(self):
        n_latent = self.C.shape[1]
        self.F = build_var_companion_matrix(self.A_matrices)
        self.G = build_varx_companion_input_matrix(B_matrices=self.B_matrices, n_states=n_latent, p=self.na)
        self.Q_aug = build_companion_process_covariance(Q=self.Q, p=self.na)
        self.C_aug = build_companion_observation_matrix(C=self.C, p=self.na)

    def _e_step(self, y, u):
        result = kalman_smooth_varx_p_companion(
            y=y,
            u=u,
            F=self.F,
            G=self.G,
            Q_aug=self.Q_aug,
            R=self.R,
            C_aug=self.C_aug,
            nb=self.nb,
            D=self.D,
            s0_mean=None,
            s0_cov=None
        )
        self.smooth_result = result

        return result
    
    # M-step using posterior sufficient statistics.
    # We estimate: x_t = H z_t + w_t
    # where: z_t = [s_{t-1}, phi_t]
    # and: H = [A1 A2 ... Ap B0 B1 ... Bq]
    def _m_step(self, y, u, smooth_result):
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.ndim == 1:
            u = u.reshape(-1, 1)

        T = y.shape[0]
        n_latent = self.C.shape[1]
        n_inputs = u.shape[1]
        n_aug = n_latent * self.na
        n_phi = n_inputs * self.nb
        z_dim = n_aug + n_phi

        mu = smooth_result["smoother"]["x_smooth"]
        P = smooth_result["smoother"]["P_smooth"]
        P_lag = smooth_result["smoother"]["P_lag_one"]

        S_zz = np.zeros((z_dim, z_dim))
        S_xz = np.zeros((n_latent, z_dim))
        S_xx = np.zeros((n_latent, n_latent))

        n_dyn = 0
        for t in range(1, T):
            mu_prev = mu[t - 1]
            P_prev = P[t - 1]

            mu_x = mu[t, :n_latent]
            P_x = P[t, :n_latent, :n_latent]

            cov_x_prev = P_lag[t, :n_latent, :]

            phi_t = make_exogenous_regressor(u=u, t=t, nb=self.nb)
            E_prev_prev = (P_prev + np.outer(mu_prev, mu_prev))
            E_x_prev = (cov_x_prev + np.outer(mu_x, mu_prev))
            E_x_x = (P_x + np.outer(mu_x, mu_x))
            
            z_mean = np.concatenate([mu_prev, phi_t])
            E_zz = np.zeros((z_dim, z_dim))

            E_zz[:n_aug, :n_aug] = E_prev_prev
            E_zz[:n_aug, n_aug:] = np.outer(mu_prev, phi_t)
            E_zz[n_aug:, :n_aug] = np.outer(phi_t, mu_prev)
            E_zz[n_aug:, n_aug:] = np.outer(phi_t, phi_t)

            E_xz = np.zeros((n_latent, z_dim))
            E_xz[:, :n_aug] = E_x_prev
            E_xz[:, n_aug:] = np.outer(mu_x, phi_t)

            S_zz += E_zz
            S_xz += E_xz
            S_xx += E_x_x

            n_dyn += 1

        Gamma = np.diag(np.diag(S_zz))

        H = (S_xz  @ np.linalg.pinv(S_zz + self.ridge_m_step * Gamma))
        S_zx = S_xz.T

        Q_num = (S_xx - H @ S_zx - S_xz @ H.T + H @ S_zz @ H.T)
        Q = Q_num / max(n_dyn, 1)
        Q = _regularize_covariance(Q, epsilon=self.covariance_floor)

        A_matrices = []
        B_matrices = []

        cursor = 0

        for _ in range(self.na):
            A_matrices.append(H[:, cursor:cursor + n_latent])
            cursor += n_latent

        for _ in range(self.nb):
            B_matrices.append(H[:, cursor:cursor + n_inputs])
            cursor += n_inputs

        '''
        # Observation covariance update.
        R_num = np.zeros_like(self.R)

        for t in range(T):
            mu_x = mu[t, :n_latent]
            P_x = P[t, :n_latent, :n_latent]

            residual = (y[t] - self.D @ u[t] - self.C @ mu_x)

            R_num += (np.outer(residual, residual) + self.C @ P_x @ self.C.T)

        R = R_num / T
        R = _regularize_covariance(R, epsilon=self.covariance_floor)
        '''

        if self.estimate_R:
            R_num = np.zeros_like(self.R)

            for t in range(T):
                mu_x = mu[t, :n_latent]
                P_x = P[t, :n_latent, :n_latent]

                residual = (y[t] - self.D @ u[t] - self.C @ mu_x)

                R_num += (np.outer(residual, residual) + self.C @ P_x @ self.C.T)

            R = R_num / T
            R = _regularize_covariance(R, epsilon=self.R_floor)
            self.R = R

        self.A_matrices = A_matrices
        self.B_matrices = B_matrices
        self.Q = Q
        # self.R = R

        self._refresh_companion_matrices()

    def fit(self, y, u):
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.ndim == 1:
            u = u.reshape(-1, 1)

        self._initialize_from_proxy(y=y, u=u)
        previous_ll = None

        for iteration in range(self.max_iter):
            smooth_result = self._e_step(y=y, u=u)
            ll = smooth_result["log_likelihood"]
            self.log_likelihoods.append(ll)

            if self.verbose:
                print(
                    f"EM iteration {iteration + 1:03d} | "
                    f"log-likelihood = {ll:.6f} | "
                    f"spectral radius = {self.spectral_radius():.6f}"
                )

            if previous_ll is not None:
                improvement = ll - previous_ll
                scale = abs(previous_ll) + 1e-12

                if abs(improvement) / scale < self.tol:
                    break

            self._m_step(y=y, u=u, smooth_result=smooth_result)
            previous_ll = ll

        # Final E-step using final parameters.
        self.smooth_result = self._e_step(y=y, u=u)

        return self
    
    def smoothed_latent_state(self):
        if self.smooth_result is None:
            raise RuntimeError("Model must be fitted before smoothing.")

        return extract_current_latent_state(
            companion_states=self.smooth_result["smoother"]["x_smooth"],
            n_latent=self.C.shape[1]
        )

    def spectral_radius(self):
        return var_companion_spectral_radius(self.A_matrices)