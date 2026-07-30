import numpy as np

from src.ssm.em_varx_p_known_c import EMVARXPSSMKnownC, _regularize_covariance
from src.ssm.ssm_varx_p_simulator import make_exogenous_regressor

# EM estimator for latent VARX(p) with known C and optional zero constraints on latent endogenous coefficients.
# A zero constraint has the form:
# {
#     "target": target_index,
#     "source": source_index,
#     "lags": "all"
# }
# Example for Y -> X in a two-variable system:
# source = 1
# target = 0
# A_k[0, 1] = 0 for all k.
class EMVARXPSSMKnownCConstrained(EMVARXPSSMKnownC):
    def __init__(self, *args, zero_constraints=None, **kwargs):
        super().__init__(*args, **kwargs)

        if zero_constraints is None:
            zero_constraints = []

        self.zero_constraints = zero_constraints

    # Return columns of H that must be zero for a given target row.
    # H columns are ordered as: [x_lag1, x_lag2, ..., x_lagna, u_lag0, ..., u_lag(nb-1)]
    def _constraint_columns_for_target(self, target, n_latent):
        forbidden = set()

        for constraint in self.zero_constraints:
            c_target = constraint["target"]
            c_source = constraint["source"]
            c_lags = constraint.get("lags", "all")

            if c_target != target:
                continue

            if c_lags == "all":
                lag_list = range(1, self.na + 1)
            else:
                lag_list = list(c_lags)

            for lag in lag_list:
                if lag < 1 or lag > self.na:
                    raise ValueError(f"Invalid lag {lag}. Valid lags are 1,...,{self.na}.")

                column = ((lag - 1) * n_latent + c_source)
                forbidden.add(column)

        return sorted(forbidden)

    # Explicitly enforce A_k[target, source] = 0.
    def _apply_zero_constraints_to_A_matrices(self):
        if self.A_matrices is None:
            return

        for constraint in self.zero_constraints:

            target = constraint["target"]
            source = constraint["source"]
            lags = constraint.get("lags", "all")

            if lags == "all":
                lag_list = range(1, self.na + 1)
            else:
                lag_list = list(lags)

            for lag in lag_list:
                self.A_matrices[lag - 1][target, source] = 0.0

    # Use parent initialization, then enforce constraints.
    def _initialize_from_proxy(self, y, u):
        super()._initialize_from_proxy(y=y, u=u)

        self._apply_zero_constraints_to_A_matrices()
        self._refresh_companion_matrices()

    # M-step with row-wise linear constraints.
    # Each target equation is estimated separately so that selected source-history coefficients can be fixed to zero.
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

        # Row-wise constrained update for H.
        H = np.zeros((n_latent, z_dim))
        Gamma = np.diag(np.diag(S_zz))

        for target in range(n_latent):
            forbidden = self._constraint_columns_for_target(target=target, n_latent=n_latent)
            allowed = [col for col in range(z_dim) if col not in forbidden]

            S_zz_allowed = S_zz[np.ix_(allowed, allowed)]
            S_xz_allowed = S_xz[target, allowed]
            Gamma_allowed = Gamma[np.ix_(allowed, allowed)]
            
            h_allowed = (S_xz_allowed @ np.linalg.pinv(S_zz_allowed + self.ridge_m_step * Gamma_allowed))

            H[target, allowed] = h_allowed

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

        self.A_matrices = A_matrices
        self.B_matrices = B_matrices
        self.Q = Q

        self._apply_zero_constraints_to_A_matrices()

        # Observation covariance update.
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

        self._refresh_companion_matrices()