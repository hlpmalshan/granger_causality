import numpy as np

from src.ssm.kalman import kalman_filter_varx, rts_smoother

# EM estimation for state-space VARX identity observation
# x_t = A x_{t-1} + B u_t + w_t
# y_t = x_t + v_t
# where: w_t ~ N(0, Q) and v_t ~ N(0, R)
class EMVARXSSM:
    def __init__(self, n_states, n_inputs, ridge=1e-6, q_floor=1e-8, r_floor=1e-8):
        self.n_states = n_states
        self.n_inputs = n_inputs

        self.ridge = ridge
        self.q_floor = q_floor
        self.r_floor = r_floor

        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        
        self.log_likelihoods = []
        self.smooth_result = None
        self.filter_result = None      

    # Initialize A and B using OLS on noisy observations
    def initialize_from_observations(self, y, u):
        y = np.asarray(y)
        u = np.asarray(u)

        Y = y[1:]
        X = []
        for t in range(1, len(y)):
            row = np.concatenate([y[t-1], u[t]])
            X.append(row)

        X = np.asarray(X)
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]

        theta = beta.T
        self.A = theta[:, :self.n_states]
        self.B = theta[:, self.n_states:]

        residuals = Y - X @ beta
        Q_init = np.cov(residuals.T, bias=True)
        y_var = np.var(y, axis=0)

        self.Q = 0.5 * Q_init + self.q_floor * np.eye(self.n_states)
        self.R = 0.5 * np.diag(y_var) + self.r_floor * np.eye(self.n_states)

    def e_step(self, y, u):
        self.filter_result = kalman_filter_varx(y=y, u=u, A=self.A, B=self.B, Q=self.Q, R=self.R, C=np.eye(self.n_states))
        self.smooth_result = rts_smoother(self.filter_result, A=self.A)

        return self.smooth_result
    
    # M-step using smoothed posterior sufficient statistics
    def m_step(self, y, u):
        y = np.asarray(y)
        u = np.asarray(u)

        mu = self.smooth_result["x_smooth"]
        P = self.smooth_result["P_smooth"]
        P_lag = self.smooth_result["P_lag_one"]

        T = len(y)

        n = self.n_states
        m = self.n_inputs
        n_features = n + m

        S_xphi = np.zeros((n, n_features))
        S_phiphi = np.zeros((n_features, n_features))
        S_xx = np.zeros((n, n))
        for t in range(1, T):
            mu_t = mu[t]
            mu_prev = mu[t-1]
            u_t = u[t]

            Exx_t = P[t] + np.outer(mu_t, mu_t)
            Exprevprev = P[t-1] + np.outer(mu_prev, mu_prev)
            Ext_prev = (P_lag[t] + np.outer(mu_t, mu_prev))
            E_x_phi = np.hstack([Ext_prev, np.outer(mu_t, u_t)])
            E_phi_phi = np.block([
                [Exprevprev, np.outer(mu_prev, u_t)],
                [np.outer(u_t, mu_prev), np.outer(u_t, u_t)]
            ])

            S_xphi += E_x_phi
            S_phiphi += E_phi_phi
            S_xx += Exx_t

        theta = S_xphi @ np.linalg.pinv(S_phiphi + self.ridge * np.eye(n_features))

        self.A = theta[:, :n]
        self.B = theta[:, n:]

        n_transitions = T - 1

        Q = (S_xx - theta @ S_xphi.T) / n_transitions
        Q = 0.5 * (Q + Q.T)
        Q += self.q_floor * np.eye(n)
        self.Q = Q

        R = np.zeros((n, n))

        for t in range(T):
            err = (y[t] - mu[t])
            R += np.outer(err, err) + P[t]

        R /= T
        R = 0.5 * (R + R.T)
        R += self.r_floor * np.eye(n)
        self.R = R

    def fit(self, y, u, max_iter=50, tol=1e-4, verbose=True):
        y = np.asarray(y)
        u = np.asarray(u)

        self.initialize_from_observations(y, u)
        
        previous_ll = -np.inf
        for iteration in range(max_iter):
            self.e_step(y, u)
            ll = self.filter_result["log_likelihood"]
            self.log_likelihoods.append(ll)

            self.m_step(y, u)
            improvement = ll - previous_ll

            if verbose:
                print(
                    f"EM iter {iteration:03d} | "
                    f"log-likelihood = {ll:.3f} | "
                    f"improvement = {improvement:.6f}"
                )
            if (iteration > 0 and abs(improvement) < tol):
                break

            previous_ll = ll

        # Final E-step using final parameters
        self.e_step(y, u)

        return self
    