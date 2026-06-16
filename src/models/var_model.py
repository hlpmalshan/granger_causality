import numpy as np

class VARModel:
    def __init__(self, order):
        self.order = order
        self.coef_matrices = None
        self.residual_covariance = None
        self.residuals = None
        self.n_vars = None
        self.n_samples = None

    def fit(self, data):
        T, n_vars = data.shape
        self.n_vars = n_vars
        self.n_samples = T
        p = self.order
        Y = data[p:]
        X = []

        for t in range(p, T):
            row = []
            for lag in range(1, p+1):
                row.extend(data[t-lag])
            
            X.append(row)

        X = np.asarray(X)
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ B

        Sigma = np.cov(
            residuals.T,
            bias=True
        )

        coef_matrices = []

        for lag in range(p):
            start = lag * n_vars
            stop = (lag + 1)*n_vars

            Ak = B[start:stop].T
            coef_matrices.append(Ak)

        self.coef_matrices = coef_matrices
        self.residual_covariance = Sigma
        self.residuals = residuals

        return self