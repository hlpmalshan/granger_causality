import numpy as np

# Ordiniary least-squares VARX model
# y_t = sum_{k=1}^{na} A_k y_{t-k} + sum_{l=0}^{nb-1} B_l u_{t-l} + e_t
class VARXModel:
    def __init__(self, na, nb, ridge=0.0):
        self.na = na
        self.nb = nb
        self.ridge = ridge

        self.A_matrices = None
        self.B_matrices = None
        self.residuals = None
        self.residual_coavariance = None
        self.beta = None

    # Build regression design matrix
    # target_indices / source_indices determine which endogenous lagged variables are included
    # If source_indices is None, all endogenous variables are included
    def _build_design(
            self,
            y,
            u,
            target_indices=None,
            source_indices=None,
            include_exogenous=True
    ):
        y = np.asarray(y)
        u = np.asarray(u)

        T, n_y = y.shape
        _, n_u = u.shape

        max_lag = max(self.na, self.nb-1)

        if source_indices is None:
            source_indices = list(range(n_y))

        source_indices = list(source_indices)

        Y = y[max_lag:]
        X_rows = []

        for t in range(max_lag, T):
            row = []
            # Endogenous lagged predictors
            for lag in range(1, self.na + 1):
                row.extend(y[t-lag, source_indices])

            # Exogenous lagged predictors
            if include_exogenous:
                for lag in range(self.nb):
                    row.extend(u[t-lag])

            X_rows.append(row)

        X = np.asarray(X_rows)

        return X, Y
    
    def fit(self, y, u, source_indices=None, include_exogenous=True):
        y = np.asarray(y)
        u = np.asarray(u)

        T, n_y = y.shape
        _, n_u = u.shape

        X, Y = self._build_design(y, u, source_indices=source_indices, include_exogenous=include_exogenous)

        if self.ridge > 0:
            XtX = X.T @ X
            XtY = X.T @ Y

            beta = np.linalg.solve(XtX + self.ridge * np.eye(XtX.shape[0]), XtY)

        else:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]

        residuals = Y - X @ beta

        self.beta = beta
        self.residuals = residuals
        self.residual_coavariance = np.cov(residuals.T, bias=True)

        return self
    
    def residual_variance(self, target):
        return np.var(self.residuals[:, target], ddof=0)
    
    