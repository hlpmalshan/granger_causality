import numpy as np

class AdaptiveVARRLSP:
    def __init__(self, n_vars, order, forgetting_factor=0.995):
        self.n_vars = n_vars
        self.order = order
        self.lambda_ = forgetting_factor

        n_features = n_vars * order

        self.theta = np.zeros((n_vars, n_features))
        self.P = 1000 * np.eye(n_features)
        self.history = []

    def update(self, lagged_vector, current_sample):
        phi = lagged_vector.reshape(-1, 1)
        y = current_sample.reshape(-1, 1)
        gain = (self.P @ phi) / (self.lambda_ + phi.T @ self.P @ phi)

        prediction_error = y - self.theta @ phi
        self.theta += prediction_error @ gain.T
        self.P = (self.P - gain @ phi.T @ self.P) / self.lambda_

        self.history.append(self.theta.copy())        

    def fit(self, data):
        T = len(data)
        for t in range(self.order, T):
            lagged = []
            for lag in range(1, self.order + 1):
                lagged.append(data[t-lag])

            lagged = np.concatenate(lagged)
            self.update(lagged, data[t])

        return np.asarray(self.history)