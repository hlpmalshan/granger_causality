import numpy as np

class AdaptiveVARRLS:
    def __init__(self, n_vars, forgetting_factor=0.99):
        self.n_vars = n_vars
        self.lambda_ = forgetting_factor
        self.theta = np.zeros((n_vars, n_vars))
        self.P = 1000 * np.eye(n_vars)
        self.history = []

    def update(self, x_prev, x_curr):
        phi = x_prev.reshape(-1, 1)
        y = x_curr.reshape(-1, 1)

        gain = (self.P @ phi) / (self.lambda_ + phi.T @ self.P @ phi)

        prediction_error = y - self.theta @ phi

        self.theta += prediction_error @ gain.T

        self.P = (self.P - gain @ phi.T @ self.P) / self.lambda_

        self.history.append(self.theta.copy())

    def fit(self, data):
        for t in range(1, len(data)):
            self.update(data[t-1], data[t])

        return np.array(self.history)
    