import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.var_model import VARModel
from src.utils.var_diagnostics import check_stability, compute_aic, compute_bic, count_var_parameters, residual_autocorrelation
from src.utils.plotting import plot_residual_acf

A1 = np.array([
    [0.5, 0.4],
    [0.0, 0.8]
])

data = generate_var_data([A1], np.eye(2), 10000)

model = VARModel(order=1)
model.fit(data)

# Stability
stability = check_stability(model.coef_matrices)
print(stability)

# Information Criteria
n_params = count_var_parameters(n_vars=2, order=1)
aic = compute_aic(model.residual_covariance, n_params, len(data))
bic = compute_bic(model.residual_covariance, n_params, len(data))
print("AIC:", aic)
print("BIC:", bic)

# Residual ACF
acf = residual_autocorrelation(model.residuals)
plot_residual_acf(acf[0], title="Residual ACF Variable X")
plot_residual_acf(acf[1], title="Residual ACF Variable Y")

