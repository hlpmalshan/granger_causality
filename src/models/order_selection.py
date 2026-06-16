import numpy as np

from src.models.var_model import VARModel
from src.utils.var_diagnostics import compute_aic, compute_bic, count_var_parameters

def select_var_order(data, max_order=20):
    n_vars = data.shape[1]
    results = []

    for p in range(1, max_order+1):
        model = VARModel(order=p)
        model.fit(data)

        n_params = count_var_parameters(n_vars, p)

        aic = compute_aic(model.residual_covariance, n_params, len(data))
        bic = compute_bic(model.residual_covariance, n_params, len(data))

        results.append({"order": p, "aic": aic, "bic": bic})

    return results

def best_order(results, criterion="bic"):
    values = [ r[criterion] for r in results ]
    
    idx = np.argmin(values)

    return results[idx]["order"]