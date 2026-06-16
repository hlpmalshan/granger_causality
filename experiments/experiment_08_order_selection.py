import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.order_selection import select_var_order, best_order
from src.utils.plotting import plot_information_criteria

A1 = np.array([
    [0.5, 0.3],
    [0.0, 1.2]
])

A2 = np.array([
    [0.0, 0.0],
    [0.0, -0.5]
])

data = generate_var_data([A1, A2], np.eye(2), 10000)

results = select_var_order(data, max_order=10)
print("Best AIC:", best_order(results, "aic"))
print("Best BIC:", best_order(results, "bic"))
plot_information_criteria(results)