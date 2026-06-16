import numpy as np

from src.simulation.var_generator import generate_var_data
from src.models.adaptive_var_rls_p import AdaptiveVARRLSP
from src.utils.adaptive_utils import extract_var_coefficients
from src.utils.plotting import plot_parameter_convergence

r = 0.95
omega0 = 0.3 * np.pi 

a1 = 2 * r * np.cos(omega0)
a2 = -r**2

A1_true = np.array([
    [0.5, 0.3],
    [0.0, a1]
])
A2_true = np.array([
    [0.0, 0.0],
    [0.0, a2]
])

data = generate_var_data([A1_true, A2_true], np.eye(2), 10000)

rls = AdaptiveVARRLSP(n_vars=2, order=2, forgetting_factor=0.999)
history = rls.fit(data)

theta_final = history[-1]

A1_est, A2_est = extract_var_coefficients(theta_final, n_vars=2, order=2)

print("\nTrue A1")
print(A1_true)

print("\nEstimated A1")
print(A1_est)

print("\nTrue A2")
print(A2_true)

print("\nEstimated A2")
print(A2_est)

plot_parameter_convergence(
    history,
    row=0,
    col=1,
    title="A1(0,1) Convergence"
)