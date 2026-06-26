import numpy as np

from src.varx.varx_generator import generate_colored_input, generate_varx_data
from src.varx.varx_gc import varx_gc, ordinary_var_gc_for_comparison

# Configuration
N_SAMPLES = 10000
na = 1
nb = 3

# Generate exogenous input
u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

# Simulate system with NO true endogenous Y -> X link
# y[:, 0] = X
# y[:, 1] = Y

# Both are driven by u. Y receives u earlier. X receives delayed u.
# Therefore ordinary VAR may incorrectly infer Y -> X
# because Y carries information about past u.

# VARX should remove this spurious effect because u is modeled.

A1 = np.array([
    [0.5, 0.3],
    [0.0, 0.5]
])

B0 = np.array([
    [0.0],
    [0.9]
])

B1 = np.array([
    [0.0],
    [0.0]
])

B2 = np.array([
    [0.9],
    [0.0]
])

noise_cov = 0.5 * np.eye(2)

y = generate_varx_data(A_matrices=[A1], B_matrices=[B0, B1, B2], u=u, noise_cov=noise_cov, random_seed=2)

# Ordinary VAR GC: ignores exogenous input
ordinary_gc_y_to_x = ordinary_var_gc_for_comparison(y, source=1, target=0, order=1)
ordinary_gc_x_to_y = ordinary_var_gc_for_comparison(y, source=0, target=1, order=1)

# VARX GC: includes exogenous input
varx_result_y_to_x = varx_gc(y=y, u=u, source=1, target=0, na=na, nb=nb, ridge=0.0)
varx_result_x_to_y = varx_gc(y=y, u=u, source=0, target=1, na=na, nb=nb, ridge=0.0)

print("\nGround truth:")
print("Y -> X endogenous link: present")
print("X -> Y endogenous link: absent")
print("u drives both X and Y with different delays")

print("\nOrdinary VAR GC, ignoring u:")
print("Y -> X:", ordinary_gc_y_to_x)
print("X -> Y:", ordinary_gc_x_to_y)

print("\nVARX GC, including u:")
print("Y -> X:", varx_result_y_to_x["gc"])
print("X -> Y:", varx_result_x_to_y["gc"])

print("\nResidual variances for Y -> X test:")
print("VARX full sigma:", varx_result_y_to_x["sigma_full"])
print("VARX reduced sigma:", varx_result_y_to_x["sigma_reduced"])