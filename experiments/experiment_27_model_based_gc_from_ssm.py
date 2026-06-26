import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_simulator import generate_linear_ssm_varx
from src.ssm.em_varx_ssm_known_c import EMVARXSSMKnownC
from src.varx.varx_gc import varx_gc, ordinary_var_gc_for_comparison
from src.gc.model_based_gc import model_based_pairwise_gc, model_based_gc_matrix
from src.utils.plotting import plot_method_comparison_bar

# Shared configuration
N_SAMPLES = 3000
B_true = np.array([
    [0.50],
    [0.40]
])
Q_true = 0.15 * np.eye(2)
R_true = 0.60 * np.eye(2)
C_true = np.array([
    [1.00, 0.35],
    [0.25, 1.00]
])
u =generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

def run_case(A_true, case_name):
    print('\n'+case_name)

    # Generate latent state and mixed observations
    x_true, y_obs = generate_linear_ssm_varx(A=A_true, B=B_true, u=u, Q=Q_true, R=R_true, C=C_true, random_seed=2)

    # Pseudo-inverse proxy
    C_pinv = np.linalg.pinv(C_true)
    x_proxy = (C_pinv @ y_obs.T).T

    # Fit state-space VARX with known C
    model = EMVARXSSMKnownC(n_states=2, n_inputs=1, C=C_true, ridge=1e-6)
    model.fit(y=y_obs, u=u, max_iter=75, tol=1e-3, verbose=False)
    x_smooth = model.smooth_result["x_smooth"]

    # Existing sample-based GC values
    latent_sample_gc = varx_gc(y=x_true, u=u, source=1, target=0, na=1, nb=1, ridge=0.0)["gc"]
    observed_var_gc = ordinary_var_gc_for_comparison(y_obs, source=1, target=0, order=1)
    observed_varx_gc = varx_gc(y=y_obs, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]
    proxy_varx_gc = varx_gc(y=x_proxy, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]
    smoothed_varx_gc = varx_gc(y=x_smooth, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

    # New model-based GC values
    true_model_gc = model_based_pairwise_gc(A=A_true, Q=Q_true, source=1, target=0)["gc"]
    estimated_model_gc = model_based_pairwise_gc(A=model.A, Q=model.Q, source=1, target=0)["gc"]
    true_gc_matrix = model_based_gc_matrix(A_true, Q_true)
    estimated_gc_matrix = model_based_gc_matrix(model.A, model.Q)

    # Print results
    print("\nTrue A")
    print(A_true)

    print("\nEstimated A")
    print(model.A)

    print("\nTrue Q")
    print(Q_true)

    print("\nEstimated Q")
    print(model.Q)

    print("\nSample-based GC values Y -> X")
    print("Latent sample VARX GC:", latent_sample_gc)
    print("Observed ordinary VAR GC:", observed_var_gc)
    print("Observed VARX GC:", observed_varx_gc)
    print("Proxy VARX GC:", proxy_varx_gc)
    print("SSM smooth VARX GC:", smoothed_varx_gc)

    print("\nModel-based GC values Y -> X")
    print("True model-based GC:", true_model_gc)
    print("Estimated model-based GC:", estimated_model_gc)

    print("\nTrue model-based GC matrix")
    print(true_gc_matrix)

    print("\nEstimated model-based GC matrix")
    print(estimated_gc_matrix)

    print("\nParameter-level interpretation")
    print("True A[0,1] Y->X:", A_true[0, 1])
    print("Estimated A[0,1] Y->X:", model.A[0, 1])
    print("True A[1,0] X->Y:", A_true[1, 0])
    print("Estimated A[1,0] X->Y:", model.A[1, 0])

    # Plot comparison
    labels = ["latent", "obs VAR", "obs VARX", "proxy", "SSM smooth", "true model", "est model"]
    values = [latent_sample_gc, observed_var_gc, observed_varx_gc, proxy_varx_gc, smoothed_varx_gc, true_model_gc, estimated_model_gc]

    plot_method_comparison_bar(labels=labels, values=values, title=f"Y - X GC Comparison {case_name}", ylabel="GC")

# Case 1: No true endogenous Y -> X
A_no_link = np.array([
    [0.60, 0.00],
    [0.00, 0.55]
])

run_case(A_true=A_no_link, case_name="Case 1 - No true endogenous Y - X")

# Case 2: True endogenous Y -> X
A_true_link = np.array([
    [0.60, 0.30],
    [0.00, 0.55]
])

run_case(A_true=A_true_link, case_name="Case 2 - True endogenous Y - X")