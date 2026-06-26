import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.ssm.ssm_varx_simulator import generate_linear_ssm_varx
from src.ssm.em_varx_ssm_known_c import EMVARXSSMKnownC
from src.varx.varx_gc import varx_gc, ordinary_var_gc_for_comparison
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

u = generate_colored_input(n_samples=N_SAMPLES, ar_coeff=0.95, noise_std=1.0, random_seed=1)

def run_case(A_true, case_name):
    print(case_name)

    # Generate latent state and mixed observations
    x_true, y_obs = generate_linear_ssm_varx(
        A=A_true,
        B=B_true,
        u=u,
        Q=Q_true,
        R=R_true,
        C=C_true,
        random_seed=2
    )

    # Pseudo-inverse proxy
    C_pinv = np.linalg.pinv(C_true)
    x_proxy = (C_pinv @ y_obs.T).T

    # Fit state-space VARX with known C
    model = EMVARXSSMKnownC(n_states=2, n_inputs=1, C=C_true, ridge=1e-6)
    model.fit(y=y_obs, u=u, max_iter=75, tol=1e-3, verbose=False)
    x_smooth = model.smooth_result["x_smooth"]

    # GC estimates
    #1. Ground truth latent data
    true_latent_gc = varx_gc(y=x_true, u=u, source=1, target=0, na=1, nb=1, ridge=0.0)["gc"]

    #2. Ordinary VAR on observed y, ignores u
    observed_var_gc = ordinary_var_gc_for_comparison(y_obs, source=1, target=0, order=1)

    #3. VARX directly on observed y
    observed_varx_gc = varx_gc(y=y_obs, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

    #4. VARX on pseudo-inverse source proxy
    proxy_varx_gc = varx_gc(y=x_proxy, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

    #5. VARX on smoothed latent posterior mean
    smoothed_varx_gc = varx_gc(y=x_smooth, u=u, source=1, target=0, na=1, nb=1, ridge=1e-3)["gc"]

    # MSE estimates
    proxy_mse = np.mean((x_proxy - x_true)**2)
    smooth_mse = np.mean((x_smooth - x_true)**2)

    # Print results
    print("\nTrue A")
    print(A_true)

    print("\nEstimated A from state-space VARX")
    print(model.A)

    print("\nTrue B")
    print(B_true)

    print("\nEstimated B from state-space VARX")
    print(model.B)

    print("\nLatent recovery")
    print("Pseudo-inverse proxy MSE:", proxy_mse)
    print("State-space smoothed MSE:", smooth_mse)

    print("\nY -> X estimates")
    print("Ground-truth latent VARX GC:", true_latent_gc)
    print("Observed ordinary VAR GC:", observed_var_gc)
    print("Observed VARX GC:", observed_varx_gc)
    print("Pseudo-inverse proxy VARX GC:", proxy_varx_gc)
    print("State-space smoothed VARX GC:", smoothed_varx_gc)

    print("\nParameter-level interpretation")
    print("True A[0,1] Y->X:", A_true[0, 1])
    print("Estimated A[0,1] Y->X:", model.A[0, 1])
    print("True A[1,0] X->Y:", A_true[1, 0])
    print("Estimated A[1,0] X->Y:", model.A[1, 0])

    # Plot method comparison
    labels = ["latent", "obs VAR", "obs VARX", "proxy VARX", "SSM smooth"]
    values = [true_latent_gc, observed_var_gc, observed_varx_gc, proxy_varx_gc, smoothed_varx_gc]
    plot_method_comparison_bar(labels=labels, values=values, title=f"Y - X GC Comparison {case_name}", ylabel="GC")

    return {
        "model": model,
        "x_true": x_true,
        "y_obs": y_obs,
        "x_proxy": x_proxy,
        "x_smooth": x_smooth,
        "gc_values": values
    }

# Case 1: No true endogenous Y -> X
A_no_link = np.array([
    [0.60, 0.00],
    [0.00, 0.55]
])

case_1 = run_case(A_true=A_no_link, case_name="Case 1- No true endogenous Y - X")

# Case 2: True endogenous Y -> X
A_true_link = np.array([
    [0.60, 0.30],
    [0.00, 0.55]
])

case_2 = run_case(A_true=A_true_link, case_name="Case 2- True endogenous Y - X")