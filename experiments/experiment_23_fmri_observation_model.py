import numpy as np

from src.varx.varx_generator import generate_colored_input
from src.fmri.fmri_simulator import simulate_fmri_from_varx
from src.varx.varx_gc import varx_gc, ordinary_var_gc_for_comparison
from src.utils.plotting import plot_latent_vs_fmri

# Configuration
N_NEURAL = 20000
TR_NEURAL = 0.1
TR_FMRI = 1

na = 1
# Because fMRI is downsampled, the exogenous history in fMRI samples
# should be longer than in the neural simulation.
nb_fmri = 8     

#Exogenous input 
u_highres = generate_colored_input(
    n_samples=N_NEURAL,
    ar_coeff=0.97,
    noise_std=1.0,
    random_seed=1
)

# Latent VARX system
# X = channel 0
# Y = channel 1
# True endohenous Y -> X link is present.
# Exogenous input also drives both channels.

A1 = np.array([
    [0.5, 0.3],
    [0.0, 0.5]
])

B0 = np.array([
    [0.0],
    [0.8]
])

B1 = np.array([
    [0.0],
    [0.0]
])

B2 = np.array([
    [0.8],
    [0.0]
])

neural_noise_cov = 0.5 * np.eye(2)

# Generate fMRI-like observations
sim = simulate_fmri_from_varx(
    A_matrices=[A1],
    B_matrices=[B0, B1, B2],
    u_highres=u_highres,
    neural_noise_cov=neural_noise_cov,
    tr_neural=TR_NEURAL,
    tr_fmri=TR_FMRI,
    hrf_duration=32.0,
    bold_noise_std=0.2,
    random_seed=2
)

x_neural = sim["x_neural"]
y_fmri = sim["y_fmri"]
u_fmri = sim["u_fmri"]

print("\nShapes")
print("x_neural:", x_neural.shape)
print("y_fmri:", y_fmri.shape)
print("u_fmri:", u_fmri.shape)

# Visual check
plot_latent_vs_fmri(
    x_neural=x_neural,
    y_fmri=y_fmri,
    channel=0,
    downsample_factor=sim["downsample_factor"],
    n_points=500,
    title="Latent Neural State vs fMRI-like Observation X"
)

# GC on latent neural state
latent_var_gc = ordinary_var_gc_for_comparison(x_neural, source=1, target=0, order=1)
latent_varx_gc = varx_gc(
    y=x_neural,
    u=u_highres,
    source=1,
    target=0,
    na=1,
    nb=3,
    ridge=0.0
)["gc"]

# GC on fMRI observations
fmri_var_gc = ordinary_var_gc_for_comparison(y_fmri, source=1, target=0, order=1)
fmri_varx_gc = varx_gc(
    y=y_fmri,
    u=u_fmri,
    source=1,
    target=0,
    na=1,
    nb=nb_fmri,
    ridge=1e-3
)["gc"]

print("\nGround truth:")
print("Latent endogenous Y -> X link: present")
print("Exogenous input drives both X and Y")

print("\nLatent neural data:")
print("Ordinary VAR GC Y -> X:", latent_var_gc)
print("VARX GC Y -> X:", latent_varx_gc)

print("\nfMRI-like observed data:")
print("Ordinary VAR GC Y -> X:", fmri_var_gc)
print("VARX GC Y -> X:", fmri_varx_gc)