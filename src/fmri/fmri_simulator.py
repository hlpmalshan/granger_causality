import numpy as np

from src.varx.varx_generator import generate_varx_data
from src.fmri.hrf import spm_hrf, convolve_with_hrf

# Downsample by integer factor
def downsample(data, factor):
    return data[::factor]

# Channel-wise z-scoring
def zscore(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)

    return (data - mean) / (std + 1e-12)

# Simulate fMRI-like observations from latent VARX dynamics.

#     Steps:
#         1. Generate latent neural VARX state x_t.
#         2. Convolve each channel with HRF.
#         3. Downsample to fMRI TR.
#         4. Add observation noise.
#         5. Downsample exogenous input to match fMRI observations.
def simulate_fmri_from_varx(
        A_matrices,
        B_matrices,
        u_highres,
        neural_noise_cov,
        tr_neural=0.1,
        tr_fmri=1.0,
        hrf_duration=32.0,
        bold_noise_std=0.2,
        random_seed=123
):
    rng = np.random.default_rng(random_seed)
    
    if int(tr_fmri % tr_neural) != 0:
        raise ValueError(
            "For this simple simulator, tr_fmri must be an integer multiple of tr_neural."
        )
    
    downsample_factor = int(tr_fmri / tr_neural)

    x_neural = generate_varx_data(
        A_matrices=A_matrices,
        B_matrices=B_matrices,
        u=u_highres,
        noise_cov=neural_noise_cov,
        random_seed=random_seed
    )

    hrf = spm_hrf(tr=tr_neural, duration=hrf_duration)

    bold_highres = convolve_with_hrf(x_neural, hrf)

    y_fmri = downsample(bold_highres, downsample_factor)
    u_fmri = downsample(u_highres, downsample_factor)

    obs_noise = rng.normal(loc=0.0, scale=bold_noise_std, size=y_fmri.shape)

    y_fmri = y_fmri + obs_noise

    return {
        "x_neural": zscore(x_neural),
        "bold_highres": zscore(bold_highres),
        "y_fmri": zscore(y_fmri),
        "u_fmri": zscore(u_fmri),
        "hrf": hrf,
        "downsample_factor": downsample_factor
    }