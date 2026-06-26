import numpy as np

from scipy.stats import gamma

# Generate and SPM-like canonical HRF
def spm_hrf(
        tr,
        duration=32.0,
        dt=0.1,
        peak_delay=6.0,
        undershoot_delay=16.0,
        peak_dispersion=1.0,
        undershoot_dispersion=1.0,
        p_u_ratio=6.0
):
    time = np.arange(0, duration, dt)

    peak = gamma.pdf(time, peak_delay / peak_dispersion, scale=peak_dispersion)
    undershoot = gamma.pdf(time, undershoot_delay / undershoot_dispersion, scale=undershoot_dispersion)

    hrf_high_res = peak - undershoot / p_u_ratio
    hrf_high_res /= np.sum(hrf_high_res)

    sample_times = np.arange(0, duration, tr)

    hrf = np.interp(sample_times, time, hrf_high_res)
    hrf /= np.sum(hrf)

    return hrf

# Convolve each channel with HRF
def convolve_with_hrf(neural_data, hrf):
    neural_data = np.asarray(neural_data)
    n_samples, n_channels = neural_data.shape
    
    bold = np.zeros_like(neural_data, dtype=float)

    for ch in range(n_channels):
        conv = np.convolve(neural_data[:, ch], hrf, mode="full")

        bold[:, ch] = conv[:n_samples]

    return bold
