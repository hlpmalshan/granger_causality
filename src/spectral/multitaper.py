import numpy as np

from scipy.signal.windows import dpss

# Estimates cross-spectral density matrix using multitaper method
def multitaper_csd(
        data,
        fs=1.0,             # Sample frequency
        time_bandwidth=3.0, # Time-bandwidth product NW
        n_tapers=None,      # Number of DPSS tapers. If None, uses int(2*NW - 1)
        n_fft=None,         # FFT length. If None, uses n_samples
        demean=True         # Whether to remove channel-wise mean before spectral estimation
):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must have shape (n_sammples, n_channels)")
    
    n_samples, n_channels = data.shape

    if demean:
        data = data - np.mean(data, axis=0, keepdims=True)

    if n_tapers is None:
        n_tapers = int(2 * time_bandwidth - 1)

    if n_fft is None:
        n_fft = n_samples

    if n_fft < n_samples:
        raise ValueError(
            f"n_fft must be >= n_samples."
            f"Got n_fft={n_fft}, n_samples={n_samples}."
            "Use n_fft=None or a larger FFT length"
        )

    tapers = dpss(n_samples, NW=time_bandwidth, Kmax=n_tapers, sym=False)

    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0/fs)
    freqs_rad = 2 * np.pi * freqs_hz / fs

    n_freqs = len(freqs_rad)
    S = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for taper in tapers:
        tapered_data = data * taper[:, None]
        Xf = np.fft.rfft(tapered_data, n=n_fft, axis=0)

        for f_idx in range(n_freqs):
            x = Xf[f_idx, :].reshape(n_channels, 1)
            S[f_idx] += x @ np.conj(x.T)

    S /= n_tapers

    # Enforce Hermittian symmetry numerically
    S = 0.5 * ( S + np.conj(np.swapaxes(S, 1, 2)))

    return freqs_rad, S     # S = Cross spectral density matrix

# Multitaper CSD averaged over trials
def multitaper_csd_trials(
        trials,
        fs=1.0,
        time_bandwidth=3.0,
        n_tapers=None,
        n_fft=None,
        demean=True
):
    trials = np.asarray(trials)

    if trials.ndim != 3:
        raise ValueError("trials must have shape (n_trials, n_samples, n_channels)")
    
    S_sum = None
    freqs = None

    for trial in trials:
        freqs, S = multitaper_csd(
            trial,
            fs=fs,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            n_fft=n_fft,
            demean=demean
        )
        if S_sum is None:
            S_sum = np.zeros_like(S)

        S_sum += S

    S_avg = S_sum / trials.shape[0]

    return freqs, S_avg

# Extract autospectrum for one chennel
def extract_psd(S, channel):
    return np.real(S[:, channel, channel])