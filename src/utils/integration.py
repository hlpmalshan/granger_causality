import numpy as np

def integrate_spectral_gc(
        frequencies,
        spectral_gc
):
    integral = np.trapezoid(spectral_gc, frequencies)

    return integral / np.pi