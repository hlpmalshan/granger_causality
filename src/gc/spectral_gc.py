import numpy as np

from src.utils.matrix_utils import (transfer_function)

def spectral_gc_bivariate(
        coef_matrices,
        Sigma,
        n_freqs=512
):
    frequencies = np.linspace(0, np.pi, n_freqs)
    H_freq = transfer_function(coef_matrices, frequencies)

    gc_y_to_x = []
    for H in H_freq:
        Hxx = H[0, 0]
        Hxy = H[0, 1]

        sigma_x = Sigma[0, 0]
        sigma_y = Sigma[1, 1]

        Sxx = ( 
            abs(Hxx)**2 * sigma_x 
            +
            abs(Hxy)**2 * sigma_y
        )

        intrinsic = ( abs(Hxx)**2 * sigma_x )

        f = np.log(Sxx/intrinsic)

        gc_y_to_x.append(np.real(f))

    return frequencies, np.array(gc_y_to_x)
