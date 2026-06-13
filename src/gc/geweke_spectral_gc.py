import numpy as np

from src.utils.matrix_utils import transfer_function

def geweke_spectral_gc(
        coef_matrices,
        Sigma,
        n_freqs=512
):
    freqs = np.linspace(0, np.pi, n_freqs)

    H_freq = transfer_function(coef_matrices, freqs)

    sigma_x = Sigma[0, 0]
    sigma_y = Sigma[1, 1]
    upsilon = Sigma[0, 1]

    gc_y_to_x = []
    for H in H_freq:
        Hxx = H[0, 0]
        Hxy = H[0, 1]

        Sxx = (
            sigma_x * abs(Hxx)**2
            +
            sigma_y * abs(Hxy)**2
            +
            2 * upsilon * np.real(Hxx * np.conj(Hxy))
        )

        Hxx_tilde = (
            Hxx
            +
            (upsilon / sigma_x) * Hxy
        )

        intrinsic = sigma_x * abs(Hxx_tilde)**2

        gc = np.log(np.real(Sxx / intrinsic))

        gc_y_to_x.append(gc)

    return ( freqs, np.asarray(gc_y_to_x) )
