import numpy as np

from src.ssm.em_varx_p_known_c_warmstart import (
    EMVARXPSSMKnownCConstrainedWarmStart
)

from src.ssm.em_varx_p_known_c import (
    _regularize_covariance
)


def shrink_covariance_matrix(
        S,
        alpha,
        target="spherical",
        covariance_floor=1e-6
):
    """
    Shrink a covariance matrix toward a structured target.

    target="spherical":
        target = trace(S) / d * I

    target="diagonal":
        target = diag(diag(S))

    alpha=0:
        no shrinkage

    alpha=1:
        full target
    """

    S = np.asarray(
        S,
        dtype=float
    )

    S = 0.5 * (
        S + S.T
    )

    d = S.shape[0]

    if alpha <= 0.0:
        return _regularize_covariance(
            S,
            epsilon=covariance_floor
        )

    if target == "spherical":

        tau = float(
            np.trace(S) / d
        )

        if not np.isfinite(tau) or tau <= 0:
            tau = covariance_floor

        target_matrix = tau * np.eye(
            d
        )

    elif target == "diagonal":

        target_matrix = np.diag(
            np.diag(S)
        )

    else:
        raise ValueError(
            f"Unknown shrinkage target: {target}"
        )

    S_shrink = (
        (1.0 - alpha) * S
        + alpha * target_matrix
    )

    S_shrink = 0.5 * (
        S_shrink
        + S_shrink.T
    )

    S_shrink = _regularize_covariance(
        S_shrink,
        epsilon=covariance_floor
    )

    return S_shrink


class EMVARXPSSMKnownCConstrainedWarmStartShrinkage(
        EMVARXPSSMKnownCConstrainedWarmStart
):
    """
    Warm-started constrained EM with covariance shrinkage applied
    after each M-step update of Q and R.

    This is used for Experiment 31G.
    """

    def __init__(
            self,
            *args,
            alpha_Q=0.0,
            alpha_R=0.0,
            shrinkage_target_Q="spherical",
            shrinkage_target_R="spherical",
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.alpha_Q = float(
            alpha_Q
        )

        self.alpha_R = float(
            alpha_R
        )

        self.shrinkage_target_Q = shrinkage_target_Q
        self.shrinkage_target_R = shrinkage_target_R

    def _m_step(
            self,
            y,
            u,
            smooth_result
    ):
        """
        Run the parent M-step, then shrink Q and R.
        """

        super()._m_step(
            y=y,
            u=u,
            smooth_result=smooth_result
        )

        self.Q = shrink_covariance_matrix(
            S=self.Q,
            alpha=self.alpha_Q,
            target=self.shrinkage_target_Q,
            covariance_floor=self.covariance_floor
        )

        if self.estimate_R:

            self.R = shrink_covariance_matrix(
                S=self.R,
                alpha=self.alpha_R,
                target=self.shrinkage_target_R,
                covariance_floor=self.R_floor
            )

        self._refresh_companion_matrices()