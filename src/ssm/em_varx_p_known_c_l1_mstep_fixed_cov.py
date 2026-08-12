import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import (
    EMVARXPSSMKnownCConstrainedWarmStartL1Mstep
)


class EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov(
        EMVARXPSSMKnownCConstrainedWarmStartL1Mstep
):
    """
    Row-wise L1 VARX EM with optionally fixed Q and R.

    The A/B update is inherited unchanged from the Experiment 32F
    estimator.  With estimate_Q=False and estimate_R=False, the covariance
    matrices remain equal to Q_init and R_init throughout EM.
    """

    def __init__(
            self,
            *args,
            estimate_Q=False,
            estimate_R=False,
            Q_init=None,
            R_init=None,
            **kwargs
    ):
        if not estimate_Q and Q_init is None:
            raise ValueError(
                "Q_init is required when estimate_Q=False."
            )

        if not estimate_R and R_init is None:
            raise ValueError(
                "R_init is required when estimate_R=False."
            )

        super().__init__(
            *args,
            estimate_Q=estimate_Q,
            estimate_R=estimate_R,
            Q_init=Q_init,
            R_init=R_init,
            **kwargs
        )

        self.fixed_Q_trace = float(
            np.trace(
                np.asarray(Q_init, dtype=float)
            )
        ) if not estimate_Q else np.nan

        self.fixed_R_trace = float(
            np.trace(
                np.asarray(R_init, dtype=float)
            )
        ) if not estimate_R else np.nan
