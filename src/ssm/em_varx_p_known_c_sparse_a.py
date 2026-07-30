import numpy as np

from src.ssm.em_varx_p_known_c_shrinkage import (
    EMVARXPSSMKnownCConstrainedWarmStartShrinkage
)


def soft_threshold(
        value,
        threshold
):
    return np.sign(value) * max(
        abs(value) - threshold,
        0.0
    )


def companion_spectral_radius_from_A(
        A_matrices
):
    A_matrices = [
        np.asarray(A, dtype=float)
        for A in A_matrices
    ]

    p = len(A_matrices)
    n = A_matrices[0].shape[0]

    companion = np.zeros(
        (n * p, n * p)
    )

    companion[:n, :n * p] = np.hstack(
        A_matrices
    )

    if p > 1:
        companion[n:, :-n] = np.eye(
            n * (p - 1)
        )

    eigvals = np.linalg.eigvals(
        companion
    )

    return float(
        np.max(
            np.abs(
                eigvals
            )
        )
    )


def stabilize_A_matrices_if_needed(
        A_matrices,
        target_radius=0.98
):
    radius = companion_spectral_radius_from_A(
        A_matrices
    )

    if radius <= target_radius:
        return A_matrices, radius, 1.0

    scale = target_radius / (
        radius + 1e-12
    )

    A_stable = [
        scale * A
        for A in A_matrices
    ]

    new_radius = companion_spectral_radius_from_A(
        A_stable
    )

    return A_stable, new_radius, scale


class EMVARXPSSMKnownCConstrainedWarmStartSparseA(
        EMVARXPSSMKnownCConstrainedWarmStartShrinkage
):
    """
    VARX state-space EM with sparse-A proximal regularization.

    This is an NLGC-inspired first implementation.

    It applies L1-style soft thresholding to off-diagonal
    endogenous A coefficients after each dense M-step.

    Model:
        x_t = sum_k A_k x_{t-k}
              + sum_l B_l u_{t-l}
              + w_t

        y_t = C x_t + v_t

    Regularization:
        off-diagonal A terms:
            L1/proximal shrinkage

        diagonal A terms:
            optionally weak shrinkage, default none

        B terms:
            not L1-shrunk in this experiment
    """

    def __init__(
            self,
            *args,
            lambda_A_offdiag=0.0,
            lambda_A_diag=0.0,
            lambda_A_lag_decay=1.0,
            stabilize_A=True,
            target_radius=0.98,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.lambda_A_offdiag = float(
            lambda_A_offdiag
        )

        self.lambda_A_diag = float(
            lambda_A_diag
        )

        self.lambda_A_lag_decay = float(
            lambda_A_lag_decay
        )

        self.stabilize_A = bool(
            stabilize_A
        )

        self.target_radius = float(
            target_radius
        )

        self.A_sparse_scale_history = []
        self.A_radius_after_sparse_history = []
        self.A_nonzero_offdiag_history = []

    def _lag_penalty_scale(
            self,
            lag_index
    ):
        """
        lag_index is zero-based.
        Larger lambda_A_lag_decay penalizes later lags more.
        """

        return self.lambda_A_lag_decay ** lag_index

    def _apply_sparse_A_proximal_step(
            self
    ):
        if self.lambda_A_offdiag <= 0.0 and self.lambda_A_diag <= 0.0:
            radius = companion_spectral_radius_from_A(
                self.A_matrices
            )

            self.A_sparse_scale_history.append(
                1.0
            )

            self.A_radius_after_sparse_history.append(
                radius
            )

            self.A_nonzero_offdiag_history.append(
                self.count_nonzero_offdiag_A()
            )

            return

        new_A_matrices = []

        for lag_index, A in enumerate(self.A_matrices):

            A = np.asarray(
                A,
                dtype=float
            ).copy()

            n_sources = A.shape[0]

            lag_scale = self._lag_penalty_scale(
                lag_index
            )

            offdiag_threshold = (
                self.lambda_A_offdiag
                * lag_scale
            )

            diag_threshold = (
                self.lambda_A_diag
                * lag_scale
            )

            for target in range(n_sources):

                for source in range(n_sources):

                    if target == source:

                        if diag_threshold > 0.0:
                            A[target, source] = soft_threshold(
                                A[target, source],
                                diag_threshold
                            )

                    else:

                        if offdiag_threshold > 0.0:
                            A[target, source] = soft_threshold(
                                A[target, source],
                                offdiag_threshold
                            )

            new_A_matrices.append(
                A
            )

        self.A_matrices = new_A_matrices

        if hasattr(
                self,
                "_apply_zero_constraints_to_A_matrices"
        ):
            self._apply_zero_constraints_to_A_matrices()

        if self.stabilize_A:

            self.A_matrices, radius, scale = stabilize_A_matrices_if_needed(
                self.A_matrices,
                target_radius=self.target_radius
            )

        else:

            radius = companion_spectral_radius_from_A(
                self.A_matrices
            )

            scale = 1.0

        self.A_sparse_scale_history.append(
            scale
        )

        self.A_radius_after_sparse_history.append(
            radius
        )

        self.A_nonzero_offdiag_history.append(
            self.count_nonzero_offdiag_A()
        )

    def count_nonzero_offdiag_A(
            self,
            threshold=1e-8
    ):
        count = 0

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source and abs(A[target, source]) > threshold:
                        count += 1

        return int(
            count
        )

    def count_nonzero_total_A(
            self,
            threshold=1e-8
    ):
        count = 0

        for A in self.A_matrices:

            count += int(
                np.sum(
                    np.abs(A) > threshold
                )
            )

        return count

    def mean_abs_offdiag_A(
            self
    ):
        values = []

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source:
                        values.append(
                            abs(A[target, source])
                        )

        return float(
            np.mean(
                values
            )
        )

    def max_abs_offdiag_A(
            self
    ):
        values = []

        for A in self.A_matrices:

            A = np.asarray(
                A,
                dtype=float
            )

            n_sources = A.shape[0]

            for target in range(n_sources):

                for source in range(n_sources):

                    if target != source:
                        values.append(
                            abs(A[target, source])
                        )

        return float(
            np.max(
                values
            )
        )

    def _m_step(
            self,
            y,
            u,
            smooth_result
    ):
        """
        Dense/shrinkage M-step from parent class,
        followed by sparse-A proximal step.
        """

        super()._m_step(
            y=y,
            u=u,
            smooth_result=smooth_result
        )

        self._apply_sparse_A_proximal_step()

        self._refresh_companion_matrices()