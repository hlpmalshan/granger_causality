import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep import (
    NUMBA_AVAILABLE,
    _coordinate_descent_core,
    _coordinate_descent_core_python,
    extract_smoothed_x
)
from src.ssm.em_varx_p_known_c_l1_mstep_fixed_cov import (
    EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov
)


def test_extract_smoothed_x_from_nested_smoother_result():
    x_smooth = np.arange(24, dtype=float).reshape(4, 6)
    smooth_result = {
        "filter": {},
        "smoother": {
            "x_smooth": x_smooth
        },
        "log_likelihood": -10.0
    }

    result = extract_smoothed_x(
        smooth_result=smooth_result,
        n_sources=3
    )

    np.testing.assert_array_equal(
        result,
        x_smooth[:, :3]
    )


def test_extract_smoothed_x_still_accepts_direct_smoother_result():
    x_smooth = np.arange(15, dtype=float).reshape(3, 5)

    result = extract_smoothed_x(
        smooth_result={"x_smooth": x_smooth},
        n_sources=2
    )

    np.testing.assert_array_equal(
        result,
        x_smooth[:, :2]
    )


def test_numba_coordinate_descent_matches_python_reference():
    if not NUMBA_AVAILABLE:
        return

    rng = np.random.default_rng(456)
    X = np.asfortranarray(rng.normal(size=(80, 9)))
    y = np.ascontiguousarray(rng.normal(size=80))
    l1 = np.ascontiguousarray(np.full(9, 1e-4))
    ridge = np.ascontiguousarray(np.full(9, 1e-4))
    beta = np.ascontiguousarray(rng.normal(scale=0.01, size=9))
    column_norms = np.ascontiguousarray(np.mean(X ** 2, axis=0))

    reference = _coordinate_descent_core_python(
        X.copy(), y.copy(), l1, ridge, beta.copy(), column_norms,
        200, 1e-8
    )
    accelerated = _coordinate_descent_core(
        X.copy(), y.copy(), l1, ridge, beta.copy(), column_norms,
        200, 1e-8
    )

    np.testing.assert_allclose(accelerated[0], reference[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(accelerated[1], reference[1], rtol=1e-12, atol=1e-12)
    assert accelerated[2:] == reference[2:]


def test_fixed_covariance_l1_mstep_updates_ab_but_not_qr():
    rng = np.random.default_rng(123)
    n_sources = 2
    n_samples = 30
    Q_fixed = 0.50 * np.eye(n_sources)
    R_fixed = 0.60 * np.eye(n_sources)

    model = EMVARXPSSMKnownCConstrainedWarmStartL1MstepFixedCov(
        na=1,
        nb=1,
        C=np.eye(n_sources),
        Q_init=Q_fixed,
        R_init=R_fixed,
        estimate_Q=False,
        estimate_R=False,
        zero_constraints=[],
        alpha_Q=0.0,
        alpha_R=0.0,
        lambda_A_offdiag=1e-5,
        lasso_max_iter=100,
        lasso_tol=1e-8
    )

    model.A_matrices = [np.zeros((n_sources, n_sources))]
    model.B_matrices = [np.zeros((n_sources, 1))]
    model.Q = Q_fixed.copy()
    model.R = R_fixed.copy()

    x_smooth = rng.normal(size=(n_samples, n_sources))
    u = rng.normal(size=(n_samples, 1))

    model._m_step(
        y=x_smooth + 0.1 * rng.normal(size=x_smooth.shape),
        u=u,
        smooth_result={
            "smoother": {
                "x_smooth": x_smooth
            }
        }
    )

    np.testing.assert_array_equal(model.Q, Q_fixed)
    np.testing.assert_array_equal(model.R, R_fixed)
    assert model.estimate_Q is False
    assert model.estimate_R is False
    assert model.fixed_Q_trace == 1.0
    assert model.fixed_R_trace == 1.2
    assert np.any(np.abs(model.A_matrices[0]) > 0.0)
    assert np.any(np.abs(model.B_matrices[0]) > 0.0)
