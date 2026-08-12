import numpy as np

from src.ssm.em_varx_p_known_c_l1_mstep_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartL1PosteriorMomentMstepFixedCov,
    build_posterior_moment_statistics,
    solve_standardized_covariance_lasso
)


def test_posterior_moments_use_lag_one_covariance_with_correct_orientation():
    n_sources = 1
    na = 2
    nb = 1
    T = 3
    mu = np.zeros((T, 2))
    covariance = np.zeros((T, 2, 2))
    covariance[1] = np.array([[2.0, 0.3], [0.3, 1.0]])
    covariance[2, 0, 0] = 4.0

    # Cov(s_2, s_1). Its first row contains Cov(x_2, s_1), so after
    # transpose the first column is Cov(s_1, x_2) = [0.7, -0.2].
    lag_one = np.zeros((T, 2, 2))
    lag_one[2] = np.array([[0.7, -0.2], [0.1, 0.4]])
    u = np.array([[1.0], [2.0], [3.0]])

    result = build_posterior_moment_statistics(
        smooth_result={
            "smoother": {
                "x_smooth": mu,
                "P_smooth": covariance,
                "P_lag_one": lag_one
            }
        },
        u=u,
        na=na,
        nb=nb,
        n_sources=n_sources
    )

    expected_G = np.array([
        [2.0, 0.3, 0.0],
        [0.3, 1.0, 0.0],
        [0.0, 0.0, 9.0]
    ])
    expected_H = np.array([[0.7], [-0.2], [0.0]])

    np.testing.assert_allclose(result["G"], expected_G)
    np.testing.assert_allclose(result["H"], expected_H)
    np.testing.assert_allclose(result["E_xx"], np.array([[4.0]]))
    assert result["start"] == 2
    assert result["n_valid"] == 1


def test_covariance_solver_matches_standardized_ridge_closed_form():
    G = np.array([
        [4.0, 0.5, 0.2],
        [0.5, 2.0, 0.1],
        [0.2, 0.1, 1.0]
    ])
    h = np.array([1.0, -0.4, 0.3])
    ridge = np.full(3, 1e-4)

    fit = solve_standardized_covariance_lasso(
        G=G,
        h=h,
        l1_mask=np.array([True, True, False]),
        lambda_fraction=0.0,
        ridge_penalty=ridge,
        max_iter=10000,
        tol=1e-12
    )

    scale = np.sqrt(np.diag(G))
    G_std = G / np.outer(scale, scale)
    h_std = h / scale
    expected_std = np.linalg.solve(G_std + np.diag(ridge), h_std)
    expected = expected_std / scale

    np.testing.assert_allclose(fit["beta"], expected, rtol=1e-9, atol=1e-9)
    assert fit["converged"]


def test_fraction_penalty_excludes_diagonal_and_exogenous_coefficients():
    G = np.eye(4)
    h = np.array([0.5, 2.0, 3.0, 4.0])

    fit = solve_standardized_covariance_lasso(
        G=G,
        h=h,
        l1_mask=np.array([True, True, False, False]),
        lambda_fraction=0.5,
        ridge_penalty=np.zeros(4),
        max_iter=100,
        tol=1e-12
    )

    assert fit["lambda_max"] == 2.0
    assert fit["lambda_effective"] == 1.0
    np.testing.assert_allclose(fit["beta"], np.array([0.0, 1.0, 3.0, 4.0]))


def test_posterior_moment_mstep_updates_ab_but_keeps_qr_fixed():
    rng = np.random.default_rng(789)
    n_sources = 2
    T = 20
    Q_fixed = 0.50 * np.eye(n_sources)
    R_fixed = 0.60 * np.eye(n_sources)

    model = (
        EMVARXPSSMKnownCConstrainedWarmStartL1PosteriorMomentMstepFixedCov(
            na=1,
            nb=1,
            C=np.eye(n_sources),
            Q_init=Q_fixed,
            R_init=R_fixed,
            estimate_Q=False,
            estimate_R=False,
            zero_constraints=[],
            lambda_A_fraction=0.01,
            ridge_A_offdiag=1e-4,
            ridge_A_diag=1e-4,
            ridge_B=1e-4,
            lasso_max_iter=1000,
            lasso_tol=1e-9
        )
    )
    model.A_matrices = [np.zeros((n_sources, n_sources))]
    model.B_matrices = [np.zeros((n_sources, 1))]
    model.Q = Q_fixed.copy()
    model.R = R_fixed.copy()

    mu = rng.normal(size=(T, n_sources))
    covariance = np.repeat(
        (0.2 * np.eye(n_sources))[None, :, :],
        T,
        axis=0
    )
    lag_one = np.zeros_like(covariance)
    u = rng.normal(size=(T, 1))

    model._m_step(
        y=mu,
        u=u,
        smooth_result={
            "smoother": {
                "x_smooth": mu,
                "P_smooth": covariance,
                "P_lag_one": lag_one
            }
        }
    )

    np.testing.assert_array_equal(model.Q, Q_fixed)
    np.testing.assert_array_equal(model.R, R_fixed)
    assert np.any(np.abs(model.A_matrices[0]) > 0.0)
    assert np.any(np.abs(model.B_matrices[0]) > 0.0)
    assert np.isfinite(model.mean_lambda_A_effective)
    assert np.isfinite(model.moment_solver_condition_number_mean)
