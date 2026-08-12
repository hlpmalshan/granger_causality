import numpy as np

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
    solve_standardized_covariance_group_lasso
)


def test_group_solver_applies_joint_l2_threshold_and_leaves_ungrouped_free():
    G = np.eye(5)
    h = np.array([3.0, 4.0, 0.6, 0.8, 2.0])
    groups = [np.array([0, 1]), np.array([2, 3])]

    fit = solve_standardized_covariance_group_lasso(
        G=G,
        h=h,
        groups=groups,
        lambda_fraction=0.5,
        ridge_penalty=np.zeros(5),
        max_iter=100,
        tol=1e-12
    )

    # lambda_max=5 and lambda=2.5. The first group shrinks jointly by 0.5,
    # the second is removed, and the ungrouped coefficient is unchanged.
    np.testing.assert_allclose(
        fit["beta"],
        np.array([1.5, 2.0, 0.0, 0.0, 2.0]),
        atol=1e-12
    )
    assert fit["lambda_max"] == 5.0
    assert fit["lambda_effective"] == 2.5
    assert fit["converged"]


def test_zero_fraction_matches_standardized_ridge_closed_form():
    G = np.array([
        [3.0, 0.4, 0.2, 0.0],
        [0.4, 2.0, 0.1, 0.2],
        [0.2, 0.1, 1.5, 0.1],
        [0.0, 0.2, 0.1, 1.0]
    ])
    h = np.array([0.8, -0.4, 0.2, 0.5])
    ridge = np.full(4, 1e-4)

    fit = solve_standardized_covariance_group_lasso(
        G=G,
        h=h,
        groups=[np.array([0, 2]), np.array([1, 3])],
        lambda_fraction=0.0,
        ridge_penalty=ridge,
        max_iter=20000,
        tol=1e-12
    )

    scale = np.sqrt(np.diag(G))
    G_std = G / np.outer(scale, scale)
    expected_std = np.linalg.solve(
        G_std + np.diag(ridge),
        h / scale
    )
    np.testing.assert_allclose(
        fit["beta"],
        expected_std / scale,
        rtol=1e-8,
        atol=1e-8
    )
    assert fit["converged"]


def test_group_helpers_count_lag_groups_not_individual_coefficients():
    model = (
        EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
            na=2,
            nb=1,
            C=np.eye(2),
            Q_init=0.50 * np.eye(2),
            R_init=0.60 * np.eye(2),
            estimate_Q=False,
            estimate_R=False,
            zero_constraints=[]
        )
    )
    model.A_matrices = [
        np.array([[0.2, 0.3], [0.0, 0.4]]),
        np.array([[0.1, 0.0], [0.5, 0.2]])
    ]

    assert model.count_nonzero_offdiag_A_coefficients() == 2
    assert model.count_nonzero_total_A_coefficients() == 6
    assert model.count_nonzero_offdiag_A_groups() == 2
    assert model.count_nonzero_total_A_groups() == 4
    np.testing.assert_allclose(
        model._A_group_norms(False),
        np.array([0.3, 0.5])
    )


def test_group_posterior_mstep_updates_ab_and_keeps_qr_fixed():
    rng = np.random.default_rng(987)
    n_sources = 3
    T = 25
    Q_fixed = 0.50 * np.eye(n_sources)
    R_fixed = 0.60 * np.eye(n_sources)
    model = (
        EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
            na=1,
            nb=1,
            C=np.eye(n_sources),
            Q_init=Q_fixed,
            R_init=R_fixed,
            estimate_Q=False,
            estimate_R=False,
            zero_constraints=[],
            lambda_A_group_fraction=0.01,
            group_solver_max_iter=5000,
            group_solver_tol=1e-9
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
    assert model.group_solver_converged_history[-1]
    assert np.isfinite(model.mean_lambda_A_group_effective)
