import numpy as np

from src.ssm.em_varx_p_known_c_group_lasso_posterior_moments import (
    EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov,
    make_group_zero_constraints
)


def test_group_constraint_expands_to_every_requested_lag():
    constraints = make_group_zero_constraints(
        source=1,
        target=2,
        na=3
    )

    assert constraints == [
        {
            "source": 1,
            "target": 2,
            "lags": [1, 2, 3]
        }
    ]


def test_group_lasso_mstep_solves_with_constrained_group_fixed_at_zero():
    rng = np.random.default_rng(321)
    n_sources = 2
    order = 2
    n_samples = 35
    Q_fixed = 0.50 * np.eye(n_sources)
    R_fixed = 0.60 * np.eye(n_sources)

    model = EMVARXPSSMKnownCConstrainedWarmStartGroupLassoPosteriorMomentMstepFixedCov(
        na=order,
        nb=1,
        C=np.eye(n_sources),
        Q_init=Q_fixed,
        R_init=R_fixed,
        estimate_Q=False,
        estimate_R=False,
        zero_constraints=make_group_zero_constraints(0, 1, order),
        alpha_Q=0.0,
        alpha_R=0.0,
        lambda_A_group_fraction=0.01,
        group_solver_max_iter=500,
        group_solver_tol=1e-8
    )
    model.A_matrices = [np.zeros((n_sources, n_sources)) for _ in range(order)]
    model.B_matrices = [np.zeros((n_sources, 1))]
    model.Q = Q_fixed.copy()
    model.R = R_fixed.copy()

    n_aug = n_sources * order
    smooth_result = {
        "smoother": {
            "x_smooth": rng.normal(size=(n_samples, n_aug)),
            "P_smooth": np.repeat(
                (0.1 * np.eye(n_aug))[None, :, :],
                n_samples,
                axis=0
            ),
            "P_lag_one": np.zeros((n_samples, n_aug, n_aug))
        }
    }
    u = rng.normal(size=(n_samples, 1))

    model._m_step(
        y=np.zeros((n_samples, n_sources)),
        u=u,
        smooth_result=smooth_result
    )

    for A in model.A_matrices:
        assert A[1, 0] == 0.0

    np.testing.assert_array_equal(model.Q, Q_fixed)
    np.testing.assert_array_equal(model.R, R_fixed)
