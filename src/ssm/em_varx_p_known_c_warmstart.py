import numpy as np

from src.ssm.em_varx_p_known_c_constrained import EMVARXPSSMKnownCConstrained

# Extract parameters from a fitted EM model so that another model can be warm-started from them.
def extract_model_parameters(model):
    params = {
        "A_matrices": [np.array(A, dtype=float).copy() for A in model.A_matrices],
        "B_matrices": [np.array(B, dtype=float).copy() for B in model.B_matrices],
        "Q": np.array(model.Q, dtype=float).copy(),
        "R": np.array(model.R, dtype=float).copy()
    }

    if hasattr(model, "D") and model.D is not None:
        params["D"] = np.array(model.D, dtype=float).copy()
    else:
        params["D"] = None

    return params

# Same as EMVARXPSSMKnownCConstrained, but allows EM to be initialized from an existing fitted model.
# This is useful for observed-likelihood full/reduced comparisons.
# Example: reduced model -> full model warm start
# This ensures the full model starts from a parameter point equivalent to the reduced model, so the full-model likelihood should be able to match or improve the reduced-model likelihood.
class EMVARXPSSMKnownCConstrainedWarmStart(EMVARXPSSMKnownCConstrained):
    def __init__(
            self,
            *args,
            initial_parameters=None,
            jitter_scale=0.0,
            random_seed=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.jitter_scale = jitter_scale
        self.random_seed = random_seed

    def _jitter_array(self, array, rng):
        if self.jitter_scale <= 0:
            return array

        return (array + self.jitter_scale * rng.standard_normal(size=array.shape))

    # Parent fit() will call this method.
    # If initial_parameters is None: use the usual proxy initialization.
    # If initial_parameters is provided: first call parent initialization to set all auxiliary defaults, then overwrite A, B, Q, R, D using the supplied parameters.
    def _initialize_from_proxy(self, y, u):
        if self.initial_parameters is None:
            super()._initialize_from_proxy(y=y, u=u)
            return

        # First run the parent initialization to make sure all internal default attributes are set.
        super()._initialize_from_proxy(y=y, u=u)
        params = self.initial_parameters

        rng = np.random.default_rng(self.random_seed)
        self.A_matrices = [self._jitter_array(np.array(A, dtype=float).copy(), rng) for A in params["A_matrices"]]
        self.B_matrices = [self._jitter_array(np.array(B, dtype=float).copy(), rng) for B in params["B_matrices"]]
        self.Q = np.array(params["Q"], dtype=float).copy()
        self.R = np.array(params["R"], dtype=float).copy()

        if params.get("D", None) is not None:
            self.D = np.array(params["D"], dtype=float).copy()

        # If this is a reduced model, force the constrained coefficients to zero after copying the initial parameters.
        self._apply_zero_constraints_to_A_matrices()
        self._refresh_companion_matrices()