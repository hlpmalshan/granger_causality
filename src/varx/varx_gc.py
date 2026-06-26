import numpy as np

from src.varx.varx_model import VARXModel

# VARX Granger-style effect:
#   source endogenous variable -> target endogenous variable
# Full model:
#   target predicted from all endogenous histories + exogenous input
# Reduced model:
#   target predicted from all endogenous histories except source + exogenous input
def varx_gc(y, u, source, target, na, nb, ridge=0.0):
    y = np.asarray(y)
    n_y = y.shape[1]

    full_sources = list(range(n_y))
    reduced_sources = [idx for idx in range(n_y) if idx != source]

    full = VARXModel(na=na, nb=nb, ridge=ridge)
    full.fit(y, u, source_indices=full_sources, include_exogenous=True)

    reduced = VARXModel(na=na, nb=nb, ridge=ridge)
    reduced.fit(y, u, source_indices=reduced_sources, include_exogenous=True)

    sigma_full = full.residual_variance(target)
    sigma_reduced = reduced.residual_variance(target)

    gc = np.log(sigma_reduced / sigma_full)

    return {
        "gc": gc,
        "sigma_full": sigma_full,
        "sigma_reduced": sigma_reduced,
        "full_model": full,
        "reduced_model": reduced
    }

# Wrapper around the existing pairwise GC function
def ordinary_var_gc_for_comparison(y, source, target, order):
    from src.gc.pairwise_gc import pairwise_gc

    return pairwise_gc(data=y, source=source, target=target, order=order)
