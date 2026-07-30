import numpy as np

# p = (1 + number of null statistics >= observed statistic) / (1 + B)
# The +1 correction prevents zero p-values when the bootstrap count is finite.
def empirical_upper_tail_p_value(observed_statistic, null_statistics):
    null_statistics = np.asarray(null_statistics, dtype=float)
    observed_statistic = float(observed_statistic)

    B = len(null_statistics)
    if B < 1:
        raise ValueError("null_statistics must contain at least one value.")

    count = np.sum(null_statistics >= observed_statistic)

    p_value = (1.0 + count) / (1.0 + B)

    return float(p_value)

# Use the same statistic that was used for chi-square p-values: max(D_debiased, 0)
# Negative deviance means no evidence for the tested direction.
def clipped_deviance_statistic(result):
    if "deviance_for_p" in result:
        return float(result["deviance_for_p"])
    
    return float(max(result["deviance_debiased"], 0.0))