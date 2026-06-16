import numpy as np 

from src.gc.pairwise_gc import pairwise_gc

def sliding_window_gc(
        data,
        source,
        target,
        order,
        window_size,
        step_size
):
    gc_values = []
    centers = []
    start = 0

    while (start + window_size <= len(data)):
        stop = start + window_size
        segment = data[start:stop]

        gc = pairwise_gc(segment, source, target, order)

        gc_values.append(gc)
        centers.append(start + window_size//2)

        start += step_size

    return (np.array(centers), np.array(gc_values))