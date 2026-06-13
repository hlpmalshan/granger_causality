from src.gc.pairwise_gc import pairwise_gc

def time_domain_gc(
        data,
        source,
        target,
        order
):
    
    return pairwise_gc(
        data=data,
        source=source,
        target=target,
        order=order
    )