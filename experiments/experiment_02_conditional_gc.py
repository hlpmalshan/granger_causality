import numpy as np

from src.simulation.var_generator import generate_var_data
from src.gc.pairwise_gc import pairwise_gc
from src.gc.conditional_gc import conditional_gc

A1 = np.array([
    [0.5, 0.0, 0.4],
    [0.0, 0.8, 0.0],
    [0.0, 0.5, 0.5]
])

data = generate_var_data([A1], np.eye(3), 5000)

print("Pairwise Y->X",
    pairwise_gc(
        data,
        source=1,
        target=0,
        order=1
    )
)

print("Conditional Y->X|Z",
      conditional_gc(
          data,
          source=1,
          target=0,
          conditioning=[2],
          order=1
      )
)
