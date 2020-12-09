from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Sequence, Any
from functools import partial

import numpy as np

from . import visualize


@dataclass
class KNN(visualize.DataView):
    def predict(self, x, k=1):
        x = np.array(x)
        if x.ndim == 0:
            x = x[(np.newaxis, ) * 3]
        elif x.ndim == 1:
            x = x[np.newaxis, ..., np.newaxis]
        elif x.ndim == 2:
            x = x[..., np.newaxis]
        elif x.ndim != 3:
            raise IndexError('Too many dimensions')

        return np.array([
            self.y[np.argpartition(
                np.sqrt(np.sum((query - self.x)**2, axis=1)), k)[:k]]
            for query in x
        ]).squeeze()

        # return self.y[np.partition(distances, k)[:k]]
