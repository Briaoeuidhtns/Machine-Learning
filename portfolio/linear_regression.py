from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import List, Sequence

import numpy as np
from . import visualize


@dataclass
class LinearRegression(visualize.DataView):
    """Least Squares regression adapted from book section 9.2.1"""
    x: Sequence[Number]
    y: Sequence[Number]
    weights: List[float] = field(init=False)

    def __post_init__(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        newx = np.array([np.ones(self.x.shape[0]), self.x])
        A = newx @ newx.T
        b = newx @ self.y
        self.weights = np.linalg.pinv(A) @ b

    def plot(self, axes):
        intercept, slope = self.weights
        ypred = slope * self.x + intercept
        axes.plot(self.x, ypred, label=fr'$y = {slope}x + {intercept}$')
