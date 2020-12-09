from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from operator import itemgetter
from typing import Sequence
import matplotlib.pyplot as plt

import numpy as np

import visualize


@dataclass
class Soft(visualize.DataView):
    rate: Real = 0.001
    lmbda: Real = 0.001
    niter: int = 100
    w: Sequence[Real] = field(init=False, default=None)
    b: Real = field(init=False, default=None)

    def __post_init__(self, *args, **kwargs):
        n_samples, n_features = self.x.shape
        self.w = np.random.rand(n_features)
        self._fit()

    def _correct(self, x, y):
        return y * (np.dot(x, self.w) - self.b) >= 1

    def _fit(self):
        self.b = 0

        for _ in range(self.niter):
            for x_i, y_i in zip(self.x, self.y):
                if _correct(x_i, y_i):
                    self.w -= self.rate * (2 * self.lmbda * self.w)
                else:
                    self.w -= self.rate * (2 * self.lmbda * self.w -
                                           np.dot(x_i, y_i))
                    self.b -= self.rate * y_i

    def predict(self, X):
        return np.sign(X @ self.w - self.b)
