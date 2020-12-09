from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from operator import itemgetter
from typing import Sequence
import matplotlib.pyplot as plt

import numpy as np

from . import visualize


@dataclass
class Soft(visualize.DataView):
    rate: Real = 0.001
    lmbda: Real = 0.002
    niter: int = 1000
    w: Sequence[Real] = field(init=False, default=None)
    b: Real = field(init=False, default=None)

    def __post_init__(self, *args, **kwargs):
        self._fit()

    def _hinge_loss(self, x, y):
        return y * self._classifier_score(x)

    def _classifier_score(self, x):
        return x @ self.w - self.b

    def _fit(self):
        self.b = 0
        n_samples, n_features = self.x.shape
        self.w = np.random.rand(n_features)

        for _ in range(self.niter):
            for x_i, y_i in zip(self.x, self.y):
                if self._hinge_loss(x_i, y_i) >= 1:
                    self.w -= self.rate * (self.lmbda * self.w)
                else:
                    self.w -= self.rate * (self.lmbda * self.w -
                                           np.dot(x_i, y_i))
                    self.b -= self.rate * y_i

    def predict(self, X):
        return np.sign(self._classifier_score(X))

    def plot(self, ax):
        visualize.plot_decision_regions(ax, self.x, self.y, self)
