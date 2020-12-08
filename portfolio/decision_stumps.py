from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from operator import itemgetter
from typing import Sequence

import numpy as np

from . import visualize


@dataclass
class DecisionStumps(visualize.DataView):
    """Least Squares regression adapted from book section 9.2.1"""
    theta: Real = field(init=False)
    x_j: Sequence[Real] = field(init=False)
    error: Real = field(init=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # was a bit confused on the optimized pseudocode in the book,
        # but to the best of my knowledge this is what the book wanted
        # for the O(dm^2) method
        self.error, self.x_j, self.theta = min(
            ((np.sum(np.sign(x_j - theta) != self.y), x_j, theta)
             for x_j in self.x.T for theta in x_j),
            key=itemgetter(0))

    def plot(self, axes):
        axes.plot(self.x_j, self.y, 'o')
        axes.axvline(self.theta, label=r'$\theta$ on optimal dimension')
