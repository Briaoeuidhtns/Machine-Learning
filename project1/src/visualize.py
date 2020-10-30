from __future__ import annotations

from functools import singledispatchmethod
from typing import Type

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Plot:
    def __init__(self, x, y, label='data'):
        self.x = x
        self.y = y
        self.fig = plt.figure()
        self.__axes = self.fig.add_subplot()
        self.__axes.plot(self.x, self.y, 'o', label=label)

    def add_learner(self, L: Type[Learner]):
        L(self.x, self.y).plot(self.__axes)
        self.__axes.legend()


class Learner:
    def __init__(x, y):
        pass

    def plot(self, axes: Axes):
        pass
