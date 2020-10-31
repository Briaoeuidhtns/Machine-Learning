from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Type

import matplotlib.pyplot as plt
import mpld3
from matplotlib.axes import Axes


class Plot:
    def __init__(self, x, y, label='data'):
        self.x = x
        self.y = y
        self.fig = plt.figure()
        self.__axes = self.fig.add_subplot()
        # self.__axes.plot(self.x, self.y, 'o', label=label)

    def add_view(self, L: Type[DataView]):
        L(self.x, self.y).plot(self.__axes)
        self.__axes.legend()

    def save(self):
        with NamedTemporaryFile(delete=False, suffix='.png', dir='../data') as f:
            self.fig.savefig(f)
            return f.name

    def embed(self):
        return mpld3.fig_to_html(self.fig)


class DataView:
    def __init__(x, y):
        pass

    def plot(self, axes: Axes):
        pass
