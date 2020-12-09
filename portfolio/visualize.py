from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from tempfile import NamedTemporaryFile
from typing import Sequence, Type

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mpld3
import numpy as np
from matplotlib.axes import Axes
from base64 import standard_b64encode
import io


class Plot:
    def __init__(self, x, y, label='data'):
        self.x = np.array(x)
        self.y = np.array(y)
        self.fig = plt.figure()
        self.__axes = self.fig.add_subplot()

    def add_view(self, L: Type[DataView], *args, **kwargs):
        L(self.x, self.y, *args, **kwargs).plot(self.__axes)
        self.__axes.legend()

    def save(self, dir='../data'):
        with NamedTemporaryFile(delete=False, suffix='.png',
                                dir='../data') as f:
            self.fig.savefig(f)
            return f.name

    def show(self):
        self.fig.show()

    def embed(maybe_self, static=True):
        """Exports as a string suitable for html embedding

        Can also be run as a static function to export a fig.
        """
        if isinstance(maybe_self, Plot):
            self = maybe_self
            fig = self.fig
        else:
            self = None
            fig = maybe_self

        if static:
            with io.BytesIO() as img:
                fig.savefig(img, format='png')
                with img.getbuffer() as bimg:
                    return (
                        b'<img src="data:image/png;charset=utf-8;base64, %b" />'
                        % standard_b64encode(bimg)).decode()
        else:
            return mpld3.fig_to_html(fig)


@dataclass
class DataView:
    x: Sequence[Real]
    y: Sequence[Real]
    label: str = 'data'

    def plot(self, axes: Axes):
        pass


class Points(DataView):
    def plot(self, axes: Axes):
        axes.plot(self.x, self.y, 'o', label=self.label)


def plot_decision_regions(ax, X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   color=cmap(idx),
                   marker=markers[idx],
                   label=cl)
