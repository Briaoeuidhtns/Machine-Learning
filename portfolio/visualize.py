from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from tempfile import NamedTemporaryFile
from typing import Sequence, Type

import matplotlib.pyplot as plt
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

    def add_view(self, L: Type[DataView], **kwargs):
        L(self.x, self.y, **kwargs).plot(self.__axes)
        self.__axes.legend()

    def save(self, dir='../data'):
        with NamedTemporaryFile(delete=False, suffix='.png',
                                dir='../data') as f:
            self.fig.savefig(f)
            return f.name

    def show(self):
        plt.show(self.fig)

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
