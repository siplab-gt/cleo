from abc import ABC, abstractmethod
import warnings

import numpy as np

"""Delays should all assume the unit of milliseconds.
"""


class Delay(ABC):
    @abstractmethod
    def compute(self) -> float:
        pass


class ConstantDelay(Delay):
    def __init__(self, delay_ms):
        self.delay = delay_ms

    def compute(self):
        return self.delay


class GaussianDelay(Delay):
    """Generates normal-distributed delay.

    Will return 0 when a negative value is sampled.

    Parameters
    ----------
    Delay : [type]
        [description]
    """

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def compute(self) -> float:
        out = np.random.normal(self.loc, self.scale)
        if out < 0:
            warnings.warn("Negative delay sampled. Returning 0.")
            out = 0
        return out
