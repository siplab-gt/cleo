"""Classes facilitating adding latency to :class:`~cleo.ioproc.ProcessingBlock` computations"""
from abc import ABC, abstractmethod
import warnings

import numpy as np


class Delay(ABC):
    """Abstract base class for computing delays."""

    @abstractmethod
    def compute(self) -> float:
        """Compute delay."""
        pass


class ConstantDelay(Delay):
    """Simply adds a constant delay to the computation"""

    def __init__(self, delay_ms: float):
        """
        Parameters
        ----------
        delay_ms : float
            Desired delay in milliseconds
        """
        self.delay = delay_ms

    def compute(self):
        return self.delay


class GaussianDelay(Delay):
    """Generates normal-distributed delay.

    Will return 0 when a negative value is sampled.
    """

    def __init__(self, loc: float, scale: float):
        """
        Parameters
        ----------
        loc : float
            Center of distribution
        scale : float
            Standard deviation of delay distribution
        """
        self.loc = loc
        self.scale = scale

    def compute(self) -> float:
        out = np.random.normal(self.loc, self.scale)
        if out < 0:
            warnings.warn("Negative delay sampled. Returning 0.")
            out = 0
        return out
