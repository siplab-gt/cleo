from . import LoopComponent
import numpy as np
from nptyping import NDArray, Int32
from typing import Any
from .delays import Delay

class FiringRateEstimator(LoopComponent):
    """Exponential filter to estimate firing rate.

    Requires `sample_time_ms` kwarg when process is called.

    Parameters
    ----------
    LoopComponent : [type]
        [description]
    """
    def __init__(self, tau_ms: float, sample_period_ms: float, **kwargs):
        '''
        See `LoopComponent` for `**kwargs` options
        '''
        super().__init__(**kwargs)
        self.tau_s = tau_ms / 1000
        self.T_s = sample_period_ms / 1000
        self.alpha = np.exp(-sample_period_ms / tau_ms)
        self.prev_rate = None
        self.prev_time_ms = None

    # def _process(self, data: NDArray[(1, Any), Int32], time_ms=None) \
    def _process(self, input: NDArray[(1, Any), Int32], **kwargs) \
            -> NDArray[(1, Any), float]:
        '''
        `input` should be a vector of spike counts.
        '''
        time_ms = kwargs['sample_time_ms']
        if self.prev_rate is None:
            self.prev_rate = np.zeros(input.shape)
        if self.prev_time_ms is None:
            self.prev_time_ms = time_ms - self.T_s*1000

        intersample_period_s = (time_ms - self.prev_time_ms)/1000
        alpha = np.exp(-intersample_period_s / self.tau_s)
        curr_rate = self.prev_rate*alpha + (1-alpha)*input/intersample_period_s
        self.prev_rate = curr_rate
        self.prev_time_ms = time_ms
        return curr_rate