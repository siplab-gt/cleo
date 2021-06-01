from . import LoopComponent
import numpy as np
from nptyping import NDArray, Int32
from typing import Any
from .delays import Delay

class FiringRateEstimator(LoopComponent):
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

    def _process_data(self, data: NDArray[(1, Any), Int32], time_ms=None) \
            -> NDArray[(1, Any), float]:
        '''
        `data` should be a vector of spike counts.
        '''
        if self.prev_rate is None:
            self.prev_rate = np.zeros(data.shape)
        if self.prev_time_ms is None:
            self.prev_time_ms = time_ms - self.T_s*1000

        intersample_period_s = (time_ms - self.prev_time_ms)/1000
        alpha = np.exp(-intersample_period_s / self.tau_s)
        curr_rate = self.prev_rate*alpha + (1-alpha)*data/intersample_period_s
        self.prev_rate = curr_rate
        self.prev_time_ms = time_ms
        return curr_rate