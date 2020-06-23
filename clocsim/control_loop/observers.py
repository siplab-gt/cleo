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

    def _process_data(self, data: NDArray[(1, Any), Int32], time_ms=None) \
            -> NDArray[(1, Any), float]:
        '''
        `data` should be a vector of spike counts.
        '''
        if self.prev_rate is None:
            self.prev_rate = np.zeros(data.shape)

        curr_rate = self.prev_rate*self.alpha + (1-self.alpha)*data/self.T_s
        self.prev_rate = curr_rate
        return curr_rate