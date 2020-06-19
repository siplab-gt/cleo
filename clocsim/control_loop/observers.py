from . import LoopComponent
import numpy as np
from .delays import Delay

class FiringRateEstimator(LoopComponent):
    def __init__(self, tau_ms: float, sample_period_ms: float, **kwargs):
        '''
        See `LoopComponent` for `**kwargs` options
        '''
        super().__init__(**kwargs)
        self.tau = tau_ms
        self.T = sample_period_ms
        self.alpha = np.exp(-sample_period_ms / tau_ms)
        self.prev_rate = None

    def _process_data(self, data: np.array) -> np.array:
        '''
        `data` should be a vector of spike counts.
        '''
        if self.prev_rate is None:
            self.prev_rate = np.zeros(data.shape)

        curr_rate = self.prev_rate*self.alpha + (1-self.alpha)*data/self.T
        self.prev_rate = curr_rate
        return curr_rate