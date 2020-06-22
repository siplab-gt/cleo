from . import LoopComponent
from typing import Any
from nptyping import NDArray

class Controller(LoopComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class PIController(Controller):
    def __init__(self, ref_signal, Kp, Ki=0, sample_period_ms=0, **kwargs):
        '''
        ref_signal is a function of time
        '''
        super().__init__(**kwargs)
        self.ref_signal = ref_signal
        self.Kp = Kp
        self.Ki = Ki
        self.sample_period = sample_period_ms
        self.integrated_error = 0
    
    def _process_data(self, data, time: float):
        error = self.ref_signal(time) - data
        self.integrated_error += error*self.sample_period
        return self.Kp*error + self.Ki * self.integrated_error
