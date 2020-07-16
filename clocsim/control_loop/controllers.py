from . import LoopComponent
from typing import Any, Callable
from nptyping import NDArray

class Controller(LoopComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class PIController(Controller):
    ref_signal: Callable[[float], Any]

    def __init__(self, ref_signal, Kp, Ki=0, sample_period_ms=0, **kwargs):
        '''
        ref_signal is a function of time in milliseconds
        '''
        super().__init__(**kwargs)
        self.ref_signal = ref_signal
        self.Kp = Kp
        self.Ki = Ki
        self.sample_period_s = sample_period_ms / 1000
        self.integrated_error = 0
        self.prev_time_ms = None
    
    def _process_data(self, data, time_ms: float):
        if self.prev_time_ms = None:
            self.prev_time_ms = time_ms - self.sample_period_s*1000
        intersample_period_s = (time_ms - self.prev_time_ms)/1000
        error = self.ref_signal(time_ms) - data
        self.integrated_error += error*intersample_period_s
        self.prev_time_ms = time_ms
        return self.Kp*error + self.Ki * self.integrated_error
