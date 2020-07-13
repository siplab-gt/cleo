from abc import ABC, abstractmethod
from typing import Tuple, Any
from collections import deque

from brian2 import ms

from ..base import ControlLoop
from .delays import Delay

class LoopComponent(ABC):
    def __init__(self, **kwargs):
        self.delay = kwargs.get('delay', None)
        self.save_history = kwargs.get('save_history', False)
        if self.save_history is True:
            self.t = []
            self.out_t = []
            self.values = []

    def process_data(self, data, time_ms: float) -> Tuple[Any, float]:
        out = self._process_data(data, time_ms)
        if self.delay is not None:
            out_time_ms = self.delay.add_delay_to_time(time_ms)
        else:
            out_time_ms = time_ms
        if self.save_history:
            self.t.append(time_ms)
            self.out_t.append(out_time_ms)
            self.values.append(out)
        return (out, out_time_ms)

    @abstractmethod
    def _process_data(self, data, time_ms: float = None) -> Any:
        '''
        This is the method that must be implemented, which will process
        the data without needing to account for time delay.
        '''
        pass


class DelayControlLoop(ControlLoop):
    '''
    The unit for keeping track of time in the control loop is milliseconds.
    To deal in quantities relative to seconds (e.g., defining a target firing
    rate in Hz), the component involved must make the conversion.
    '''
    def __init__(self):
        self.out_buffer = deque([])

    def put_state(self, state_dict: dict, t):
        time_ms = t / ms
        out, time_ms = self.compute_ctrl_signal(state_dict, time_ms)
        self.out_buffer.append((out, time_ms))

    def get_ctrl_signal(self, time):
        time_ms = time / ms
        if len(self.out_buffer) == 0:
            return None
        next_out_signal, next_out_time_ms = self.out_buffer[0]
        if time_ms >= next_out_time_ms:
            self.out_buffer.popleft()
            return next_out_signal
        else:
            return None
    
    @abstractmethod
    def compute_ctrl_signal(self, state_dict: dict, time_ms: float) -> Tuple[dict, float]:
        ''' 
        Must return a tuple (dict, delayed_time). This function is where you set
        up the data processing pipeline. The output dictionary must have one name-value
        pair for each stimulator you want to control.
        '''
        pass

class FixedSamplingSerialProcessingControlLoop(DelayControlLoop):
    '''
    Samples on a fixed schedule, even if a computation exceeds the sampling period.
    The computation delay for each sample is added on to the output
    time for the previous signal (hence, processing is serial).
    '''

    def __init__(self, sampling_period_ms):
        self.sampling_period_ms = sampling_period_ms
        super().__init__()

    def is_sampling_now(self, time):
        # current sample
        pass
