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

    For non-serial processing, the order of input/output is preserved even
    if the computed output time for a sample is sooner than that for a previous
    sample.

    Fixed sampling: on a fixed schedule no matter what
    Wait for computation sampling: Can't sample during computation. Samples ASAP
    after an over-period computation: otherwise remains on schedule.
    '''
    def __init__(self, sampling_period_ms, **kwargs):
        self.out_buffer = deque([])
        self.sampling_period_ms = sampling_period_ms
        self.sampling = kwargs.get('sampling', 'fixed')
        if self.sampling not in ['fixed', 'wait for computation']:
            raise ValueError('Invalid sampling scheme:', self.sampling)
        self.processing = kwargs.get('processing', 'serial')
        if self.processing not in ['serial', 'parallel']:
            raise ValueError('Invalid processing scheme:', self.processing)

    def put_state(self, state_dict: dict, t):
        in_time_ms = t / ms
        out, out_time_ms = self.compute_ctrl_signal(state_dict, in_time_ms)
        if self.processing == 'serial' and len(self.out_buffer) > 0:
            prev_out_time_ms = self.out_buffer[-1][1]
            # add delay onto the output time of the last computation
            out_time_ms = prev_out_time_ms + out_time_ms - in_time_ms
        self.out_buffer.append((out, out_time_ms))

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
    
    def is_sampling_now(self, time):
        time_ms = time / ms
        if self.sampling == 'fixed':
            if time_ms % self.sampling_period_ms == 0:
                return True
        elif self.sampling == 'wait for computation':
            if time_ms % self.sampling_period_ms == 0:
                # if done computing
                if len(self.out_buffer) == 0 or self.out_buffer[0][1] <= time_ms:
                    self.overrun = False
                    return True
                else:  # if not done computing
                    self.overrun = True
                    return False
            else:  # off-schedule, only sample if the last sampling period was missed (there was an overrun)
                return self.overrun
        return False

    @abstractmethod
    def compute_ctrl_signal(self, state_dict: dict, time_ms: float) -> Tuple[dict, float]:
        ''' 
        Must return a tuple (dict, delayed_time). This function is where you set
        up the data processing pipeline. The output dictionary must have one name-value
        pair for each stimulator you want to control.
        '''
        pass
