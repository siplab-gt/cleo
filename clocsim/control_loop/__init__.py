from abc import ABC, abstractmethod
from typing import Tuple, Any
from collections import deque

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

    def process_data(self, data, time: float) -> Tuple(Any, float):
        out = self._process_data(data)
        if self.delay is not None:
            out_time = self.delay.add_delay_to_time(time)
        else:
            out_time = time
        self.t.append(time)
        self.out_t.append(out_time)
        self.values.append(out)
        return (out, out_time)

    @abstractmethod
    def _process_data(self, data) -> Any:
        '''
        This is the method that must be implemented, which will process
        the data without needing to account for time delay.
        '''
        pass


class DelayControlLoop(ControlLoop):
    def __init__(self):
        self.out_buffer = deque([])

    def put_state(self, state_dict: dict, time: float):
        out, time = self.compute_ctrl_signal(state_dict, time)
        self.out_buffer.append((out, time))

    def get_ctrl_signal(self, time):
        next_out_signal, next_out_time = self.out_buffer[0]
        if time >= next_out_time:
            self.out_buffer.popleft()
            return next_out_signal
        else:
            return None
    
    @abstractmethod
    def compute_ctrl_signal(self, state_dict: dict, time: float) -> Tuple(Any, float):
        ''' 
        Must return a tuple (data, delayed_time). This function is where you'd pipe data
        through multiple components.
        '''
        pass