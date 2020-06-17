from abc import ABC, abstractmethod
from typing import Tuple, Any
from ..base import ControlLoop
from collections import deque

class LoopComponent(ABC):
    @abstractmethod
    def process_data(self, data, time) -> Tuple(Any, float):
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
    
    ''' Must return a tuple (data, delayed_time). This function is where you'd pipe data
    through multiple components.'''
    @abstractmethod
    def compute_ctrl_signal(self, state_dict: dict, time: float) -> Tuple(Any, float):
        pass