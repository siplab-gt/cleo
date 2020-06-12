from .. import LoopComponent

class Controller(LoopComponent):
    def __init__(self):
        pass


from ..base import Controller
from abc import abstractmethod
from collections import deque

class SimpleDelayController(Controller):
    def __init__(self, delay):
        self.delay = delay
        self.out_buffer = deque()

    def put_state(self, state_dict, time):
        out = self.compute_ctr_signal(state_dict)
        self.out_buffer.append((time, out))

    def get_control_signal(self, time):
        next_out_time, next_out_signal = out_buffer[0]
        if time - next_out_time >= self.delay:
            self.out_buffer.popleft()
            return next_out_signal
        else:
            return None
    
    @abstractmethod
    def compute_ctr_signal(self, state_dict):
        pass