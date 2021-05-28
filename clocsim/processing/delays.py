from abc import ABC, abstractmethod

'''Delays should all assume the unit of milliseconds.
'''
class Delay(ABC):
    @abstractmethod
    def add_delay_to_time(self, time) -> float:
        pass


class ConstantDelay(Delay):
    def __init__(self, delay_ms):
        self.delay = delay_ms

    def add_delay_to_time(self, time):
        return time + self.delay