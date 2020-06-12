from abc import ABC, abstractmethod

class LoopComponent(ABC):
    out = None

    @abstractmethod
    def process_data(self, data, time):
        pass