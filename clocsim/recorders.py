from brian2 import PopulationRateMonitor, StateMonitor
from .base import Recorder


class RateRecorder(Recorder):
    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.mon = None

    def connect_to_neurons(self, neuron_group):
        self.mon = PopulationRateMonitor(neuron_group[self.i])
        self.brian_objects.add(self.mon)

    def get_state(self):
        try:
            return self.mon.rate[-1]
        except IndexError:
            return 0

class VoltageRecorder(Recorder):
    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.mon = None

    def connect_to_neurons(self, neuron_group):
        self.mon = StateMonitor(neuron_group, 'v', self.i)
        self.brian_objects.add(self.mon)

    def get_state(self):
        try:
            return self.mon.v[:, -1]
        except IndexError:
            return None
