from brian2 import PopulationRateMonitor, StateMonitor, SpikeMonitor
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

class SpikeRecorder(Recorder):
    '''
    Reports the number of spikes seen since last queried, so effectively
    the number of spikes per control period.
    '''
    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.mon = None
        self.num_spikes_seen = 0

    def connect_to_neurons(self, neuron_group):
        self.mon = SpikeMonitor(neuron_group, record=self.i)
        self.brian_objects.add(self.mon)
    
    def get_state(self):
        num_new_spikes = len(self.mon.t) - self.num_spikes_seen
        self.num_spikes_seen += num_new_spikes
        return num_new_spikes