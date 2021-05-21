from brian2 import PopulationRateMonitor, StateMonitor, SpikeMonitor
import numpy as np

from .. import Recorder


class RateRecorder(Recorder):
    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.mon = None

    def connect_to_neuron_group(self, neuron_group):
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

    def connect_to_neuron_group(self, neuron_group):
        self.mon = StateMonitor(neuron_group, "v", self.i)
        self.brian_objects.add(self.mon)

    def get_state(self):
        try:
            return self.mon.v[:, -1]
        except IndexError:
            return None


class GroundTruthSpikeRecorder(Recorder):
    """
    Reports the number of spikes seen since last queried for each neuron,
    so effectively the number of spikes per control period.
    Note: this will only work for one neuron group at the moment.
    """

    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.mon = None
        self.num_spikes_seen = 0
        if isinstance(self.i, int):
            self.out_template = np.zeros(1)
        else:
            self.out_template = np.zeros(len(index))
            # i is index with respect to neuron group
            # j is index with respect to output array
            # self.i_to_j = dict(zip(self.i, range(len(index))))

    def connect_to_neuron_group(self, neuron_group):
        # self.mon = SpikeMonitor(neuron_group, record=self.i)
        self.mon = SpikeMonitor(neuron_group[self.i])
        self.brian_objects.add(self.mon)

    def get_state(self):
        num_new_spikes = len(self.mon.t) - self.num_spikes_seen
        self.num_spikes_seen += num_new_spikes
        if len(self.out_template) == 1:
            out = np.array([num_new_spikes])
        else:
            out = self.out_template.copy()
            for spike_i in self.mon.i[-num_new_spikes:]:
                out[spike_i] += 1
        return out
