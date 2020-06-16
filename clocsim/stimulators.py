from .base import Stimulator
from brian2 import NeuronGroup

class CurrentPulseStimulator(Stimulator):
    def __init__(self, name, index):
        super().__init__(name)
        self.i = index
        self.neuron_group = None

    def connect_to_neurons(self, neuron_group):
        self.neurons = neuron_group[self.i]

    def update(self, ctrl_signal):
        self.neurons.I = ctrl_signal