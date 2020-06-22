from .base import Stimulator
from brian2 import NeuronGroup

class StateVariableSetter(Stimulator):
    def __init__(self, name, index, variable_to_ctrl, unit):
        super().__init__(name)
        self.i = index
        self.neurons = None
        self.var = variable_to_ctrl
        self.unit = unit

    def connect_to_neurons(self, neuron_group):
        self.neurons = neuron_group[self.i]

    def update(self, ctrl_signal):
        setattr(self.neurons, self.var, ctrl_signal*self.unit)