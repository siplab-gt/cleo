from cleosim.base import Stimulator


class StateVariableSetter(Stimulator):
    def __init__(self, name, variable_to_ctrl, unit, start_value=0):
        super().__init__(name, start_value)
        self.neuron_groups = []
        self.var = variable_to_ctrl
        self.unit = unit
        self.value = start_value

    def connect_to_neuron_group(self, neuron_group):
        self.neuron_groups.append(neuron_group)
        self.update(self.value)

    def update(self, ctrl_signal):
        for ng in self.neuron_groups:
            setattr(ng, self.var, ctrl_signal * self.unit)
