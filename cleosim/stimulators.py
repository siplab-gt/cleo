"""Contains basic stimulators."""
from brian2 import Unit
from cleosim.base import Stimulator


class StateVariableSetter(Stimulator):
    """Sets the given state variable of target neuron groups."""

    def __init__(
        self, name: str, variable_to_ctrl: str, unit: Unit, start_value: float = 0
    ):
        """
        Parameters
        ----------
        name : str
            Unique device name
        variable_to_ctrl : str
            Name of state variable to control
        unit : Unit
            Unit of that state variable: will be used in :meth:`update`
        start_value : float, optional
            Starting variable value, by default 0
        """
        super().__init__(name, start_value)
        self.neuron_groups = []
        self.var = variable_to_ctrl
        self.unit = unit
        self.value = start_value

    def connect_to_neuron_group(self, neuron_group):
        self.neuron_groups.append(neuron_group)
        self.update(self.value)

    def update(self, ctrl_signal: float) -> None:
        """Set state variable of target neuron groups

        Parameters
        ----------
        ctrl_signal : float
            Value to update variable to, without unit. The unit
            provided on initialization is automatically multiplied.
        """
        for ng in self.neuron_groups:
            setattr(ng, self.var, ctrl_signal * self.unit)
