"""Contains basic stimulators."""
from __future__ import annotations
from attrs import define, field
from brian2 import Unit, NeuronGroup
from cleo.base import Stimulator


@define(eq=False)
class StateVariableSetter(Stimulator):
    """Sets the given state variable of target neuron groups."""

    variable_to_ctrl: str = field(kw_only=True)
    """Name of state variable to control"""

    unit: Unit = field(kw_only=True)
    """Unit of controlled variable: will be used in :meth:`update`"""

    neuron_groups: list[NeuronGroup] = field(init=False, factory=list)

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
            setattr(ng, self.variable_to_ctrl, ctrl_signal * self.unit)
