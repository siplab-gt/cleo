"""Contains basic recorders."""
from typing import Any
from attrs import define, field
from brian2 import (
    PopulationRateMonitor,
    StateMonitor,
    SpikeMonitor,
    Quantity,
    NeuronGroup,
)
import numpy as np
from nptyping import NDArray

from cleo.base import Recorder


@define(eq=False)
class RateRecorder(Recorder):
    """Records firing rate from a single neuron.

    Firing rate comes from Brian's :class:`~brian2.monitors.ratemonitor.PopulationRateMonitor`
    """

    i: int
    """index of neuron to record"""

    mon: PopulationRateMonitor = field(init=False)

    def connect_to_neuron_group(self, neuron_group):
        self.mon = PopulationRateMonitor(neuron_group[self.i])
        self.brian_objects.add(self.mon)

    def get_state(self):
        try:
            return self.mon.rate[-1]
        except IndexError:
            return 0


@define(eq=False)
class VoltageRecorder(Recorder):
    """Records the voltage of a single neuron group."""

    voltage_var_name: str = "v"
    """Name of variable representing membrane voltage"""

    mon: StateMonitor = field(init=False, default=None)

    def connect_to_neuron_group(self, neuron_group):
        if self.mon is not None:
            raise UserWarning(
                "Recorder was already connected to a neuron group. "
                "Can only record from one at a time."
            )
        self.mon = StateMonitor(neuron_group, self.voltage_var_name, record=True)
        self.brian_objects.add(self.mon)

    def get_state(self) -> Quantity:
        """
        Returns
        -------
        Quantity
            Current voltage of target neuron group
        """
        try:
            return self.mon.v[:, -1]
        except IndexError:
            return None


@define(eq=False)
class GroundTruthSpikeRecorder(Recorder):
    """Reports the number of spikes seen since last queried for each neuron.

    This amounts effectively to the number of spikes per control period.
    Note: this will only work for one neuron group at the moment.
    """

    _mon: SpikeMonitor = field(init=False, default=None)

    _num_spikes_seen: int = field(init=False, default=0)

    neuron_group: NeuronGroup = field(init=False, default=None)

    def connect_to_neuron_group(self, neuron_group):
        if self._mon is not None:
            raise UserWarning(
                "Recorder was already connected to a neuron group. "
                "Can only record from one at a time."
            )
        self._mon = SpikeMonitor(neuron_group)
        self.brian_objects.add(self._mon)
        self.neuron_group = neuron_group

    def get_state(self) -> NDArray[(Any,), np.uint]:
        """
        Returns
        -------
        NDArray[(n_neurons,), np.uint]
            n_neurons-length array with spike counts over the latest
            control period.
        """
        num_new_spikes = len(self._mon.t) - self._num_spikes_seen
        self._num_spikes_seen += num_new_spikes
        out = np.zeros(len(self.neuron_group), dtype=np.uint)
        if num_new_spikes > 0:
            for spike_i in self._mon.i[-num_new_spikes:]:
                out[spike_i] += 1
        return out
