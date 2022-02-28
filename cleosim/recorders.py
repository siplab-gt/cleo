"""Contains basic recorders."""
from typing import Any
from brian2 import PopulationRateMonitor, StateMonitor, SpikeMonitor, Quantity
import numpy as np
from nptyping import NDArray

from cleosim.base import Recorder


class RateRecorder(Recorder):
    """Records firing rate from a single neuron.

    Firing rate comes from Brian's :class:`~brian2.monitors.ratemonitor.PopulationRateMonitor`"""

    def __init__(self, name: str, index: int):
        """
        Parameters
        ----------
        name : str
            Unique device name
        index : int
            index of neuron to record
        """
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
    """Records the voltage of a single neuron group."""

    def __init__(self, name: str, voltage_var_name: str = "v"):
        """
        Parameters
        ----------
        name : str
            Unique device name
        voltage_var_name : str, optional
            Name of variable representing membrane voltage, by default "v"
        """
        super().__init__(name)
        self.voltage_var_name = voltage_var_name
        self.mon = None

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


class GroundTruthSpikeRecorder(Recorder):
    """Reports the number of spikes seen since last queried for each neuron.

    This amounts effectively to the number of spikes per control period.
    Note: this will only work for one neuron group at the moment.
    """

    def __init__(self, name):
        super().__init__(name)
        self.mon = None
        self.num_spikes_seen = 0

    def connect_to_neuron_group(self, neuron_group):
        if self.mon is not None:
            raise UserWarning(
                "Recorder was already connected to a neuron group. "
                "Can only record from one at a time."
            )
        self.mon = SpikeMonitor(neuron_group)
        self.brian_objects.add(self.mon)
        self.out_template = np.zeros(len(neuron_group))

    def get_state(self) -> NDArray[(Any,), np.uint]:
        """
        Returns
        -------
        NDArray[(n_neurons,), np.uint]
            n_neurons-length array with spike counts over the latest
            control period.
        """
        num_new_spikes = len(self.mon.t) - self.num_spikes_seen
        self.num_spikes_seen += num_new_spikes
        if len(self.out_template) == 1:
            out = np.array([num_new_spikes])
        else:
            out = self.out_template.copy()
            for spike_i in self.mon.i[-num_new_spikes:]:
                out[spike_i] += 1
        return out
