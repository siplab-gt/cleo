"""Contains multi-unit and sorted spiking signals"""
from typing import Any

from brian2 import NeuronGroup, Quantity
from numpy import half
import numpy.typing as npt

from cleosim.ephys import Signal


class Spiking(Signal):
    perfect_detection_radius: Quantity
    half_detection_radius: Quantity
    cutoff_probability: float

    def __init__(
        self,
        name: str,
        perfect_detection_radius: Quantity,
        half_detection_radius: Quantity = None,
        cutoff_probability: float = 0.01,
    ) -> None:
        super().__init__(name)
        self.perfect_detection_radius = perfect_detection_radius
        self.half_detection_radius = half_detection_radius
        self.cutoff_probability = cutoff_probability

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        super().connect_to_neuron_group(neuron_group, **kwparams)
        distances = 0
        # get probabilties for each channel, neuron
        # cut off to get indices of neurons to monitor
        # create monitor

    def _get_new_spikes(self) -> npt.NDArray:
        pass


class MultiUnitSpiking(Spiking):

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # map from neuron indices to probs/channels rather than
        # channel to indices/probs since the data coming from the monitor
        # will be in terms of indices
        # maybe an array of len(indices) X num. channels?
        # thinking a sparse array would be better than an {index: prob per channel} dict
        return super().connect_to_neuron_group(neuron_group, **kwparams)


    def get_state(self) -> Any:
        return super().get_state()


class SortedSpiking(Spiking):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # combine probabilities for each neuron to get just one representing
        # when a spike is detected on any channel
        return super().connect_to_neuron_group(neuron_group, **kwparams)
