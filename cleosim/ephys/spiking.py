"""Contains multi-unit and sorted spiking signals"""
from __future__ import annotations
from typing import Any, Tuple

from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor
import numpy as np
import numpy.typing as npt

from cleosim.ephys import Signal


class Spiking(Signal):
    perfect_detection_radius: Quantity
    half_detection_radius: Quantity
    cutoff_probability: float
    save_history: bool
    t_ms: npt.NDArray
    i: npt.NDArray[np.uint]
    t_samp_ms: Quantity
    monitors: list[SpikeMonitor]
    _mon_spikes_already_seen: list[int]
    _dtct_prob_array: npt.NDArray = None
    i_eg_by_i_ng: bidict

    def __init__(
        self,
        name: str,
        perfect_detection_radius: Quantity,
        half_detection_radius: Quantity = None,
        cutoff_probability: float = 0.01,
        save_history: bool = False,
    ) -> None:
        super().__init__(name)
        self.perfect_detection_radius = perfect_detection_radius
        self.half_detection_radius = half_detection_radius
        self.cutoff_probability = cutoff_probability
        self.save_history = save_history
        if save_history:
            self.t_ms = np.array([], dtype=float)
            self.i = np.array([], dtype=np.uint)
            self.t_samp_ms = np.array([], dtype=float)

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        super().connect_to_neuron_group(neuron_group, **kwparams)
        # n_neurons X n_channels X 3
        dist2 = np.zeros((len(neuron_group), self.electrode_group.n))
        for dim in ["x", "y", "z"]:
            dim_ng, dim_eg = np.meshgrid(
                getattr(neuron_group, dim), getattr(self.electrode_group, f"{dim}s")
            )
            dist2 += (dim_ng - dim_eg) ** 2
        # distances[:, :, 0] = (self.electrode_group.xs - neuron_group.x) ** 2
        # distances[:, :, 1] = (self.electrode_group.ys - neuron_group.y) ** 2
        # distances[:, :, 2] = (self.electrode_group.zs - neuron_group.z) ** 2
        # distances = np.sqrt(np.sum(distances, axis=2))
        distances = np.sqrt(dist2)
        # probs is n_neurons by n_channels
        probs = self._detection_prob_for_distance(distances)
        # cut off to get indices of neurons to monitor
        i_ng_to_keep = np.nonzero(np.all(probs > self.cutoff_probability, axis=1))
        # create monitor
        self.monitors.append(SpikeMonitor(neuron_group, record=i_ng_to_keep))
        # update bidict from electrode group index to (neuron group, index)
        i_eg_start = len(self.i_eg_by_i_ng)
        new_i_eg = range(i_eg_start, i_eg_start + len(i_ng_to_keep))
        self.i_eg_by_i_ng.update(
            {i_eg: (neuron_group, i_ng) for i_eg, i_ng in zip(new_i_eg, i_ng_to_keep)}
        )
        # return neuron-channel detection probabilities array for use in subclass
        return probs[i_ng_to_keep]

    def _detection_prob_for_distance(self, d) -> float:
        # p(d) = h/(d-c)
        a = self.perfect_detection_radius
        b = self.half_detection_radius
        # solving for p(a) = 1 and p(b) = .5 yields:
        c = 2 * a - b
        h = b - a
        decaying_p = h / (d - c)
        return 1 * (d < a) + decaying_p * (d >= a)

    def _i_ng_to_i_eg(self, i_ng, monitor):
        ng = monitor.source
        return np.array([self.i_eg_by_i_ng[(ng, k)] for k in i_ng])

    def _get_new_spikes(self) -> Tuple[npt.NDArray, npt.NDarray]:
        i_eg = np.array([], dtype=np.uint)
        t = np.array([], dtype=float)
        for j in range(len(self.monitors)):
            mon = self.monitors[j]
            spikes_already_seen = self._mon_spikes_already_seen[j]
            i_ng = mon.i[spikes_already_seen:]
            i_eg = np.concatenate(i_eg, self._i_ng_to_i_eg(i_ng, mon))
            t = np.concatenate(t, mon.t[spikes_already_seen:])
            self._mon_spikes_already_seen[j] = len(mon.i)

        return (i_eg, t)

    def _get_neuron_per_channel_probs(self, ng) -> npt.NDArray:
        pass


class MultiUnitSpiking(Spiking):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        neuron_channel_dtct_probs = super().connect_to_neuron_group(
            neuron_group, **kwparams
        )
        if self._dtct_prob_array is None:
            self._dtct_prob_array = neuron_channel_dtct_probs
        else:
            self._dtct_prob_array = np.concatenate(
                (self._dtct_prob_array, neuron_channel_dtct_probs), axis=0
            )

    def get_state(
        self,
    ) -> tuple[npt.NDArray[np.uint], npt.NDArray[np.uint]]:
        i_eg, t = self._get_new_spikes()
        i_c, t, y = self._noisily_detect_spikes_per_channel(i_eg, t)
        if self.save_history:
            self.i = np.concatenate(self.i, i_c)
        return (i_c, t, y)

    def _noisily_detect_spikes_per_channel(
        self, i_eg, t
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        probs_for_spikes = self._dtct_prob_array[i_eg]
        detected_spikes = np.random.random(probs_for_spikes.shape) < probs_for_spikes
        y = np.sum(detected_spikes, axis=0)
        # ⬇ nonzero gives row, column indices of each nonzero element
        i_eg_detected, i_c_detected = detected_spikes.nonzero()
        t_detected = t[i_eg_detected]
        return i_c_detected, t_detected, y


class SortedSpiking(Spiking):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # combine probabilities for each neuron to get just one representing
        # when a spike is detected on any channel
        return super().connect_to_neuron_group(neuron_group, **kwparams)
