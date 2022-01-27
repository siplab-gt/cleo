"""Contains multi-unit and sorted spiking signals"""
from __future__ import annotations
from typing import Any, Tuple

from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor, meter, ms
import numpy as np
import numpy.typing as npt

from cleosim.electrodes.probes import Signal


class Spiking(Signal):
    perfect_detection_radius: Quantity
    half_detection_radius: Quantity
    cutoff_probability: float
    save_history: bool
    t_ms: npt.NDArray
    i: npt.NDArray[np.uint]
    t_samp_ms: Quantity
    _monitors: list[SpikeMonitor]
    _mon_spikes_already_seen: list[int]
    _dtct_prob_array: npt.NDArray
    i_probe_by_i_ng: bidict

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
        self._monitors = []
        self._mon_spikes_already_seen = []
        self._dtct_prob_array = None
        self.i_probe_by_i_ng = bidict()

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        super().connect_to_neuron_group(neuron_group, **kwparams)
        # n_neurons X n_channels X 3
        dist2 = np.zeros((len(neuron_group), self.probe.n))
        for dim in ["x", "y", "z"]:
            dim_ng, dim_probe = np.meshgrid(
                getattr(neuron_group, dim),
                getattr(self.probe, f"{dim}s"),
                indexing="ij",
            )
            dist2 += (dim_ng - dim_probe) ** 2
        distances = np.sqrt(dist2) * meter  # since units stripped
        # probs is n_neurons by n_channels
        probs = self._detection_prob_for_distance(distances)
        # cut off to get indices of neurons to monitor
        # [0] since nonzero returns tuple of array per axis
        i_ng_to_keep = np.nonzero(np.all(probs > self.cutoff_probability, axis=1))[0]

        if len(i_ng_to_keep) > 0:
            # create monitor
            mon = SpikeMonitor(neuron_group)
            self._monitors.append(mon)
            self.brian_objects.add(mon)
            self._mon_spikes_already_seen.append(0)

            # update bidict from electrode group index to (neuron group, index)
            i_probe_start = len(self.i_probe_by_i_ng)
            new_i_probe = range(i_probe_start, i_probe_start + len(i_ng_to_keep))
            self.i_probe_by_i_ng.update(
                {
                    (neuron_group, i_ng): i_probe
                    for i_probe, i_ng in zip(new_i_probe, i_ng_to_keep)
                }
            )

        # return neuron-channel detection probabilities array for use in subclass
        return probs[i_ng_to_keep]

    def _detection_prob_for_distance(self, r) -> float:
        # p(d) = h/(r-c)
        a = self.perfect_detection_radius
        b = self.half_detection_radius
        # solving for p(a) = 1 and p(b) = .5 yields:
        c = 2 * a - b
        h = b - a
        with np.errstate(divide="ignore"):
            decaying_p = h / (r - c)
        decaying_p[decaying_p == np.inf] = 1  # to fix NaNs caused by /0
        p = 1 * (r <= a) + decaying_p * (r > a)
        return p

    def _i_ng_to_i_probe(self, i_ng, monitor):
        ng = monitor.source
        # assign value of -1 to neurons we aren't recording to filter out
        i_probe_unfilt = np.array([self.i_probe_by_i_ng.get((ng, k), -1) for k in i_ng])
        return i_probe_unfilt[i_probe_unfilt != -1].astype(np.uint)

    def _get_new_spikes(self) -> Tuple[npt.NDArray, npt.NDarray]:
        i_probe = np.array([], dtype=np.uint)
        t_ms = np.array([], dtype=float)
        for j in range(len(self._monitors)):
            mon = self._monitors[j]
            spikes_already_seen = self._mon_spikes_already_seen[j]
            i_ng = mon.i[spikes_already_seen:]  # can contain spikes we don't care about
            i_probe = np.concatenate((i_probe, self._i_ng_to_i_probe(i_ng, mon)))
            # get all time in terms of ms
            t_ms = np.concatenate((t_ms, mon.t[spikes_already_seen:] / ms))
            self._mon_spikes_already_seen[j] = mon.num_spikes

        return (i_probe, t_ms)

    def reset(self, **kwargs):
        # crucial that this be called after network restore
        # since that would reset monitors
        for j in range(len(self._monitors)):
            mon = self._monitors[j]
            self._mon_spikes_already_seen[j] = mon.num_spikes
        if self.save_history:
            self.t_ms = np.array([], dtype=float)
            self.i = np.array([], dtype=np.uint)
            self.t_samp_ms = np.array([], dtype=float)


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
    ) -> tuple[npt.NDArray[np.uint], npt.NDArray, npt.NDArray[np.uint]]:
        i_probe, t_ms = self._get_new_spikes()
        i_c, t_ms, y = self._noisily_detect_spikes_per_channel(i_probe, t_ms)
        if self.save_history:
            self.i = np.concatenate((self.i, i_c))
            self.t_ms = np.concatenate((self.t_ms, t_ms))
        return (i_c, t_ms, y)

    def _noisily_detect_spikes_per_channel(
        self, i_probe, t
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        probs_for_spikes = self._dtct_prob_array[i_probe]
        detected_spikes = np.random.random(probs_for_spikes.shape) < probs_for_spikes
        y = np.sum(detected_spikes, axis=0)
        # ⬇ nonzero gives row, column indices of each nonzero element
        i_spike_detected, i_c_detected = detected_spikes.nonzero()
        t_detected = t[i_spike_detected]
        return i_c_detected, t_detected, y


class SortedSpiking(Spiking):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        neuron_channel_dtct_probs = super().connect_to_neuron_group(
            neuron_group, **kwparams
        )
        neuron_multi_channel_probs = self._combine_channel_probs(
            neuron_channel_dtct_probs
        )
        if self._dtct_prob_array is None:
            self._dtct_prob_array = neuron_multi_channel_probs
        else:
            self._dtct_prob_array = np.concatenate(
                (self._dtct_prob_array, neuron_multi_channel_probs)
            )

    def _combine_channel_probs(self, neuron_channel_dtct_probs: np.array) -> np.array:
        # combine probabilities for each neuron to get just one representing
        # when a spike is detected on at least 1 channel
        # p(at least one detected) = 1 - p(none detected) = 1 - q1*q2*q3...
        return 1 - np.prod(1 - neuron_channel_dtct_probs, axis=1)

    def get_state(
        self,
    ) -> tuple[npt.NDArray[np.uint], npt.NDArray, npt.NDArray[np.uint]]:
        i_probe, t_ms = self._get_new_spikes()
        i_probe, t_ms = self._noisily_detect_spikes(i_probe, t_ms)
        y = np.zeros(len(self.i_probe_by_i_ng), dtype=bool)
        y[i_probe] = 1
        if self.save_history:
            self.i = np.concatenate((self.i, i_probe))
            self.t_ms = np.concatenate((self.t_ms, t_ms))
        return (i_probe, t_ms, y)

    def _noisily_detect_spikes(self, i_probe, t) -> Tuple[npt.NDArray, npt.NDArray]:
        probs_for_spikes = self._dtct_prob_array[i_probe]
        detected_spikes = np.random.random(probs_for_spikes.shape) < probs_for_spikes
        i_spike_detected = detected_spikes.nonzero()
        i_probe_out = i_probe[i_spike_detected]
        t_out = t[i_spike_detected]
        return i_probe_out, t_out
