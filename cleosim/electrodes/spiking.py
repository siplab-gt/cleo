"""Contains multi-unit and sorted spiking signals"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any, Tuple

from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor, meter, ms
import numpy as np

# import numpy.typing as npt
from nptyping import NDArray

from cleosim.electrodes.probes import Signal


class Spiking(Signal):
    """Base class for probabilistically detecting spikes"""

    perfect_detection_radius: Quantity
    """Radius (with Brian unit) within which all spikes
    are detected"""
    half_detection_radius: Quantity
    """Radius (with Brian unit) within which half of all spikes
    are detected"""
    cutoff_probability: float
    """Spike detection probability below which neurons will not be
    considered. For computational efficiency."""
    save_history: bool
    """Determines whether :attr:`t_ms`, :attr:`i`, and :attr:`t_samp_ms` are recorded"""
    t_ms: NDArray[(Any,), float]
    """Spike times in ms, stored if :attr:`save_history`"""
    i: NDArray[(Any,), np.uint]
    """Channel (for multi-unit) or neuron (for sorted) indices
    of spikes, stored if :attr:`save_history`"""
    t_samp_ms: NDArray[(Any,), float]
    """Sample times in ms when each spike was recorded, stored 
    if :attr:`save_history`"""
    i_probe_by_i_ng: bidict
    """(neuron_group, i_ng) keys,  i_probe values. bidict for converting between
    neuron group indices and the indices the probe uses"""
    _monitors: list[SpikeMonitor]
    _mon_spikes_already_seen: list[int]
    _dtct_prob_array: NDArray[(Any, Any), float]

    def __init__(
        self,
        name: str,
        perfect_detection_radius: Quantity,
        half_detection_radius: Quantity = None,
        cutoff_probability: float = 0.01,
        save_history: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Unique identifier for signal
        perfect_detection_radius : Quantity
            Radius (with Brian unit) within which all spikes
            are detected
        half_detection_radius : Quantity, optional
            Radius (with Brian unit) within which half of all spikes
            are detected
        cutoff_probability : float, optional
            Spike detection probability below which neurons will not be
            considered, by default 0.01. For computational efficiency.
        save_history : bool, optional
            If True, will save t_ms (spike times), i (neuron or
            channel index), and t_samp_ms (sample times) as attributes.
            By default False
        """
        super().__init__(name)
        self.perfect_detection_radius = perfect_detection_radius
        self.half_detection_radius = half_detection_radius
        self.cutoff_probability = cutoff_probability
        self.save_history = save_history
        self._init_saved_vars()
        self._monitors = []
        self._mon_spikes_already_seen = []
        self._dtct_prob_array = None
        self.i_probe_by_i_ng = bidict()

    def _init_saved_vars(self):
        if self.save_history:
            self.t_ms = np.array([], dtype=float)
            self.i = np.array([], dtype=np.uint)
            self.t_samp_ms = np.array([], dtype=float)

    def _update_saved_vars(self, t_ms, i, t_samp_ms):
        if self.save_history:
            self.i = np.concatenate([self.i, i])
            self.t_ms = np.concatenate([self.t_ms, t_ms])
            self.t_samp_ms = np.concatenate([self.t_samp_ms, [t_samp_ms]])

    def connect_to_neuron_group(
        self, neuron_group: NeuronGroup, **kwparams
    ) -> np.ndarray:
        """Configure signal to record from specified neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            Neuron group to record from

        Returns
        -------
        np.ndarray
            num_neurons_to_consider x num_channels array of spike
            detection probabilities, for use in subclasses
        """
        super().connect_to_neuron_group(neuron_group, **kwparams)
        # could support separate detection probabilities per group using kwparams
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

    @abstractmethod
    def get_state(
        self,
    ) -> tuple[NDArray[np.uint], NDArray[float], NDArray[np.uint]]:
        """Return spikes since method was last called (i, t_ms, y)

        Returns
        -------
        tuple[NDArray[np.uint], NDArray[float], NDArray[np.uint]]
            (i, t_ms, y) where i is channel (for multi-unit) or neuron (for sorted) spike
            indices, t_ms is spike times, and y is a spike count vector suitable for control-
            theoretic uses---i.e., a 0 for every channel/neuron that hasn't spiked and a 1
            for a single spike.
        """
        pass

    def _detection_prob_for_distance(self, r: Quantity) -> float:
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

    def reset(self, **kwargs) -> None:
        # crucial that this be called after network restore
        # since that would reset monitors
        for j, mon in enumerate(self._monitors):
            self._mon_spikes_already_seen[j] = mon.num_spikes
        self._init_saved_vars()


class MultiUnitSpiking(Spiking):
    """Detects spikes per channel, that is, unsorted."""

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Configure signal to record from specified neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            group to record from
        """
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
    ) -> tuple[NDArray[np.uint], NDArray[float], NDArray[np.uint]]:
        # inherit docstring
        i_probe, t_ms = self._get_new_spikes()
        t_samp_ms = self.probe.sim.network.t / ms
        i_c, t_ms, y = self._noisily_detect_spikes_per_channel(i_probe, t_ms)
        self._update_saved_vars(t_ms, i_c, t_samp_ms)
        return (i_c, t_ms, y)

    def _noisily_detect_spikes_per_channel(
        self, i_probe, t
    ) -> Tuple[NDArray, NDArray, NDArray]:
        probs_for_spikes = self._dtct_prob_array[i_probe]
        detected_spikes = np.random.random(probs_for_spikes.shape) < probs_for_spikes
        y = np.sum(detected_spikes, axis=0)
        # ⬇ nonzero gives row, column indices of each nonzero element
        i_spike_detected, i_c_detected = detected_spikes.nonzero()
        t_detected = t[i_spike_detected]
        return i_c_detected, t_detected, y


class SortedSpiking(Spiking):
    """Detect spikes identified by neuron indices.

    The indices used by the probe do not correspond to those
    coming from neuron groups, since the probe must consider
    multiple potential groups and within a group ignores those
    neurons that are too far away to be easily detected."""

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Configure sorted spiking signal to record from given neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            group to record from
        """
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

    def _combine_channel_probs(self, neuron_channel_dtct_probs: NDArray) -> NDArray:
        # combine probabilities for each neuron to get just one representing
        # when a spike is detected on at least 1 channel
        # p(at least one detected) = 1 - p(none detected) = 1 - q1*q2*q3...
        return 1 - np.prod(1 - neuron_channel_dtct_probs, axis=1)

    def get_state(
        self,
    ) -> tuple[NDArray[np.uint], NDArray[float], NDArray[np.uint]]:
        # inherit docstring
        i_probe, t_ms = self._get_new_spikes()
        i_probe, t_ms = self._noisily_detect_spikes(i_probe, t_ms)
        y = np.zeros(len(self.i_probe_by_i_ng), dtype=bool)
        y[i_probe] = 1
        t_samp_ms = self.probe.sim.network.t / ms
        self._update_saved_vars(t_ms, i_probe, t_samp_ms)
        return (i_probe, t_ms, y)

    def _noisily_detect_spikes(self, i_probe, t) -> Tuple[NDArray, NDArray]:
        probs_for_spikes = self._dtct_prob_array[i_probe]
        detected_spikes = np.random.random(probs_for_spikes.shape) < probs_for_spikes
        i_spike_detected = detected_spikes.nonzero()
        i_probe_out = i_probe[i_spike_detected]
        t_out = t[i_spike_detected]
        return i_probe_out, t_out
