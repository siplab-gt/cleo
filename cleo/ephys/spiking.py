"""Contains multi-unit and sorted spiking signals"""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any, Tuple

import neo
import numpy as np
import numpy.typing as npt
import quantities as pq
from attrs import define, field, fields
from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor, mm, ms

from cleo.base import NeoExportable
from cleo.ephys.probes import Signal
from cleo.utilities import rng, unit_safe_cat


@define(eq=False)
class Spiking(Signal, NeoExportable):
    """Base class for probabilistically detecting spikes"""

    r_perfect_detection: Quantity
    """Radius (with Brian unit) within which all spikes
    are detected"""
    r_half_detection: Quantity
    """Radius (with Brian unit) within which half of all spikes
    are detected"""
    cutoff_probability: float = 0.01
    """Spike detection probability below which neurons will not be
    considered. For computational efficiency."""
    t: Quantity = field(
        init=False, factory=lambda: ms * np.array([], dtype=float), repr=False
    )
    """Spike times in ms, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    i: UInt[np.ndarray, "n_recorded_spikes"] = field(
        init=False, factory=lambda: np.array([], dtype=np.uint), repr=False
    )
    """Channel (for multi-unit) or neuron (for sorted) indices
    of spikes, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    t_samp: Quantity = field(
        init=False, factory=lambda: ms * np.array([], dtype=float), repr=False
    )
    """Sample times in ms when each spike was recorded, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    i_probe_by_i_ng: bidict = field(init=False, factory=bidict, repr=False)
    """(neuron_group, i_ng) keys,  i_probe values. bidict for converting between
    neuron group indices and the indices the probe uses"""
    _monitors: list[SpikeMonitor] = field(init=False, factory=list, repr=False)
    _mon_spikes_already_seen: list[int] = field(init=False, factory=list, repr=False)
    _dtct_prob_array: Float[np.ndarray, "n_neurons n_channels"] = field(
        init=False, default=None, repr=False
    )

    def _init_saved_vars(self):
        if self.probe.save_history:
            self.t = fields(type(self)).t.default.factory()
            self.i = fields(type(self)).i.default.factory()
            self.t_samp = fields(type(self)).t_samp.default.factory()

    def _update_saved_vars(self, t, i, t_samp):
        if self.probe.save_history:
            self.i = np.concatenate([self.i, i])
            self.t = unit_safe_cat([self.t, t])
            t_samp_rep = np.full_like(t, t_samp)
            self.t_samp = unit_safe_cat([self.t_samp, t_samp_rep])

    @property
    @abstractmethod
    def n(self):
        """Number of spike sources"""
        pass

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
        super(Spiking, self).connect_to_neuron_group(neuron_group, **kwparams)
        # could support separate detection probabilities per group using kwparams
        # n_neurons X n_channels X 3
        dist2 = np.zeros((len(neuron_group), self.probe.n))
        for dim in ["x", "y", "z"]:
            dim_ng, dim_probe = np.meshgrid(
                getattr(neuron_group, dim),
                getattr(self.probe, f"{dim}s"),
                indexing="ij",
            )
            # proactively strip units to avoid numpy maybe doing so
            dist2 += (dim_ng / mm - dim_probe / mm) ** 2
        distances = np.sqrt(dist2) * mm
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
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        """Return spikes since method was last called (i, t, y)

        Returns
        -------
        tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]
            (i, t, y) where i is channel (for multi-unit) or neuron (for sorted) spike
            indices, t is spike times, and y is a spike count vector suitable for control-
            theoretic uses---i.e., a 0 for every channel/neuron that hasn't spiked and a 1
            for a single spike.
        """
        pass

    def _detection_prob_for_distance(self, r: Quantity) -> float:
        # p(d) = h/(r-c)
        a = self.r_perfect_detection
        b = self.r_half_detection
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

    def _get_new_spikes(self) -> Tuple[UInt[np.ndarray, "n_spikes"], Quantity]:
        i_probe = np.array([], dtype=np.uint)
        t = ms * np.array([], dtype=float)
        for j in range(len(self._monitors)):
            mon = self._monitors[j]
            spikes_already_seen = self._mon_spikes_already_seen[j]
            i_ng = mon.i[spikes_already_seen:]  # can contain spikes we don't care about
            i_probe = np.concatenate((i_probe, self._i_ng_to_i_probe(i_ng, mon)))
            t = unit_safe_cat([t, mon.t[spikes_already_seen:]])
            self._mon_spikes_already_seen[j] = mon.num_spikes

        return i_probe, t

    def reset(self, **kwargs) -> None:
        # crucial that this be called after network restore
        # since that would reset monitors
        for j, mon in enumerate(self._monitors):
            self._mon_spikes_already_seen[j] = mon.num_spikes
        self._init_saved_vars()

    def to_neo(self) -> neo.Group:
        group = neo.Group(allowed_types=[neo.SpikeTrain])
        for i in set(self.i):
            st = neo.SpikeTrain(
                times=self.t[self.i == i] / ms * pq.ms,
                t_stop=self.probe.sim.network.t / ms * pq.ms,
            )
            st.annotate(i=int(i))
            group.add(st)

        group.annotate(export_datetime=datetime.now())
        group.name = f"{self.probe.name}.{self.name}"
        group.description = f"Exported from Cleo {self.__class__.__name__} object"
        return group


@define(eq=False)
class MultiUnitSpiking(Spiking):
    """Detects (unsorted) spikes per channel."""

    @property
    def n(self):
        """Number of channels on probe"""
        return self.probe.n

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Configure signal to record from specified neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            group to record from
        """
        neuron_channel_dtct_probs = super(
            MultiUnitSpiking, self
        ).connect_to_neuron_group(neuron_group, **kwparams)
        if self._dtct_prob_array is None:
            self._dtct_prob_array = neuron_channel_dtct_probs
        else:
            self._dtct_prob_array = np.concatenate(
                (self._dtct_prob_array, neuron_channel_dtct_probs), axis=0
            )

    def get_state(
        self,
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        # inherit docstring
        i_probe, t = self._get_new_spikes()
        t_samp = self.probe.sim.network.t
        i_c, t, y = self._noisily_detect_spikes_per_channel(i_probe, t)
        self._update_saved_vars(t, i_c, t_samp)
        return (i_c, t, y)

    def _noisily_detect_spikes_per_channel(
        self, i_probe, t
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        probs_for_spikes = self._dtct_prob_array[i_probe]
        detected_spikes = rng.random(probs_for_spikes.shape) < probs_for_spikes
        y = np.sum(detected_spikes, axis=0)
        # ⬇ nonzero gives row, column indices of each nonzero element
        i_spike_detected, i_c_detected = detected_spikes.nonzero()
        t_detected = t[i_spike_detected]
        return i_c_detected, t_detected, y

    def to_neo(self) -> neo.Group:
        group = super(MultiUnitSpiking, self).to_neo()
        for st in group.spiketrains:
            i = int(st.annotations["i"])
            st.annotate(
                i_channel=i,
                x_contact=self.probe.coords[i, 0] / mm * pq.mm,
                y_contact=self.probe.coords[i, 1] / mm * pq.mm,
                z_contact=self.probe.coords[i, 2] / mm * pq.mm,
            )
        return group


@define(eq=False)
class SortedSpiking(Spiking):
    """Detect spikes identified by neuron indices.

    The indices used by the probe do not correspond to those
    coming from neuron groups, since the probe must consider
    multiple potential groups and within a group ignores those
    neurons that are too far away to be easily detected."""

    @property
    def n(self):
        """Number of recorded neurons"""
        return len(self.i_probe_by_i_ng)

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Configure sorted spiking signal to record from given neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            group to record from
        """
        neuron_channel_dtct_probs = super(SortedSpiking, self).connect_to_neuron_group(
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

    def _combine_channel_probs(
        self, neuron_channel_dtct_probs: np.ndarray
    ) -> np.ndarray:
        # combine probabilities for each neuron to get just one representing
        # when a spike is detected on at least 1 channel
        # p(at least one detected) = 1 - p(none detected) = 1 - q1*q2*q3...
        return 1 - np.prod(1 - neuron_channel_dtct_probs, axis=1)

    def get_state(
        self,
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        # inherit docstring
        i_probe, t = self._get_new_spikes()
        i_probe, t = self._noisily_detect_spikes(i_probe, t)
        y = np.bincount(i_probe.astype(int))
        # include 0s for upper indices not seen:
        y = np.concatenate([y, np.zeros(len(self.i_probe_by_i_ng) - len(y))])
        t_samp = self.probe.sim.network.t / ms
        self._update_saved_vars(t, i_probe, t_samp)
        return (i_probe, t, y)

    def _noisily_detect_spikes(self, i_probe, t) -> Tuple[np.ndarray, np.ndarray]:
        # dtct_prob_array: n_nrns x n_channels
        probs_for_spikes = self._dtct_prob_array[i_probe]
        # now n_spks x n_channels
        detected_spikes = rng.random(probs_for_spikes.shape) < probs_for_spikes
        i_spike_detected = detected_spikes.nonzero()
        i_probe_out = i_probe[i_spike_detected]
        t_out = t[i_spike_detected]
        return i_probe_out, t_out
