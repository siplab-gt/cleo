"""Contains multi-unit and sorted spiking signals."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Callable, Tuple

import neo
import numpy as np
import quantities as pq
from attrs import define, field, fields
from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor, mm, ms, um
from jaxtyping import Bool, Float, UInt
from scipy.stats import norm

from cleo.base import NeoExportable
from cleo.ephys.probes import Signal
from cleo.utilities import rng, unit_safe_cat


@define(eq=False)
class Spiking(Signal, NeoExportable):
    """Base class for probabilistically detecting spikes.

    See `notebooks/spike_detection.py` for an interactive explanation of the methods
    and parameters involved."""

    r_noise_floor: Quantity = 80 * um
    """Radius (with Brian unit) at which the measured spike amplitude
    equals the background noise standard deviation. 
    i.e., `spike_amplitude(r_noise_floor) = sigma_noise = 1`"""
    threshold_sigma: int = 4
    """Spike detection threshold, as a multiple of sigma_noise. 
    Values in real experiments typically range from 3 to 6."""
    spike_amplitude_cv: float = 0.05
    """Coefficient of variation of the spike amplitude, i.e., `|sigma_amp/mu_amp|`. 
    From what we have seen in Allen Cell Types data, this ranges from 0 to 0.2, 
    but is most often very low."""
    r0: Quantity = 5 * um
    """A small distance added to r before computing the amplitude to avoid division
    by 0 for the power law decay. 
    
    It also makes some physical sense as the minimum distance from the current source
    it is possible to place an electrode, 5 μm being reasonable as the radius of a typical soma."""
    cutoff_probability: float = 0.01
    """Spike detection probability below which neurons will not be
    considered. For computational efficiency."""
    eap_decay_fn: Callable[[Quantity], float] = lambda r: r**-2
    """The function describing the decay of the measured extracellular action potential
    amplitude. By default 1/r^2.
    
    This inverse square decay is a good approximation in accordance with the detailed
    simulations by `Pettersen et al. (2008) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2186261/>`_,
    though they find the exponent ranges from 2 to 3 depending on the cell type and distance. """
    t: Quantity = field(
        init=False, factory=lambda: ms * np.array([], dtype=float), repr=False
    )
    """Spike times with Brian units, stored if
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
    """Sample times with Brian units when each spike was recorded, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    i_probe_by_i_ng: bidict = field(init=False, factory=bidict, repr=False)
    """(neuron_group, i_ng) keys,  i_probe values. bidict for converting between
    neuron group indices and the indices the probe uses"""
    _monitors: list[SpikeMonitor] = field(init=False, factory=list, repr=False)
    _mon_spikes_already_seen: list[int] = field(init=False, factory=list, repr=False)
    _mu_eap: Float[np.ndarray, "n_neurons n_channels"] = field(
        init=False, default=None, repr=False
    )
    _sigma_eap: Float[np.ndarray, "n_neurons n_channels"] = field(
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

    @property
    def n_channels(self) -> int:
        """Number of channels on probe"""
        return self.probe.n

    def mean_eap_amplitude(self, r: Quantity) -> float:
        """The mean extracellular action potential amplitude, as a multiple
        of background noise standard deviation."""
        return self.eap_decay_fn(r + self.r0) / self.eap_decay_fn(
            self.r0 + self.r_noise_floor
        )

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        """Configure signal to record from specified neuron group

        Parameters
        ----------
        neuron_group : NeuronGroup
            Neuron group to record from
        """
        super(Spiking, self).connect_to_neuron_group(neuron_group, **kwparams)
        # could support separate detection probabilities per group using kwparams
        # n_neurons X n_channels X 3
        dist2 = np.zeros((len(neuron_group), self.n_channels))
        for dim in ["x", "y", "z"]:
            dim_ng, dim_probe = np.meshgrid(
                getattr(neuron_group, dim),
                getattr(self.probe, f"{dim}s"),
                indexing="ij",
            )
            # proactively strip units to avoid numpy maybe doing so
            dist2 += (dim_ng / mm - dim_probe / mm) ** 2
        distances = np.sqrt(dist2) * mm
        mu_eap = self.mean_eap_amplitude(distances)
        # 1 from baseline noise, mu * cv from spike amp variability
        sigma_eap = 1 + mu_eap * self.spike_amplitude_cv
        probs = norm.sf(self.threshold_sigma, loc=mu_eap, scale=sigma_eap)
        assert probs.shape == (len(neuron_group), self.n_channels)
        # probs is n_neurons by n_channels
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

        # store neuron-channel mean and stdev of EAPs to use in subclasses
        if self._mu_eap is None:
            self._mu_eap = mu_eap[i_ng_to_keep]
            self._sigma_eap = sigma_eap[i_ng_to_keep]
        else:
            self._mu_eap = np.concatenate((self._mu_eap, mu_eap[i_ng_to_keep]), axis=0)
            self._sigma_eap = np.concatenate(
                (self._sigma_eap, sigma_eap[i_ng_to_keep]), axis=0
            )

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

    def _noisily_detect_spikes(
        self, i_probe, t
    ) -> Tuple[Bool[np.ndarray, "n_spikes {self.n_channels}"], Quantity]:
        n_spks = len(i_probe)
        # mu and sigma arrays: n_nrns x n_channels
        mu_eap_for_spikes = self._mu_eap[i_probe]
        sigma_eap_for_spikes = self._sigma_eap[i_probe]
        assert (
            mu_eap_for_spikes.shape
            == sigma_eap_for_spikes.shape
            == (n_spks, self.n_channels)
        )
        spike_amps = rng.standard_normal(n_spks)
        measured_eaps = (
            spike_amps[:, np.newaxis] * sigma_eap_for_spikes + mu_eap_for_spikes
        )
        spk_per_chan_dtct = measured_eaps > self.threshold_sigma
        return spk_per_chan_dtct, t

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

    def get_state(
        self,
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        # inherit docstring
        t_samp = self.probe.sim.network.t
        i_probe, t = self._get_new_spikes()
        spk_per_chan_dtct, t = self._noisily_detect_spikes(i_probe, t)
        # ⬇ nonzero gives row, column indices of each nonzero element
        i_spk_detected, i_chan_detected = spk_per_chan_dtct.nonzero()
        t_detected = t[i_spk_detected]
        y = np.sum(spk_per_chan_dtct, axis=0)
        self._update_saved_vars(t_detected, i_chan_detected, t_samp)
        return i_chan_detected, t_detected, y

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

    def get_state(
        self,
    ) -> tuple[UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n}"]]:
        # inherit docstring

        t_samp = self.probe.sim.network.t
        i_nrn_probe, t = self._get_new_spikes()
        spk_per_chan_dtct, t = self._noisily_detect_spikes(i_nrn_probe, t)
        # get spikes detected on any channel
        spk_detected_any_channel = spk_per_chan_dtct.sum(axis=1) > 0
        i_nrn_probe_detected = i_nrn_probe[spk_detected_any_channel]
        t_detected = t[spk_detected_any_channel]
        y = np.bincount(i_nrn_probe_detected.astype(int))
        # include 0s for upper indices not seen:
        y = np.concatenate([y, np.zeros(len(self.i_probe_by_i_ng) - len(y))])
        self._update_saved_vars(t_detected, i_nrn_probe_detected, t_samp)
        return i_nrn_probe_detected, t_detected, y
