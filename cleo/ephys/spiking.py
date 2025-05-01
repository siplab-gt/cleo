"""Contains multi-unit and sorted spiking signals."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from functools import cache
from typing import Callable, Tuple

import brian2.only as b2
import neo
import numpy as np
import quantities as pq
from attrs import define, field, fields
from bidict import bidict
from brian2 import NeuronGroup, Quantity, SpikeMonitor, mm, ms, um
from jaxtyping import Bool, Float, UInt
from scipy import signal
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
    i.e., ``spike_amplitude(r_noise_floor) = sigma_noise = 1``.
    80 μm by default."""
    threshold_sigma: int = 4
    """Spike detection threshold, as a multiple of sigma_noise. 
    Values in real experiments typically range from 3 to 6. 4 by default."""
    spike_amplitude_cv: float = 0.05
    """Coefficient of variation of the spike amplitude, i.e., `|sigma_amp/mu_amp|`. 
    From what we have seen in Allen Cell Types data, this ranges from 0 to 0.2, 
    but is most often very low. 0.05 by default."""
    r0: Quantity = 5 * um
    """A small distance added to r before computing the amplitude to avoid division
    by 0 for the power law decay. 5 μm by default.
    
    It also makes some physical sense as the minimum distance from the current source
    it is possible to place an electrode, 5 μm being reasonable as the radius of a typical soma."""
    recall_cutoff: float = None
    """Spike detection probability (recall) above which neurons will be considered.
    
    Mainly for efficiency in the case of :class:`MultiUnitSpiking`.
    Should be somewhat high for :class:`SortedSpiking` since sorter should only identify
    spikes from high-SNR neurons."""
    eap_decay_fn: Callable[[Quantity], float] = lambda r: r**-2
    """The function describing the decay of the measured extracellular action potential
    amplitude. By default 1/r^2.
    
    This inverse square decay is a good approximation in accordance with the detailed
    simulations by `Pettersen et al. (2008) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2186261/>`_,
    though they find the exponent ranges from 2 to 3 depending on the cell type and distance. """
    collision_prob_fn: Callable[[Quantity], float] = field(default=None)
    """The probability of failing to detect the latter of two overlapping threshold
    crossings on a given channel, as a function of ``t2 - t1``.
    Values for ``t2 - t1 < 0`` are ignored.

    For :class:`SortedSpiking`, the default is a decaying exponential.
    See `Garcia et al. (2022) <https://www.eneuro.org/content/9/5/ENEURO.0105-22.2022>`_
    for what this might look like for different sorters.

    By default simply enforces a hard 1 ms refractory period
    per channel for :class:`MultiUnitSpiking`.
    """

    @collision_prob_fn.validator
    def _validate_coll_prob_fn(self, attribute, value):
        if value is not None:
            assert callable(value), "collision_prob_fn must be callable"
            assert np.all(0 <= value([0, 1, 10] * ms) <= 1), (
                "collision_prob_fn must return a value between 0 and 1"
            )

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
    _prev_t: Quantity = field(init=False, default=None, repr=False)
    _prev_zi: Float[np.ndarray, "n_sos 2"] = field(init=False, default=None, repr=False)

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

    @property
    def snr_cutoff(self) -> float:
        """The minimum SNR required for spikes to factor into detection.
        Derived from :attr:`~Spiking.recall_cutoff`."""
        return self.snr_by_recall(self.recall_cutoff)

    @property
    def r_threshold(self, resolution: Quantity = um / 10) -> Quantity:
        """The distance from a contact at which the SNR equals the detection threshold.
        This also means 50% recall."""
        return self.r_for_snr(self.threshold_sigma, resolution=resolution)

    @property
    def r_cutoff(self, resolution: Quantity = um / 10) -> Quantity:
        """The distance from a contact at which the recall is high enough for a neuron
        to be included."""
        return self.r_for_recall(self.recall_cutoff, resolution=resolution)

    def r_for_recall(
        self,
        recall: float,
        resolution: Quantity = um / 10,
        upper_limit: Quantity = None,
    ) -> Quantity:
        """The distance from a contact at which the recall (detection probability)
        equals the specified value."""
        if upper_limit is None:
            upper_limit = 10 * self.r_noise_floor
        rr = np.arange(0, upper_limit / um, resolution / um) * um
        recalls = self.recall_by_distance(rr)
        try:
            return rr[recalls <= recall][0]
        except IndexError:
            return self.r_for_recall(
                recall, resolution=resolution, upper_limit=upper_limit * 2
            )

    def r_for_snr(
        self, snr: float, resolution: Quantity = um, upper_limit: Quantity = None
    ) -> Quantity:
        """The distance from a contact at which the SNR equals the specified value."""
        if upper_limit is None:
            upper_limit = 10 * self.r_noise_floor
        rr = np.arange(0, upper_limit / um, resolution / um) * um
        try:
            return rr[self.snr_by_distance(rr) <= snr][0]
        except IndexError:
            return self.r_for_snr(
                snr, resolution=resolution, upper_limit=upper_limit * 2
            )

    def recall_by_snr(self, snr: float) -> float:
        """Probability of detecting a spike at distance r from the neuron
        as a function of SNR."""
        # 1 - P(spike not detected)
        mu = snr
        sigma = 1 + mu * self.spike_amplitude_cv
        return norm.sf(self.threshold_sigma, loc=mu, scale=sigma)

    def recall_by_distance(self, r: Quantity) -> float:
        """Probability of detecting a spike at distance r from the neuron
        as a function of distance."""
        return self.recall_by_snr(self.snr_by_distance(r))

    def snr_by_distance(self, r: Quantity) -> float:
        """The mean extracellular action potential amplitude as a function of distance
        from the neuron, in units of background noise standard deviation."""
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
        mu_eap = self.snr_by_distance(distances)
        # 1 from baseline noise, mu * cv from spike amp variability
        sigma_eap = 1 + mu_eap * self.spike_amplitude_cv
        probs = norm.sf(self.threshold_sigma, loc=mu_eap, scale=sigma_eap)
        assert probs.shape == (len(neuron_group), self.n_channels)
        # probs is n_neurons by n_channels
        # cut off to get indices of neurons to monitor
        # [0] since nonzero returns tuple of array per axis
        i_ng_to_keep = np.nonzero(np.any(probs >= self.recall_cutoff, axis=1))[0]

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

        if not self._prev_t:
            self._prev_t = self.probe.sim.network.t

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
            # filter out spikes we don't care about
            i_probe_unfilt = np.array(
                [self.i_probe_by_i_ng.get((mon.source, k), -1) for k in i_ng]
            )
            i2keep = i_probe_unfilt != -1
            i_probe = np.concatenate((i_probe, i_probe_unfilt[i2keep].astype(np.uint)))
            t = unit_safe_cat([t, mon.t[spikes_already_seen:][i2keep]])

            self._mon_spikes_already_seen[j] = mon.num_spikes

        return i_probe, t

    @staticmethod
    @cache
    def _prep_noise(dt_s: float) -> np.ndarray:
        lowcut_Hz = 300
        fs_Hz = 1 / dt_s
        highcut_Hz = min(3000, 0.45 * fs_Hz)
        sos = signal.butter(
            6, [lowcut_Hz, highcut_Hz], fs=fs_Hz, btype="band", output="sos"
        )
        w, h = signal.sosfreqz(sos, 2**14, fs=fs_Hz)
        enbw = np.trapz(np.abs(h) ** 2, w)
        assert np.isclose(enbw, highcut_Hz - lowcut_Hz, rtol=0.01), (
            f"{enbw} != {highcut_Hz - lowcut_Hz}"
        )
        nyq = 0.5 * fs_Hz
        pre_filter_factor = np.sqrt(nyq / enbw)

        # inline test:
        white_noise = rng.standard_normal((400, 32))
        # scale noise so RMS after filtering is 1
        white_noise *= pre_filter_factor
        filtered_noise = signal.sosfilt(sos, white_noise, axis=0)
        assert np.isclose(np.std(filtered_noise), 1, rtol=0.05), (
            f"Filtered noise RMS {np.std(filtered_noise)} != 1"
        )

        return sos, pre_filter_factor

    def _generate_noise(self) -> tuple[Float[np.ndarray, "n_t n_channels"], Quantity]:
        """generate noise in spiking band"""
        dt = b2.defaultclock.dt
        n_t = int(round((self.probe.sim.network.t - self._prev_t) / dt))
        t_window = np.arange(n_t) * dt + self._prev_t

        sos, pre_filter_factor = self._prep_noise(dt / b2.second)
        if n_t <= 0:
            return np.zeros((0, self.n_channels)), [] * ms
        # generate white noise
        white_noise = rng.standard_normal((n_t, self.n_channels))
        # scale noise so RMS after filtering is 1
        white_noise *= pre_filter_factor
        if self._prev_zi is None:
            self._prev_zi = np.zeros((sos.shape[0], 2, self.n_channels))
        noise_filt, zi = signal.sosfilt(sos, white_noise, axis=0, zi=self._prev_zi)
        # I assume we won't have spikes at the current timestep here
        # since Cleo's NetworkOperation is scheduled for start of timestep

        self._prev_zi = zi
        return noise_filt, t_window

    def _noisily_get_true_tcs(
        self, i_probe, t
    ) -> Tuple[
        Quantity,
        UInt[np.ndarray, "n_tcs"],
        UInt[np.ndarray, "n_tcs"],
        Float[np.ndarray, "n_tcs"],
        Float[np.ndarray, "n_t_window {self.n_channels}"],
        Quantity,
    ]:
        """"""
        n_spks = len(i_probe)
        noise, t_noise = self._generate_noise()
        # mu and sigma arrays: n_nrns x n_channels
        mu_eap_for_spikes = self._mu_eap[i_probe]
        # TODO: probably remove this self._sigma_eap
        # sigma_eap_for_spikes = self._sigma_eap[i_probe]
        sigma_spike_amps = mu_eap_for_spikes * self.spike_amplitude_cv

        # add noise at right timesteps
        noise_at_spks = noise[
            ((t - self._prev_t) / b2.defaultclock.dt).round().astype(int)
        ]

        assert (
            mu_eap_for_spikes.shape
            == sigma_spike_amps.shape
            == noise_at_spks.shape
            == (n_spks, self.n_channels)
        )
        amps = (
            rng.standard_normal((n_spks, 1)) * sigma_spike_amps
            + mu_eap_for_spikes
            + noise_at_spks
        )

        # ⬇ nonzero gives row, column indices of each nonzero element
        i_spk_tcs, i_chan_tcs = (amps > self.threshold_sigma).nonzero()
        i_probe_tcs = i_probe[i_spk_tcs]
        t_tcs = t[i_spk_tcs]
        amp_tcs = amps[i_spk_tcs, i_chan_tcs]
        self._prev_t = self.probe.sim.network.t

        return t_tcs, i_probe_tcs, i_chan_tcs, amp_tcs, noise, t_noise

    @staticmethod
    @cache
    def _max_collision_interval(dt_ms, collision_prob_fn):
        intervals = np.arange(10 / dt_ms) * dt_ms * ms
        i = np.searchsorted(-collision_prob_fn(intervals).astype(float), -1e-3)
        if i == len(intervals):
            raise NotImplementedError("not looking for max collision interval >= 10 ms")
        else:
            return intervals[i]

    def _sample_collisions(self, t, i_chan, amps) -> Bool[np.ndarray, "n_spikes"]:
        """Filter out spikes that are too close together in time on the same channel.
        For simplicity, the first spike is kept, or the largest if simultaneous.

        Note this operates on candidate threshold crossings, not called spikes.
        This is mainly for computational efficiency, so we don't have to iterate."""
        assert np.all(np.diff(t) >= 0), "should be time-sorted"

        # need to combine with previous t, i_chan, amps
        try:
            where_window_starts = len(self._prev_t_tcs)
            t = unit_safe_cat([self._prev_t_tcs, t])
            i_chan = np.concatenate([self._prev_i_chan_tcs, i_chan])
            amps = np.concatenate([self._prev_amp_tcs, amps])
        except AttributeError:
            where_window_starts = 0

        # rows=spike 2, cols=spike 1
        t_diff = t[:, None] - t[None, :]
        amp_diff = amps[:, None] - amps[None, :]
        same_chan = i_chan[:, None] == i_chan[None, :]

        collision_prob = self.collision_prob_fn(t_diff) * same_chan
        # remove self-pairs
        np.fill_diagonal(collision_prob, 0)
        # only consider same-channel pairs, and only consider where t_spk2 >= t_spk1
        collision_prob *= t_diff >= 0
        # should be roughly lower triangular at this point if spikes are ordered by time
        # for simultaneous spikes, make sure biggest amplitude wins
        # by removing simultaneous pairs where the second is bigger
        collision_prob[(t_diff == 0) & (amp_diff > 0)] = 0

        which_collided = np.any(
            rng.uniform(size=collision_prob.shape) < collision_prob, axis=1
        )

        # save t, i_chan, amps for next call
        # earliest time needed from current time (more than enough for next sample)
        t_needed = self.probe.sim.network.t - self._max_collision_interval(
            b2.defaultclock.dt / ms, self.collision_prob_fn
        )
        # TODO: use searchsorted elsewhere
        i_oldest_needed = max(np.searchsorted(t, t_needed) - 1, 0)
        self._prev_t_tcs = t[i_oldest_needed:]
        self._prev_i_chan_tcs = i_chan[i_oldest_needed:]
        self._prev_amp_tcs = amps[i_oldest_needed:]

        return which_collided[where_window_starts:]

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

    recall_cutoff: float = 0.01
    collision_prob_fn: Callable[[Quantity], float] = lambda t: t < 1 * ms
    simulate_false_positives: bool = True

    @property
    def n(self):
        """Number of channels on probe"""
        return self.probe.n

    def get_state(
        self,
    ) -> tuple[
        UInt[np.ndarray, "n_spikes"], Quantity, UInt[np.ndarray, "{self.n_channels}"]
    ]:
        # inherit docstring
        t_samp = self.probe.sim.network.t
        i_probe, t = self._get_new_spikes()
        t_tcs, _, i_chan_tcs, amp_tcs, noise, t_noise = self._noisily_get_true_tcs(
            i_probe, t
        )
        # get false positives from noise
        if self.simulate_false_positives:
            i_t_fps, i_chan_fps = (noise > self.threshold_sigma).nonzero()
            t_fps = t_noise[i_t_fps]
            amp_fps = noise[i_t_fps, i_chan_fps]
        else:
            t_fps, i_chan_fps, amp_fps = [] * ms, [], []

        i_chan = np.concatenate([i_chan_tcs, i_chan_fps])
        t = unit_safe_cat([t_tcs, t_fps])
        amps = np.concatenate([amp_tcs, amp_fps])
        # sort by time
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        i_chan = i_chan[sort_idx]
        amps = amps[sort_idx]

        # sample collisions
        which_collided = self._sample_collisions(t, i_chan, amps)
        t_detected = t[~which_collided]
        i_chan_detected = i_chan[~which_collided]

        y = np.bincount(i_chan_detected.astype(int))
        # include 0s for upper indices not seen:
        y = np.concatenate([y, np.zeros(self.n_channels - len(y))])
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

    recall_cutoff: float = 0.8
    collision_prob_fn: Callable[[Quantity], float] = lambda t: 0.2 * np.exp(
        -t / (0.3 * ms)
    )

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
        t_tcs, i_probe_tcs, i_chan_tcs, amp_tcs, _, _ = self._noisily_get_true_tcs(
            i_nrn_probe, t
        )
        # no false positives from noise in sorted spiking
        which_collided = self._sample_collisions(t_tcs, i_chan_tcs, amp_tcs)
        t_detected = t_tcs[~which_collided]
        i_nrn_probe_detected = i_probe_tcs[~which_collided]

        # remove repeat t, i_nrn spikes (spikes detected on >1 channel)
        _, spk_detected_any_channel = np.unique(
            np.array([t_detected / ms, i_nrn_probe_detected]), axis=1, return_index=True
        )
        # get spikes detected on any channel
        # spk_detected_any_channel = spk_per_chan_dtct.sum(axis=1) > 0
        i_nrn_probe_detected = i_nrn_probe_detected[spk_detected_any_channel]
        t_detected = t_detected[spk_detected_any_channel]
        y = np.bincount(i_nrn_probe_detected.astype(int))
        # include 0s for upper indices not seen:
        y = np.concatenate([y, np.zeros(len(self.i_probe_by_i_ng) - len(y))])
        self._update_saved_vars(t_detected, i_nrn_probe_detected, t_samp)
        return i_nrn_probe_detected, t_detected, y
