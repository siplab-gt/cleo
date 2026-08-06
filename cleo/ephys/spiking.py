"""Contains multi-unit and sorted spiking signals."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from datetime import datetime
from functools import cache
from typing import Callable, Tuple

import brian2.only as b2
import neo
import numpy as np
import quantities as pq
from attrs import define, field, fields
from brian2 import NeuronGroup, Quantity, SpikeMonitor, mm, ms, um
from jaxtyping import Bool, Float, Int
from scipy import signal
from scipy.stats import norm

from cleo.base import NeoExportable
from cleo.ephys.probes import Signal
from cleo.utilities import rng, unit_safe_cat


@define(eq=False)
class Spiking(Signal, NeoExportable):
    """Base class for probabilistically detecting spikes.

    See ``notebooks/spike_detection.py`` for an interactive explanation of the methods
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
    recording_recall_cutoff: float = 0.001
    """*Multi-channel* recall, above which neurons will be considered.
    I.e., the probability a spike is detected on at least one channel.
    
    You shouldn't need to change this; it's mainly for efficiency, allowing amplitude 
    sampling and threshold crossing to operate on fewer spikes by ignoring neurons
    very unlikely to produce a spike that crosses the threshold."""
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
    per channel for :class:`MultiUnitActivity`.
    """

    @collision_prob_fn.validator
    def _validate_coll_prob_fn(self, attribute, value):
        if value is not None:
            assert callable(value), "collision_prob_fn must be callable"
            assert np.all(0 <= value([0, 1, 10] * ms) <= 1), (
                "collision_prob_fn must return a value between 0 and 1"
            )

    simulate_false_positives: bool = True
    """Whether to simulate false positives from noise. In the case of :class:`SortedSpiking`,
    these aren't reported, but still affect collision sampling."""

    t: Quantity = field(
        init=False, factory=lambda: ms * np.array([], dtype=float), repr=False
    )
    """Spike times with Brian units, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    i: Int[np.ndarray, "n_recorded_spikes"] = field(
        init=False, factory=lambda: np.array([], dtype=int), repr=False
    )
    """Channel (for multi-unit) or neuron (for sorted) indices
    of spikes, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    t_samp: Quantity = field(
        init=False, factory=lambda: ms * np.array([], dtype=float), repr=False
    )
    """Sample times with Brian units when each spike was recorded, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    i_probe_by_ng: dict[NeuronGroup, Int[np.ndarray, "ng_N"]] = field(
        init=False, factory=dict, repr=False
    )
    """neuron_group keys, i_probe values for every neuron in group."""
    i_ng_by_i_probe: list[tuple[NeuronGroup, int]] = field(
        init=False, factory=list, repr=False
    )
    """n_neurons-length list indexed by i_probe returning a neuron_group, i_ng tuple
    to map from i_probe to neuron group and index."""
    _monitors: list[SpikeMonitor] = field(init=False, factory=list, repr=False)
    _mon_spikes_already_seen: list[int] = field(init=False, factory=list, repr=False)
    _mu_eap: Float[np.ndarray, "n_neurons n_channels"] = field(
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
        """Number of spike sources: channels for :class:`MultiUnitActivity` or sorted neurons
        for :class:`SortedSpiking`."""
        pass

    @property
    def n_channels(self) -> int:
        """Number of channels on probe"""
        return self.probe.n

    @property
    def n_neurons(self) -> int:
        """Number of neurons recorded by probe"""
        return sum([np.sum(i_probe != -2) for i_probe in self.i_probe_by_ng.values()])

    @property
    def r_threshold(self, resolution: Quantity = um / 10) -> Quantity:
        """The distance from a contact at which the SNR equals the detection threshold.
        This also means 50% single-channel recall."""
        return self.r_for_snr(self.threshold_sigma, resolution=resolution)

    def r_for_recall(
        self,
        recall: float,
        resolution: Quantity = um / 10,
        upper_limit: Quantity = None,
    ) -> Quantity:
        """The distance from a contact at which the single-channel recall
        (detection probability) equals the specified value."""
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
        self, snr: float, resolution: Quantity = um / 10, upper_limit: Quantity = None
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
        sigma = np.sqrt(1 + (mu * self.spike_amplitude_cv) ** 2)
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
        if neuron_group in self.i_probe_by_ng:
            raise ValueError(
                f"Spiking signal {self.name} already connected to NeuronGroup {neuron_group.name}"
            )
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
        snr = self.snr_by_distance(distances)
        # 1 from baseline noise, mu * cv from spike amp variability
        sigma_eap = np.sqrt(1 + (snr * self.spike_amplitude_cv) ** 2)
        probs_miss = norm.cdf(self.threshold_sigma, loc=snr, scale=sigma_eap)
        assert probs_miss.shape == (len(neuron_group), self.n_channels)
        multi_channel_recall = 1 - np.prod(probs_miss, axis=1)
        where_to_keep = multi_channel_recall >= self.recording_recall_cutoff
        i2keep = np.nonzero(where_to_keep)[0]
        # spike sorters don't sort by multi-channel recall; we only do this
        # for computational efficiency, to not get super distant neurons

        n2keep = np.sum(where_to_keep)
        if n2keep == 0:
            warnings.warn(
                f"NeuronGroup {neuron_group.name} has no neurons with multi-channel recall >= {self.recording_recall_cutoff}."
                " Skipping this group."
            )
            return

        # create monitor
        mon = SpikeMonitor(neuron_group)
        self._monitors.append(mon)
        self.brian_objects.add(mon)
        self._mon_spikes_already_seen.append(0)

        # update mapping from neuron group to probe index
        i_probe_start = self.n_neurons
        # -2 means not recorded
        new_i_probe = np.arange(i_probe_start, i_probe_start + n2keep)
        i_probe_for_ng = np.full(neuron_group.N, -2, dtype=int)
        i_probe_for_ng[where_to_keep] = new_i_probe
        self.i_probe_by_ng[neuron_group] = i_probe_for_ng

        # update mapping from probe index to neuron group & index
        assert len(self.i_ng_by_i_probe) == i_probe_start
        self.i_ng_by_i_probe.extend(list(zip([neuron_group] * n2keep, i2keep)))

        if not self._prev_t:
            self._prev_t = self.probe.sim.network.t

        # store neuron-channel mean and stdev of EAPs to use in subclasses
        if self._mu_eap is None:
            self._mu_eap = snr[where_to_keep]
        else:
            self._mu_eap = np.concatenate((self._mu_eap, snr[where_to_keep]), axis=0)

        return snr[where_to_keep].max(axis=1), new_i_probe

    @abstractmethod
    def get_state(
        self,
    ) -> tuple[Int[np.ndarray, "n_spikes"], Quantity, Int[np.ndarray, "{self.n}"]]:
        """Return spikes since method was last called (i, t, y)

        Returns
        -------
        tuple[Int[np.ndarray, "n_spikes"], Quantity, Int[np.ndarray, "{self.n}"]]
            (i, t, y) where i is channel (for multi-unit) or neuron (for sorted) spike
            indices, t is spike times, and y is a spike count vector suitable for control-
            theoretic uses---i.e., a 0 for every channel/neuron that hasn't spiked and a 1
            for a single spike.
        """
        pass

    def _get_new_spikes(self) -> Tuple[Int[np.ndarray, "n_spikes"], Quantity]:
        i_probe = np.array([], dtype=int)
        t = ms * np.array([], dtype=float)
        for j in range(len(self._monitors)):
            mon = self._monitors[j]
            spikes_already_seen = self._mon_spikes_already_seen[j]
            i_ng = mon.i[spikes_already_seen:]  # can contain spikes we don't care about
            # filter out spikes we don't care about
            i_probe_unfilt = self.i_probe_by_ng[mon.source][i_ng]
            i2keep = i_probe_unfilt != -2
            i_probe = np.concatenate((i_probe, i_probe_unfilt[i2keep]))
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
        enbw = np.trapezoid(np.abs(h) ** 2, w)
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

    @staticmethod
    @cache
    def _cannot_simulate_noise(dt_s: float) -> bool:
        fs_Hz = 1 / dt_s
        cannot_simulate_noise = fs_Hz / 2 < 300
        if cannot_simulate_noise:
            warnings.warn(
                f"Sampling frequency {fs_Hz} Hz is too low to simulate spiking band noise"
            )
        return cannot_simulate_noise

    def _generate_noise(self) -> tuple[Float[np.ndarray, "n_t n_channels"], Quantity]:
        """generate noise in spiking band"""
        dt = b2.defaultclock.dt
        n_t = int(round((self.probe.sim.network.t - self._prev_t) / dt))
        t_window = np.arange(n_t) * dt + self._prev_t

        if self._cannot_simulate_noise(dt / b2.second):
            return np.zeros((n_t, self.n_channels)), t_window

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
        Int[np.ndarray, "n_tcs"],
        Int[np.ndarray, "n_tcs"],
        Float[np.ndarray, "n_tcs"],
        Float[np.ndarray, "n_t_window {self.n_channels}"],
        Quantity,
    ]:
        """"""
        n_spks = len(i_probe)
        noise, t_noise = self._generate_noise()
        # mu and sigma arrays: n_nrns x n_channels
        mu_eap_for_spikes = self._mu_eap[i_probe]
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
        # allows for function to return scale (like 0)
        coll_probs = np.broadcast_to(
            collision_prob_fn(intervals), intervals.shape
        ).astype(float)
        i = np.searchsorted(-coll_probs, -1e-3)
        if i == len(intervals):
            warnings.warn(
                "collision_prob_fn(10 ms) > 1e-3. "
                "Will not look for collisions over 10 ms in the past."
            )
            return 10 * ms
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
class MultiUnitActivity(Spiking):
    """Detects (unsorted) spikes per channel."""

    collision_prob_fn: Callable[[Quantity], float] = lambda t: t < 1 * ms

    @property
    def n(self):
        return self.probe.n

    def get_state(
        self,
    ) -> tuple[
        Int[np.ndarray, "n_spikes"], Quantity, Int[np.ndarray, "{self.n_channels}"]
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
        group = super(MultiUnitActivity, self).to_neo()
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

    snr_cutoff: float = 6
    """The signal-to-noise ratio a unit must have for its spikes to be reported.
    SNR is defined as the mean spike amplitude divided by the standard deviation of
    the background noise for the peak (closest) channel.
    
    Should be higher than :attr:`~Spiking.threshold_sigma`.
    Spikes from units with SNR < snr_cutoff still factor into collision sampling and
    are reported as unsorted (index -1), essentially "multi-unit activity"."""
    collision_prob_fn: Callable[[Quantity], float] = lambda t: 0.2 * np.exp(
        -t / (0.3 * ms)
    )

    @property
    def n(self):
        return self.n_sorted

    @property
    def n_sorted(self):
        """Number of sorted neurons"""
        return len(self._i_probe_by_i_sorted)

    def i_sorted_by_ng(self, ng: NeuronGroup) -> Int[np.ndarray, "{ng.N}"]:
        """Get the sorted indices for a given neuron group.

        -1 means recorded, but not sorted. -2 means not recorded."""
        i_probe = self.i_probe_by_ng[ng][ng.i]
        i_sorted = self._i_sorted_by_i_probe[i_probe]
        # pass through the -2s so they don't index i_sorted
        i_sorted[i_probe == -2] = -2
        return i_sorted

    @property
    def i_ng_by_i_sorted(self) -> list[tuple[NeuronGroup, int]]:
        """Get a list of (ng, i_ng) tuples for all sorted neurons, in order.

        That is, this maps from sorted indices back to the original neuron group and
        indices."""
        i_probe = self._i_probe_by_i_sorted[np.arange(self.n_sorted)]
        return [self.i_ng_by_i_probe[i] for i in i_probe]

    _i_sorted_by_i_probe: Int[np.ndarray, "{self.n_neurons}"] = field(
        init=False, factory=lambda: np.zeros(0, dtype=int), repr=False
    )
    _i_probe_by_i_sorted: Int[np.ndarray, "{self.n_sorted}"] = field(
        init=False, factory=lambda: np.zeros(0, dtype=int), repr=False
    )

    @property
    def r_cutoff(self, resolution: Quantity = um / 10) -> Quantity:
        """The distance from a contact at which the SNR is high enough for a neuron
        to be included."""
        return self.r_for_snr(self.snr_cutoff, resolution=resolution)

    @property
    def sorted_units_snr(self) -> Float[np.ndarray, "{self.n_sorted}"]:
        """The SNR for each sorted neuron, in order."""
        return self._mu_eap[self._i_probe_by_i_sorted]

    def connect_to_neuron_group(self, neuron_group, **kwparams):
        snr, i_probe = super().connect_to_neuron_group(neuron_group, **kwparams)
        n_recorded_ng = len(snr)
        # filter by SNR (measured on peak channel)
        above_cutoff = snr >= self.snr_cutoff
        n_above_cutoff = np.sum(above_cutoff)
        n_prev_sorted = self.n_sorted
        i_srt_new_range = np.arange(self.n_sorted, n_prev_sorted + n_above_cutoff)

        # update map from i_probe to i_sorted
        # -1 means not reported
        i_srt_by_i_probe_for_ng = np.full(n_recorded_ng, -1, dtype=int)
        i_srt_by_i_probe_for_ng[above_cutoff] = i_srt_new_range
        self._i_sorted_by_i_probe = np.concatenate(
            [self._i_sorted_by_i_probe, i_srt_by_i_probe_for_ng]
        )

        # update map from i_sorted to i_probe
        i_probe_by_i_sorted_for_ng = i_probe[above_cutoff]
        assert len(i_probe_by_i_sorted_for_ng) == n_above_cutoff
        self._i_probe_by_i_sorted = np.concatenate(
            [self._i_probe_by_i_sorted, i_probe_by_i_sorted_for_ng]
        )
        assert n_prev_sorted + n_above_cutoff == self.n_sorted

    def get_state(
        self,
    ) -> tuple[Int[np.ndarray, "n_spikes"], Quantity, Int[np.ndarray, "{self.n}"]]:
        # inherit docstring

        t_samp = self.probe.sim.network.t
        i_probe, t = self._get_new_spikes()
        t_tcs, i_probe_tcs, i_chan_tcs, amp_tcs, noise, t_noise = (
            self._noisily_get_true_tcs(i_probe, t)
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
        i_probe = np.concatenate([i_probe_tcs, np.full_like(i_chan_fps, -3)]).astype(
            int
        )
        # sort by time
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        i_chan = i_chan[sort_idx]
        amps = amps[sort_idx]
        i_probe = i_probe[sort_idx]

        which_collided = self._sample_collisions(t, i_chan, amps)
        # filter out false positives
        t_detected = t[~which_collided & (i_probe != -3)]
        i_probe_detected = i_probe[~which_collided & (i_probe != -3)]

        # remove repeat t, i_nrn spikes (spikes detected on >1 channel)
        _, spk_detected_any_channel = np.unique(
            np.array([t_detected / ms, i_probe_detected]), axis=1, return_index=True
        )
        # get spikes detected on any channel
        i_probe_detected = i_probe_detected[spk_detected_any_channel]
        t_detected = t_detected[spk_detected_any_channel]
        # convert to sorted indices, including getting -1s for unsorted
        i_sorted_detected = self._i_sorted_by_i_probe[i_probe_detected]
        # filter out -1s
        to_keep = i_sorted_detected != -1
        i_srt_dtct_filt = i_sorted_detected[to_keep]
        t_dtct_filt = t_detected[to_keep]

        y = np.bincount(i_srt_dtct_filt.astype(int))
        # include 0s for upper indices not seen:
        y = np.concatenate([y, np.zeros(self.n_sorted - len(y))])
        self._update_saved_vars(t_dtct_filt, i_srt_dtct_filt, t_samp)
        return i_srt_dtct_filt, t_dtct_filt, y
