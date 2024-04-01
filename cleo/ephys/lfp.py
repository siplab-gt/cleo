"""Contains LFP signals"""
from __future__ import annotations

import warnings
from collections import deque
from datetime import datetime
from itertools import chain
from math import ceil
from numbers import Number
from typing import Any, Union

import neo
import quantities as pq
import wslfp
from attrs import define, field
from brian2 import NeuronGroup, Quantity, Subgroup, Synapses, mm, ms, np, um
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.synapses.synapses import SynapticSubgroup
from brian2.units import Unit, uvolt
from nptyping import NDArray
from scipy import sparse
from tklfp import TKLFP

import cleo.utilities
from cleo.base import NeoExportable
from cleo.coords import coords_from_ng
from cleo.ephys.probes import Signal


class LFPSignalBase(Signal, NeoExportable):
    """Base class for LFP Signals.

    Injection kwargs
    ----------------
    orientation : np.ndarray, optional
        Array of shape (n_neurons, 3) representing which way is "up," that is, towards
        the surface of the cortex, for each neuron. If a single vector is given, it is
        taken to be the orientation for all neurons in the group. [0, 0, -1] is the
        default, meaning the negative z axis is "up."
    """

    t_ms: NDArray[(Any,), float] = field(init=False, repr=False)
    """Times at which LFP is recorded, in ms, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    lfp: Union[NDArray[(Any, Any), Quantity]] = field(init=False, repr=False)
    """Approximated LFP from every call to :meth:`get_state`.
    Shape is (n_samples, n_channels). Stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    _elec_coords: np.ndarray = field(init=False, repr=False)
    _lfp_unit: Unit

    def _post_init_for_probe(self):
        self._elec_coords = self.probe.coords.copy()
        # need to invert z coords since cleo uses an inverted z axis and
        # tklfp and wslfp do not
        self._elec_coords[:, 2] *= -1
        self._init_saved_vars()

    def _init_saved_vars(self):
        if self.probe.save_history:
            self.t_ms = np.empty((0,))
            self.lfp = np.empty((0, self.probe.n)) * self._lfp_unit

    def _update_saved_vars(self, t_ms, lfp):
        if self.probe.save_history:
            # self.t_ms = np.concatenate([self.t_ms, [t_ms]])
            # self.lfp = np.vstack([self.lfp, lfp])
            lfp = np.reshape(lfp, (1, -1))
            t_ms = np.reshape(t_ms, (1,))
            self.t_ms = cleo.utilities.unit_safe_append(self.t_ms, t_ms)
            self.lfp = cleo.utilities.unit_safe_append(self.lfp, lfp)

    def to_neo(self) -> neo.AnalogSignal:
        # inherit docstring
        try:
            signal = cleo.utilities.analog_signal(
                self.t_ms,
                self.lfp,
                str(self._lfp_unit),
            )
        except AttributeError:
            return
        signal.name = self.probe.name + "." + self.name
        signal.description = f"Exported from Cleo {self.__class__.__name__} object"
        signal.annotate(export_datetime=datetime.now())
        # broadcast in case of uniform direction
        signal.array_annotate(
            x=self.probe.coords[..., 0] / mm * pq.mm,
            y=self.probe.coords[..., 1] / mm * pq.mm,
            z=self.probe.coords[..., 2] / mm * pq.mm,
            i_channel=np.arange(self.probe.n),
        )
        return signal


@define(eq=False)
class TKLFPSignal(LFPSignalBase):
    """Records the Teleńczuk kernel LFP approximation.

    Requires ``tklfp_type='exc'|'inh'`` to specify cell type
    on injection.

    TKLFP is computed from spikes using the `tklfp package <https://github.com/kjohnsen/tklfp/>`_.

    Injection kwargs
    ----------------
    tklfp_type : str
        Either 'exc' or 'inh' to specify the cell type.
    """

    uLFP_threshold_uV: float = 1e-3
    """Threshold, in microvolts, above which the uLFP for a single spike is guaranteed
    to be considered, by default 1e-3.
    This determines the buffer length of past spikes, since the uLFP from a long-past
    spike becomes negligible and is ignored."""
    _tklfps: list[TKLFP] = field(init=False, factory=list, repr=False)
    _monitors: list[SpikeMonitor] = field(init=False, factory=list, repr=False)
    _mon_spikes_already_seen: list[int] = field(init=False, factory=list, repr=False)
    _i_buffers: list[list[np.ndarray]] = field(init=False, factory=list, repr=False)
    _t_ms_buffers: list[list[np.ndarray]] = field(init=False, factory=list, repr=False)
    _buffer_positions: list[int] = field(init=False, factory=list, repr=False)
    _lfp_unit: Unit = uvolt

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # inherit docstring
        # prep tklfp object
        tklfp_type = kwparams.pop("tklfp_type", "not given")
        if tklfp_type not in ["exc", "inh"]:
            raise Exception(
                "tklfp_type ('exc' or 'inh') must be passed as a keyword argument to "
                "inject() when injecting an electrode with a TKLFPSignal."
            )
        orientation = kwparams.pop("orientation", np.array([[0, 0, -1]])).copy()
        orientation[:, 2] *= -1

        tklfp = TKLFP(
            neuron_group.x / mm,
            neuron_group.y / mm,
            -neuron_group.z / mm,  # invert neuron zs as well
            is_excitatory=tklfp_type == "exc",
            elec_coords_mm=self._elec_coords / mm,
            orientation=orientation,
        )

        # determine buffer length necessary for given neuron group
        # if 0 (neurons are too far away to much influence LFP)
        # then we ignore this neuron group
        buf_len = self._get_buffer_length(tklfp, **kwparams)

        if buf_len > 0:
            # prep buffers
            self._tklfps.append(tklfp)
            self._i_buffers.append([np.array([], dtype=int, ndmin=1)] * buf_len)
            self._t_ms_buffers.append([np.array([], dtype=float, ndmin=1)] * buf_len)
            self._buffer_positions.append(0)

            # prep SpikeMonitor
            mon = SpikeMonitor(neuron_group)
            self._monitors.append(mon)
            self._mon_spikes_already_seen.append(0)
            self.brian_objects.add(mon)

    def get_state(self) -> np.ndarray:
        tot_tklfp = 0
        now_ms = self.probe.sim.network.t / ms
        # loop over neuron groups (monitors, tklfps)
        for i_mon in range(len(self._monitors)):
            self._update_spike_buffer(i_mon)
            tot_tklfp += self._tklfp_for_monitor(i_mon, now_ms)
        tot_tklfp = np.reshape(tot_tklfp, (-1,)) * uvolt  # return 1D array (vector)
        self._update_saved_vars(now_ms, tot_tklfp)
        return tot_tklfp

    def reset(self, **kwargs) -> None:
        for i_mon in range(len(self._monitors)):
            self._reset_buffer(i_mon)
        self._init_saved_vars()

    def _reset_buffer(self, i_mon):
        buf_len = len(self._i_buffers[i_mon])
        self._i_buffers[i_mon] = [np.array([], dtype=int, ndmin=1)] * buf_len
        self._t_ms_buffers[i_mon] = [np.array([], dtype=float, ndmin=1)] * buf_len
        self._buffer_positions[i_mon] = 0

    def _update_spike_buffer(self, i_mon):
        mon = self._monitors[i_mon]
        n_prev = self._mon_spikes_already_seen[i_mon]
        buf_pos = self._buffer_positions[i_mon]

        # insert new spikes into buffer (overwriting anything previous)
        self._i_buffers[i_mon][buf_pos] = mon.i[n_prev:]
        self._t_ms_buffers[i_mon][buf_pos] = mon.t[n_prev:] / ms

        self._mon_spikes_already_seen[i_mon] = mon.num_spikes
        # update buffer position
        buf_len = len(self._i_buffers[i_mon])
        self._buffer_positions[i_mon] = (buf_pos + 1) % buf_len

    def _tklfp_for_monitor(self, i_mon, now_ms):
        i = np.concatenate(self._i_buffers[i_mon])
        t_ms = np.concatenate(self._t_ms_buffers[i_mon])
        return self._tklfps[i_mon].compute(i, t_ms, [now_ms])

    def _get_buffer_length(self, tklfp, **kwparams):
        # need sampling period
        sample_period_ms = kwparams.get("sample_period_ms", None)
        if sample_period_ms is None:
            try:
                sample_period_ms = self.probe.sim.io_processor.sample_period_ms
            except AttributeError:  # probably means sim doesn't have io_processor
                raise Exception(
                    "TKLFP needs to know the sampling period. Either set the simulator's "
                    f"IO processor before injecting {self.probe.name} or "
                    f"specify it on injection: .inject({self.probe.name} "
                    ", tklfp_type=..., sample_period_ms=...)"
                )
        return np.ceil(
            tklfp.compute_min_window_ms(self.uLFP_threshold_uV) / sample_period_ms
        ).astype(int)


@define(eq=False)
class RWSLFPSignalBase(LFPSignalBase):
    """Base class for :class:`RWSLFPSignalFromSpikes` and :class:`RWSLFPSignalFromPSCs`.

    These signals should only be injected into neurons representing pyramidal cells with
    standard synaptic structure (see `Mazzoni, Lindén et al., 2015
    <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004584>`_).

    RWSLFP is computed using the `wslfp package <https://github.com/siplab-gt/wslfp/>`_.

    ``amp_func`` and ``pop_aggregate`` can be overridden on injection.
    """

    # note these set defaults that can be overrriden on injection
    amp_func: callable = wslfp.mazzoni15_nrn
    """Function to calculate LFP amplitudes, by default ``wslfp.mazzoni15_nrn``.
    
    See `wslfp documentation <https://github.com/siplab-gt/wslfp/blob/master/notebooks/amplitude_comparison.ipynb>`_ for more info."""
    pop_aggregate: bool = False
    """Whether to aggregate currents across the population (as opposed to neurons having 
    differential contributions to LFP depending on their location). False by default."""

    wslfp_kwargs: dict = field(factory=dict)
    """Keyword arguments to pass to the WSLFP calculator, e.g., ``alpha``,
    ``tau_ampa_ms``, ``tau_gaba_ms````source_coords_are_somata``,
    ``source_dendrite_length_um``, ``amp_kwargs``, ``strict_boundaries``.
    """

    _wslfps: dict[NeuronGroup, wslfp.WSLFPCalculator] = field(
        init=False, factory=dict, repr=False
    )
    _lfp_unit: Unit = 1

    def _init_wslfp_calc(self, neuron_group: NeuronGroup, **kwparams):
        nrn_coords_um = coords_from_ng(neuron_group) / um
        nrn_coords_um[:, 2] *= -1

        orientation = np.copy(kwparams.pop("orientation", np.array([[0, 0, -1]])))
        orientation = orientation.reshape((-1, 3))
        assert np.shape(orientation)[-1] == 3
        orientation[..., 2] *= -1

        if self.pop_aggregate:
            nrn_coords_um = np.mean(nrn_coords_um, axis=0)
            orientation = np.mean(orientation, axis=0)

        wslfp_kwargs = {}
        for key in [
            "source_coords_are_somata",
            "source_dendrite_length_um",
            "amp_kwargs",
            "alpha",
            "tau_ampa_ms",
            "tau_gaba_ms",
            "strict_boundaries",
        ]:
            if key in kwparams:
                wslfp_kwargs[key] = kwparams.pop(key)
            elif key in self.wslfp_kwargs:
                wslfp_kwargs[key] = self.wslfp_kwargs[key]

        self._wslfps[neuron_group] = wslfp.from_xyz_coords(
            self._elec_coords / um,
            nrn_coords_um,
            amp_func=kwparams.pop("amp_func", self.amp_func),
            source_orientation=orientation,
            **wslfp_kwargs,
        )

    def get_state(self) -> np.ndarray:
        # round to avoid floating point errors
        now_ms = round(self.probe.sim.network.t / ms, 3)
        lfp = np.zeros((1, self.probe.n))
        for ng, wslfp_calc in self._wslfps.items():
            t_ampa_ms = now_ms - wslfp_calc.tau_ampa_ms
            t_gaba_ms = now_ms - wslfp_calc.tau_gaba_ms
            I_ampa, I_gaba = self._needed_current(ng, t_ampa_ms, t_gaba_ms)
            lfp += wslfp_calc.calculate(
                [now_ms], t_ampa_ms, I_ampa, t_gaba_ms, I_gaba, normalize=False
            )
        out = np.reshape(lfp, (-1,))  # return 1D array (vector)
        self._update_saved_vars(now_ms, out)
        return out

    def _needed_current(self, ng, t_ampa_ms: float, t_gaba_ms: float) -> np.ndarray:
        """output must have shape (n_t, n_current_sources)"""
        raise NotImplementedError

    def reset(self, **kwargs) -> None:
        self._init_saved_vars()

    def to_neo(self) -> neo.AnalogSignal:
        # inherit docstring
        try:
            signal = cleo.utilities.analog_signal(
                self.t_ms,
                self.lfp,
            )
        except AttributeError:
            return
        signal.name = self.probe.name + "." + self.name
        signal.description = f"Exported from Cleo {self.__class__.__name__} object"
        signal.annotate(export_datetime=datetime.now())
        # broadcast in case of uniform direction
        signal.array_annotate(
            x=self.probe.coords[..., 0] / mm * pq.mm,
            y=self.probe.coords[..., 1] / mm * pq.mm,
            z=self.probe.coords[..., 2] / mm * pq.mm,
            i_channel=np.arange(self.probe.n),
        )
        return signal


@define
class SpikeToCurrentSource:
    """Stores info needed to calculate synaptic currents from spikes for a given spike source."""

    J: Union[np.ndarray, sparse.sparray]
    mon: SpikeMonitor
    biexp_kernel_params: dict[str, Any]


@define(eq=False)
class RWSLFPSignalFromSpikes(RWSLFPSignalBase):
    """Computes RWSLFP from the spikes onto pyramidal cell.

    Use this if your model does not simulate synaptic current dynamics directly.
    The parameters of this class are used to synthesize biexponential synaptic currents
    using ``wslfp.spikes_to_biexp_currents()``.
    ``ampa_syns`` and ``gaba_syns`` are lists of Synapses or SynapticSubgroup objects
    and must be passed as kwargs on injection, or else this signal will not be recorded
    for the target neurons (useful for ignoring interneurons).
    Attributes set on the signal object serve as the default, but can be overridden on injection.
    Also, in the case that parameters (e.g., ``tau1_ampa`` or ``weight``) vary by synapse,
    these can be overridden by passing a tuple of the Synapses or SynapticSubgroup object and
    a dictionary of the parameters to override.

    RWSLFP refers to the Reference Weighted Sum of synaptic currents LFP proxy from
    `Mazzoni, Lindén et al., 2015 <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004584>`_.

    Injection kwargs
    ----------------
    ampa_syns : list[Synapses | SynapticSubgroup | tuple[Synapses|SynapticSubgroup, dict]]
        Synapses or SynapticSubgroup objects representing AMPA synapses (delivering excitatory currents).
        Or a tuple of the Synapses or SynapticSubgroup object and a dictionary of parameters to override.
    gaba_syns : list[Synapses | SynapticSubgroup | tuple[Synapses|SynapticSubgroup, dict]]
        Synapses or SynapticSubgroup objects representing GABA synapses (delivering inhibitory currents).
        Or a tuple of the Synapses or SynapticSubgroup object and a dictionary of parameters to override.
    weight : str | float, optional
        Name of the weight variable or parameter in the Synapses or SynapticSubgroup objects, or a float
        in the case of a single weight for all synapses. Default is 'w'.
    """

    # can override on injection: tau1|2_ampa|gaba, syn_delay, I_threshold
    tau1_ampa: Quantity = 2 * ms
    """The fall time constant of the biexponential current kernel for AMPA synapses.
    2 ms by default."""
    tau2_ampa: Quantity = 0.4 * ms
    """The time constant of subtracted part of the biexponential current kernel for AMPA synapses.
    0.4 ms by default."""
    tau1_gaba: Quantity = 5 * ms
    """The fall time constant of the biexponential current kernel for GABA synapses.
    5 ms by default."""
    tau2_gaba: Quantity = 0.25 * ms
    """The time constant of subtracted part of the biexponential current kernel for GABA synapses.
    0.25 ms by default."""
    syn_delay: Quantity = 1 * ms
    """The synaptic transmission delay, i.e., between a spike and the onset of the postsynaptic current.
    1 ms by default."""
    I_threshold: float = 1e-3
    """Threshold, as a proportion of the peak current, below which spikes' contribution
    to synaptic currents (and thus LFP) is ignored, by default 1e-3."""
    weight: str = "w"
    """Name of the weight variable or parameter in the Synapses or SynapticSubgroup objects.
    Default is 'w'."""
    _ampa_sources: dict[NeuronGroup, dict[Synapses, SpikeToCurrentSource]] = field(
        init=False, factory=dict, repr=False
    )
    _gaba_sources: dict[NeuronGroup, dict[Synapses, SpikeToCurrentSource]] = field(
        init=False, factory=dict, repr=False
    )

    def _get_weight(self, syn, weight):
        assert isinstance(weight, (Number, Quantity, str))
        if isinstance(weight, (Number, Quantity)):
            return weight

        if isinstance(syn, Synapses):
            syn_name = syn.name
            if weight in syn.variables:
                return getattr(syn, weight)
            elif weight in syn.namespace:
                return syn.namespace[weight]
        elif isinstance(syn, SynapticSubgroup):
            syn_name = syn.synapses.name
            if weight in syn.synapses.variables:
                return getattr(syn.synapses, weight)[syn._stored_indices]
            elif weight in syn.synapses.namespace:
                return syn.synapses.namespace[weight]

        raise ValueError(
            f"weight {weight} not found in {syn_name} variables or namespace"
        )

    def _create_spk2curr_source(self, syn, neuron_group, weight, biexp_kwparams):
        # need source_ng, syn_i, syn_j
        if isinstance(syn, Synapses):
            source_ng = syn.source
            syn_i, syn_j = syn.i, syn.j
        elif isinstance(syn, SynapticSubgroup):
            source_ng = syn.synapses.source
            syn_i = syn.synapses.i[syn._stored_indices]
            syn_j = syn.synapses.j[syn._stored_indices]
        else:
            raise TypeError(
                "ampa_syns and gaba_syns only take Synapses or SynapticSubgroup objects"
            )
        mon = SpikeMonitor(source_ng, record=list(np.unique(syn_i)))
        self.brian_objects.add(mon)

        J = sparse.lil_array((source_ng.N, neuron_group.N))
        w = self._get_weight(syn, weight)
        J[syn_i, syn_j] = w
        J = J.tocsr()
        if self.pop_aggregate:
            J = J.sum(axis=1).reshape((-1, 1))

        return SpikeToCurrentSource(J, mon, biexp_kwparams)

    def _process_syn(self, syn, kwparams) -> tuple[Synapses, dict]:
        """handles the case when a tuple of Synapses, kwargs is passed in"""
        if isinstance(syn, (tuple, list)):
            syn, override_kwargs = syn
            kwparams = {**kwparams, **override_kwargs}
        else:
            assert isinstance(syn, (Synapses, SynapticSubgroup))

        return syn, kwparams

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # inherit docstring
        # this dict structure should allow multiple injections with no problem;
        # just the most recent injection will be used

        # prep wslfp calculator object
        if neuron_group not in self._wslfps:
            self._init_wslfp_calc(neuron_group, **kwparams)

        default_weight = kwparams.pop("weight", self.weight)
        ampa_syns = kwparams.pop("ampa_syns", [])
        gaba_syns = kwparams.pop("gaba_syns", [])

        biexp_kwparams = {}
        for key in [
            "tau1_ampa",
            "tau2_ampa",
            "tau1_gaba",
            "tau2_gaba",
            "syn_delay",
            "I_threshold",
        ]:
            biexp_kwparams[key] = kwparams.pop(key, getattr(self, key))

        if neuron_group not in self._ampa_sources:
            self._ampa_sources[neuron_group] = {}
            self._gaba_sources[neuron_group] = {}
        for ampa_syn in ampa_syns:
            ampa_syn, updated_kwparams = self._process_syn(ampa_syn, biexp_kwparams)
            weight = updated_kwparams.pop("weight", default_weight)
            self._ampa_sources[neuron_group][ampa_syn] = self._create_spk2curr_source(
                ampa_syn, neuron_group, weight, updated_kwparams
            )
        for gaba_syn in gaba_syns:
            gaba_syn, updated_kwparams = self._process_syn(gaba_syn, biexp_kwparams)
            weight = updated_kwparams.pop("weight", default_weight)
            self._gaba_sources[neuron_group][gaba_syn] = self._create_spk2curr_source(
                gaba_syn, neuron_group, weight, updated_kwparams
            )

    def _get_biexp_kwargs_from_s2cs(self, s2cs: SpikeToCurrentSource, syn_type: str):
        # check overrides, fall back on Signal-level defaults
        return {
            "tau1_ms": s2cs.biexp_kernel_params.get(
                f"tau1_{syn_type}", getattr(self, f"tau1_{syn_type}")
            )
            / ms,
            "tau2_ms": s2cs.biexp_kernel_params.get(
                f"tau2_{syn_type}", getattr(self, f"tau2_{syn_type}")
            )
            / ms,
            "syn_delay_ms": s2cs.biexp_kernel_params.get("syn_delay", self.syn_delay)
            / ms,
            "threshold": s2cs.biexp_kernel_params.get("I_threshold", self.I_threshold),
        }

    def _needed_current(self, ng, t_ampa_ms: float, t_gaba_ms: float) -> np.ndarray:
        """output must have shape (n_t, n_current_sources)"""
        n_sources = 1 if self.pop_aggregate else ng.N
        I_ampa = np.zeros((1, n_sources))
        for ampa_src in self._ampa_sources[ng].values():
            biexp_kwargs = self._get_biexp_kwargs_from_s2cs(ampa_src, "ampa")
            I_ampa += wslfp.spikes_to_biexp_currents(
                [t_ampa_ms],
                ampa_src.mon.t / ms,
                ampa_src.mon.i,
                ampa_src.J,
                **biexp_kwargs,
            )

        I_gaba = np.zeros((1, n_sources))
        for gaba_src in self._gaba_sources[ng].values():
            biexp_kwargs = self._get_biexp_kwargs_from_s2cs(gaba_src, "gaba")
            I_gaba += wslfp.spikes_to_biexp_currents(
                [t_gaba_ms],
                gaba_src.mon.t / ms,
                gaba_src.mon.i,
                gaba_src.J,
                **biexp_kwargs,
            )

        return I_ampa, I_gaba


@define(eq=False)
class RWSLFPSignalFromPSCs(RWSLFPSignalBase):
    """Computes RWSLFP from the currents onto pyramidal cells.

    Use this if your model already simulates synaptic current dynamics.
    ``Iampa_var_names`` and ``Igaba_var_names`` are lists of variable names to include
    and must be passed in as kwargs on injection or else the target neuron group will
    not contribute to this signal (desirable for interneurons).

    RWSLFP refers to the Reference Weighted Sum of synaptic currents LFP proxy from
    `Mazzoni, Lindén et al., 2015 <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004584>`_.

    Injection kwargs
    ----------------
    Iampa_var_names : list[str]
        List of variable names in the neuron group representing AMPA currents.
    Igaba_var_names : list[str]
        List of variable names in the neuron group representing GABA currents.
    """

    _ampa_vars: dict[NeuronGroup, list] = field(init=False, factory=dict, repr=False)
    _gaba_vars: dict[NeuronGroup, list] = field(init=False, factory=dict, repr=False)
    _t_ampa_bufs: dict[NeuronGroup, deque[float]] = field(
        init=False, factory=dict, repr=False
    )
    _I_ampa_bufs: dict[NeuronGroup, deque[np.ndarray]] = field(
        init=False, factory=dict, repr=False
    )
    _t_gaba_bufs: dict[NeuronGroup, deque[float]] = field(
        init=False, factory=dict, repr=False
    )
    _I_gaba: dict[NeuronGroup, deque[np.ndarray]] = field(
        init=False, factory=dict, repr=False
    )

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # ^ is XOR
        if ("Iampa_var_names" in kwparams) ^ ("Igaba_var_names" in kwparams):
            raise ValueError(
                "Iampa_var_names and Igaba_var_names must be included together."
            )
        if "Iampa_var_names" not in kwparams:
            return

        I_ampa_names = kwparams.pop("Iampa_var_names")
        I_gaba_names = kwparams.pop("Igaba_var_names")
        for varname in chain(I_ampa_names, I_gaba_names):
            if not hasattr(neuron_group, varname):
                raise ValueError(
                    f"NeuronGroup {neuron_group.name} does not have a variable {varname}"
                )

        # prep wslfp calculator object
        if neuron_group in self._wslfps:
            warnings.warn(
                f"{self.name} previously connected to {neuron_group.name}."
                " Reconnecting will overwrite previous connection."
            )

        self._init_wslfp_calc(neuron_group, **kwparams)

        buf_len_ampa, buf_len_gaba = self._get_buf_lens_for_wslfp(
            self._wslfps[neuron_group]
        )
        self._t_ampa_bufs[neuron_group] = deque(maxlen=buf_len_ampa)
        self._I_ampa_bufs[neuron_group] = deque(maxlen=buf_len_ampa)
        self._t_gaba_bufs[neuron_group] = deque(maxlen=buf_len_gaba)
        self._I_gaba[neuron_group] = deque(maxlen=buf_len_gaba)

        # add underscores to get values without units
        self._ampa_vars[neuron_group] = [varname + "_" for varname in I_ampa_names]
        self._gaba_vars[neuron_group] = [varname + "_" for varname in I_gaba_names]

    def _buf_len(self, tau, dt):
        return ceil((tau + dt) / dt)

    def _get_buf_lens_for_wslfp(self, wslfp_calc, **kwparams):
        # need sampling period
        sample_period_ms = kwparams.get("sample_period_ms", None)
        if sample_period_ms is None:
            try:
                sample_period_ms = self.probe.sim.io_processor.sample_period_ms
            except AttributeError:  # probably means sim doesn't have io_processor
                raise Exception(
                    "RSWLFPSignalFromPSCs needs to know the sampling period."
                    " Either set the simulator's IO processor before injecting"
                    f" {self.probe.name} or "
                    f" specify it on injection: .inject({self.probe.name}"
                    ", sample_period_ms=...)"
                )
        buf_len_ampa = self._buf_len(wslfp_calc.tau_ampa_ms, sample_period_ms)
        buf_len_gaba = self._buf_len(wslfp_calc.tau_gaba_ms, sample_period_ms)
        return buf_len_ampa, buf_len_gaba

    def _curr_from_buffer(self, t_buf_ms, I_buf, t_eval_ms: float, n_sources):
        # t_eval_ms is not iterable
        empty = np.zeros((1, n_sources))
        if len(t_buf_ms) == 0 or t_buf_ms[0] > t_eval_ms or t_buf_ms[-1] < t_eval_ms:
            return empty
        # when tau is multiple of sample time, current should be collected
        # right when needed, at the left end of the buffer
        elif np.isclose(t_eval_ms, t_buf_ms[0]):
            return I_buf[0]

        # if not, should only need to interpolate between first and second positions
        # if buffer length is correct
        assert len(t_buf_ms) > 1
        if t_buf_ms[0] < t_eval_ms < t_buf_ms[1]:
            i_l, i_r = 0, 1
        else:
            warnings.warn(
                f"Time buffer is unexpected. Did a sample get skipped? "
                f"t_buf_ms={t_buf_ms}, t_eval_ms={t_eval_ms}"
            )
            i_l, i_r = None, None
            for i, t in enumerate(t_buf_ms):
                if t < t_eval_ms:
                    i_l = i
                if t >= t_eval_ms:
                    i_r = i
                    break
            if i_l is None or i_r is None or i_l >= i_r:
                warnings.warn(
                    "Signal buffer does not contain currents at needed timepoints. "
                    "Returning 0. "
                    f"t_buf_ms={t_buf_ms}, t_eval_ms={t_eval_ms}"
                )
                return empty

        I_interp = I_buf[i_l] + (I_buf[i_r] - I_buf[i_l]) * (
            t_eval_ms - t_buf_ms[i_l]
        ) / (t_buf_ms[i_r] - t_buf_ms[i_l])

        I_interp = np.reshape(I_interp, (1, n_sources))
        I_interp = np.nan_to_num(I_interp, nan=0)
        return I_interp

    def _needed_current(
        self, ng, t_ampa_ms, t_gaba_ms
    ) -> tuple[np.ndarray, np.ndarray]:
        """outputs must have shape (n_t, n_current_sources)"""
        # First add current currents to history
        # -- need to round to avoid floating point errors
        now_ms = round(self.probe.sim.network.t / ms, 3)
        self._t_ampa_bufs[ng].append(now_ms)
        self._t_gaba_bufs[ng].append(now_ms)

        I_ampa = np.zeros((1, ng.N))
        for I_ampa_name in self._ampa_vars[ng]:
            I_ampa += getattr(ng, I_ampa_name)

        I_gaba = np.zeros((1, ng.N))
        for I_gaba_name in self._gaba_vars[ng]:
            I_gaba += getattr(ng, I_gaba_name)

        if self.pop_aggregate:
            I_ampa = np.sum(I_ampa)
            I_gaba = np.sum(I_gaba)
        self._I_ampa_bufs[ng].append(I_ampa)
        self._I_gaba[ng].append(I_gaba)

        # Then interpolate history to get currents at the requested times
        n_sources = 1 if self.pop_aggregate else ng.N
        I_ampa = self._curr_from_buffer(
            self._t_ampa_bufs[ng], self._I_ampa_bufs[ng], t_ampa_ms, n_sources
        )
        I_gaba = self._curr_from_buffer(
            self._t_gaba_bufs[ng], self._I_gaba[ng], t_gaba_ms, n_sources
        )

        return I_ampa, I_gaba

    def reset(self, **kwargs) -> None:
        self._init_saved_vars()
        for ng in self._t_ampa_bufs:
            buf_len_ampa, buf_len_gaba = self._get_buf_lens_for_wslfp(self._wslfps[ng])
            self._t_ampa_bufs[ng] = deque(maxlen=buf_len_ampa)
            self._I_ampa_bufs[ng] = deque(maxlen=buf_len_ampa)
            self._t_gaba_bufs[ng] = deque(maxlen=buf_len_gaba)
            self._I_gaba[ng] = deque(maxlen=buf_len_gaba)
