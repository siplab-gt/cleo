"""Contains LFP signals"""
from __future__ import annotations

from datetime import datetime
from numbers import Number
from typing import Any, Union

import neo
import numpy as np
import quantities as pq
import wslfp
from attrs import define, field
from brian2 import NeuronGroup, Quantity, Subgroup, Synapses, mm, ms, um
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.synapses.synapses import SynapticSubgroup
from nptyping import NDArray
from scipy import interpolate, sparse
from tklfp import TKLFP

import cleo.utilities
from cleo.base import NeoExportable
from cleo.coords import coords_from_ng
from cleo.ephys.probes import Signal


@define(eq=False)
class TKLFPSignal(Signal, NeoExportable):
    """Records the Teleńczuk kernel LFP approximation.

    Requires ``tklfp_type='exc'|'inh'`` to specify cell type
    on injection.

    An ``orientation`` keyword argument can also be specified on
    injection, which should be an array of shape ``(n_neurons, 3)``
    representing which way is "up," that is, towards the surface of
    the cortex, for each neuron. If a single vector is given, it is
    taken to be the orientation for all neurons in the group. [0, 0, -1]
    is the default, meaning the negative z axis is "up." As stated
    elsewhere, Cleo's convention is that z=0 corresponds to the
    cortical surface and increasing z values represent increasing depth.

    TKLFP is computed from spikes using the `tklfp package <https://github.com/kjohnsen/tklfp/>`_.
    """

    uLFP_threshold_uV: float = 1e-3
    """Threshold, in microvolts, above which the uLFP for a single spike is guaranteed
    to be considered, by default 1e-3.
    This determines the buffer length of past spikes, since the uLFP from a long-past
    spike becomes negligible and is ignored."""
    t_ms: NDArray[(Any,), float] = field(init=False, repr=False)
    """Times at which LFP is recorded, in ms, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    lfp_uV: NDArray[(Any, Any), float] = field(init=False, repr=False)
    """Approximated LFP from every call to :meth:`get_state`.
    Shape is (n_samples, n_channels). Stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    _elec_coords_mm: np.ndarray = field(init=False, repr=False)
    _tklfps: list[TKLFP] = field(init=False, factory=list, repr=False)
    _monitors: list[SpikeMonitor] = field(init=False, factory=list, repr=False)
    _mon_spikes_already_seen: list[int] = field(init=False, factory=list, repr=False)
    _i_buffers: list[list[np.ndarray]] = field(init=False, factory=list, repr=False)
    _t_ms_buffers: list[list[np.ndarray]] = field(init=False, factory=list, repr=False)
    _buffer_positions: list[int] = field(init=False, factory=list, repr=False)

    def _post_init_for_probe(self):
        self._elec_coords_mm = self.probe.coords / mm
        # need to invert z coords since cleo uses an inverted z axis and
        # tklfp does not
        self._elec_coords_mm[:, 2] *= -1
        self._init_saved_vars()

    def _init_saved_vars(self):
        if self.probe.save_history:
            self.t_ms = np.empty((0,))
            self.lfp_uV = np.empty((0, self.probe.n))

    def _update_saved_vars(self, t_ms, lfp_uV):
        if self.probe.save_history:
            self.t_ms = np.concatenate([self.t_ms, [t_ms]])
            self.lfp_uV = np.vstack([self.lfp_uV, lfp_uV])

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
            elec_coords_mm=self._elec_coords_mm,
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
        out = np.reshape(tot_tklfp, (-1,))  # return 1D array (vector)
        self._update_saved_vars(now_ms, out)
        return out

    def reset(self, **kwargs) -> None:
        super(TKLFPSignal, self).reset(**kwargs)
        for i_mon in range(len(self._monitors)):
            self._reset_buffer(i_mon)
        self._init_saved_vars()

    def _reset_buffer(self, i_mon):
        mon = self._monitors[i_mon]
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

    def to_neo(self) -> neo.AnalogSignal:
        # inherit docstring
        try:
            signal = cleo.utilities.analog_signal(
                self.t_ms,
                self.lfp_uV,
                "uV",
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
class RWSLFPSignalBase(Signal, NeoExportable):
    """Records the weighted sum of synaptic current LFP proxy from spikes.

    Requires list of ``ampa_syns`` and ``gaba_syns` on injection.

    An ``orientation`` keyword argument can also be specified on
    injection, which should be an array of shape ``(n_neurons, 3)``
    representing which way is "up," that is, towards the surface of
    the cortex, for each neuron. If a single vector is given, it is
    taken to be the orientation for all neurons in the group. [0, 0, -1]
    is the default, meaning the negative z axis is "up." As stated
    elsewhere, Cleo's convention is that z=0 corresponds to the
    cortical surface and increasing z values represent increasing depth.

    RWSLFP is computed from spikes using the `wslfp package <https://github.com/siplab-gt/wslfp/>`_.
    WSLFPCalculator params can be specified on injection with alpha, tau_ampa_ms, tau_gaba_ms.
    """

    # note these set defaults that can be overrriden on injection
    amp_func: callable = wslfp.mazzoni15_nrn
    pop_aggregate: bool = False
    """Threshold, as a proportion of the peak current, below which spikes' contribution
    to synaptic currents (and thus LFP) is ignored, by default 1e-3."""
    t_ms: NDArray[(Any,), float] = field(init=False, repr=False)
    """Times at which LFP is recorded, in ms, stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    lfp: NDArray[(Any, Any), float] = field(init=False, repr=False)
    """Approximated LFP from every call to :meth:`get_state`.
    Shape is (n_samples, n_channels). Stored if
    :attr:`~cleo.InterfaceDevice.save_history` on :attr:`~Signal.probe`"""
    _elec_coords_um: np.ndarray = field(init=False, repr=False)
    _wslfps: dict[NeuronGroup, wslfp.WSLFPCalculator] = field(
        init=False, factory=dict, repr=False
    )

    def _post_init_for_probe(self):
        self._elec_coords_um = self.probe.coords / um
        # need to invert z coords since cleo uses an inverted z axis and
        # tklfp does not
        self._elec_coords_um[:, 2] *= -1
        self._init_saved_vars()

    def _init_saved_vars(self):
        if self.probe.save_history:
            self.t_ms = np.empty((0,))
            self.lfp = np.empty((0, self.probe.n))

    def _update_saved_vars(self, t_ms, lfp):
        if self.probe.save_history:
            self.t_ms = np.concatenate([self.t_ms, [t_ms]])
            self.lfp = np.vstack([self.lfp, lfp])

    def _init_wslfp_calc(self, neuron_group: NeuronGroup, **kwparams):
        nrn_coords_um = coords_from_ng(neuron_group) / um
        nrn_coords_um[:, 2] *= -1

        orientation = kwparams.pop("orientation", np.array([[0, 0, -1]])).copy()
        orientation[:, 2] *= -1

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

        self._wslfps[neuron_group] = wslfp.from_xyz_coords(
            self._elec_coords_um,
            nrn_coords_um,
            amp_func=kwparams.pop("amp_func", self.amp_func),
            source_orientation=orientation,
            **wslfp_kwargs,
        )

    def get_state(self) -> np.ndarray:
        now_ms = self.probe.sim.network.t / ms
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
        super(RWSLFPSignalBase, self).reset(**kwargs)
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
    J: Union[np.ndarray, sparse.sparray]
    mon: SpikeMonitor
    biexp_kernel_params: dict[str, Any]


@define(eq=False)
class RWSLFPSignalFromSpikes(RWSLFPSignalBase):
    # can override on injection: tau1|2_ampa|gaba, syn_delay, I_threshold
    tau1_ampa: Quantity = 2 * ms
    tau2_ampa: Quantity = 0.4 * ms
    tau1_gaba: Quantity = 5 * ms
    tau2_gaba: Quantity = 0.25 * ms
    syn_delay: Quantity = 1 * ms
    I_threshold: float = 1e-3
    weight: str = "w"
    # for each source, need spike monitor, J, and biexp kernel params
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
            assert isinstance(syn, Synapses)

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
    _ampa_vars: dict[NeuronGroup, str] = field(init=False, factory=dict, repr=False)
    _gaba_vars: dict[NeuronGroup, str] = field(init=False, factory=dict, repr=False)
    _t: dict[NeuronGroup, list[float]] = field(init=False, factory=dict, repr=False)
    _I_ampa: dict[NeuronGroup, list[np.ndarray]] = field(
        init=False, factory=dict, repr=False
    )
    _I_gaba: dict[NeuronGroup, list[np.ndarray]] = field(
        init=False, factory=dict, repr=False
    )

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        if ("Iampa_var_name" in kwparams or "Igaba_var_name" in kwparams) and not (
            "Iampa_var_name" in kwparams and "Igaba_var_name" in kwparams
        ):
            raise ValueError(
                "Iampa_var_name and Igaba_var_name must be included together."
            )
        if "Iampa_var_name" not in kwparams:
            return

        I_ampa_name = kwparams.pop("Iampa_var_name")
        if not hasattr(neuron_group, I_ampa_name):
            raise ValueError(
                f"NeuronGroup {neuron_group.name} does not have a variable {I_ampa_name}"
            )

        I_gaba_name = kwparams.pop("Igaba_var_name")
        if not hasattr(neuron_group, I_gaba_name):
            raise ValueError(
                f"NeuronGroup {neuron_group.name} does not have a variable {I_ampa_name}"
            )

        # prep wslfp calculator object
        if neuron_group not in self._wslfps:
            self._init_wslfp_calc(neuron_group, **kwparams)

        self._ampa_vars[neuron_group] = I_ampa_name
        self._gaba_vars[neuron_group] = I_gaba_name

        self._t[neuron_group] = []
        self._I_ampa[neuron_group] = []
        self._I_gaba[neuron_group] = []

    def _interp_currents(self, t_ms, I, t_eval_ms: float, n_sources):
        # TODO: super slow , bogging down the whole simulation
        # do own interpolation. easy with searchsorted
        timestart = datetime.now()
        empty = np.zeros((1, n_sources))
        if len(t_ms) == 0:
            return empty

        if len(t_ms) > 1:
            interpolator = interpolate.PchipInterpolator(t_ms, I, extrapolate=False)
        elif len(t_ms) == 1:
            interpolator = lambda t_eval: np.multiply((t_eval == t_ms[0]), I[0:1])

        I_interp = interpolator(t_eval_ms)
        I_interp = np.reshape(I_interp, (1, n_sources))
        I_interp = np.nan_to_num(I_interp, nan=0)
        print("interp took", datetime.now() - timestart)
        return I_interp

    def _needed_current(
        self, ng, t_ampa_ms, t_gaba_ms
    ) -> tuple[np.ndarray, np.ndarray]:
        """outputs must have shape (n_t, n_current_sources)"""
        # First add current currents to history
        self._t[ng].append(self.probe.sim.network.t / ms)
        I_ampa = getattr(ng, self._ampa_vars[ng])
        I_gaba = getattr(ng, self._gaba_vars[ng])
        if self.pop_aggregate:
            I_ampa = np.sum(I_ampa)
            I_gaba = np.sum(I_gaba)
        self._I_ampa[ng].append(I_ampa)
        self._I_gaba[ng].append(I_gaba)

        # Then interpolate history to get currents at the requested times
        n_sources = 1 if self.pop_aggregate else ng.N
        I_ampa = self._interp_currents(
            self._t[ng], self._I_ampa[ng], t_ampa_ms, n_sources
        )
        I_gaba = self._interp_currents(
            self._t[ng], self._I_gaba[ng], t_gaba_ms, n_sources
        )

        return I_ampa, I_gaba

    def reset(self, **kwargs) -> None:
        super(RWSLFPSignalBase, self).reset(**kwargs)
        self._init_saved_vars()
        for ng in self._t:
            self._t[ng] = []
            self._I_ampa[ng] = []
            self._I_gaba[ng] = []
