"""Contains LFP signals"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import quantities as pq
from attrs import define, field
from brian2 import NeuronGroup, mm, ms
from brian2.monitors.spikemonitor import SpikeMonitor
from nptyping import NDArray
from tklfp import TKLFP

import cleo.utilities
from cleo.base import NeoExportable
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
