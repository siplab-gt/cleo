"""Contains LFP signals"""
from __future__ import annotations

from brian2 import NeuronGroup, mm, ms
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
from tklfp import TKLFP

from cleosim.ephys.electrodes import Signal, ElectrodeGroup


class TKLFPSignal(Signal):
    uLFP_threshold_uV: float
    _elec_coords_mm: np.ndarray
    _tklfps: list[TKLFP]
    _monitors: list[SpikeMonitor]
    _mon_spikes_already_seen: list[int]
    _i_buffers: list[list[np.ndarray]]
    _t_ms_buffers: list[list[np.ndarray]]

    def __init__(self, name: str, uLFP_threshold_uV: float = 1e-3) -> None:
        super().__init__(name)
        self.uLFP_threshold_uV = uLFP_threshold_uV
        self._tklfps = []
        self._monitors = []
        self._mon_spikes_already_seen = []
        self._i_buffers = []
        self._t_ms_buffers = []
        self._buffer_positions = []

    def init_for_electrode_group(self, eg: ElectrodeGroup):
        super().init_for_electrode_group(eg)
        self.elec_coords_mm = eg.coords / mm
        # need to invert z coords since cleosim uses an inverted z axis and
        # tklfp does not
        self.elec_coords_mm[:, 2] *= -1

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        # prep tklfp object
        tklfp_type = kwparams.get("tklfp_type", "not given")
        if tklfp_type not in ["exc", "inh"]:
            raise Exception(
                "tklfp_type ('exc' or 'inh') must be passed as a keyword argument to "
                "inject_recorder() when injecting an electrode with a TKLFPSignal."
            )

        tklfp = TKLFP(
            neuron_group.x / mm,
            neuron_group.y / mm,
            -neuron_group.z / mm,  # invert neuron zs as well
            is_excitatory=tklfp_type == "exc",
            elec_coords_mm=self.elec_coords_mm,
        )
        self._tklfps.append(tklfp)

        # prep buffers
        buf_len = self._get_buffer_length(tklfp, **kwparams)
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
        now_ms = self.electrode_group.sim.network.t / ms
        # loop over neuron groups (monitors, tklfps)
        for i_mon in range(len(self._monitors)):
            self._update_spike_buffer(i_mon)
            tot_tklfp += self._tklfp_for_monitor(i_mon, now_ms)
        return tot_tklfp.reshape((-1,))  # return 1D array (vector)

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
        sampling_period_ms = kwparams.get("sampling_period_ms", None)
        if sampling_period_ms is None:
            try:
                sampling_period_ms = (
                    self.electrode_group.sim.proc_loop.sampling_period_ms
                )
            except AttributeError:  # probably means sim doesn't have proc_loop
                raise Exception(
                    "TKLFP needs to know the sampling period. Either set the simulator's "
                    f"processing loop before injecting {self.electrode_group.name} or "
                    f"specify it on injection: .inject_recorder({self.electrode_group.name} "
                    ", tklfp_type=..., sampling_period_ms=...)"
                )
        return np.ceil(
            tklfp.compute_min_window_ms(self.uLFP_threshold_uV) / sampling_period_ms
        ).astype(int)
