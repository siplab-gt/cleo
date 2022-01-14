"""Tests for electrodes module"""
from typing import Tuple, Any

import numpy as np
import pytest
from brian2 import NeuronGroup, mm, Network, StateMonitor

import cleosim
from cleosim import CLSimulator
from cleosim.ephys import ElectrodeGroup, get_1D_probe_coords, Signal


def test_ElectrodeGroup():
    eg = ElectrodeGroup("eg", [0, 0, 0]*mm)
    assert eg.n == 1
    eg = ElectrodeGroup("eg", [[0, 0, 0], [1, 1, 1]]*mm)
    assert eg.n == 2
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [0, 0]*mm)
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [0, 0, 0, 0]*mm)
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [[0, 0], [1, 1], [2, 2], [3, 3]]*mm)


def test_electrode_injection():
    class DummySignal(Signal):
        def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
            self.ng = neuron_group
            self.brian_objects.add(StateMonitor(ng, "v", record=True))

        def get_state(self) -> Any:
            return self.ng.name

    ng = NeuronGroup(1, "v : 1")
    sim = CLSimulator(Network(ng))

    dumb = DummySignal("dumb")
    dumber = DummySignal("dumber")
    eg = ElectrodeGroup("eg", [0, 0, 0]*mm, signals=[dumb])
    eg.add_signals(dumber)
    sim.inject_recorder(eg, ng)

    assert dumb.brian_objects.issubset(sim.network.objects)
    assert dumber.brian_objects.issubset(sim.network.objects)
    assert sim.get_state()["eg"]["dumb"] == ng.name
    assert sim.get_state()["eg"]["dumber"] == ng.name

    with pytest.raises(ValueError):
        # cannot use same signal object for two electrodes
        ElectrodeGroup("eg2", [0, 0, 0]*mm, signals=[dumb])


def test_probe_coords():
    coords = get_1D_probe_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        length=1 * mm,
        channel_count=32,
    )
    assert coords.shape == (32, 3)
    # not using norm because it strips units
    assert np.sqrt(np.sum((coords[0] - coords[-1]) ** 2)) == 1 * mm

    coords2 = get_1D_probe_coords(
        base_location=(1, 1, 0.1) * mm,
        direction=(0, 0, 0.01),
        length=1 * mm,
        channel_count=32,
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))
