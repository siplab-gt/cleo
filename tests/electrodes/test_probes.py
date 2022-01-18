"""Tests for electrodes module"""
from typing import Tuple, Any

import numpy as np
import pytest
from brian2 import NeuronGroup, mm, Network, StateMonitor, umeter

import cleosim
from cleosim import CLSimulator
from cleosim.electrodes import ElectrodeGroup, linear_shank_coords, Signal
from cleosim.electrodes.probes import (
    poly2_shank_coords,
    poly3_shank_coords,
    tetrode_shank_coords,
)


def test_ElectrodeGroup():
    eg = ElectrodeGroup("eg", [0, 0, 0] * mm)
    assert eg.n == 1
    eg = ElectrodeGroup("eg", [[0, 0, 0], [1, 1, 1]] * mm)
    assert eg.n == 2
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [0, 0] * mm)
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [0, 0, 0, 0] * mm)
    with pytest.raises(ValueError):
        ElectrodeGroup("eg", [[0, 0], [1, 1], [2, 2], [3, 3]] * mm)


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
    eg = ElectrodeGroup("eg", [0, 0, 0] * mm, signals=[dumb])
    eg.add_signals(dumber)
    sim.inject_recorder(eg, ng)

    assert dumb.brian_objects.issubset(sim.network.objects)
    assert dumber.brian_objects.issubset(sim.network.objects)
    assert sim.get_state()["eg"]["dumb"] == ng.name
    assert sim.get_state()["eg"]["dumber"] == ng.name

    with pytest.raises(ValueError):
        # cannot use same signal object for two electrodes
        ElectrodeGroup("eg2", [0, 0, 0] * mm, signals=[dumb])


def _dist(c1, c2):
    # not using norm because it strips units
    return np.sqrt(np.sum((c1 - c2) ** 2))


def test_linear_shank_coords():
    coords = linear_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=1 * mm,
        channel_count=32,
    )
    assert coords.shape == (32, 3)
    assert _dist(coords[0], coords[-1]) == 1 * mm

    coords2 = linear_shank_coords(
        base_location=(1, 1, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=1 * mm,
        channel_count=32,
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_tetrode_shank_coords():
    coords = tetrode_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=1.5 * mm,
        tetrode_count=5,
    )
    assert coords.shape == (20, 3)
    # check length (length + tiny bit for contacts above and below tetrode centers)
    assert _dist(coords[0], coords[-1]) == 1.5 * mm + 25 * umeter * np.sqrt(2)
    # check contact spacing (tetrode width)
    assert _dist(coords[0], coords[1]) == 25 * umeter

    coords2 = tetrode_shank_coords(
        base_location=(1, 1, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=1.5 * mm,
        tetrode_count=5,
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_poly2_shank_coords():
    coords = poly2_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=3 * mm,
        channel_count=32,
        intercol_space=50 * umeter,
    )
    assert coords.shape == (32, 3)
    # check length
    assert _dist(coords[0, 2], coords[-1, 2]) == 3 * mm
    # check spacing
    assert _dist(coords[0, :2], coords[1, :2]) == 50 * umeter

    coords2 = poly2_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=3 * mm,
        channel_count=32,
        intercol_space=51 * umeter,  # <- only difference
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_poly3_shank_coords():
    coords = poly3_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=0.55 * mm,
        channel_count=32,
        intercol_space=50 * umeter,
    )
    assert coords.shape == (32, 3)
    # check length
    assert _dist(coords[0, 2], coords[-1, 2]) == 0.55 * mm
    # check spacing, assuming first contact is in middle and 2nd and 3rd
    # are on the sides of it
    assert _dist(coords[0, :2], coords[1, :2]) == 50 * umeter
    assert _dist(coords[0, :2], coords[2, :2]) == 50 * umeter

    coords2 = poly2_shank_coords(
        base_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        recording_length=0.55 * mm,
        channel_count=32,
        intercol_space=51 * umeter,  # <- only difference
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))
