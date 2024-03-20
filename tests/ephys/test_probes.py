"""Tests for electrodes module"""
from typing import Any, Tuple

import neo
import pytest
import quantities as pq
from brian2 import Network, NeuronGroup, StateMonitor, mm, np, umeter

import cleo
from cleo import CLSimulator
from cleo.base import NeoExportable
from cleo.coords import concat_coords
from cleo.ephys import (
    Probe,
    Signal,
    linear_shank_coords,
    poly2_shank_coords,
    poly3_shank_coords,
    tetrode_shank_coords,
    tile_coords,
)


def test_Probe():
    probe = Probe([0, 0, 0] * mm)
    assert probe.n == 1
    probe = Probe([[0, 0, 0], [1, 1, 1]] * mm)
    assert probe.n == 2
    with pytest.raises(ValueError):
        Probe([0, 0] * mm)
    with pytest.raises(ValueError):
        Probe([0, 0, 0, 0] * mm)
    with pytest.raises(ValueError):
        Probe([[0, 0], [1, 1], [2, 2], [3, 3]] * mm)


class DummySignal(Signal, NeoExportable):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        self.ng = neuron_group
        self.value = 1
        self.brian_objects.add(StateMonitor(self.ng, "v", record=True))

    def get_state(self) -> Any:
        return self.ng.name

    def reset(self):
        self.value = 0

    def to_neo(self):
        return neo.core.AnalogSignal([1, 2, 3], sampling_period=1 * pq.ms, units="uV")


def test_Probe_injection():
    ng = NeuronGroup(1, "v : 1")
    sim = CLSimulator(Network(ng))

    dumb = DummySignal(name="dumb")
    dumber = DummySignal(name="dumber")
    probe = Probe([0, 0, 0] * mm, signals=[dumb])
    probe.add_signals(dumber)
    sim.inject(probe, ng)

    assert dumb.brian_objects.issubset(sim.network.objects)
    assert dumber.brian_objects.issubset(sim.network.objects)
    assert sim.get_state()["Probe"]["dumb"] == ng.name
    assert sim.get_state()["Probe"]["dumber"] == ng.name

    with pytest.raises(ValueError):
        # cannot use same signal object for two electrodes
        Probe([0, 0, 0] * mm, signals=[dumb], name="probe2")

    assert dumb.value == dumber.value == 1
    probe.reset()
    assert dumb.value == dumber.value == 0


def _dist(c1, c2):
    # not using norm because it strips units
    return np.sqrt(np.sum((c1 - c2) ** 2))


def test_linear_shank_coords():
    coords = linear_shank_coords(
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=1 * mm,
        channel_count=32,
    )
    assert coords.shape == (32, 3)
    assert _dist(coords[0], coords[-1]) == 1 * mm

    coords2 = linear_shank_coords(
        start_location=(1, 1, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=1 * mm,
        channel_count=32,
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_tetrode_shank_coords():
    tetr_width = 30 * umeter  # default is 25 um
    coords = tetrode_shank_coords(
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=1.5 * mm,
        tetrode_count=5,
        tetrode_width=tetr_width,
    )
    assert coords.shape == (20, 3)
    # check length (length + tiny bit for contacts above and below tetrode centers)
    assert _dist(coords[0], coords[-1]) == 1.5 * mm + tetr_width * np.sqrt(2)
    # check contact spacing (tetrode width)
    assert np.allclose(_dist(coords[0], coords[1]) / umeter, tetr_width / umeter)

    coords2 = tetrode_shank_coords(
        start_location=(1, 1, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=1.5 * mm,
        tetrode_count=5,
        tetrode_width=tetr_width,
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_poly2_shank_coords():
    coords = poly2_shank_coords(
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=3 * mm,
        channel_count=32,
        intercol_space=50 * umeter,
    )
    assert coords.shape == (32, 3)
    # check length
    assert _dist(coords[0, 2], coords[-1, 2]) == 3 * mm
    # check spacing
    assert _dist(coords[0, :2], coords[1, :2]) == 50 * umeter

    coords2 = poly2_shank_coords(
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=3 * mm,
        channel_count=32,
        intercol_space=51 * umeter,  # <- only difference
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_poly3_shank_coords():
    coords = poly3_shank_coords(
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=0.55 * mm,
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
        start_location=(0, 0, 0.1) * mm,
        direction=(0, 0, 0.01),
        array_length=0.55 * mm,
        channel_count=32,
        intercol_space=51 * umeter,  # <- only difference
    )
    # no matching rows between them
    assert not np.any(np.all(coords == coords2, axis=1))


def test_concat_tile_coords():
    poly2 = poly2_shank_coords(0.5 * mm, 16, 50 * umeter)
    # tiling 3 shanks over .4 mm creates a .2 mm intershank distance
    multi_poly2 = tile_coords(poly2, 3, (0.4, 0, 0) * mm)
    assert multi_poly2.shape == (48, 3)
    # 3 shanks should have same coords except for in X
    assert np.all(np.equal(multi_poly2[:16, 1:], multi_poly2[16:32, 1:]))
    assert np.all(np.equal(multi_poly2[:16, 1:], multi_poly2[32:, 1:]))

    # create a tetrode multi-shank probe further out at X=.6mm
    tetr = tetrode_shank_coords(0.5 * mm, 3, (0.6, 0, 0) * mm)
    multi_tetr = tile_coords(tetr, 3, (0.4, 0, 0) * mm)
    # similar coordinate checks:
    assert multi_tetr.shape == (36, 3)
    assert np.all(np.equal(multi_tetr[:12, 1:], multi_tetr[12:24, 1:]))
    assert np.all(np.equal(multi_tetr[:12, 1:], multi_tetr[24:, 1:]))

    # now combine and repeat along Y axis for a 3D "matrix array"
    multi_hybrid = concat_coords(multi_poly2, multi_tetr)
    n_hybr = 48 + 36
    assert multi_hybrid.shape == (n_hybr, 3)
    matrix_array = tile_coords(multi_hybrid, 4, (0, 1, 0) * mm)
    assert matrix_array.shape == (n_hybr * 4, 3)
    # check for matching z axis between different rows of shanks
    for i_row in range(2):
        i_coord1 = i_row * n_hybr
        i_coord2 = (i_row + 1) * n_hybr
        i_coord3 = (i_row + 2) * n_hybr
        assert np.all(
            np.equal(
                matrix_array[i_coord1:i_coord2, 2], matrix_array[i_coord2:i_coord3, 2]
            )
        )


@pytest.mark.parametrize("n_signals", [1, 3])
def test_probe_to_neo(n_signals):
    probe = Probe([0, 0, 0] * mm)
    if n_signals > 1:
        with pytest.raises(ValueError, match="Signal names must be unique"):
            probe.add_signals(*[DummySignal() for _ in range(n_signals)])
    probe.add_signals(*[DummySignal(name=f"sig{i}") for i in range(n_signals)])
    probe_neo = probe.to_neo()

    assert type(probe_neo) == neo.core.Group
    assert probe_neo.name == probe.name
    assert len(probe_neo.children) == n_signals
