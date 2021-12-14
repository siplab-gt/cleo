"""Tests for ephys.spiking module"""
import numpy as np

from brian2 import SpikeGeneratorGroup, ms, mm, Network, prefs
from cleosim import CLSimulator
from cleosim.ephys import *


def test_MultiUnitSpiking():
    np.random.seed(1830)
    prefs.codegen.target = "numpy"
    # raster of test spikes: each character is 1-ms bin
    # | || |  <- neuron 0, 0mm     contact at .25mm
    #    |||  <- neuron 1, 0.5mm   contact at .75mm
    #     ||  <- neuron 2, 1mm
    sgg = SpikeGeneratorGroup(
        3, [0, 0, 0, 1, 1, 2, 0, 1, 2], [1, 2.1, 3.5, 3.5, 4.1, 4.9, 5.1, 5.3, 5.5] * ms
    )
    # assign_coords_grid_rect_prism(sgg, (0, 0), (0, 0), (0, 1), shape=(1, 1, 3))
    for var in ["x", 'y', 'z']:
        sgg.add_attribute(var)
    sgg.x = 0*mm; sgg.y = 0*mm; sgg.z = (0, .5, 1)*mm
    net = Network(sgg)
    sim = CLSimulator(net)
    mus = MultiUnitSpiking(
        "mus",
        perfect_detection_radius=0.3 * mm,
        half_detection_radius=1.2 * mm,
        save_history=True,
    )
    eg = ElectrodeGroup("eg", [[0, 0, .25], [0, 0, .75]], [mus])
    sim.inject_recorder(eg, sgg)

    # remember i here is channel, no longer neuron
    i, t = mus.get_state()
    assert len(i) == 0
    assert len(t) == 0

    sim.run(1 * ms)  # 1 ms
    i, t = mus.get_state()
    assert (0 in i) and (1 in t)

    sim.run(1 * ms)  # 2 ms
    i, t = mus.get_state()
    assert len(i) == 0 and len(t) == 0

    sim.run(1 * ms)  # 3 ms
    i, t = mus.get_state()
    assert (0 in i) and (2.1 in t)

    sim.run(1 * ms)  # 4 ms
    i, t = mus.get_state()
    # should pick up 2 spikes on first and 1 or 2 on second channel
    assert 1 in i and len(i) >= 3
    assert 3.5 in t

    # skip to 6 ms
    sim.run(2 * ms)
    i, t = mus.get_state()
    assert (0 in i) and (1 in i) and len(i) >= 4
    assert all(t_i in t for t_i in [5.1, 5.3, 5.5])

    # should have picked up at least one but not all spikes outside perfect radius
    assert len(mus.i) > 12
    assert len(mus.i) <= len(sgg.indices)
    assert len(mus.i) == len(mus.t)


def test_SortedSpiking():
    pass
