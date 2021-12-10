"""Tests for ephys.spiking module"""
from brian2 import SpikeGeneratorGroup, ms, mm, Network, prefs
from cleosim import CLSimulator
from cleosim.ephys import *


def test_MultiUnitSpiking():
    prefs.codegen.target = "numpy"
    # raster of test spikes: each character is 1-ms bin
    # | || |  <- neuron 0, 0mm     contact at .25mm
    #    |||  <- neuron 1, 0.5mm   contact at .75mm
    #     ||  <- neuron 2, 1mm
    sgg = SpikeGeneratorGroup(
        3, [0, 0, 0, 1, 1, 2, 0, 1, 2], [1, 2.1, 3.5, 3.5, 4.1, 4.9, 5.1, 5.3, 5.5] * ms
    )
    net = Network(sgg)
    sim = CLSimulator(net)
    mus = MultiUnitSpiking(
        "mus", perfect_detection_radius=0.3 * mm, half_detection_radius=1 * mm
    )
    eg = ElectrodeGroup("eg", [[0, 0, 0.25], [0, 0, 0.75]] * mm, [mus])
    sim.inject_recorder(eg, sgg)

    # remember i here is channel, no longer neuron
    i, t = mus.get_state()
    assert len(i) == 0
    assert len(t) == 0

    sim.run(1 * ms)  # 1 ms
    i, t = mus.get_state()
    assert 0 in i
    assert 1 * ms in t
    #TODO: make spikes more predictable so I can run and assert in a for loop


def test_SortedSpiking():
    pass
