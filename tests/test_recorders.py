import pytest
from brian2 import NeuronGroup, Network, ms

from cleosim.recorders import (
    RateRecorder,
    VoltageRecorder,
    GroundTruthSpikeRecorder,
)
from cleosim import CLSimulator


@pytest.fixture
def neurons():
    # will spike every time
    return NeuronGroup(1, "v = -70 : 1", threshold="v > -80")


@pytest.fixture
def sim(neurons):
    net = Network(neurons)
    return CLSimulator(net)


def test_RateRecorder(sim, neurons):
    rate_rec = RateRecorder("rate_rec", 0)
    sim.inject_recorder(rate_rec, neurons)
    assert rate_rec.get_state() == 0

    sim.run(0.3 * ms)  # will spike at every 0.1 ms timestep

    assert rate_rec.get_state() == 10 / ms


def test_VoltageRecorder(sim, neurons):
    v_rec = VoltageRecorder("v_rec")
    sim.inject_recorder(v_rec, neurons)

    sim.run(0.1 * ms)  # will spike at every 0.1 ms timestep
    assert v_rec.get_state() == -70


def test_GroundTruthSpikeRecorder(sim, neurons):
    spike_rec = GroundTruthSpikeRecorder("spike_rec")
    sim.inject_recorder(spike_rec, neurons)

    sim.run(0.1 * ms)  # will spike at every 0.1 ms timestep
    assert spike_rec.get_state() == 1

    sim.run(0.2 * ms)
    assert spike_rec.get_state() == 2
