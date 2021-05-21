import pytest
from brian2 import NeuronGroup, Network, ms

from clocsim.recorders import RateRecorder, VoltageRecorder, GroundTruthSpikeRecorder
from clocsim import CLOCSimulator


@pytest.fixture
def neurons():
    return NeuronGroup(1, "v = -70 : 1", threshold="v > -50")


@pytest.fixture
def sim(neurons):
    net = Network(neurons)
    return CLOCSimulator(net)


def test_RateRecorder(sim, neurons, mocker):
    rate_rec = RateRecorder("rate_rec", 0)
    sim.inject_recorder(sim, neurons)
    assert rate_rec.get_state() == 0

    sim.run(1 * ms)

    mocker.patch("rate_rec.mon.rate", return_value=[1])
    assert rate_rec.get_state() == 1
