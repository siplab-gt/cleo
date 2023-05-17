import pytest
from brian2 import NeuronGroup, Network, mV, pamp, namp
from brian2.core.base import BrianObjectException

from cleo import CLSimulator
from cleo.opto import ChR2_four_state
from cleo.coords import assign_coords_grid_rect_prism

model = """
    dv/dt = (-(v - -70*mV) + 100*Mohm*Iopto) / (10*ms) : volt
    Iopto : amp
"""


@pytest.fixture
def neurons():
    ng = NeuronGroup(10, model, reset="v=-70*mV", threshold="v>-50*mV")
    ng.v = -70 * mV
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 1), shape=(1, 1, 10))
    return ng


neurons2 = neurons


@pytest.fixture
def opsin():
    return ChR2_four_state()


def test_multi_inject(opsin, neurons):
    """Test multiple injections of the same opsin into the same neuron group"""
    sim = CLSimulator(Network(neurons))
    sim.inject(opsin, neurons)
    with pytest.raises(ValueError):
        sim.inject(opsin, neurons)
