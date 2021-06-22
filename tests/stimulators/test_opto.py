"""Tests for opto module"""

import pytest
from brian2 import NeuronGroup, Network, mV, prefs, pamp
from brian2.core.base import BrianObjectException

prefs.codegen.target = "numpy"  # to avoid cython overhead for short tests

from clocsim import CLOCSimulator
from clocsim.stimulators.opto import *
from clocsim.coordinates import assign_coords_rect_prism

model = """
        dv/dt = (-(v - -70*mV) - 100*Mohm*Iopto) / (10*ms) : volt
        Iopto : amp
        """


@pytest.fixture
def neurons():
    ng = NeuronGroup(10, model, reset="v=-70*mV", threshold="v>-50*mV")
    ng.v = -70 * mV
    assign_coords_rect_prism(
        ng, "grid", (0, 0), (0, 0), (0, 1), xyz_grid_shape=(1, 1, 10)
    )
    return ng


neurons2 = neurons


@pytest.fixture
def opto():
    return OptogeneticIntervention("opto", four_state, ChR2_four_state, default_blue)


opto2 = opto


def test_inject_opto(opto, neurons, neurons2):
    sim = CLOCSimulator(Network(neurons))
    sim.inject_stimulator(opto, neurons, rho_rel=2)
    # channel density setting
    assert all(opto.opto_syns[neurons.name].rho_rel == 2)
    # one-to-one connections
    assert len(opto.opto_syns[neurons.name]) == neurons.N

    # not in network
    with pytest.raises(Exception):
        sim.inject_stimulator(opto, neurons2)

    # p_expression
    sim.network.add(neurons2)
    sim.inject_stimulator(opto, neurons2, p_expression=0.1)
    assert len(opto.opto_syns[neurons2.name]) < neurons2.N


## not implemented yet. Will have to use multiple postsynaptic variables.
## E.g., Iopto = I_ChR2 + I_Jaws
## https://brian2.readthedocs.io/en/stable/user/synapses.html#summed-variables
# def test_two_optos(opto, opto2, neurons):
#     sim = CLOCSimulator(Network(neurons))
#     sim.inject_stimulator(opto, neurons)
#     assert neurons.Iopto == 0
#     opto.update(1)
#     assert neurons.Iopto ==
#     sim.inject_stimulator(opto2, neurons)


def test_v_and_Iopto_in_model(opto):
    ng = NeuronGroup(1, "v = -70*mV : volt")
    sim = CLOCSimulator(Network(ng))
    with pytest.raises(BrianObjectException):
        sim.inject_stimulator(opto, ng)
    ng = NeuronGroup(
        1,
        """du/dt = (-70*mV + 100*Mohm*Iopto) / (10*ms) : volt
        Iopto : amp""",
    )
    sim = CLOCSimulator(Network(ng))
    with pytest.raises(BrianObjectException):
        sim.inject_stimulator(opto, ng)


def test_opsin_model(opto, neurons):
    sim = CLOCSimulator(Network(neurons))
    sim.inject_stimulator(opto, neurons)
    opsyn = opto.opto_syns[neurons.name]
    assert all(neurons.Iopto) == 0
    assert all(neurons.v == -70 * mV)
    assert all(opsyn.C1 == 1)
    assert all(opsyn.O1 == 0)
    assert all(opsyn.C2 == 0)
    assert all(opsyn.O2 == 0)
    # light on
    opto.update(1)
    sim.run(1 * ms)
    # current flowing, channels opened
    assert all(neurons.Iopto < 0)
    assert all(neurons.v > -70 * mV)  # depolarized
    assert all(opsyn.C1 < 1)
    assert all(opsyn.O1 > 0)
    assert all(opsyn.C2 > 0)
    assert all(opsyn.O2 > 0)
    # light off: should go back to (close to) resting state
    opto.update(0)
    sim.run(49 * ms)
    assert all(neurons.Iopto > -100 * pamp)
    assert all(np.abs(neurons.v + 70 * mV) < 2 * mV)  # within 2 mV of -70
    assert all(opsyn.C1 > 0.99)
    assert all(opsyn.O1 < 0.01)
    assert all(opsyn.C2 < 0.01)
    assert all(opsyn.O2 < 0.01)


def test_light_model(opto, neurons):
    opto.connect_to_neuron_group(neurons)
    opsyn = opto.opto_syns[neurons.name]
    # neurons are in line from 0 to 1 mm from fiber
    # so we expect descending order of transmittance (T)
    assert all(np.logical_and(opsyn.T > 0, opsyn.T < 1))
    assert all(opsyn.T[::-1] == np.sort(opsyn.T))
    # another set of neurons just off-center should have lower T
    neurons2 = NeuronGroup(len(neurons), model)
    assign_coords_rect_prism(
        neurons2,
        "grid",
        (0.1, 0.1),  # just 100 um off from center in x and y
        (0.1, 0.1),
        (0, 1),
        xyz_grid_shape=(1, 1, len(neurons)),
    )
    opto.connect_to_neuron_group(neurons2)
    assert all(np.greater(opsyn.T, opto.opto_syns[neurons2.name].T))
