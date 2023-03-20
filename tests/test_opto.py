"""Tests for opto module"""

import pytest
from brian2 import NeuronGroup, Network, mV, pamp, namp
from brian2.core.base import BrianObjectException


from cleo import CLSimulator
from cleo.opto import *
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
def opto() -> OptogeneticIntervention:
    return OptogeneticIntervention(
        "opto", ChR2_four_state(), default_blue
    )


opto2 = opto


def test_inject_opto(opto, neurons, neurons2):
    sim = CLSimulator(Network(neurons))
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
#     sim = CLSimulator(Network(neurons))
#     sim.inject_stimulator(opto, neurons)
#     assert neurons.Iopto == 0
#     opto.update(1)
#     assert neurons.Iopto ==
#     sim.inject_stimulator(opto2, neurons)


def test_v_and_Iopto_in_model(opto, opto2):
    ng = NeuronGroup(
        1,
        """v = -70*mV : volt
        I_wumbo : amp""",
    )
    sim = CLSimulator(Network(ng))
    with pytest.raises(BrianObjectException):
        sim.inject_stimulator(opto, ng)
    # ok when Iopto name specified
    # ...but missing coords
    with pytest.raises(AttributeError):
        sim.inject_stimulator(opto, ng, Iopto_var_name="I_wumbo")
    # now it should work
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), (1, 1, 1))
    sim.inject_stimulator(opto, ng, Iopto_var_name="I_wumbo")

    ng = NeuronGroup(
        1,
        """du/dt = (-70*mV + 100*Mohm*Iopto) / (10*ms) : volt
        Iopto : amp""",
    )
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), (1, 1, 1))
    sim = CLSimulator(Network(ng))
    # can't inject device into different simulator
    with pytest.raises(Exception):
        sim.inject_stimulator(opto, ng, v_var_name="u")
    # so use different device, opto2
    # missing v in model
    with pytest.raises(BrianObjectException):
        sim.inject_stimulator(opto2, ng)
    # should work with new, un-injected opto
    sim.inject_stimulator(opto2, ng, v_var_name="u")


@pytest.mark.slow
def test_markov_opsin_model(opto, neurons):
    sim = CLSimulator(Network(neurons))
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
    assert all(neurons.Iopto > 0)
    assert all(neurons.v > -70 * mV)  # depolarized
    assert all(opsyn.C1 < 1)
    assert all(opsyn.O1 > 0)
    assert all(opsyn.C2 > 0)
    assert all(opsyn.O2 > 0)
    # light off: should go back to (close to) resting state
    opto.update(0)
    sim.run(49 * ms)
    assert all(neurons.Iopto > -100 * pamp)
    assert np.allclose(neurons.v, -70 * mV, atol=2 * mV)  # within 2 mV of -70
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
    assign_coords_grid_rect_prism(
        neurons2,
        (0.1, 0.1),  # just 100 um off from center in x and y
        (0.1, 0.1),
        (0, 1),
        shape=(1, 1, len(neurons)),
    )
    opto.connect_to_neuron_group(neurons2)
    assert all(np.greater(opsyn.T, opto.opto_syns[neurons2.name].T))


@pytest.mark.slow
def test_opto_reset(opto, neurons, neurons2):
    sim = CLSimulator(Network(neurons, neurons2))
    sim.inject_stimulator(opto, neurons, neurons2)
    opsyn1 = opto.opto_syns[neurons.name]
    opsyn2 = opto.opto_syns[neurons2.name]

    init_values = {"Irr0": 0, "C1": 1, "O1": 0, "O2": 0}
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert np.all(getattr(opsyn, varname) == value)

    opto.update(1)
    sim.run(5 * ms)
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert not np.all(getattr(opsyn, varname) == value)

    opto.reset()
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert np.all(getattr(opsyn, varname) == value)


def _prep_simple_opto(ng_model, gain):
    ng = NeuronGroup(1, ng_model)
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), shape=(1, 1, 1))
    # assuming 0 starting voltage
    opto = OptogeneticIntervention("opto", ProportionalCurrentModel(gain), default_blue)
    sim = CLSimulator(Network(ng))
    sim.inject_stimulator(opto, ng)
    return ng, opto, sim


@pytest.mark.slow
def test_simple_opto_unitless():
    ng_model = "dv/dt = (-v + Iopto) / (10*ms) : 1"
    # since Iopto not in model
    with pytest.raises(BrianObjectException):
        ng, opto, sim = _prep_simple_opto(ng_model, 1)

    ng_model += "\n Iopto : 1"
    ng, opto, sim = _prep_simple_opto(ng_model, 1)
    opto.update(1)
    sim.run(1 * ms)
    assert ng.v > 0


@pytest.mark.slow
def test_simple_opto_amps():
    ng_model = "dv/dt = (-v + 50*Mohm * Iopto) / (10*ms) : volt"
    # since Iopto not in model
    with pytest.raises(BrianObjectException):
        ng, opto, sim = _prep_simple_opto(ng_model, 1 * namp)

    ng_model += "\n Iopto : ampere"
    ng, opto, sim = _prep_simple_opto(ng_model, 1 * namp)
    opto.update(1)
    sim.run(1 * ms)
    assert ng.v > 0 * volt


def _prep_markov_opto(ng_model, opto):
    ng = NeuronGroup(3, ng_model)
    sim = CLSimulator(Network(ng))
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), shape=(3, 1, 1))
    sim.inject_stimulator(opto, ng)
    return ng, sim


@pytest.mark.slow
def test_opto_syn_var_name_conflict(opto):
    ng, sim = _prep_markov_opto(
        """
        dHp/dt = 0*Hz : 1  # diff eq
        dfv/dt = 0*Hz : 1
        Ga1 = O1*Hz : hertz  # constant
        O1 : 1
        dv/dt = 0*mV/ms : volt
        Iopto : amp
        """,
        opto,
    )
    opto_syn_vars = opto.opto_syns[ng.name].equations.names
    for var in ["Hp", "fv", "Ga1", "O1"]:
        assert not var in opto_syn_vars
        assert f"{var}_syn" in opto_syn_vars
    sim.run(0.1 * ms)


@pytest.mark.slow
def test_opto_syn_param_name_conflict(opto):
    # put 4-state model param names in neuron group vars
    ng, sim = _prep_markov_opto(
        """
        dg0/dt = 0*Hz : 1  # diff eq
        dphim/dt = 0*Hz : 1
        p = v1*Hz : hertz  # constant
        v1 : 1
        dv/dt = 0*mV/ms : volt
        Iopto : amp
        """,
        opto,
    )
    opto_syn = opto.opto_syns[ng.name]
    for param in ["g0", "phim", "p", "v1"]:
        assert not param in opto_syn.equations.names
        assert f"{param}_syn" in opto_syn.namespace
    sim.run(0.1 * ms)
