import pytest
from brian2 import (
    NeuronGroup,
    Network,
    mV,
    pamp,
    ms,
    meter,
    second,
    np,
    seed,
    namp,
    mwatt,
)
from brian2.core.base import BrianObjectException

from cleo import CLSimulator
from cleo.opto import chr2_4s, Opsin, ProportionalCurrentOpsin
from cleo.coords import assign_coords_grid_rect_prism

model = """
    dv/dt = (-(v - -70*mV) + 100*Mohm*Iopto) / (10*ms) : volt
    Iopto : amp
"""


@pytest.fixture
def neurons() -> NeuronGroup:
    ng = NeuronGroup(15, model, reset="v=-70*mV", threshold="v>-50*mV")
    ng.v = -70 * mV
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 1), shape=(1, 1, 15))
    return ng


neurons2 = neurons


@pytest.fixture
def opsin() -> Opsin:
    return chr2_4s()


@pytest.fixture
def opsin2() -> Opsin:
    ops = chr2_4s()
    ops.name = "ChR2_2"
    return ops


def test_inject_opsin(opsin, neurons, neurons2, rand_seed):
    np.random.seed(rand_seed)
    seed(rand_seed)
    sim = CLSimulator(Network(neurons))
    sim.inject(opsin, neurons, rho_rel=2)
    # channel density setting
    assert all(opsin.synapses[neurons.name].rho_rel == 2)
    # one-to-one connections
    assert opsin.light_agg_ngs.keys() == opsin.synapses.keys()
    assert len(opsin.light_agg_ngs[neurons.name]) == neurons.N
    assert len(opsin.synapses[neurons.name]) == neurons.N

    # not in network
    with pytest.raises(Exception):
        sim.inject(opsin, neurons2)

    # p_expression
    sim.network.add(neurons2)
    sim.inject(opsin, neurons2, p_expression=0.5)
    assert (
        len(opsin.light_agg_ngs[neurons2.name])
        == len(opsin.synapses[neurons2.name])
        < neurons2.N
    )

    # added to net
    for obj in [*opsin.light_agg_ngs.values(), *opsin.synapses.values()]:
        assert obj in sim.network.objects


def test_multi_inject_opsin(opsin, neurons):
    """Test multiple injections of the same opsin into the same neuron group"""
    sim = CLSimulator(Network(neurons))
    sim.inject(opsin, neurons)
    with pytest.raises(ValueError):
        sim.inject(opsin, neurons)


def test_v_and_Iopto_in_model(opsin, opsin2):
    ng = NeuronGroup(
        1,
        """v = -70*mV : volt
        I_wumbo : amp""",
    )
    sim = CLSimulator(Network(ng))
    with pytest.raises(BrianObjectException):
        sim.inject(opsin, ng)
    # ok when Iopto name specified
    # ...but missing coords
    with pytest.raises(AttributeError):
        sim.inject(opsin, ng, Iopto_var_name="I_wumbo")
    # now it should work
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), (1, 1, 1))
    sim.inject(opsin, ng, Iopto_var_name="I_wumbo")

    ng = NeuronGroup(
        1,
        """du/dt = (-70*mV + 100*Mohm*Iopto) / (10*ms) : volt
        Iopto : amp""",
    )
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), (1, 1, 1))
    sim = CLSimulator(Network(ng))
    # can't inject device into different simulator
    with pytest.raises(Exception):
        sim.inject(opsin, ng, v_var_name="u")
    # so use different device: opto2
    # missing v in model
    with pytest.raises(BrianObjectException):
        sim.inject(opsin2, ng)
    # should work with new, un-injected opto
    sim.inject(opsin2, ng, v_var_name="u")


@pytest.mark.slow
def test_markov_opsin_model(opsin, neurons):
    sim = CLSimulator(Network(neurons))
    sim.inject(opsin, neurons)
    opsyn = opsin.synapses[neurons.name]
    light_agg = opsin.light_agg_ngs[neurons.name]
    assert all(neurons.Iopto) == 0
    assert all(neurons.v == -70 * mV)
    assert all(opsyn.C1 == 1)
    assert all(opsyn.O1 == 0)
    assert all(opsyn.C2 == 0)
    assert all(opsyn.O2 == 0)
    # light on
    light_agg.phi = 1e10 / second / meter**2
    sim.run(1 * ms)
    # current flowing, channels opened
    assert all(neurons.Iopto > 0)
    assert all(neurons.v > -70 * mV)  # depolarized
    assert all(opsyn.C1 < 1)
    assert all(opsyn.O1 > 0)
    assert all(opsyn.C2 > 0)
    assert all(opsyn.O2 > 0)
    # light off: should go back to (close to) resting state
    light_agg.phi = 0
    sim.run(49 * ms)
    assert all(neurons.Iopto > -100 * pamp)
    assert np.allclose(neurons.v, -70 * mV, atol=2 * mV)  # within 2 mV of -70
    assert all(opsyn.C1 > 0.99)
    assert all(opsyn.O1 < 0.01)
    assert all(opsyn.C2 < 0.01)
    assert all(opsyn.O2 < 0.01)


@pytest.mark.slow
def test_opto_reset(opsin, neurons, neurons2):
    sim = CLSimulator(Network(neurons, neurons2))
    sim.inject(opsin, neurons, neurons2)
    opsyn1 = opsin.synapses[neurons.name]
    opsyn2 = opsin.synapses[neurons2.name]

    init_values = {"C1": 1, "O1": 0, "O2": 0}
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert np.all(getattr(opsyn, varname) == value)

    for light_agg_ng in opsin.light_agg_ngs.values():
        light_agg_ng.phi = 1e10 / second / meter**2
    sim.run(5 * ms)
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert not np.all(getattr(opsyn, varname) == value)

    opsin.reset()
    for opsyn in [opsyn1, opsyn2]:
        for varname, value in init_values.items():
            assert np.all(getattr(opsyn, varname) == value)


def _prep_simple_opsin(ng_model, gain):
    ng = NeuronGroup(1, ng_model)
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), shape=(1, 1, 1))
    opsin = ProportionalCurrentOpsin(I_per_Irr=gain)
    sim = CLSimulator(Network(ng))
    sim.inject(opsin, ng)
    return ng, opsin, sim


@pytest.mark.slow
def test_simple_opsin_unitless():
    ng_model = "dv/dt = (-v + Iopto) / (10*ms) : 1"
    # since Iopto not in model
    with pytest.raises(BrianObjectException):
        ng, opsin, sim = _prep_simple_opsin(ng_model, 1)

    ng_model += "\n Iopto : 1"
    ng, opsin, sim = _prep_simple_opsin(ng_model, 1)

    opsin.source_ngs[ng.name].Irr = 1 * mwatt / meter**2
    sim.run(1 * ms)
    assert ng.v > 0


@pytest.mark.slow
def test_simple_opsin_amps():
    ng_model = "dv/dt = (-v + 50*Mohm * Iopto) / (10*ms) : volt"
    # since Iopto not in model
    with pytest.raises(BrianObjectException):
        ng, opsin, sim = _prep_simple_opsin(ng_model, 1 * namp)

    ng_model += "\n Iopto : ampere"
    ng, opsin, sim = _prep_simple_opsin(ng_model, 1 * namp)
    opsin.source_ngs[ng.name].Irr = 1 * mwatt / meter**2
    sim.run(1 * ms)
    assert ng.v > 0


def _prep_markov_opto(ng_model, opsin):
    ng = NeuronGroup(3, ng_model)
    sim = CLSimulator(Network(ng))
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0, 0), shape=(3, 1, 1))
    sim.inject(opsin, ng)
    return ng, sim


@pytest.mark.slow
def test_opto_syn_var_name_conflict(opsin):
    ng, sim = _prep_markov_opto(
        """
        dHp/dt = 0*Hz : 1  # diff eq
        dfv/dt = 0*Hz : 1
        Ga1 = O1*Hz : hertz  # constant
        O1 : 1
        dv/dt = 0*mV/ms : volt
        Iopto : amp
        """,
        opsin,
    )
    opto_syn_vars = opsin.synapses[ng.name].equations.names
    for var in ["Hp", "fv", "Ga1", "O1"]:
        assert not var in opto_syn_vars
        assert f"{var}_syn" in opto_syn_vars
    sim.run(0.1 * ms)


@pytest.mark.slow
def test_opto_syn_param_name_conflict(opsin):
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
        opsin,
    )
    opto_syn = opsin.synapses[ng.name]
    for param in ["g0", "phim", "p", "v1"]:
        assert not param in opto_syn.equations.names
        assert f"{param}_syn" in opto_syn.namespace
    sim.run(0.1 * ms)
