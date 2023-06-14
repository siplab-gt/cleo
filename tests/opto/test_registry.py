import pytest
from brian2 import np, NeuronGroup, mV, Network, umeter, ms, mwatt, mm

from cleo import CLSimulator
from cleo.coords import assign_coords_grid_rect_prism
from cleo.opto import Light, fiber473nm, ChR2_four_state, Opsin

model = """
    dv/dt = (-(v - -70*mV) + 100*Mohm*Iopto) / (10*ms) : volt
    Iopto : amp
"""


@pytest.fixture
def ng1() -> NeuronGroup:
    ng = NeuronGroup(15, model, reset="v=-70*mV", threshold="v>-50*mV", method="euler")
    ng.v = -70 * mV
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0.1, 1), shape=(1, 1, 15))
    return ng


ng2 = ng1


@pytest.fixture
def ops1() -> Opsin:
    return ChR2_four_state()


@pytest.fixture
def ops2() -> Opsin:
    return ChR2_four_state(name="ChR2_2")


@pytest.fixture
def sim_ng1_ng2(ng1, ng2):
    return CLSimulator(Network(ng1, ng2)), ng1, ng2


@pytest.mark.slow
@pytest.mark.parametrize("coords", [(0, 0, 0), [[0, 0, 0]], [[0, 0, 0], [1, 1, -5]]])
def test_light_opsin_interaction(sim_ng1_ng2, ops1, ops2, coords):
    sim, ng1, ng2 = sim_ng1_ng2
    coords = coords * umeter
    light = Light(coords=coords, light_model=fiber473nm())
    sim.inject(light, ng1)
    assert all(ng1.v == -70 * mV)
    assert all(ng2.v == -70 * mV)

    # light without opsins
    light.update(30)
    assert np.all(light.source_ng.Irr0 == 30 * mwatt / mm**2)
    sim.run(0.5 * ms)
    assert all(ng1.v == -70 * mV)
    assert all(ng2.v == -70 * mV)

    # light then opsin on ng1
    sim.inject(ops1, ng1)
    sim.run(0.5 * ms)
    assert np.all(ng1.v > -70 * mV)
    assert np.all(ng2.v == -70 * mV)

    # opsin without light (ng2)
    sim.inject(ops1, ng2)
    sim.run(0.5 * ms)
    assert np.all(ng2.v == -70 * mV)

    # opsin then light (ng2)
    sim.inject(light, ng2)
    sim.run(0.5 * ms)
    assert np.all(ng2.v > -70 * mV)

    # second light on ng1
    sim.reset()
    light2 = Light(coords=coords, light_model=fiber473nm(), name="light2")
    sim.inject(light2, ng1)
    sim.run(0.5 * ms)
    assert np.all(ng1.v > ng2.v)

    # second opsin on ng2


def test_multi_channel(sim_ng1_ng2, ops1):
    sim, ng1, ng2 = sim_ng1_ng2


def test_epsilon(sim_ng1_ng2, ops1):
    sim, ng1, ng2 = sim_ng1_ng2
