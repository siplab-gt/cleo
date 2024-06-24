import pytest
from brian2 import Network, NeuronGroup, mm, ms, mV, mwatt, nmeter, np, umeter

from cleo import CLSimulator
from cleo.coords import assign_coords_grid_rect_prism
from cleo.light import Light, fiber473nm
from cleo.opto import Opsin, chr2_4s, vfchrimson_4s
from cleo.registry import registry_for_sim


@pytest.fixture
def ng1() -> NeuronGroup:
    ng = NeuronGroup(
        15,
        model="""dv/dt = (-(v - -70*mV) + 100*Mohm*(Iopto+Iopto2)) / (10*ms) : volt
            Iopto : amp
            Iopto2 : amp
        """,
        reset="v=-70*mV",
        threshold="v>-20*mV",
        method="euler",
    )
    ng.v = -70 * mV
    assign_coords_grid_rect_prism(ng, (0, 0), (0, 0), (0.1, 1), shape=(1, 1, 15))
    return ng


ng2 = ng1


@pytest.fixture
def ops1() -> Opsin:
    return chr2_4s()


@pytest.fixture
def ops2() -> Opsin:
    ops = chr2_4s()
    ops.name = "ChR2_2"
    return ops


@pytest.fixture
def sim_ng1_ng2(ng1, ng2):
    return CLSimulator(Network(ng1, ng2)), ng1, ng2


def reset(sim, ng1, ng2):
    sim.reset()
    ng1.v = -70 * mV
    ng2.v = -70 * mV


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
    light.update(10)
    assert np.all(light.source.Irr0 == 10 * mwatt / mm**2)
    sim.run(0.2 * ms)
    assert all(ng1.v == -70 * mV)
    assert all(ng2.v == -70 * mV)

    # light then opsin on ng1
    sim.inject(ops1, ng1)

    sim.run(0.3 * ms)
    assert np.all(ops1.light_agg_ngs[ng1.name].Irr > 0)
    assert np.all(ops1.light_agg_ngs[ng1.name].phi > 0)
    assert np.all(ng1.v > -70 * mV)
    assert np.all(ng2.v == -70 * mV)

    # opsin without light (ng2)
    sim.inject(ops1, ng2)
    sim.run(0.3 * ms)
    assert np.all(ng2.v == -70 * mV)

    # opsin then light (ng2)
    sim.inject(light, ng2)
    sim.run(0.3 * ms)
    assert np.all(ng2.v > -70 * mV)

    # second light on ng1
    reset(sim, ng1, ng2)
    light2 = Light(coords=coords, light_model=fiber473nm(), name="light2")
    sim.inject(light2, ng1)
    light.update(10)
    light2.update(10)
    sim.run(0.3 * ms)
    assert np.all(ng2.v > -70 * mV)
    assert np.all(ng1.v > ng2.v)

    # second opsin on ng2
    reset(sim, ng1, ng2)
    assert np.all(light2.value == 0)
    light.update(10)
    sim.inject(ops2, ng2, Iopto_var_name="Iopto2")
    sim.run(0.3 * ms)
    assert np.all(ops2.light_agg_ngs[ng2.name].Irr > 0)
    assert np.all(ops2.light_agg_ngs[ng2.name].phi > 0)
    assert np.all(ng2.v > ng1.v)

    # remove light
    sim.remove(light)
    assert False  # TODO


@pytest.mark.slow
def test_multi_channel(sim_ng1_ng2, ops1):
    sim, ng1, _ = sim_ng1_ng2
    light = Light(
        coords=[[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, -50]] * umeter,
        light_model=fiber473nm(),
    )
    sim.inject(ops1, ng1)
    sim.inject(light, ng1)
    v_result = {}
    for i_channel, case in enumerate(["baseline", "x", "y", "z"]):
        sim.reset()
        u = np.ones(light.n)
        u[i_channel] = 10
        light.update(u)
        sim.run(0.3 * ms)
        v_result[case] = np.array(ng1.v)

    # stronger effect for baseline than for x, y, z offsets
    assert np.all(v_result["x"] == v_result["y"])
    assert np.all(v_result["baseline"] > v_result["x"])
    assert np.all(v_result["baseline"] > v_result["z"])

    # 2 channels on at once
    sim.reset()
    light.update([10, 10, 0, 0])
    sim.run(0.3 * ms)
    v_base_x = np.array(ng1.v)

    sim.reset()
    light.update([0, 0, 10, 10])
    sim.run(0.3 * ms)
    v_yz = np.array(ng1.v)

    assert np.all(v_base_x > v_yz)


@pytest.mark.slow
def test_multi_light_opsin(sim_ng1_ng2):
    sim, ng1, ng2 = sim_ng1_ng2
    chr2 = chr2_4s()
    blue = Light(light_model=fiber473nm(), name="blue")
    vfchrimson = vfchrimson_4s()
    amber = Light(light_model=fiber473nm(), wavelength=590 * nmeter, name="amber")

    sim.inject(chr2, ng1, Iopto_var_name="Iopto")
    sim.inject(vfchrimson, ng2, Iopto_var_name="Iopto2")
    sim.inject(blue, ng1, ng2)
    # amber on chr2 no longer issues warning since we have 2p spectrum data
    sim.inject(amber, ng1)
    sim.inject(amber, ng2)

    # warning when going outside action spectrum data
    uv = Light(light_model=fiber473nm(), wavelength=300 * nmeter, name="uv")
    with pytest.warns(
        UserWarning,
        match="outside the range of the action spectrum data.*extrapolate=False",
    ):
        sim.inject(uv, ng1)
    # but not when extrapolation enabled
    vfchrimson.extrapolate = True
    with pytest.warns(
        UserWarning,
        match="outside the range of the action spectrum data.*Extrapolating:",
    ):
        sim.inject(uv, ng2)

    blue.update(10)
    sim.run(0.3 * ms)
    # main effect on ng1
    assert np.all(ng1.v > ng2.v)
    # some cross-talk on ng2
    assert np.all(ng2.v > -70 * mV)
    v_crosstalk = ng2.v

    sim.reset()
    amber.update(10)
    sim.run(0.3 * ms)
    # main effect on ng2
    assert np.all(ng2.v > ng1.v)
    # very little cross-talk on ng1
    assert chr2.epsilon(amber.wavelength) < 0.01
    assert np.all((-70 * mV < ng1.v) & (ng1.v < v_crosstalk))

    sim.reset()
    uv.update(10)
    sim.run(0.3 * ms)
    # small effect on ng2 (extrapolating action spectrum)
    assert np.all(ng2.v > ng1.v)
    # no cross-talk on ng1
    assert chr2.epsilon(uv.wavelength) == 0
    assert np.all(ng1.v == -70 * mV)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
