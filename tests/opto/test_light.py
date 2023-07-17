import pytest
from brian2 import mm, np, asarray, mwatt, mm2
import neo
import quantities as pq

from cleo.opto import Light, fiber473nm, LightModel
from cleo.utilities import normalize_coords


def rand_coords(rows, squeeze, add_units=True):
    coords = np.random.rand(rows, 3)
    if squeeze:
        coords = coords.squeeze()
    if add_units:
        coords = coords * mm
    return coords


@pytest.mark.parametrize("light_model", [fiber473nm])
@pytest.mark.parametrize("m, squeeze_target", [(1, True), (1, False), (4, False)])
@pytest.mark.parametrize("n, squeeze_source", [(1, True), (1, False), (6, False)])
def test_transmittance_decrease_with_distance(
    m, n, squeeze_source, squeeze_target, light_model, rand_seed
):
    # TODO: add other light models when implemented
    np.random.seed(rand_seed)

    # range lights from close to far along a narrow rod, same direction
    coords = (
        1e-3 * rand_coords(m, squeeze_source)
        + np.linspace(-1, -3, m)[:, np.newaxis] * mm
    )
    light = Light(light_model=light_model(), coords=coords, direction=(1, 1, 1))
    # neurons can be widely distributed (1mm^3 box)
    target_coords = rand_coords(n, squeeze_target)
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert np.all(np.diff(T, axis=0) <= 0)

    # keep random light sources constant, test neurons close to far
    coords = rand_coords(m, squeeze_source) - 1 * mm
    light = Light(light_model=fiber473nm(), coords=coords, direction=(1, 1, 1))
    # keep in small box (to ensure transmittances all decrease with further fibers)
    target_coords = (
        1e-3 * rand_coords(n, squeeze_target) + np.linspace(1, 4, n)[:, np.newaxis] * mm
    )
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=1) <= 0)

    # randomize both, adjust direction of random light sources to point toward neurons
    coords = (
        1e-1 * rand_coords(m, squeeze_source)
        + np.linspace(-1, -3, m)[:, np.newaxis] * mm
    )
    direction = asarray((0.5, 0.5, 0.5) * mm - coords)
    light = Light(light_model=fiber473nm(), coords=coords, direction=direction)
    target_coords = (
        1e-1 * rand_coords(n, squeeze_target) + np.linspace(0, 1, n)[:, np.newaxis] * mm
    )
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=0) <= 0)
    assert np.all(np.diff(T, axis=1) <= 0)

    # decrease with radius from light axis (z-axis in this test)
    coords = 1e-2 * rand_coords(m, squeeze_source)
    coords[..., 0] += np.linspace(0, 1, m).squeeze() * mm  # series of lights on x axis
    direction = (0, 0, 1)
    light = Light(light_model=fiber473nm(), coords=coords, direction=direction)
    target_coords = 1e-2 * rand_coords(n, squeeze_target)
    target_coords[..., 1] += (
        np.linspace(0, 1, n).squeeze() * mm
    )  # series of neurons on y axis
    target_coords[..., 2] += 1 * mm  # move neurons up 1mm
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=0) <= 0)
    assert np.all(np.diff(T, axis=1) <= 0)


def test_FiberModel():
    # range from close to far
    source_coords = (np.random.rand(4, 3) + np.arange(1, 5)[:, np.newaxis]) * mm
    target_coords = np.random.rand(10, 3) * mm
    source_direction = normalize_coords([1, 1, 1])
    T = fiber473nm().transmittance(source_coords, source_direction, target_coords)
    assert T.shape == (4, 10)
    # will all be 0 since fiber is pointing away from neurons
    assert np.all(T == 0)


def test_reset():
    light = Light(light_model=fiber473nm())
    assert light.value == 0


def test_coords():
    light = Light(light_model=fiber473nm(), coords=[[0, 0, 0], [1, 1, 1]] * mm)
    assert light.n == 2
    assert light.coords.shape == (2, 3)


@pytest.mark.parametrize("light_model", [fiber473nm()])
@pytest.mark.parametrize(
    "m, squeeze_coords, squeeze_dir",
    [
        (1, True, True),
        (1, True, False),
        (1, False, True),
        (1, False, False),
        (4, False, False),
    ],
)
@pytest.mark.parametrize("n_points_per_source", [1, 100, 10000])
def test_viz_points(
    light_model: LightModel,
    m,
    n_points_per_source,
    squeeze_coords,
    squeeze_dir,
    rand_seed,
):
    np.random.seed(rand_seed)
    # TODO: add other light models when implemented
    light_coords = rand_coords(m, squeeze_coords)
    light_direction = rand_coords(m, squeeze_dir, False)

    def check_viz_points(coords):
        assert coords.shape[0] <= m * n_points_per_source
        assert coords.shape[-1] == 3

    viz_points = light_model.viz_points(
        light_coords,
        light_direction,
        n_points_per_source,
        T_threshold=0.5,
    )
    check_viz_points(viz_points)
    n_to_plot = len(viz_points)

    for T_threshold in [1e-1, 1e-3, 0]:
        viz_points = light_model.viz_points(
            light_coords,
            light_direction,
            n_points_per_source,
            T_threshold,
        )
        check_viz_points(viz_points)
        assert len(viz_points) >= n_to_plot
        n_to_plot = len(viz_points)

    assert len(viz_points) == n_to_plot


@pytest.mark.parametrize("squeeze", [True, False])
@pytest.mark.parametrize("n_light, n_direction", [(1, 1), (4, 1), (4, 4)])
def test_light_to_neo(n_light, n_direction, squeeze):
    light = Light(
        coords=rand_coords(n_light, squeeze),
        direction=rand_coords(n_direction, squeeze, False),
        light_model=fiber473nm(),
    )
    t = 5
    light.t_ms = list(range(t))
    light.values = np.random.rand(t, n_light)
    sig = light.to_neo()

    assert np.all(sig.array_annotations["x"] / pq.mm == light.coords[..., 0] / mm)
    assert np.all(sig.array_annotations["y"] / pq.mm == light.coords[..., 1] / mm)
    assert np.all(sig.array_annotations["z"] / pq.mm == light.coords[..., 2] / mm)

    assert np.all(sig.array_annotations["direction_x"] == light.direction[..., 0])
    assert np.all(sig.array_annotations["direction_y"] == light.direction[..., 1])
    assert np.all(sig.array_annotations["direction_z"] == light.direction[..., 2])

    assert np.all(sig.array_annotations["i_channel"] == np.arange(light.n))
    assert np.all(sig.magnitude == light.values)
    assert sig.name == light.name
