import neo
import pytest
import quantities as pq
from brian2 import asarray, mm, mm2, mwatt, nmeter, np, um

from cleo.light import GaussianEllipsoid, Koehler, Light, LightModel, fiber473nm
from cleo.utilities import normalize_coords


def rand_coords(rows, squeeze, repr_dist=1 * mm):
    coords = np.random.rand(rows, 3)
    if squeeze:
        coords = coords.squeeze()
    coords = coords * repr_dist
    return coords


@pytest.mark.parametrize(
    "light_model, repr_dist",
    [(fiber473nm(), 1 * mm), (GaussianEllipsoid(), 50 * um)],
)
@pytest.mark.parametrize("m, squeeze_target", [(1, True), (1, False), (4, False)])
@pytest.mark.parametrize("n, squeeze_source", [(1, True), (1, False), (6, False)])
def test_transmittance_decrease_with_distance(
    m, n, squeeze_source, squeeze_target, light_model, repr_dist, rand_seed
):
    """repr_dist refers to a representative distance between light sources and neurons,
    specific to each light model."""
    np.random.seed(rand_seed)

    # range lights from close to far along a narrow rod, same direction
    coords = (
        1e-3 * rand_coords(m, squeeze_source, repr_dist)
        + np.linspace(-1, -3, m)[:, np.newaxis] * repr_dist
    )
    light = Light(light_model=light_model, coords=coords, direction=(1, 1, 1))
    # neurons can be widely distributed (1mm^3 box)
    target_coords = rand_coords(n, squeeze_target, repr_dist)
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert np.all(np.diff(T, axis=0) <= 0)

    # keep random light sources constant, test neurons close to far
    coords = rand_coords(m, squeeze_source, repr_dist) - repr_dist
    light = Light(light_model=light_model, coords=coords, direction=(1, 1, 1))
    # keep in small box (to ensure transmittances all decrease with further sources)
    target_coords = (
        1e-3 * rand_coords(n, squeeze_target, repr_dist)
        + np.linspace(1, 4, n)[:, np.newaxis] * repr_dist
    )
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=1) <= 0)

    # randomize both, adjust direction of random light sources to point toward neurons
    coords = (
        1e-1 * rand_coords(m, squeeze_source, repr_dist)
        + np.linspace(-1, -3, m)[:, np.newaxis] * repr_dist
    )
    direction = asarray((0.5, 0.5, 0.5) * mm - coords)
    light = Light(light_model=light_model, coords=coords, direction=direction)
    target_coords = (
        1e-1 * rand_coords(n, squeeze_target, repr_dist)
        + np.linspace(0, 1, n)[:, np.newaxis] * repr_dist
    )
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=0) <= 0)
    assert np.all(np.diff(T, axis=1) <= 0)

    # decrease with radius from light axis (z-axis in this test)
    coords = 1e-2 * rand_coords(m, squeeze_source, repr_dist)
    coords[..., 0] += (
        np.linspace(0, 1, m).squeeze() * repr_dist
    )  # series of lights on x axis
    direction = (0, 0, 1)
    light = Light(light_model=light_model, coords=coords, direction=direction)
    target_coords = 1e-2 * rand_coords(n, squeeze_target, repr_dist)
    target_coords[..., 1] += (
        np.linspace(0, 1, n).squeeze() * repr_dist
    )  # series of neurons on y axis
    target_coords[..., 2] += repr_dist  # move neurons up
    T = light.transmittance(target_coords)
    assert T.shape == (m, n)
    assert not np.any(T == 0)
    assert np.all(np.diff(T, axis=0) <= 0)
    assert np.all(np.diff(T, axis=1) <= 0)


def test_OpticFiber():
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


@pytest.mark.parametrize(
    "light_model", [fiber473nm(), GaussianEllipsoid(), Koehler(1 * mm)]
)
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
@pytest.mark.parametrize("n_points_per_source", [None, 1, 100, 10000])
def test_viz_params(
    light_model: LightModel,
    m,
    n_points_per_source,
    squeeze_coords,
    squeeze_dir,
    rand_seed,
):
    np.random.seed(rand_seed)
    light_coords = rand_coords(m, squeeze_coords)
    light_direction = rand_coords(m, squeeze_dir, False)

    if n_points_per_source is None:
        kwargs = {}
    else:
        kwargs = {"n_points_per_source": n_points_per_source}

    def check_viz_points(coords):
        if n_points_per_source:
            assert coords.shape[0] <= m * n_points_per_source
        assert coords.shape[-1] == 3

    viz_points, _, _ = light_model.viz_params(
        light_coords, light_direction, 0.5, **kwargs
    )
    check_viz_points(viz_points)
    n_to_plot = len(viz_points)

    for T_threshold in [1e-1, 1e-3, 0]:
        viz_points, _, _ = light_model.viz_params(
            light_coords, light_direction, T_threshold, **kwargs
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
        direction=rand_coords(n_direction, squeeze, 1),
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


if __name__ == "__main__":
    pytest.main(["-xs", __file__])
