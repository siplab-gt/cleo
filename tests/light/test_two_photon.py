import pytest
from brian2 import NeuronGroup, asarray, mm, np, nmeter
from brian2.units import Quantity, um

from cleo.coords import assign_xyz
from cleo.light import Light, GaussianEllipsoid
from cleo.utilities import normalize_coords


def rand_coords(rows, squeeze, add_units=True):
    coords = (np.random.rand(rows, 3)) - 0.5
    if squeeze:
        coords = coords.squeeze()
    if add_units:
        coords = coords * 100 * um
    return coords


# general transmittance and viz_points tested in test_light.py
def test_GaussianEllipsoid():
    repr_dist = 50 * um
    # range from close to far
    source_coords = (np.random.rand(4, 3) + np.arange(1, 5)[:, np.newaxis]) * repr_dist
    target_coords = np.random.rand(10, 3) * repr_dist
    source_direction = normalize_coords([1, 1, 1])
    T = GaussianEllipsoid().transmittance(
        source_coords, source_direction, target_coords
    )
    assert T.shape == (4, 10)
    assert np.all(T != 0)
    dir_flip = normalize_coords([-1, -1, -1])
    T_flip = GaussianEllipsoid().transmittance(source_coords, dir_flip, target_coords)
    assert np.allclose(T, T_flip)


if __name__ == "__main__":
    pytest.main(["-xs", __file__])
