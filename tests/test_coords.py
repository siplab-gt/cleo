"""Tests for coordinates module"""

import numpy as np
import pytest
from brian2 import NeuronGroup, mm

from cleo import coords


def test_rect_prism_grid():
    ng = NeuronGroup(27, "v=-70*mV : volt")
    coords.assign_coords_grid_rect_prism(ng, (0, 1), (0, 1), (0, 1), shape=(3, 3, 3))
    # check grid spacing in all directions
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.x))])
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.y))])
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.z))])


def test_rect_prism_random():
    ng = NeuronGroup(27, "v=-70*mV : volt")
    coords.assign_coords_rand_rect_prism(ng, (-2, -1), (1, 2), (4, 5))
    # check coords are all within limits
    assert all(np.logical_and(ng.x > -2 * mm, ng.x < -1 * mm))
    assert all(np.logical_and(ng.y > 1 * mm, ng.y < 2 * mm))
    assert all(np.logical_and(ng.z > 4 * mm, ng.z < 5 * mm))


def test_cylinder_random():
    ng = NeuronGroup(100, "v=-70*mV : volt")
    coords.assign_coords_rand_cylinder(ng, (1, 1, 1), (2, 2, 2), 1)
    # none past the ends
    assert not np.any(
        np.logical_and.reduce((ng.x < 1 * mm, ng.y < 1 * mm, ng.z < 1 * mm))
    )
    assert not np.any(
        np.logical_and.reduce((ng.x > 2 * mm, ng.y > 2 * mm, ng.z > 2 * mm))
    )
    # none exactly on the axis (theoretically possible but highly improbable)
    assert not np.any(ng.z == ng.x / 2 + ng.y / 2)

    coords.assign_coords_rand_cylinder(ng, (0, 0, 0), (0, 0, 1), 1)
    # none past the ends
    assert np.all(ng.z <= 1 * mm)
    assert np.all(ng.z >= 0 * mm)
    # none exactly on the axis (theoretically possible but highly improbable)
    assert not np.any(ng.z == ng.x / 2 + ng.y / 2)


def test_cylinder_uniform():
    ng = NeuronGroup(100, "v=-70*mV : volt")
    coords.assign_coords_uniform_cylinder(ng, (1, 1, 1), (2, 2, 2), 1)
    # none past the ends
    assert not np.any(
        np.logical_and.reduce((ng.x < 1 * mm, ng.y < 1 * mm, ng.z < 1 * mm))
    )
    assert not np.any(
        np.logical_and.reduce((ng.x > 2 * mm, ng.y > 2 * mm, ng.z > 2 * mm))
    )
    # none exactly on the axis (theoretically possible but highly improbable)
    assert not np.any(ng.z == ng.x / 2 + ng.y / 2)

    coords.assign_coords_rand_cylinder(ng, (0, 0, 0), (0, 0, 1), 1)
    # none past the ends
    assert np.all(ng.z <= 1 * mm)
    assert np.all(ng.z >= 0 * mm)
    # none exactly on the axis (theoretically possible but highly improbable)
    assert not np.any(ng.z == ng.x / 2 + ng.y / 2)


def test_arbitrary_coords():
    # single neuron
    ng = NeuronGroup(1, "v=0: volt")
    coords.assign_xyz(ng, 4, 4, 4)
    # lists
    ng = NeuronGroup(3, "v=0: volt")
    coords.assign_xyz(ng, [0, 1, 2], [3, 4, 5], [6, 7, 8])
    # nested lists
    ng = NeuronGroup(3, "v=0: volt")
    coords.assign_xyz(ng, [[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]])
    # np arrays
    ng = NeuronGroup(3, "v=0: volt")
    coords.assign_xyz(ng, np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]))


def test_init_vars_for_subgroup():
    ng = NeuronGroup(10, "v=0: volt")
    sg1 = ng[:8]
    sg2 = ng[8:]
    coords._init_variables(ng)
    # attributes not accessible since subgroups created before vars init
    with pytest.raises(AttributeError):
        sg1.x
        sg2.x

    ng = NeuronGroup(10, "v=0: volt")
    coords._init_variables(ng)
    sg1 = ng[:8]
    sg2 = ng[8:]
    # attributes should be accessible in this case
    sg1.x
    sg2.x

    # assign vars to subgroup directly
    ng = NeuronGroup(10, "v=0: volt")
    sg1 = ng[:8]
    sg2 = ng[8:]
    coords._init_variables(sg1)
    # should be able to handle second init
    coords._init_variables(sg1)
    coords._init_variables(sg2)
    sg1.x
    sg2.x


if __name__ == "__main__":
    pytest.main(["-xs", "--lf", __file__])
