"""Tests for coordinates module"""

import numpy as np

from brian2 import NeuronGroup, mm

from cleosim.coordinates import *


def test_rect_prism_grid():
    ng = NeuronGroup(27, "v=-70*mV : volt")
    assign_coords_grid_rect_prism(
        ng, (0, 1), (0, 1), (0, 1), shape=(3, 3, 3)
    )
    # check grid spacing in all directions
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.x))])
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.y))])
    assert all([d == 0.5 * mm for d in np.diff(np.unique(ng.z))])


def test_rect_prism_random():
    ng = NeuronGroup(27, "v=-70*mV : volt")
    assign_coords_rand_rect_prism(ng, (-2, -1), (1, 2), (4, 5))
    # check coords are all within limits
    assert all(np.logical_and(ng.x > -2 * mm, ng.x < -1 * mm))
    assert all(np.logical_and(ng.y > 1 * mm, ng.y < 2 * mm))
    assert all(np.logical_and(ng.z > 4 * mm, ng.z < 5 * mm))


def test_cylinder_random():
    ng = NeuronGroup(100, "v=-70*mV : volt")
    assign_coords_rand_cylinder(ng, (1, 1, 1), (2, 2, 2), 1)
    # none past the ends
    assert not any(np.logical_and.reduce((ng.x < 1 * mm, ng.y < 1 * mm, ng.z < 1 * mm)))
    assert not any(np.logical_and.reduce((ng.x > 2 * mm, ng.y > 2 * mm, ng.z > 2 * mm)))
    # none exactly on the axis (theoretically possible but highly improbable)
    assert not any(ng.z == ng.x / 2 + ng.y / 2)
