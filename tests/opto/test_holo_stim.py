import unittest
import pytest
from brian2 import mm, np, asarray, NeuronGroup
from brian2.units import Quantity
from cleo.opto import Light
from cleo.opto.holo_stim import GaussianBallModel
from cleo.opto.holo_stim import holo_stim_targets_from_plane
from cleo.coords import assign_coords


def rand_coords(rows, squeeze, add_units=True):
    coords = np.random.rand(rows, 3)
    if squeeze:
        coords = coords.squeeze()
    if add_units:
        coords = coords * mm
    return coords


@pytest.mark.parametrize("m, squeeze_source", [(1, True), (1, False), (4, False)])
def test_holo_stim_targets_from_plane(m, squeeze_source):
    np.random.seed(0)

    # Generate random neuron and microscope coordinates
    neuron_coords = rand_coords(m, squeeze_source)
    neuron_group = NeuronGroup(m, "dv/dt = 0 * volt/second : volt")
    assign_coords(neuron_group, neuron_coords, mm)

    microscope_location = (0 * mm, 0 * mm, 0 * mm)
    microscope_direction = (1, 1, 1)
    depth = 1 * mm
    inclusion_distance = 0.5 * mm

    # Call the holo_stim_targets_from_plane method
    targets = holo_stim_targets_from_plane(
        neuron_group,  # Pass the NeuronGroup object instead of the coordinates array
        microscope_location,
        microscope_direction,
        depth,
        inclusion_distance,
    )

    # Assertions
    assert targets.shape[0] <= m
    assert targets.shape[1] == 3


def test_transmittance():
    # Create an instance of the GaussianBallModel
    light_model = GaussianBallModel(
        std_dev=1.0 * mm, center=(0.0, 0.0, 0.0), intensity=1.0
    )
    light = Light(light_model=light_model)

    # Generate random target coordinates
    target_coords = rand_coords(10, squeeze=False)

    # Call the transmittance method
    transmittance_values = light.transmittance(target_coords)

    # Assertions
    assert transmittance_values.shape == (light.m, 10)


def test_viz_points():
    # Create an instance of the GaussianBallModel
    light_model = GaussianBallModel(
        std_dev=1.0 * mm, center=(0.0, 0.0, 0.0), intensity=1.0
    )
    light = Light(light_model=light_model)

    # Generate random light coordinates
    light_coords = rand_coords(5, squeeze=False)

    # Call the viz_points method
    viz_points = light.viz_points(light_coords, n_points_per_source=3, T_threshold=0.5)

    # Assertions
    assert viz_points.shape[0] <= light.m * 3
    assert viz_points.shape[1] == 3
