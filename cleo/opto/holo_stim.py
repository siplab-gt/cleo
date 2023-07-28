from cleo.opto.light import LightModel
from brian2 import (
    NeuronGroup,
)
from cleo.coords import coords_from_ng
from scipy.spatial.distance import cdist
from cleo.utilities import normalize_coords
from attrs import define
from brian2.units import (
    mm,
    mm2,
    nmeter,
    meter,
    kgram,
    Quantity,
    second,
    ms,
    second,
    psiemens,
    mV,
    volt,
    amp,
    mwatt,
)

from nptyping import NDArray
from typing import Tuple, Any
import numpy as np
import matplotlib
from brian2.units import um, meter


@define
class GaussianBallModel(LightModel):
    # if you get an error about the order of required attributes,try = field(kw_only=True)
    std_dev: Quantity
    """Standard deviation...
    You may want to consider increasing ``std_dev`` for deeper stimulation."""
    center: Tuple[float, float, float]
    intensity: float

    def transmittance(
        self,
        target_coords: Quantity,
    ) -> NDArray[(Any, Any), float]:
        # Compute the distances from target_coords to the center of the Gaussian ball
        distances = np.linalg.norm(target_coords - self.center, axis=-1)

        # Compute the transmittance values based on the Gaussian ball model
        transmittance_values = np.exp(-(distances**2) / (2 * self.std_dev**2))

        # Scale the transmittance values by the intensity
        transmittance_values *= self.intensity

        return transmittance_values

    def viz_points(
        self,
        coords: Quantity,
        n_points_per_source: int,
        T_threshold: float,
        **kwargs,
    ) -> Quantity:
        # Compute the transmittance values for all the provided coordinates
        transmittance_values = self.transmittance(coords)

        # Filter the points based on the transmittance threshold
        filtered_points = coords[transmittance_values >= T_threshold]

        # Randomly select n_points_per_source from the filtered points
        selected_points = np.random.choice(filtered_points, size=(n_points_per_source,))

        return selected_points  # m n 3


def holo_stim_targets_from_plane(
    neuron_group: NeuronGroup,
    microscope_location: Tuple[float, float, float],
    microscope_direction: Tuple[float, float, float],
    depth: float,
    inclusion_distance: float,
) -> NDArray[(Any, Any, Any), float]:
    """Implement the generation of targets from a plane for holo-stimulation
    based on the provided neuron group, microscope location, microscope direction,
    depth, and inclusion distance threshold. Return the generated targets as a Quantity.
    """

    # Extract neuron coordinates from neuron_group
    neuron_coords = coords_from_ng(neuron_group)

    # Compute the normal vector and the center of the plane
    plane_normal = normalize_coords(microscope_direction)
    plane_center = microscope_location + plane_normal * depth

    # Compute the distance of each neuron from the plane
    # Or: distances = abs(cdist(neuron_coords, [plane_center], lambda x, y: (x - y) @ plane_normal))
    distances = np.abs(np.dot(neuron_coords - plane_center, plane_normal))

    # Filter the neurons based on the inclusion distance threshold
    included_neurons = neuron_coords[distances <= inclusion_distance]

    return included_neurons
