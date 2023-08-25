from typing import Any, Tuple

import matplotlib
import numpy as np
from attrs import define, field
from brian2 import NeuronGroup, Quantity
from brian2.units import meter, um
from nptyping import NDArray
from scipy.spatial.distance import cdist

from cleo.coords import coords_from_ng
from cleo.opto.light import LightModel
from cleo.utilities import normalize_coords


@define(eq=False)
class GaussianEllipsoid(LightModel):
    # if you get an error about the order of required attributes,try = field(kw_only=True)
    sigma_axial: Quantity
    """Standard deviation along the focal axis.
    You may want to consider increasing ``std_dev`` for deeper stimulation."""
    sigma_lateral: Quantity
    """Standard deviation along the focal plane.
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


def targets_from_scope(scope):
    pass
