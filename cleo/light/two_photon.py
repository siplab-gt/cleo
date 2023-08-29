from typing import Any, Tuple

import matplotlib
import numpy as np
from attrs import define, field
from brian2 import NeuronGroup, Quantity
from brian2.units import meter, um, mm
from nptyping import NDArray
from scipy.spatial.distance import cdist

from cleo.coords import coords_from_ng, coords_from_xyz
from cleo.light import LightModel
from cleo.utilities import normalize_coords, uniform_cylinder_rθz, xyz_from_rθz


@define(eq=False)
class GaussianEllipsoid(LightModel):
    sigma_axial: Quantity = 18 * um
    """Standard deviation distance along the focal axis.

    Standard deviations estimated by taking point where response is ~60% of peak:
    
    ======================== ========== ============= =======
    Publication              axial      lateral       measure
    ======================== ========== ============= =======
    Rickgauer et al., 2014   8 μm       4 μm          Ca2+ dF/F response
    Packer et al., 2015      18 μm      8 μm          AP probability
    Chen et al., 2019        18/11 μm   8/? μm        AP probability/photocurrent
    ======================== ========== ============= =======
    """
    sigma_lateral: Quantity = 8 * um
    """Standard deviation distance along the focal plane."""

    def transmittance(
        self,
        source_coords: Quantity,
        source_dir_uvec: NDArray[(Any, 3), Any],
        target_coords: Quantity,
    ) -> NDArray[(Any, Any), float]:
        assert np.allclose(np.linalg.norm(source_dir_uvec, axis=-1), 1)
        r, z = self._get_rz_for_xyz(source_coords, source_dir_uvec, target_coords)
        return self._gaussian_transmittance(r, z)

    def viz_points(
        self,
        coords: Quantity,
        direction: NDArray[(Any, 3), Any],
        T_threshold: float,
        n_points_per_source: int = 3e2,
        **kwargs,
    ) -> Quantity:
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(n_points_per_source, r_thresh, zc_thresh)

        end = coords + zc_thresh * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)
        return coords_from_xyz(x, y, z)  # m x n x 3

    def _gaussian_transmittance(self, r, z):
        """r is lateral distance, z is axial distance from focal point.

        Not normalizing the Gaussian: i.e., a point perfectly in focus will
        get relative power of 1.0, not 1.0 / (2 * pi * sigma**2)."""
        return np.exp(-(r**2) / (2 * self.sigma_lateral**2)) * np.exp(
            -(z**2) / (2 * self.sigma_axial**2)
        )

    def _find_rz_thresholds(self, thresh):
        """find r and z thresholds for visualization purposes"""
        res_um = 0.1
        zc = np.arange(1000, 0, -res_um) * um  # ascending T
        T = self._gaussian_transmittance(0 * um, zc)
        try:
            zc_thresh = zc[np.searchsorted(T, thresh)]
        except IndexError:  # no points above threshold
            zc_thresh = np.max(zc)

        r = np.arange(100, 0, -res_um) * um
        T = self._gaussian_transmittance(r, 0 * um)
        try:
            r_thresh = r[np.searchsorted(T, thresh)]
        except IndexError:  # no points above threshold
            r_thresh = np.max(r)

        return r_thresh, zc_thresh


def target_coords_from_scope(scope):
    pass
