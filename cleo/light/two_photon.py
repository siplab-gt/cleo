from typing import Any

from attrs import define
from brian2 import Quantity, np
from brian2.units import nmeter, um
from nptyping import NDArray

from cleo.coords import concat_coords, coords_from_ng, coords_from_xyz
from cleo.light import Light, LightModel
from cleo.utilities import uniform_cylinder_rθz, xyz_from_rθz


@define(eq=False)
class GaussianEllipsoid(LightModel):
    sigma_axial: Quantity = 18 * um
    """Standard deviation distance along the focal axis.

    Standard deviations estimated by taking point where response is ~60% of peak:
    
    ======================== ========== ============= =======
    Publication              axial      lateral       measure
    ======================== ========== ============= =======
    Prakash et al., 2012     13 μm      7 μm          photocurrent
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

    def viz_params(
        self,
        coords: Quantity,
        direction: NDArray[(Any, 3), Any],
        T_threshold: float,
        n_points_per_source: int = 4000,
        **kwargs,
    ) -> Quantity:
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(
            n_points_per_source, r_thresh, zc_thresh * 2
        )
        zc -= zc_thresh

        end = coords + zc_thresh * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)
        # m x n x 3
        density_factor = 3
        cyl_vol = np.pi * r_thresh**2 * zc_thresh
        markersize_um = (cyl_vol / n_points_per_source * density_factor) ** (1 / 3) / um
        intensity_scale = (1000 / n_points_per_source) ** (1 / 3)
        return coords_from_xyz(x, y, z), markersize_um, intensity_scale

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


def tp_light_from_scope(scope, wavelength=1060 * nmeter, **kwargs) -> Light:
    """Creates a light object from a scope object with 2P focused laser points
    at each target.

    Parameters
    ----------
    scope : Scope
        The scope object containing the laser spots.
    wavelength : Quantity, optional
        The wavelength of the laser, by default 1060 * nmeter.
    """
    coords = []
    for ng, i_targets in zip(scope.neuron_groups, scope.i_targets_per_injct):
        coords.append(coords_from_ng(ng)[i_targets])
    coords = concat_coords(*coords)
    light = Light(
        coords=coords,
        direction=scope.direction,
        light_model=GaussianEllipsoid(),
        wavelength=wavelength,
        **kwargs,
    )
    return light
