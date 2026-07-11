import xarray as xr
import numpy as np
from attrs import define, field
from brian2.units import Quantity, nmeter, um, cm, mm
from jaxtyping import Float
from cleo.light.light import LightModel
from cleo.utilities import uniform_cylinder_rθz, xyz_from_rθz
from cleo.coords import coords_from_xyz


@define
class OptogenSIMLight(LightModel):
    """Light model from OptogenSIM Monte Carlo simulations."""

    wavelength: Quantity = 473 * nmeter
    """Light wavelength. Must be within the simulated grid range."""
    beam_radius: Quantity = 100 * um
    """Beam (1/e^2) radius. Must be within the simulated grid range."""
    data_path: str = "light_model_4d.nc"
    """Path to the 4D dataset."""
    _rz_slice: object = field(init=False, default=None, repr=False)
    _r_range: object = field(init=False, default=None, repr=False)
    _z_range: object = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self):
        rz = xr.open_dataarray(self.data_path).interp(
            wavelength=self.wavelength / nmeter, beam_size=self.beam_radius / um
        )
        # data measures z from the atlas top; re-zero to the source
        z_source = float(rz.isel(r=0).z[rz.isel(r=0).argmax("z")])
        self._rz_slice = rz.assign_coords(z=rz.z - z_source)
        self._r_range = (float(self._rz_slice.r.min()), float(self._rz_slice.r.max()))
        self._z_range = (float(self._rz_slice.z.min()), float(self._rz_slice.z.max()))

    def transmittance(self, source_coords, source_dir_uvec, target_coords):
        assert np.allclose(np.linalg.norm(source_dir_uvec, axis=-1), 1)
        r, z = self._get_rz_for_xyz(source_coords, source_dir_uvec, target_coords)
        r_cm = np.asarray(r / cm)
        z_cm = np.asarray(z / cm)
        T = self._rz_slice.interp(
            r=xr.DataArray(np.clip(r_cm, *self._r_range)),
            z=xr.DataArray(np.clip(z_cm, *self._z_range)),
        ).values
        T = np.nan_to_num(T, nan=0.0)
        T[z_cm < 0] = 0   # no light behind the source
        return T

    @property
    def area0(self):
        return np.pi * self.beam_radius**2

    def viz_params(self, coords, direction, T_threshold,
                   n_points_per_source=4000, **kwargs):
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(n_points_per_source, r_thresh, zc_thresh)
        end = coords + zc_thresh * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)
        density_factor = 3
        cyl_vol = np.pi * r_thresh**2 * zc_thresh
        markersize = (cyl_vol / n_points_per_source * density_factor) ** (1 / 3)
        intensity_scale = 1.5 * (4e3 / n_points_per_source) ** (1 / 3)
        return coords_from_xyz(x, y, z), markersize, intensity_scale

    def _find_rz_thresholds(self, thresh):
        """Find r and z extents for visualization at the given transmittance threshold."""
        # On-axis (r[0]) profile along z (re-zeroed so source at z=0).
        on_axis = self._rz_slice.isel(r=0)
        z_vals = on_axis.z.values  # cm, source at 0
        T_z = on_axis.values
        # Only look in the forward (z >= 0) direction.
        mask = z_vals >= 0
        z_pos = z_vals[mask]
        T_pos = T_z[mask]
        below = np.where(T_pos < thresh)[0]
        zc_thresh = (z_pos[below[0]] if len(below) > 0 else z_pos[-1]) * cm

        # Find r threshold at z = zc_thresh / 2.
        z_mid = float(zc_thresh / cm) / 2
        z_mid = np.clip(z_mid, float(self._rz_slice.z.min()),
                        float(self._rz_slice.z.max()))
        r_vals = self._rz_slice.r.values
        T_r = self._rz_slice.interp(z=float(z_mid)).values
        below_r = np.where(T_r < thresh)[0]
        r_thresh = (r_vals[below_r[0]] if len(below_r) > 0 else r_vals[-1]) * cm

        return r_thresh * 1.2, zc_thresh
