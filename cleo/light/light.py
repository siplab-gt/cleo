"""Contains Light device and propagation models"""
from __future__ import annotations

import datetime
import warnings
from abc import ABC, abstractmethod
from typing import Any, Union

import matplotlib as mpl
import quantities as pq
from attrs import define, field
from brian2 import (
    NeuronGroup,
    Subgroup,
    np,
)
from brian2.units import (
    Quantity,
    mm,
    mm2,
    mwatt,
    nmeter,
    um,
)
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from nptyping import NDArray

from cleo.base import CLSimulator
from cleo.coords import coords_from_xyz
from cleo.registry import registry_for_sim
from cleo.stimulators import Stimulator
from cleo.utilities import (
    analog_signal,
    normalize_coords,
    uniform_cylinder_rθz,
    wavelength_to_rgb,
    xyz_from_rθz,
)


@define
class LightModel(ABC):
    """Defines how light propagates given a source location and direction."""

    @abstractmethod
    def transmittance(
        self,
        source_coords: Quantity,
        source_direction: NDArray[(Any, 3), Any],
        target_coords: Quantity,
    ) -> NDArray[(Any, Any), float]:
        """Output must be between 0 and 1 with shape (n_sources, n_targets)."""
        pass

    @abstractmethod
    def viz_params(
        self,
        coords: Quantity,
        direction: NDArray[(Any, 3), Any],
        T_threshold: float,
        n_points_per_source: int = None,
        **kwargs,
    ) -> tuple[NDArray[(Any, Any, 3), Any], float, float]:
        """Returns info needed for visualization.
        Output is ((m, n_points_per_source, 3) viz_points array, markersize_um, intensity_scale).

        For best-looking results, implementations should scale `markersize_um` and `intensity_scale`.
        """
        pass

    def _get_rz_for_xyz(self, source_coords, source_direction, target_coords):
        """Assumes x, y, z already have units"""
        m = source_coords.reshape((-1, 3)).shape[0]
        if target_coords.ndim > 1:
            n = target_coords.shape[-2]
        elif target_coords.ndim == 1:
            n = 1
        # either shared or per-source targets
        assert target_coords.shape in [(m, n, 3), (1, n, 3), (n, 3), (3,)]

        # relative to light source(s)
        rel_coords = target_coords - source_coords.reshape((m, 1, 3))
        assert rel_coords.shape == (m, n, 3)
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        # zc = usf.dot(rel_coords, source_direction)  # mxn distance along cylinder axis
        #           m x n x 3    m x 1 x 3
        zc = np.sum(
            rel_coords * source_direction.reshape((-1, 1, 3)), axis=-1
        )  # mxn distance along cylinder axis
        assert zc.shape == (m, n)
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt(
            np.sum(
                (
                    rel_coords
                    #    m x n                 m x 3
                    # --> m x n x 1             m x 1 x 3
                    - zc.reshape((m, n, 1)) * source_direction.reshape((-1, 1, 3))
                )
                ** 2,
                axis=-1,
            )
        )
        assert r.shape == (m, n)
        return r, zc


@define
class OpticFiber(LightModel):
    """Optic fiber light model from Foutz et al., 2012.

    Defaults are from paper for 473 nm wavelength."""

    R0: Quantity = 0.1 * mm
    """optical fiber radius"""
    NAfib: float = 0.37
    """optical fiber numerical aperture"""
    K: Quantity = 0.125 / mm
    """absorbance coefficient (wavelength/tissue dependent)"""
    S: Quantity = 7.37 / mm
    """scattering coefficient (wavelength/tissue dependent)"""
    ntis: float = 1.36
    """tissue index of refraction (wavelength/tissue dependent)"""

    def transmittance(
        self,
        source_coords: Quantity,
        source_dir_uvec: NDArray[(Any, 3), Any],
        target_coords: Quantity,
    ) -> NDArray[(Any, Any), float]:
        assert np.allclose(np.linalg.norm(source_dir_uvec, axis=-1), 1)
        r, z = self._get_rz_for_xyz(source_coords, source_dir_uvec, target_coords)
        return self._Foutz12_transmittance(r, z)

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        """Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation.

        r and z should be (n_source, n_target) arrays with units"""

        if spread:
            # divergence half-angle of cone
            theta_div = np.arcsin(self.NAfib / self.ntis)
            Rz = self.R0 + z * np.tan(
                theta_div
            )  # radius as light spreads ("apparent radius" from original code)
            C = (self.R0 / Rz) ** 2
        else:
            Rz = self.R0  # "apparent radius"
            C = 1

        if gaussian:
            G = 1 / np.sqrt(2 * np.pi) * np.exp(-2 * (r / Rz) ** 2)
        else:
            G = 1

        if scatter:
            S = self.S
            a = 1 + self.K / S
            b = np.sqrt(a**2 - 1)
            dist = np.sqrt(r**2 + z**2)
            M = b / (a * np.sinh(b * S * dist) + b * np.cosh(b * S * dist))
        else:
            M = 1

        T = G * C * M
        T[z < 0] = 0
        return T

    def viz_params(
        self,
        coords: Quantity,
        direction: NDArray[(Any, 3), Any],
        T_threshold: float,
        n_points_per_source: int = 4000,
        **kwargs,
    ) -> Quantity:
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(n_points_per_source, r_thresh, zc_thresh)

        end = coords + zc_thresh * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)
        density_factor = 3
        cyl_vol = np.pi * r_thresh**2 * zc_thresh
        markersize_um = (cyl_vol / n_points_per_source * density_factor) ** (1 / 3) / um
        intensity_scale = 1.5 * (4e3 / n_points_per_source) ** (1 / 3)
        return coords_from_xyz(x, y, z), markersize_um, intensity_scale

    def _find_rz_thresholds(self, thresh):
        """find r and z thresholds for visualization purposes"""
        res_mm = 0.01
        zc = np.arange(20, 0, -res_mm) * mm  # ascending T
        T = self._Foutz12_transmittance(0 * mm, zc)
        try:
            zc_thresh = zc[np.searchsorted(T, thresh)]
        except IndexError:  # no points above threshold
            zc_thresh = np.max(zc)
        # look at half the z threshold for the r threshold
        r = np.arange(20, 0, -res_mm) * mm
        T = self._Foutz12_transmittance(r, zc_thresh / 2)
        try:
            r_thresh = r[np.searchsorted(T, thresh)]
        except IndexError:  # no points above threshold
            r_thresh = np.max(r)
        # multiply by 1.2 just in case
        return r_thresh * 1.2, zc_thresh


def fiber473nm(
    R0=0.1 * mm,  # optical fiber radius
    NAfib=0.37,  # optical fiber numerical aperture
    K=0.125 / mm,  # absorbance coefficient
    S=7.37 / mm,  # scattering coefficient
    ntis=1.36,  # tissue index of refraction
) -> OpticFiber:
    """Returns an :class:`OpticFiber` model with parameters for 473 nm light.

    Parameters from Foutz et al., 2012."""
    return OpticFiber(
        R0=R0,
        NAfib=NAfib,
        K=K,
        S=S,
        ntis=ntis,
    )


@define
class Koehler(LightModel):
    """Even illumination over a circular area, with no scattering."""

    radius: Quantity
    """The radius of the Köhler beam"""
    zmax: Quantity = 500 * um
    """The maximum extent of the Köhler beam, 500 μm by default
    (i.e., no thicker than necessary to go through a slice or culture)."""

    def transmittance(self, source_coords, source_dir_uvec, target_coords):
        r, z = self._get_rz_for_xyz(source_coords, source_dir_uvec, target_coords)
        T = np.ones_like(r)
        T[r > self.radius] = 0
        T[z > self.zmax] = 0
        T[z < 0] = 0
        return T

    def viz_params(
        self, coords, direction, T_threshold, n_points_per_source=4000, **kwargs
    ):
        r, theta, zc = uniform_cylinder_rθz(n_points_per_source, self.radius, self.zmax)

        end = coords + self.zmax * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)
        density_factor = 2
        cyl_vol = np.pi * self.radius**2 * self.zmax
        markersize_um = (cyl_vol / n_points_per_source * density_factor) ** (1 / 3) / um
        intensity_scale = (1 / n_points_per_source) ** (1 / 3)
        return coords_from_xyz(x, y, z), markersize_um, intensity_scale


@define(eq=False)
class Light(Stimulator):
    """Delivers light to the network for photostimulation and (when implemented) imaging.

    Requires neurons to have 3D spatial coordinates already assigned.

    Visualization kwargs
    --------------------
    n_points_per_source : int, optional
        The number of points per light source used to represent light intensity in
        space. Default varies by :attr:`light_model`. Alias ``n_points``.
    T_threshold : float, optional
        The transmittance below which no points are plotted. By default 1e-3.
    intensity : float, optional
        How bright the light appears, should be between 0 and 1. By default 0.5.
    rasterized : bool, optional
        Whether to render as rasterized in vector output, True by default.
        Useful since so many points makes later rendering and editing slow.
    """

    light_model: LightModel = field(kw_only=True)
    """Defines how light is emitted. See :class:`OpticFiber` for an example."""

    coords: Quantity = field(
        default=(0, 0, 0) * mm, converter=lambda x: np.reshape(x, (-1, 3)), repr=False
    )
    """(x, y, z) coords with Brian unit specifying where to place
    the base of the light source, by default (0, 0, 0)*mm.
    Can also be an nx3 array for multiple sources.
    """

    wavelength: Quantity = field(default=473 * nmeter, kw_only=True)
    """light wavelength with unit (usually nmeter)"""

    @coords.validator
    def _check_coords(self, attribute, value):
        if len(value.shape) != 2 or value.shape[1] != 3:
            raise ValueError(
                "coords must be an n by 3 array (with unit) with x, y, and z"
                "coordinates for n contact locations."
            )

    direction: NDArray[(Any, 3), Any] = field(
        default=(0, 0, 1), converter=normalize_coords
    )
    """(x, y, z) vector specifying direction in which light
    source is pointing, by default (0, 0, 1).
    
    Will be converted to unit magnitude."""

    max_Irr0_mW_per_mm2: float = field(default=None, kw_only=True)
    """The maximum irradiance the light source can emit.
    
    Usually determined by hardware in a real experiment."""

    max_Irr0_mW_per_mm2_viz: float = field(default=None, kw_only=True)
    """Maximum irradiance for visualization purposes. 
    
    i.e., the level at or above which the light appears maximally bright.
    Only relevant in video visualization.
    """

    default_value: NDArray[(Any,), float] = field(kw_only=True, repr=False)

    @default_value.default
    def _default_default(self):
        return np.zeros(self.n)

    @property
    def color(self):
        """Color of light"""
        return wavelength_to_rgb(self.wavelength / nmeter)

    def transmittance(self, target_coords) -> np.ndarray:
        """Returns :attr:`light_model` transmittance given light's coords and direction."""
        return self.light_model.transmittance(
            self.coords, self.direction, target_coords
        )

    def init_for_simulator(self, sim: CLSimulator) -> None:
        registry = registry_for_sim(sim)
        registry.init_register_light(self)
        self.reset()

    def connect_to_neuron_group(
        self, neuron_group: NeuronGroup, **kwparams: Any
    ) -> None:
        registry = registry_for_sim(self.sim)
        if self in registry.lights_for_ng.get(neuron_group, set()):
            raise ValueError(
                f"Light {self} already connected to neuron group {neuron_group}"
            )
        registry.register_light(self, neuron_group)

    @property
    def n(self):
        """Number of light sources"""
        assert len(self.coords.shape) == 2 or len(self.coords.shape) == 1
        return len(self.coords) if len(self.coords.shape) == 2 else 1

    @property
    def source(self) -> Subgroup:
        """Returns the "neuron(s)" representing the light source(s)."""
        registry = registry_for_sim(self.sim)
        return registry.source_for_light(self)

    def add_self_to_plot(self, ax, axis_scale_unit, **kwargs) -> list[PathCollection]:
        # show light with point field, assigning r and z coordinates
        # to all points
        # filter out points with <0.001 transmittance to make plotting faster

        # alias
        if "n_points" in kwargs and "n_points_per_source" not in kwargs:
            kwargs["n_points_per_source"] = kwargs.pop("n_points")
        T_threshold = kwargs.pop("T_threshold", 1e-3)

        viz_points, markersize_um, intensity_scale = self.light_model.viz_params(
            self.coords,
            self.direction,
            T_threshold,
            **kwargs,
        )
        assert viz_points.shape[0] == self.n
        assert viz_points.shape[2] == 3
        n_points_per_source = viz_points.shape[1]
        intensity = kwargs.get("intensity", 0.5 * intensity_scale)

        biggest_dim_pixels = max([ax.bbox.height, ax.bbox.width])
        dpi = 100  # default
        pt_per_in = 72
        biggest_dim_pt = biggest_dim_pixels / dpi * pt_per_in

        biggest_dim_um = (
            max(
                [
                    ax.get_xlim()[1] - ax.get_xlim()[0],
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    ax.get_zlim()[1] - ax.get_zlim()[0],
                ]
            )
            * axis_scale_unit
            / um
        )

        markersize_pt = markersize_um / biggest_dim_um * biggest_dim_pt
        markerarea = markersize_pt**2

        T = self.light_model.transmittance(self.coords, self.direction, viz_points)
        assert T.shape == (self.n, n_points_per_source)

        point_clouds = []
        for i in range(self.n):
            idx_to_plot = T[i] >= T_threshold
            point_cloud = ax.scatter(
                viz_points[i, idx_to_plot, 0] / axis_scale_unit,
                viz_points[i, idx_to_plot, 1] / axis_scale_unit,
                viz_points[i, idx_to_plot, 2] / axis_scale_unit,
                c=T[i, idx_to_plot],
                cmap=self._alpha_cmap_for_wavelength(intensity),
                vmin=0,
                vmax=1,
                marker="o",
                edgecolors="none",
                label=self.name,
                s=markerarea,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*Rasterization.*will be ignored.*"
                )
                # to make manageable in SVGs
                point_cloud.set_rasterized(kwargs.get("rasterized", True))
            point_clouds.append(point_cloud)

        handles = ax.get_legend().legend_handles
        c = wavelength_to_rgb(self.wavelength / nmeter)
        opto_patch = mpl.patches.Patch(color=c, label=self.name)
        handles.append(opto_patch)
        ax.legend(handles=handles)

        return point_clouds

    def update_artists(
        self, artists: list[Artist], value, *args, **kwargs
    ) -> list[Artist]:
        assert len(artists) == self.n
        if self.max_Irr0_mW_per_mm2_viz is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2_viz
        elif self.max_Irr0_mW_per_mm2 is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2
        else:
            raise Exception(
                f"Light'{self.name}' needs max_Irr0_mW_per_mm2_viz "
                "or max_Irr0_mW_per_mm2 "
                "set to visualize light intensity."
            )

        updated_artists = []
        for point_cloud, source_value in zip(artists, value):
            prev_value = getattr(point_cloud, "_prev_value", None)
            if source_value != prev_value:
                intensity = (
                    source_value / max_Irr0 if source_value <= max_Irr0 else max_Irr0
                )
                point_cloud.set_cmap(self._alpha_cmap_for_wavelength(intensity))
                updated_artists.append(point_cloud)
                point_cloud._prev_value = source_value

        return updated_artists

    def update(self, value: Union[float, np.ndarray]) -> None:
        """Set the light intensity, in mW/mm2 (without unit) for 1P
        excitation or laser power (mW) for 2P excitation (GaussianEllipsoid light model).

        Parameters
        ----------
        Irr0_mW_per_mm2 : float
            Desired light intensity for light source
        """
        if type(value) != np.ndarray:
            value = np.array(value).reshape((-1,))
        if value.shape not in [(), (1,), (self.n,)]:
            raise ValueError(
                f"Input to light must be a scalar or an array of"
                f" length {self.n}. Got {value.shape} instead."
            )
        if type(self.light_model) == "GaussianEllipsoid":
            # 10 microns, on upper end of what's used as spot size in Ronzitti et al., 2017
            # Irr0 = P / spot_area, as in Ronzitti et al., 2017
            cell_radius = 0.010  # mm
            cell_area = np.pi * cell_radius**2
            Irr0_mW_per_mm2 = value / cell_area
        else:
            Irr0_mW_per_mm2 = value
        if np.any(Irr0_mW_per_mm2 < 0):
            warnings.warn(f"{self.name}: negative light intensity Irr0 clipped to 0")
            Irr0_mW_per_mm2[Irr0_mW_per_mm2 < 0] = 0
        if self.max_Irr0_mW_per_mm2 is not None:
            Irr0_mW_per_mm2[
                Irr0_mW_per_mm2 > self.max_Irr0_mW_per_mm2
            ] = self.max_Irr0_mW_per_mm2
        super(Light, self).update(Irr0_mW_per_mm2)
        self.source.Irr0 = Irr0_mW_per_mm2 * mwatt / mm2

    def _alpha_cmap_for_wavelength(self, intensity):
        c = wavelength_to_rgb(self.wavelength / nmeter)
        c_dimmest = (*c, 0)
        alpha_max = 0.6
        alpha_brightest = alpha_max * intensity
        c_brightest = (*c, alpha_brightest)
        return colors.LinearSegmentedColormap.from_list(
            # breaks on newer matplotlib. leave 0 and 1 implicit
            # "incr_alpha", [(0, c_dimmest), (1, c_brightest)]
            "incr_alpha",
            [c_dimmest, c_brightest],
        )

    def to_neo(self):
        signal = analog_signal(self.t_ms, self.values, "mW/mm**2")
        signal.name = self.name
        signal.description = "Exported from Cleo Light device"
        signal.annotate(export_datetime=datetime.datetime.now())
        # broadcast in case of uniform direction
        _, direction = np.broadcast_arrays(self.coords, self.direction)
        signal.array_annotate(
            x=self.coords[..., 0] / mm * pq.mm,
            y=self.coords[..., 1] / mm * pq.mm,
            z=self.coords[..., 2] / mm * pq.mm,
            direction_x=direction[..., 0],
            direction_y=direction[..., 1],
            direction_z=direction[..., 2],
            i_channel=np.arange(self.n),
        )
        return signal
