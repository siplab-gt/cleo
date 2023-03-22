"""Contains Light device and propagation models"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any
import warnings

from attrs import define, field, asdict
from brian2 import (
    Synapses,
    NeuronGroup,
    Unit,
    BrianObjectException,
    get_unit,
    Equations,
)
from nptyping import NDArray
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
from brian2.units.allunits import meter2, radian
import brian2.units.unitsafefunctions as usf
import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection

from cleo.base import InterfaceDevice
from cleo.utilities import (
    uniform_cylinder_rθz,
    wavelength_to_rgb,
    xyz_from_rθz,
    coords_from_ng,
    normalize_coords,
)
from cleo.stimulators import Stimulator


@define
class LightModel(ABC):
    @abstractmethod
    def transmittance(
        self,
        source_coords: Quantity,
        source_direction: NDArray[(Any, 3), Any],
        target_coords: Quantity,
    ) -> NDArray[(Any,), float]:
        pass

    @abstractmethod
    def viz_points(
        self, coords: Quantity, direction: NDArray[(Any, 3), Any], **kwargs
    ) -> Quantity:
        pass


@define
class FiberModel(LightModel):
    """Optic fiber light model from Foutz et al., 2012.

    Defaults are from paper for 473 nm wavelength."""

    R0: Quantity = 0.1 * mm
    """optical fiber radius"""
    NAfib: Quantity = 0.37
    """optical fiber numerical aperture"""
    wavelength: Quantity = 473 * nmeter
    """light wavelength"""
    K: Quantity = 0.125 / mm
    """absorbance coefficient (wavelength/tissue dependent)"""
    S: Quantity = 7.37 / mm
    """scattering coefficient (wavelength/tissue dependent)"""
    ntis: Quantity = 1.36
    """tissue index of refraction (wavelength/tissue dependent)"""

    model = """
            Irr = Irr0*T : watt/meter**2
            Irr0 : watt/meter**2 
            T : 1
            phi = Irr / Ephoton : 1/second/meter**2
            """

    def transmittance(
        self,
        source_coords: Quantity,
        source_direction: NDArray[(Any, 3), Any],
        target_coords: Quantity,
    ) -> NDArray[(Any, Any), float]:
        assert np.allclose(np.linalg.norm(source_direction, axis=-1), 1)
        r, z = self._get_rz_for_xyz(source_coords, source_direction, target_coords)
        return self._Foutz12_transmittance(r, z)

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        """Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation"""

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

    def viz_points(
        self, coords: Quantity, direction: NDArray[(Any, 3), Any], **kwargs
    ) -> Quantity:
        T_threshold = kwargs.get("T_threshold", 0.001)
        n_points_per_source = kwargs.get("n_points", 1e4)
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(n_points_per_source, r_thresh, zc_thresh)

        T = self._Foutz12_transmittance(r, zc)

        end = coords + zc_thresh * direction
        x, y, z = xyz_from_rθz(r, theta, zc, coords, end)

    def _get_rz_for_xyz(self, source_coords, source_direction, target_coords):
        """Assumes x, y, z already have units"""
        m = len(source_coords) if len(source_coords.shape) == 2 else 1
        n = len(target_coords) if len(target_coords.shape) == 2 else 1

        rel_coords = (
            # target_coords[np.newaxis, :, :] - source_coords[:, np.newaxis, :]
            target_coords.reshape((1, n, 3))
            - source_coords.reshape((m, 1, 3))
        )  # relative to light source(s)
        # now m x n x 3 array, where m is number of sources, n is number of targets
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        # zc = usf.dot(rel_coords, source_direction)  # mxn distance along cylinder axis
        #           m x n x 3    m x 1 x 3
        zc = np.sum(
            rel_coords * source_direction.reshape((m, 1, 3)), axis=-1
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
                    - zc.reshape((m, n, 1)) * source_direction.reshape((m, 1, 3))
                )
                ** 2,
                axis=-1,
            )
        )
        assert r.shape == (m, n)
        return r, zc

    def _find_rz_thresholds(self, thresh):
        """find r and z thresholds for visualization purposes"""
        res_mm = 0.01
        zc = np.arange(20, 0, -res_mm) * mm  # ascending T
        T = self._Foutz12_transmittance(0 * mm, zc)
        zc_thresh = zc[np.searchsorted(T, thresh)]
        # look at half the z threshold for the r threshold
        r = np.arange(20, 0, -res_mm) * mm
        T = self._Foutz12_transmittance(r, zc_thresh / 2)
        r_thresh = r[np.searchsorted(T, thresh)]
        # multiply by 1.2 just in case
        return r_thresh * 1.2, zc_thresh


def fiber473nm(
    R0=0.1 * mm,  # optical fiber radius
    NAfib=0.37,  # optical fiber numerical aperture
    wavelength=473 * nmeter,
    K=0.125 / mm,  # absorbance coefficient
    S=7.37 / mm,  # scattering coefficient
    ntis=1.36,  # tissue index of refraction
) -> FiberModel:
    """Light parameters for 473 nm wavelength delivered via an optic fiber.

    From Foutz et al., 2012. See :class:`FiberModel` for parameter descriptions."""
    return FiberModel(
        R0=R0,
        NAfib=NAfib,
        wavelength=wavelength,
        K=K,
        S=S,
        ntis=ntis,
    )


@define
class Light(Stimulator):
    """Delivers photostimulation of the network.

    Essentially "transfects" neurons and provides a light source.
    Under the hood, it delivers current via a Brian :class:`~brian2.synapses.synapses.Synapses`
    object.

    Requires neurons to have 3D spatial coordinates already assigned.
    Also requires that the neuron model has a current term
    (by default Iopto) which is assumed to be positive (unlike the
    convention in many opsin modeling papers, where the current is
    described as negative).

    See :meth:`connect_to_neuron_group` for optional keyword parameters
    that can be specified when calling
    :meth:`cleo.CLSimulator.inject_stimulator`.

    Visualization kwargs
    --------------------
    n_points : int, optional
        The number of points used to represent light intensity in space.
        By default 1e4.
    T_threshold : float, optional
        The transmittance below which no points are plotted. By default
        1e-3.
    intensity : float, optional
        How bright the light appears, should be between 0 and 1. By default 0.5.
    rasterized : bool, optional
        Whether to render as rasterized in vector output, True by default.
        Useful since so many points makes later rendering and editing slow.
    """

    opsin_model: OpsinModel = field(kw_only=True)
    """OpsinModel object defining how light affects target
    neurons. See :class:`FourStateModel` and :class:`ProportionalCurrentModel`
    for examples."""

    light_model: LightModel = field(kw_only=True)
    """LightModel object defining how light is emitted. See
    :class:`FiberModel` for an example."""

    coords: Quantity = (0, 0, 0) * mm
    """(x, y, z) coords with Brian unit specifying where to place
    the base of the light source, by default (0, 0, 0)*mm.
    Can also be an nx3 array for multiple sources.
    """

    direction: NDArray[(Any, 3), Any] = field(
        default=(0, 0, 1), converter=normalize_coords
    )
    """(x, y, z) vector specifying direction in which light
    source is pointing, by default (0, 0, 1).
    
    Will be converted to unit magnitude."""

    opto_syns: dict[str, Synapses] = field(factory=dict, init=False)
    """Stores the synapse objects implementing the opsin model,
    with NeuronGroup name keys and Synapse values."""

    max_Irr0_mW_per_mm2: float = None
    """The maximum irradiance the light source can emit.
    
    Usually determined by hardware in a real experiment."""

    max_Irr0_mW_per_mm2_viz: float = field(default=None, kw_only=True)
    """Maximum irradiance for visualization purposes. 
    
    i.e., the level at or above which the light appears maximally bright.
    Only relevant in video visualization.
    """

    def connect_to_neuron_group(
        self, neuron_group: NeuronGroup, **kwparams: Any
    ) -> None:
        """Configure opsin and light source to stimulate given neuron group.

        Parameters
        ----------
        neuron_group : NeuronGroup
            The neuron group to stimulate with the given opsin and light source

        Keyword args
        ------------
        p_expression : float
            Probability (0 <= p <= 1) that a given neuron in the group
            will express the opsin. 1 by default.
        rho_rel : float
            The expression level, relative to the standard model fit,
            of the opsin. 1 by default. For heterogeneous expression,
            this would have to be modified in the opsin synapse post-injection,
            e.g., ``opto.opto_syns["neuron_group_name"].rho_rel = ...``
        Iopto_var_name : str
            The name of the variable in the neuron group model representing
            current from the opsin
        v_var_name : str
            The name of the variable in the neuron group model representing
            membrane potential
        """
        # get modified opsin model string (i.e., with names/units specified)
        (
            mod_opsin_model,
            mod_opsin_params,
        ) = self.opsin_model.modify_model_and_params_for_ng(neuron_group, kwparams)

        # fmt: off
        # Ephoton = h*c/lambda
        E_photon = (
            6.63e-34 * meter2 * kgram / second
            * 2.998e8 * meter / second
            / self.light_model.wavelength
        )
        # fmt: on

        light_model = Equations(
            """
            Irr = Irr0*T : watt/meter**2
            Irr0 : watt/meter**2 
            T : 1
            phi = Irr / Ephoton : 1/second/meter**2
            """
        )

        opto_syn = Synapses(
            neuron_group,
            model=mod_opsin_model + light_model,
            namespace=mod_opsin_params,
            name=f"synapses_{self.name}_{neuron_group.name}",
            method="rk2",
        )
        opto_syn.namespace["Ephoton"] = E_photon

        p_expression = kwparams.get("p_expression", 1)
        if p_expression == 1:
            opto_syn.connect(j="i")
        else:
            opto_syn.connect(condition="i==j", p=p_expression)

        self.opsin_model.init_opto_syn_vars(opto_syn)

        # relative channel density
        opto_syn.rho_rel = kwparams.get("rho_rel", 1)
        # calculate transmittance coefficient for each point
        T = self.light_model.transmittance(
            self.coords, self.direction, coords_from_ng(neuron_group)
        )
        assert T.shape == (self.m, len(neuron_group))
        # reduce to subset expressing opsin before assigning
        T = T[opto_syn.i]

        opto_syn.T = T

        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    @property
    def m(self):
        """Number of light sources"""
        assert len(self.coords.shape) == 2 or len(self.coords.shape) == 1
        return len(self.coords) if len(self.coords.shape) == 2 else 1

    def add_self_to_plot(self, ax, axis_scale_unit, **kwargs) -> PathCollection:
        # show light with point field, assigning r and z coordinates
        # to all points
        # filter out points with <0.001 transmittance to make plotting faster

        T_threshold = kwargs.get("T_threshold", 0.001)
        n_points = kwargs.get("n_points", 1e4)
        intensity = kwargs.get("intensity", 0.5)

        viz_points = self.light_model.viz_points(self.coords, self.direction, **kwargs)
        T = self.light_model.transmittance(
            self.coords, self.direction, viz_points, **kwargs
        )

        idx_to_plot = T >= T_threshold
        x = x[idx_to_plot]
        y = y[idx_to_plot]
        z = z[idx_to_plot]
        T = T[idx_to_plot]
        point_cloud = ax.scatter(
            x / axis_scale_unit,
            y / axis_scale_unit,
            z / axis_scale_unit,
            c=T,
            cmap=self._alpha_cmap_for_wavelength(intensity),
            marker="o",
            edgecolors="none",
            label=self.name,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Rasterization.*will be ignored.*"
            )
            # to make manageable in SVGs
            point_cloud.set_rasterized(kwargs.get("rasterized", True))
        handles = ax.get_legend().legendHandles
        c = wavelength_to_rgb(self.light_model.wavelength / nmeter)
        opto_patch = matplotlib.patches.Patch(color=c, label=self.name)
        handles.append(opto_patch)
        ax.legend(handles=handles)
        return [point_cloud]

    def update_artists(
        self, artists: list[Artist], value, *args, **kwargs
    ) -> list[Artist]:
        self._prev_value = getattr(self, "_prev_value", None)
        if value == self._prev_value:
            return []

        assert len(artists) == 1
        point_cloud = artists[0]

        if self.max_Irr0_mW_per_mm2_viz is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2_viz
        elif self.max_Irr0_mW_per_mm2 is not None:
            max_Irr0 = self.max_Irr0_mW_per_mm2
        else:
            raise Exception(
                f"OptogeneticIntervention '{self.name}' needs max_Irr0_mW_per_mm2_viz "
                "or max_Irr0_mW_per_mm2 "
                "set to visualize light intensity."
            )

        intensity = value / max_Irr0 if value <= max_Irr0 else max_Irr0
        point_cloud.set_cmap(self._alpha_cmap_for_wavelength(intensity))
        return [point_cloud]

    def update(self, Irr0_mW_per_mm2: float):
        """Set the light intensity, in mW/mm2 (without unit)

        Parameters
        ----------
        Irr0_mW_per_mm2 : float
            Desired light intensity for light source
        """
        if Irr0_mW_per_mm2 < 0:
            warnings.warn(f"{self.name}: negative light intensity Irr0 clipped to 0")
            Irr0_mW_per_mm2 = 0
        if (
            self.max_Irr0_mW_per_mm2 is not None
            and Irr0_mW_per_mm2 > self.max_Irr0_mW_per_mm2
        ):
            Irr0_mW_per_mm2 = self.max_Irr0_mW_per_mm2
        super().update(Irr0_mW_per_mm2)
        for opto_syn in self.opto_syns.values():
            opto_syn.Irr0 = Irr0_mW_per_mm2 * mwatt / mm2

    def reset(self, **kwargs):
        for opto_syn in self.opto_syns.values():
            self.opsin_model.init_opto_syn_vars(opto_syn)

    def _alpha_cmap_for_wavelength(self, intensity=0.5):
        c = wavelength_to_rgb(self.light_model.wavelength / nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 0.6 * intensity)
        return colors.LinearSegmentedColormap.from_list(
            "incr_alpha", [(0, c_clear), (1, c_opaque)]
        )