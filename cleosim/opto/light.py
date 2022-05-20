"""TODO"""
from __future__ import annotations
from abc import abstractmethod
from typing import Tuple, Any

import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from brian2 import Synapses, NeuronGroup
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
from brian2.core.base import BrianObjectException

from cleosim.base import Stimulator
from cleosim.utilities import wavelength_to_rgb
from cleosim.opto.opsins import OpsinModel


class LightSource(Stimulator):
    """Delivers light for optogenetic stimulation of the network.

    On injection, transfects neurons with specified opsin.
    Under the hood, it delivers current via a Brian :class:`~brian2.synapses.synapses.Synapses`
    object.

    Requires neurons to have 3D spatial coordinates already assigned.
    Also requires that the neuron model has a current term
    (by default Iopto) which is assumed to be positive (unlike the
    convention in many opsin modeling papers, where the current is
    described as negative). An :class:`cleosim.opto.OpsinModel` object must
    be specified on injection with the ``opsin=`` keyword argument for the light
    to drive photocurrents.

    See :meth:`connect_to_neuron_group` for optional keyword parameters
    that can be specified when calling
    :meth:`cleosim.CLSimulator.inject_stimulator`.
    """

    opto_syns: dict[str, Synapses]
    """Stores the synapse objects implementing the opsin model,
    with NeuronGroup name keys and Synapse values."""

    max_Irr0_mW_per_mm2: float
    """The maximum irradiance the light source can emit.
    
    Usually determined by hardware in a real experiment."""

    max_Irr0_mW_per_mm2_viz: float
    """Maximum irradiance for visualization purposes. 
    
    i.e., the level at or above which the light appears maximally bright.
    Only relevant in video visualization.
    """

    def __init__(self, name: str,
        start_value,
        max_Irr0_mW_per_mm2: float = None,
    ) -> None:
        """TODO"""
        super().__init__(name, start_value)
        self.opto_syns = {}
        self.max_Irr0_mW_per_mm2 = max_Irr0_mW_per_mm2
        self.max_Irr0_mW_per_mm2_viz = None

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
        TODO: opsin_model
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
        # TODO: no longer just one self.opsin_model. get from kwparams
        modified_opsin_model = self.opsin_model.get_modified_model_for_ng(
            neuron_group, kwparams
        )

        # fmt: off
        # Ephoton = h*c/lambda
        E_photon = (
            6.63e-34 * meter2 * kgram / second
            * 2.998e8 * meter / second
            / self.light_model_params["wavelength"]
        )
        # fmt: on

        light_model = """
            Irr = Irr0*T : watt/meter**2
            Irr0 : watt/meter**2 
            T : 1
            phi = Irr / Ephoton : 1/second/meter**2
        """

        # TODO: opsin model needs epsilon, way to take into account crosstalk
        # -- partial current for non-peak wavelength
        opto_syn = Synapses(
            neuron_group,
            model=modified_opsin_model + light_model,
            namespace=self.opsin_model.params,
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
        T = self.get_transmittance_for_neurons(neuron_group)
        # reduce to subset expressing opsin before assigning
        T = [T[k] for k in opto_syn.i]

        opto_syn.T = T

        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    def _get_transmittance_for_neurons(self, neuron_group: NeuronGroup):
        return self.get_transmittance_for_coords(neuron_group.x, neuron_group.y, neuron_group.z)

    @abstractmethod
    def get_transmittance_for_coords(self, x, y, z):
        pass

    def add_self_to_plot(self, ax, axis_scale_unit) -> PathCollection:
        # show light with point field, assigning r and z coordinates
        # to all points
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        x = np.linspace(xlim[0], xlim[1], 50)
        y = np.linspace(ylim[0], ylim[1], 50)
        z = np.linspace(zlim[0], zlim[1], 50)
        x, y, z = np.meshgrid(x, y, z) * axis_scale_unit

        r, zc = self._get_rz_for_xyz(x, y, z)
        T = self._Foutz12_transmittance(r, zc)
        # filter out points with <0.001 transmittance to make plotting faster
        plot_threshold = 0.001
        idx_to_plot = T[:, 0] >= plot_threshold
        x = x.flatten()[idx_to_plot]
        y = y.flatten()[idx_to_plot]
        z = z.flatten()[idx_to_plot]
        T = T[idx_to_plot, 0]
        point_cloud = ax.scatter(
            x / axis_scale_unit,
            y / axis_scale_unit,
            z / axis_scale_unit,
            c=T,
            cmap=self._alpha_cmap_for_wavelength(),
            marker=",",
            edgecolors="none",
            label=self.name,
        )
        handles = ax.get_legend().legendHandles
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
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

        Raises
        ------
        ValueError
            When intensity is negative
        """
        if Irr0_mW_per_mm2 < 0:
            raise ValueError(f"{self.name}: light intensity Irr0 must be nonnegative")
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
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 0.6 * intensity)
        return colors.LinearSegmentedColormap.from_list(
            "incr_alpha", [(0, c_clear), (1, c_opaque)]
        )


class OpticFiber(LightSource):
    """TODO"""

    def __init__(
        self,
        name: str,
        params: dict,
        location: Quantity = (0, 0, 0) * mm,
        direction: Tuple[float, float, float] = (0, 0, 1),
        max_Irr0_mW_per_mm2: float = None,
    ):
        """
        Parameters
        ----------
        name : str
            Unique identifier for stimulator
        opsin_model : OpsinModel
            OpsinModel object defining how light affects target
            neurons. See :class:`FourStateModel` and :class:`ProportionalCurrentModel`
            for examples.
        params : dict
            Parameters for the light propagation model in Foutz et al., 2012.
            See :attr:`fiber_params_blue` for an example.
        location : Quantity, optional
            (x, y, z) coords with Brian unit specifying where to place
            the base of the light source, by default (0, 0, 0)*mm
        direction : Tuple[float, float, float], optional
            (x, y, z) vector specifying direction in which light
            source is pointing, by default (0, 0, 1)
        max_Irr0_mW_per_mm2 : float, optional
            Set :attr:`max_Irr0_mW_per_mm2`.
        """
        super().__init__(name, 0, max_Irr0_mW_per_mm2)
        self.light_model_params = params
        self.location = location
        # direction unit vector
        self.dir_uvec = (direction / np.linalg.norm(direction)).reshape((3, 1))

    def get_transmittance_for_coords(self, x, y, z):
        """TODO"""
        r, z = self._get_rz_for_xyz(x, y, z)
        return self._Foutz12_transmittance(r, z).flatten()

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        """Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation"""

        if spread:
            # divergence half-angle of cone
            theta_div = np.arcsin(
                self.light_model_params["NAfib"] / self.light_model_params["ntis"]
            )
            Rz = self.light_model_params["R0"] + z * np.tan(
                theta_div
            )  # radius as light spreads("apparent radius" from original code)
            C = (self.light_model_params["R0"] / Rz) ** 2
        else:
            Rz = self.light_model_params["R0"]  # "apparent radius"
            C = 1

        if gaussian:
            G = 1 / np.sqrt(2 * np.pi) * np.exp(-2 * (r / Rz) ** 2)
        else:
            G = 1

        def kubelka_munk(dist):
            S = self.light_model_params["S"]
            a = 1 + self.light_model_params["K"] / S
            b = np.sqrt(a ** 2 - 1)
            dist = np.sqrt(r ** 2 + z ** 2)
            return b / (a * np.sinh(b * S * dist) + b * np.cosh(b * S * dist))

        M = kubelka_munk(np.sqrt(r ** 2 + z ** 2)) if scatter else 1

        T = G * C * M
        return T

    def _get_rz_for_xyz(self, x, y, z):
        """Assumes x, y, z already have units"""

        def flatten_if_needed(var):
            if len(var.shape) != 1:
                return var.flatten()
            else:
                return var

        # have to add unit back on since it's stripped by vstack
        coords = (
            np.vstack(
                [flatten_if_needed(x), flatten_if_needed(y), flatten_if_needed(z)]
            ).T
            * meter
        )
        rel_coords = coords - self.location  # relative to fiber location
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        zc = usf.dot(rel_coords, self.dir_uvec)  # distance along cylinder axis
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt(np.sum((rel_coords - usf.dot(zc, self.dir_uvec.T)) ** 2, axis=1))
        r = r.reshape((-1, 1))
        return r, zc


fiber_params_blue = {
    "R0": 0.1 * mm,  # optical fiber radius
    "NAfib": 0.37,  # optical fiber numerical aperture
    "wavelength": 473 * nmeter,
    # NOTE: the following depend on wavelength and tissue properties and thus would be different for another wavelength
    "K": 0.125 / mm,  # absorbance coefficient
    "S": 7.37 / mm,  # scattering coefficient
    "ntis": 1.36,  # tissue index of refraction
}
"""Light parameters for 473 nm wavelength delivered via an optic fiber.

From Foutz et al., 2012"""

