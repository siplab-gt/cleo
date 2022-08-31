"""Contains opsin models, parameters, and OptogeneticIntervention device"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any
import warnings

from brian2 import (
    Synapses,
    NeuronGroup,
    Unit,
    BrianObjectException,
    get_unit,
    Equations,
)
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

from cleo.utilities import uniform_cylinder_rθz, wavelength_to_rgb, xyz_from_rθz
from cleo.stimulators import Stimulator


ChR2_four_state = {
    "g0": 114000 * psiemens,
    "gamma": 0.00742,
    "phim": 2.33e17 / mm2 / second,  # *photon, not in Brian2
    "k1": 4.15 / ms,
    "k2": 0.868 / ms,
    "p": 0.833,
    "Gf0": 0.0373 / ms,
    "kf": 0.0581 / ms,
    "Gb0": 0.0161 / ms,
    "kb": 0.063 / ms,
    "q": 1.94,
    "Gd1": 0.105 / ms,
    "Gd2": 0.0138 / ms,
    "Gr0": 0.00033 / ms,
    "E": 0 * mV,
    "v0": 43 * mV,
    "v1": 17.1 * mV,
}
"""Parameters for the 4-state ChR2 model.

Taken from try.projectpyrho.org's default 4-state params.
"""


class OpsinModel(ABC):
    """Base class for opsin model"""

    model: str
    """Basic Brian model equations string.
    
    Should contain a `rho_rel` term reflecting relative expression 
    levels. Will likely also contain special NeuronGroup-dependent
    symbols such as V_VAR_NAME to be replaced on injection in 
    :meth:`~OpsinModel.modify_model_and_params_for_ng`."""

    params: dict
    """Parameter values for model, passed in as a namespace dict"""

    required_vars: list[Tuple[str, Unit]]
    """Default names of state variables required in the neuron group,
    along with units, e.g., [('Iopto', amp)].
    
    It is assumed that non-default values can be passed in on injection
    as a keyword argument ``[default_name]_var_name=[non_default_name]``
    and that these are found in the model string as 
    ``[DEFAULT_NAME]_VAR_NAME`` before replacement."""

    def modify_model_and_params_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict, model="class-defined"
    ) -> Tuple[Equations, dict]:
        """Adapt model for given neuron group on injection

        This enables the specification of variable names
        differently for each neuron group, allowing for custom names
        and avoiding conflicts.

        Parameters
        ----------
        neuron_group : NeuronGroup
            NeuronGroup this opsin model is being connected to
        injct_params : dict
            kwargs passed in on injection, could contain variable
            names to plug into the model

        Keyword Args
        ------------
        model : str, optional
            Model to start with, by default that defined for the class.
            This allows for prior string manipulations before it can be
            parsed as an `Equations` object.

        Returns
        -------
        Equations, dict
            A tuple containing an Equations object
            and a parameter dictionary, constructed from :attr:`~model`
            and :attr:`~params`, respectively, with modified names for use
            in :attr:`~cleo.opto.OptogeneticIntervention.opto_syns`
        """
        if model == "class-defined":
            model = self.model

        for default_name, unit in self.required_vars:
            var_name = injct_params.get(f"{default_name}_var_name", default_name)
            if var_name not in neuron_group.variables or not neuron_group.variables[
                var_name
            ].unit.has_same_dimensions(unit):
                raise BrianObjectException(
                    (
                        f"{var_name} : {unit.name} needed in the model of NeuronGroup "
                        f"{neuron_group.name} to connect OptogeneticIntervention."
                    ),
                    neuron_group,
                )
            # opsin synapse model needs modified names
            to_replace = f"{default_name}_var_name".upper()
            model = model.replace(to_replace, var_name)

        # Synapse variable and parameter names cannot be the same as any
        # neuron group variable name
        return self._fix_name_conflicts(model, neuron_group)

    def _fix_name_conflicts(
        self, modified_model: str, neuron_group: NeuronGroup
    ) -> Tuple[str, dict]:
        modified_params = self.params.copy()
        rename = lambda x: f"{x}_syn"

        # get variables to rename
        opsin_eqs = Equations(modified_model)
        substitutions = {}
        for var in opsin_eqs.names:
            if var in neuron_group.variables:
                substitutions[var] = rename(var)

        # and parameters
        for param in self.params.keys():
            if param in neuron_group.variables:
                substitutions[param] = rename(param)
                modified_params[rename(param)] = modified_params[param]
                del modified_params[param]

        mod_opsin_eqs = opsin_eqs.substitute(**substitutions)
        return mod_opsin_eqs, modified_params

    def init_opto_syn_vars(self, opto_syn: Synapses) -> None:
        """Initializes appropriate variables in Synapses implementing the model

        Can also be used to reset the variables.

        Parameters
        ----------
        opto_syn : Synapses
            The synapses object implementing this model
        """
        pass


class MarkovModel(OpsinModel):
    """Base class for Markov state models à la Evans et al., 2016"""

    required_vars: list[Tuple[str, Unit]] = [("Iopto", amp), ("v", volt)]

    def __init__(self, params: dict) -> None:
        """
        Parameters
        ----------
        params : dict
            dict defining params in the :attr:`model`
        """
        super().__init__()
        self.params = params


class FourStateModel(MarkovModel):
    """4-state model from PyRhO (Evans et al. 2016).

    rho_rel is channel density relative to standard model fit;
    modifying it post-injection allows for heterogeneous opsin expression.

    IOPTO_VAR_NAME and V_VAR_NAME are substituted on injection.
    """

    model: str = """
        dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1 (clock-driven)
        dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gf)*O1 : 1 (clock-driven)
        dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1 (clock-driven)
        C2 = 1 - C1 - O1 - O2 : 1
        # dC2/dt = Gd2*O2 - (Gr0+Ga2)*C2 : 1 (clock-driven)

        Theta = int(phi > 0*phi) : 1
        Hp = Theta * phi**p/(phi**p + phim**p) : 1
        Ga1 = k1*Hp : hertz
        Ga2 = k2*Hp : hertz
        Hq = Theta * phi**q/(phi**q + phim**q) : 1
        Gf = kf*Hq + Gf0 : hertz
        Gb = kb*Hq + Gb0 : hertz

        fphi = O1 + gamma*O2 : 1
        fv = (1 - exp(-(V_VAR_NAME_post-E)/v0)) / -2 : 1

        IOPTO_VAR_NAME_post = -g0*fphi*fv*(V_VAR_NAME_post-E)*rho_rel : ampere (summed)
        rho_rel : 1
    """

    def init_opto_syn_vars(self, opto_syn: Synapses) -> None:
        for varname, value in {"Irr0": 0, "C1": 1, "O1": 0, "O2": 0}.items():
            setattr(opto_syn, varname, value)


class ProportionalCurrentModel(OpsinModel):
    """A simple model delivering current proportional to light intensity"""

    # would be IOPTO_UNIT but that throws off Equation parsing
    model: str = """
        IOPTO_VAR_NAME_post = gain * Irr * rho_rel : IOPTO_UNIT (summed)
        rho_rel : 1
    """

    def __init__(self, Iopto_per_mW_per_mm2: Quantity) -> None:
        """
        Parameters
        ----------
        Iopto_per_mW_per_mm2 : Quantity
            How much current (in amps or unitless, depending on neuron model) to
            deliver per mW/mm2
        """
        self.params = {"gain": Iopto_per_mW_per_mm2 / (mwatt / mm2)}
        if isinstance(Iopto_per_mW_per_mm2, Quantity):
            self._Iopto_unit = get_unit(Iopto_per_mW_per_mm2.dim)
        else:
            self._Iopto_unit = radian
        self.required_vars = [("Iopto", self._Iopto_unit)]

    def modify_model_and_params_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
    ) -> Tuple[Equations, dict]:
        mod_model = self.model.replace("IOPTO_UNIT", self._Iopto_unit.name)
        return super().modify_model_and_params_for_ng(
            neuron_group, injct_params, model=mod_model
        )


default_blue = {
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


class OptogeneticIntervention(Stimulator):
    """Enables optogenetic stimulation of the network.

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

    def __init__(
        self,
        name: str,
        opsin_model: OpsinModel,
        light_model_params: dict,
        location: Quantity = (0, 0, 0) * mm,
        direction: Tuple[float, float, float] = (0, 0, 1),
        max_Irr0_mW_per_mm2: float = None,
        save_history: bool = False,
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
        light_model_params : dict
            Parameters for the light propagation model in Foutz et al., 2012.
            See :attr:`default_blue` for an example.
        location : Quantity, optional
            (x, y, z) coords with Brian unit specifying where to place
            the base of the light source, by default (0, 0, 0)*mm
        direction : Tuple[float, float, float], optional
            (x, y, z) vector specifying direction in which light
            source is pointing, by default (0, 0, 1)
        max_Irr0_mW_per_mm2 : float, optional
            Set :attr:`max_Irr0_mW_per_mm2`.
        save_history : bool, optional
            Determines whether :attr:`~values` and :attr:`~t_ms` are saved.
        """
        super().__init__(name, 0, save_history)
        self.opsin_model = opsin_model
        self.light_model_params = light_model_params
        self.location = location
        # direction unit vector
        self.dir_uvec = direction / np.linalg.norm(direction)
        self.opto_syns = {}
        self.max_Irr0_mW_per_mm2 = max_Irr0_mW_per_mm2
        self.max_Irr0_mW_per_mm2_viz = None

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        """Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation"""

        if spread:
            # divergence half-angle of cone
            theta_div = np.arcsin(
                self.light_model_params["NAfib"] / self.light_model_params["ntis"]
            )
            Rz = self.light_model_params["R0"] + z * np.tan(
                theta_div
            )  # radius as light spreads ("apparent radius" from original code)
            C = (self.light_model_params["R0"] / Rz) ** 2
        else:
            Rz = self.light_model_params["R0"]  # "apparent radius"
            C = 1

        if gaussian:
            G = 1 / np.sqrt(2 * np.pi) * np.exp(-2 * (r / Rz) ** 2)
        else:
            G = 1

        if scatter:
            S = self.light_model_params["S"]
            a = 1 + self.light_model_params["K"] / S
            b = np.sqrt(a**2 - 1)
            dist = np.sqrt(r**2 + z**2)
            M = b / (a * np.sinh(b * S * dist) + b * np.cosh(b * S * dist))
        else:
            M = 1

        T = G * C * M
        T[z < 0] = 0
        return T

    def _get_rz_for_xyz(self, x, y, z):
        """Assumes x, y, z already have units"""

        # have to add unit back on since it's stripped by vstack
        coords = np.column_stack([x, y, z]) * meter
        rel_coords = coords - self.location  # relative to fiber location
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        zc = usf.dot(rel_coords, self.dir_uvec)  # distance along cylinder axis
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt(
            np.sum((rel_coords - zc[..., np.newaxis] * self.dir_uvec.T) ** 2, axis=1)
        )
        return r, zc

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
            / self.light_model_params["wavelength"]
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
        r, z = self._get_rz_for_xyz(neuron_group.x, neuron_group.y, neuron_group.z)
        T = self._Foutz12_transmittance(r, z).flatten()
        assert len(T) == len(neuron_group)
        # reduce to subset expressing opsin before assigning
        T = T[opto_syn.i]

        opto_syn.T = T

        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    def add_self_to_plot(self, ax, axis_scale_unit, **kwargs) -> PathCollection:
        # show light with point field, assigning r and z coordinates
        # to all points
        # filter out points with <0.001 transmittance to make plotting faster

        T_threshold = kwargs.get("T_threshold", 0.001)
        n_points = kwargs.get("n_points", 1e4)
        intensity = kwargs.get("intensity", 0.5)
        r_thresh, zc_thresh = self._find_rz_thresholds(T_threshold)
        r, theta, zc = uniform_cylinder_rθz(n_points, r_thresh, zc_thresh)

        T = self._Foutz12_transmittance(r, zc)

        end = self.location + zc_thresh * self.dir_uvec
        x, y, z = xyz_from_rθz(r, theta, zc, self.location, end)

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
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
        opto_patch = matplotlib.patches.Patch(color=c, label=self.name)
        handles.append(opto_patch)
        ax.legend(handles=handles)
        return [point_cloud]

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
        c = wavelength_to_rgb(self.light_model_params["wavelength"] / nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 0.6 * intensity)
        return colors.LinearSegmentedColormap.from_list(
            "incr_alpha", [(0, c_clear), (1, c_opaque)]
        )
