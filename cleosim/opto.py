"""Contains opsin models, parameters, and OptogeneticIntervention device"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any

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
import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection

from cleosim.utilities import uniform_cylinder_rθz, wavelength_to_rgb, xyz_from_rθz
from cleosim.stimulators import Stimulator


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
    :meth:`get_modified_model_for_ng`."""

    params: dict
    """Parameter values for model, passed in as a namespace dict"""

    @abstractmethod
    def get_modified_model_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
    ) -> str:
        """Adapt model for given neuron group on injection

        This is mainly to enable the specification of variable names
        differently for each neuron group.

        Parameters
        ----------
        neuron_group : NeuronGroup
            NeuronGroup this opsin model is being connected to
        injct_params : dict
            kwargs passed in on injection, could contain variable
            names to plug into the model

        Returns
        -------
        str
            A string containing modified model equations for use
            in :attr:`~cleosim.opto.OptogeneticIntervention.opto_syns`
        """
        pass

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

    def __init__(self, params: dict) -> None:
        """
        Parameters
        ----------
        params : dict
            dict defining params in the :attr:`model`
        """
        super().__init__()
        self.params = params

    def _check_vars_on_injection(self, neuron_group, injct_params):
        Iopto_var_name = injct_params.get("Iopto_var_name", "Iopto")
        v_var_name = injct_params.get("v_var_name", "v")
        for variable, unit in zip([v_var_name, Iopto_var_name], [volt, amp]):
            if (
                variable not in neuron_group.variables
                or neuron_group.variables[variable].unit != unit
            ):
                raise BrianObjectException(
                    (
                        f"{variable} : {unit.name} needed in the model of NeuronGroup"
                        f"{neuron_group.name} to connect OptogeneticIntervention."
                    ),
                    neuron_group,
                )

    def get_modified_model_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
    ) -> str:
        self._check_vars_on_injection(neuron_group, injct_params)
        Iopto_var_name = injct_params.get("Iopto_var_name", "Iopto")
        v_var_name = injct_params.get("v_var_name", "v")
        # opsin synapse model needs modified names
        return self.model.replace("IOPTO_VAR_NAME", Iopto_var_name).replace(
            "V_VAR_NAME", v_var_name
        )


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
            self._Iopto_dim = Iopto_per_mW_per_mm2.get_best_unit().dim
            self._Iopto_dim_name = self._Iopto_dim._str_representation(python_code=True)
        else:
            self._Iopto_dim = radian.dim
            self._Iopto_dim_name = self._Iopto_dim._str_representation(
                python_code=False
            )

    def get_modified_model_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
    ) -> str:
        Iopto_var_name = injct_params.get("Iopto_var_name", "Iopto")
        if (
            Iopto_var_name not in neuron_group.variables
            or neuron_group.variables[Iopto_var_name].dim != self._Iopto_dim
        ):
            raise BrianObjectException(
                (
                    f"{Iopto_var_name} : {self._Iopto_dim_name} needed in the model of NeuronGroup"
                    f"{neuron_group.name} to connect OptogeneticIntervention."
                ),
                neuron_group,
            )
        return self.model.replace("IOPTO_VAR_NAME", Iopto_var_name).replace(
            "IOPTO_UNIT", self._Iopto_dim_name
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
    :meth:`cleosim.CLSimulator.inject_stimulator`.

    Visualization kwargs
    --------------------
    n_points : int, optional
        The number of points used to represent light intensity in space.
        By default 1e4.
    T_threshold : float, optional
        The transmittance below which no points are plotted. By default
        1e-3.
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
            b = np.sqrt(a**2 - 1)
            dist = np.sqrt(r**2 + z**2)
            return b / (a * np.sinh(b * S * dist) + b * np.cosh(b * S * dist))

        M = kubelka_munk(np.sqrt(r**2 + z**2)) if scatter else 1

        T = G * C * M
        T[z < 0] = 0
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
        r = np.sqrt(np.sum((rel_coords - zc[..., np.newaxis] * self.dir_uvec.T)**2, axis=1))
        r = r.reshape((-1, 1))
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
        r, z = self._get_rz_for_xyz(neuron_group.x, neuron_group.y, neuron_group.z)
        T = self._Foutz12_transmittance(r, z).flatten()
        # reduce to subset expressing opsin before assigning
        T = [T[k] for k in opto_syn.i]

        opto_syn.T = T

        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    def add_self_to_plot(self, ax, axis_scale_unit, **kwargs) -> PathCollection:
        # show light with point field, assigning r and z coordinates
        # to all points
        # filter out points with <0.001 transmittance to make plotting faster

        fiber_T_thresh = kwargs.get('fiber_T_thresh', 0.001)
        n_points = kwargs.get('n_points', 1e4)
        r_thresh, zc_thresh = self._find_rz_thresholds(fiber_T_thresh)
        r, theta, zc = uniform_cylinder_rθz(n_points, r_thresh, zc_thresh)

        T = self._Foutz12_transmittance(r, zc)

        end = self.location + zc_thresh*self.dir_uvec
        x, y, z = xyz_from_rθz(r, theta, zc, self.location, end)

        idx_to_plot = T >= fiber_T_thresh
        x = x[idx_to_plot]
        y = y[idx_to_plot]
        z = z[idx_to_plot]
        T = T[idx_to_plot]
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

    def _find_rz_thresholds(self, thresh):
        """find r and z thresholds for visualization purposes"""
        res_mm = 0.1
        zc = np.arange(20, 0, -res_mm) * mm  # ascending T
        T = self._Foutz12_transmittance(0 * mm, zc)
        zc_thresh = zc[np.searchsorted(T, thresh)]
        # look at half the z threshold for the r threshold
        r = np.arange(20, 0, -res_mm) * mm
        T = self._Foutz12_transmittance(r, zc_thresh / 2)
        r_thresh = r[np.searchsorted(T, thresh)]
        return r_thresh, zc_thresh


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
