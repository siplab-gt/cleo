"""Contains opsin models and default parameters"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any
import warnings

from attrs import define, field, asdict, fields_dict
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
from cleo.coords import assign_coords, coords_from_ng
from cleo.opto.registry import lor_for_sim
from cleo.utilities import (
    uniform_cylinder_rθz,
    wavelength_to_rgb,
    xyz_from_rθz,
    normalize_coords,
)
from cleo.stimulators import Stimulator


@define(eq=False)
class Opsin(InterfaceDevice):
    """Base class for opsin model.

    We approximate dynamics under multiple wavelengths using a weighted sum
    of photon fluxes, where the :math:`\\varepsilon` factor indicates the activation
    relative to the peak-sensitivy wavelength for an equivalent number of photons
    (see Mager et al, 2018). This weighted sum is an approximation of a nonlinear
    peak-non-peak wavelength relation; see notebooks/multi_wavelength_model.ipynb
    for details."""

    model: str = field(init=False)
    """Basic Brian model equations string.
    
    Should contain a `rho_rel` term reflecting relative expression 
    levels. Will likely also contain special NeuronGroup-dependent
    symbols such as V_VAR_NAME to be replaced on injection in 
    :meth:`~OpsinModel.modify_model_and_params_for_ng`."""

    per_ng_unit_replacements: list[Tuple[str, str]] = field(factory=list, init=False)
    """List of (UNIT_NAME, neuron_group_specific_unit_name) tuples to be substituted
    in the model string on injection and before checking required variables."""

    required_vars: list[Tuple[str, Unit]] = field(factory=list, init=False)
    """Default names of state variables required in the neuron group,
    along with units, e.g., [('Iopto', amp)].
    
    It is assumed that non-default values can be passed in on injection
    as a keyword argument ``[default_name]_var_name=[non_default_name]``
    and that these are found in the model string as 
    ``[DEFAULT_NAME]_VAR_NAME`` before replacement."""

    light_agg_ngs: dict[NeuronGroup, NeuronGroup] = field(factory=dict, init=False)
    """{target_ng: light_agg_ng} dict of light aggregator neuron groups."""

    opto_syns: dict[NeuronGroup, Synapses] = field(factory=dict, init=False)
    """Stores the synapse objects implementing the opsin model,
    with NeuronGroup keys and Synapse values."""

    epsilon: Callable = field(default=(lambda wl: 1))
    """A callable that returns a value between 0 and 1 when given a wavelength
    representing the relative sensitivity of the opsin to that wavelength."""

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Transfect neuron group with opsin.

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
        if neuron_group.name in self.light_agg_ngs:
            assert neuron_group.name in self.opto_syns
            raise ValueError(
                f"Opsin {self.name} already connected to neuron group {neuron_group.name}"
            )

        # get modified opsin model string (i.e., with names/units specified)
        mod_opsin_model, mod_opsin_params = self.modify_model_and_params_for_ng(
            neuron_group, kwparams
        )

        # handle p_expression
        p_expression = kwparams.get("p_expression", 1)
        i_expression_bool = np.random.rand(neuron_group.N) < p_expression
        i_expression = np.where(i_expression_bool)[0]
        if len(i_expression) == 0:
            return

        # create light aggregator neurons
        light_agg_ng = NeuronGroup(
            len(i_expression),
            model="""
            phi : 1/second/meter**2
            Irr : watt/meter**2
            """,
            name=f"light_agg_{self.name}_{neuron_group.name}",
        )
        assign_coords(
            light_agg_ng,
            neuron_group.x[i_expression] / mm,
            neuron_group.y[i_expression] / mm,
            neuron_group.z[i_expression] / mm,
            unit=mm,
        )

        # create opsin synapses
        opto_syn = Synapses(
            light_agg_ng,
            neuron_group,
            model=mod_opsin_model,
            namespace=mod_opsin_params,
            name=f"opto_syn_{self.name}_{neuron_group.name}",
        )
        opto_syn.connect(i=range(len(i_expression)), j=i_expression)
        self.init_opto_syn_vars(opto_syn)
        # relative channel density
        opto_syn.rho_rel = kwparams.get("rho_rel", 1)

        lor = lor_for_sim(self.sim)
        lor.register_opsin(self, neuron_group)

        # store at the end, after all checks have passed
        self.light_agg_ngs[neuron_group.name] = light_agg_ng
        self.brian_objects.add(light_agg_ng)
        self.opto_syns[neuron_group.name] = opto_syn
        self.brian_objects.add(opto_syn)

    def modify_model_and_params_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
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
        model = self.model

        # perform unit substitutions
        for unit_name, neuron_group_unit_name in self.per_ng_unit_replacements:
            model = model.replace(unit_name, neuron_group_unit_name)

        # check required variables/units and replace placeholder names
        for default_name, unit in self.required_vars:
            var_name = injct_params.get(f"{default_name}_var_name", default_name)
            if var_name not in neuron_group.variables or not neuron_group.variables[
                var_name
            ].unit.has_same_dimensions(unit):
                raise BrianObjectException(
                    (
                        f"{var_name} : {unit.name} needed in the model of NeuronGroup "
                        f"{neuron_group.name} to connect Opsin {self.name}."
                    ),
                    neuron_group,
                )
            # opsin synapse model needs modified names
            to_replace = f"{default_name}_var_name".upper()
            model = model.replace(to_replace, var_name)

        # Synapse variable and parameter names cannot be the same as any
        # neuron group variable name
        return self._fix_name_conflicts(model, neuron_group)

    @property
    def params(self) -> dict:
        """Returns a dictionary of parameters for the model"""
        params = asdict(self)
        # remove generic fields that are not parameters
        for field in fields_dict(Opsin):
            params.pop(field)
        # remove private attributes
        for key in list(params.keys()):
            if key.startswith("_"):
                params.pop(key)
        return params

    def _fix_name_conflicts(
        self, modified_model: str, neuron_group: NeuronGroup
    ) -> Tuple[Equations, dict]:
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

    def reset(self, **kwargs):
        for opto_syn in self.opto_syns.values():
            self.init_opto_syn_vars(opto_syn)

    def init_opto_syn_vars(self, opto_syn: Synapses) -> None:
        """Initializes appropriate variables in Synapses implementing the model

        Can also be used to reset the variables.

        Parameters
        ----------
        opto_syn : Synapses
            The synapses object implementing this model
        """
        pass


@define(eq=False)
class MarkovOpsin(Opsin):
    """Base class for Markov state models à la Evans et al., 2016"""

    required_vars: list[Tuple[str, Unit]] = field(
        factory=lambda: [("Iopto", amp), ("v", volt)],
        init=False,
    )


@define(eq=False)
class FourStateOpsin(MarkovOpsin):
    """4-state model from PyRhO (Evans et al. 2016).

    rho_rel is channel density relative to standard model fit;
    modifying it post-injection allows for heterogeneous opsin expression.

    IOPTO_VAR_NAME and V_VAR_NAME are substituted on injection.

    Defaults are for ChR2.
    """

    g0: Quantity = 114000 * psiemens
    gamma: Quantity = 0.00742
    phim: Quantity = 2.33e17 / mm2 / second  # *photon, not in Brian2
    k1: Quantity = 4.15 / ms
    k2: Quantity = 0.868 / ms
    p: Quantity = 0.833
    Gf0: Quantity = 0.0373 / ms
    kf: Quantity = 0.0581 / ms
    Gb0: Quantity = 0.0161 / ms
    kb: Quantity = 0.063 / ms
    q: Quantity = 1.94
    Gd1: Quantity = 0.105 / ms
    Gd2: Quantity = 0.0138 / ms
    Gr0: Quantity = 0.00033 / ms
    E: Quantity = 0 * mV
    v0: Quantity = 43 * mV
    v1: Quantity = 17.1 * mV
    model: str = field(
        init=False,
        default="""
        dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1 (clock-driven)
        dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gf)*O1 : 1 (clock-driven)
        dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1 (clock-driven)
        C2 = 1 - C1 - O1 - O2 : 1

        Theta = int(phi_pre > 0*phi_pre) : 1
        Hp = Theta * phi_pre**p/(phi_pre**p + phim**p) : 1
        Ga1 = k1*Hp : hertz
        Ga2 = k2*Hp : hertz
        Hq = Theta * phi_pre**q/(phi_pre**q + phim**q) : 1
        Gf = kf*Hq + Gf0 : hertz
        Gb = kb*Hq + Gb0 : hertz

        fphi = O1 + gamma*O2 : 1
        fv = (1 - exp(-(V_VAR_NAME_post-E)/v0)) / -2 : 1

        IOPTO_VAR_NAME_post = -g0*fphi*fv*(V_VAR_NAME_post-E)*rho_rel : ampere (summed)
        rho_rel : 1""",
    )

    def init_opto_syn_vars(self, opto_syn: Synapses) -> None:
        for varname, value in {"C1": 1, "O1": 0, "O2": 0}.items():
            setattr(opto_syn, varname, value)


def ChR2_four_state(
    g0=114000 * psiemens,
    gamma=0.00742,
    phim=2.33e17 / mm2 / second,  # *photon, not in Brian2
    k1=4.15 / ms,
    k2=0.868 / ms,
    p=0.833,
    Gf0=0.0373 / ms,
    kf=0.0581 / ms,
    Gb0=0.0161 / ms,
    kb=0.063 / ms,
    q=1.94,
    Gd1=0.105 / ms,
    Gd2=0.0138 / ms,
    Gr0=0.00033 / ms,
    E=0 * mV,
    v0=43 * mV,
    v1=17.1 * mV,
    name="ChR2",
) -> FourStateOpsin:
    """Returns a 4-state ChR2 model.

    Params taken from try.projectpyrho.org's default 4-state configuration.
    """
    return FourStateOpsin(
        g0=g0,
        gamma=gamma,
        phim=phim,
        k1=k1,
        k2=k2,
        p=p,
        Gf0=Gf0,
        kf=kf,
        Gb0=Gb0,
        kb=kb,
        q=q,
        Gd1=Gd1,
        Gd2=Gd2,
        Gr0=Gr0,
        E=E,
        v0=v0,
        v1=v1,
        name=name,
    )


@define(eq=False)
class ProportionalCurrentOpsin(Opsin):
    """A simple model delivering current proportional to light intensity"""

    I_per_Irr: Quantity = field(kw_only=True)
    """ How much current (in amps or unitless, depending on neuron model)
    to deliver per mW/mm2.
    """
    # would be IOPTO_UNIT but that throws off Equation parsing
    model: str = field(
        init=False,
        default="""
            IOPTO_VAR_NAME_post = I_per_Irr / (mwatt / mm2) 
                * Irr_pre * rho_rel : IOPTO_UNIT (summed)
            rho_rel : 1
        """,
    )

    required_vars: list[Tuple[str, Unit]] = field(factory=list, init=False)

    def __attrs_post_init__(self):
        if isinstance(self.I_per_Irr, Quantity):
            Iopto_unit = get_unit(self.I_per_Irr.dim)
        else:
            Iopto_unit = radian
        self.per_ng_unit_replacements = [("IOPTO_UNIT", Iopto_unit.name)]
        self.required_vars = [("Iopto", Iopto_unit)]
