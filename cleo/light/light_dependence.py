from __future__ import annotations
from typing import Callable, Tuple
import warnings

from attrs import define, field, asdict, fields_dict
from brian2 import (
    np,
    Synapses,
    Function,
    NeuronGroup,
    Unit,
    BrianObjectException,
    get_unit,
    Equations,
    implementation,
    check_units,
)
from nptyping import NDArray
from brian2.units import (
    mm,
    mm2,
    nmeter,
    Quantity,
    second,
    ms,
    second,
    psiemens,
    nsiemens,
    mV,
    volt,
    amp,
    mM,
)
from brian2.units.allunits import radian
import numpy as np
from scipy.interpolate import CubicSpline

from cleo.base import InterfaceDevice
from cleo.coords import assign_coords
from cleo.registry import registry_for_sim
from cleo.utilities import wavelength_to_rgb


def linear_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return np.interp(lambda_new_nm, lambdas_nm, epsilons)


def cubic_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return CubicSpline(lambdas_nm, epsilons)(lambda_new_nm)


@define(eq=False)
class LightDependentDevice(InterfaceDevice):
    """Base class for opsin and indicator. TODO

    We approximate dynamics under multiple wavelengths using a weighted sum
    of photon fluxes, where the ε factor indicates the activation
    relative to the peak-sensitivy wavelength for an equivalent number of photons
    (see Mager et al, 2018). This weighted sum is an approximation of a nonlinear
    peak-non-peak wavelength relation; see ``notebooks/multi_wavelength_model.ipynb``
    for details."""

    model: str = field(init=False)
    """Basic Brian model equations string.
    
    Should contain a `rho_rel` term reflecting relative expression 
    levels. Will likely also contain special NeuronGroup-dependent
    symbols such as V_VAR_NAME to be replaced on injection in 
    :meth:`modify_model_and_params_for_ng`."""

    per_ng_unit_replacements: list[Tuple[str, str]] = field(
        factory=list, init=False, repr=False
    )
    """List of (UNIT_NAME, neuron_group_specific_unit_name) tuples to be substituted
    in the model string on injection and before checking required variables."""

    required_vars: list[Tuple[str, Unit]] = field(factory=list, init=False, repr=False)
    """Default names of state variables required in the neuron group,
    along with units, e.g., [('Iopto', amp)].
    
    It is assumed that non-default values can be passed in on injection
    as a keyword argument ``[default_name]_var_name=[non_default_name]``
    and that these are found in the model string as 
    ``[DEFAULT_NAME]_VAR_NAME`` before replacement."""

    light_agg_ngs: dict[str, NeuronGroup] = field(factory=dict, init=False, repr=False)
    """{target_ng.name: light_agg_ng} dict of light aggregator neuron groups."""

    ldsyns: dict[NeuronGroup, Synapses] = field(factory=dict, init=False, repr=False)
    """Stores the synapse objects implementing the model, connecting from
    :attr:`light_agg_ngs` to target neuron groups, with NeuronGroup keys and Synapse
    values."""

    spectrum: list[tuple[float, float]] = field(factory=lambda: [(-1e10, 1), (1e10, 1)])
    """List of (wavelength, epsilon) tuples representing the action (opsin) or
    excitation (indicator) spectrum."""

    action_spectrum_interpolator: Callable = field(
        default=cubic_interpolator, repr=False
    )
    """Function of signature (lambdas_nm, epsilons, lambda_new_nm) that interpolates
    the action spectrum data and returns :math:`\\varepsilon \\in [0,1]` for the new
    wavelength."""

    extra_namespace: dict = field(factory=dict, repr=False)
    """Additional items (beyond parameters) to be added to the opto synapse namespace"""

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Transfect neuron group with device.

        Parameters
        ----------
        neuron_group : NeuronGroup
            The neuron group to transform

        Keyword args
        ------------
        p_expression : float
            Probability (0 <= p <= 1) that a given neuron in the group
            will express the protein. 1 by default.
        i_targets : array-like
            Indices of neurons in the group to transfect. recommended for efficiency
            when stimulating or imaging a small subset of the group.
            Incompatible with ``p_expression``.
        rho_rel : float
            The expression level, relative to the standard model fit,
            of the protein. 1 by default. For heterogeneous expression,
            this would have to be modified in the light-dependent synapse
            post-injection, e.g., ``opto.ldsyns["neuron_group_name"].rho_rel = ...``
        Iopto_var_name : str
            The name of the variable in the neuron group model representing
            current from the opsin
        v_var_name : str
            The name of the variable in the neuron group model representing
            membrane potential
        """
        if neuron_group.name in self.light_agg_ngs:
            assert neuron_group.name in self.ldsyns
            raise ValueError(
                f"{self.__class__.__name__} {self.name} already connected to neuron group"
                f" {neuron_group.name}"
            )

        # get modified synapse model string (i.e., with names/units specified)
        mod_ldsyn_model, mod_ldsyn_params = self.modify_model_and_params_for_ng(
            neuron_group, kwparams
        )

        # handle p_expression
        if "p_expression" in kwparams:
            if "i_targets" in kwparams:
                raise ValueError("p_expression and i_targets are incompatible")
            p_expression = kwparams.get("p_expression", 1)
            expr_bool = np.random.rand(neuron_group.N) < p_expression
            i_targets = np.where(expr_bool)[0]
        elif "i_targets" in kwparams:
            i_targets = kwparams["i_targets"]
        else:
            i_targets = list(range(neuron_group.N))
        if len(i_targets) == 0:
            return

        # create light aggregator neurons
        light_agg_ng = NeuronGroup(
            len(i_targets),
            model="""
            phi : 1/second/meter**2
            Irr : watt/meter**2
            """,
            name=f"light_agg_{self.name}_{neuron_group.name}",
        )
        assign_coords(
            light_agg_ng,
            neuron_group.x[i_targets] / mm,
            neuron_group.y[i_targets] / mm,
            neuron_group.z[i_targets] / mm,
            unit=mm,
        )

        ldsyn = Synapses(
            light_agg_ng,
            neuron_group,
            model=mod_ldsyn_model,
            namespace=mod_ldsyn_params,
            name=f"ldsyn_{self.name}_{neuron_group.name}",
        )
        ldsyn.namespace.update(self.extra_namespace)
        ldsyn.connect(i=range(len(i_targets)), j=i_targets)
        self.init_opto_syn_vars(ldsyn)
        # relative protein density
        ldsyn.rho_rel = kwparams.get("rho_rel", 1)

        # store at the end, after all checks have passed
        self.light_agg_ngs[neuron_group.name] = light_agg_ng
        self.brian_objects.add(light_agg_ng)
        self.ldsyns[neuron_group.name] = ldsyn
        self.brian_objects.add(ldsyn)

        registry = registry_for_sim(self.sim)
        registry.register_ldd(self, neuron_group)

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
            in :attr:`~cleo.opto.OptogeneticIntervention.ldsyns`
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
        params = asdict(self, recurse=False)
        # remove generic fields that are not parameters
        for field in fields_dict(LightDependentDevice):
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
        for opto_syn in self.ldsyns.values():
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

    def epsilon(self, lambda_new) -> float:
        """Returns the epsilon value for a given lambda (in nm)
        representing the relative sensitivity of the opsin to that wavelength."""
        action_spectrum = np.array(self.spectrum)
        lambdas = action_spectrum[:, 0]
        epsilons = action_spectrum[:, 1]
        if lambda_new < min(lambdas) or lambda_new > max(lambdas):
            warnings.warn(
                f"λ = {lambda_new} nm is outside the range of the action spectrum data"
                f" for {self.name}. Assuming ε = 0."
            )
            return 0
        return self.action_spectrum_interpolator(lambdas, epsilons, lambda_new)


def plot_spectra(*ldds: LightDependentDevice):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for ldd in ldds:
        spectrum = np.array(ldd.spectrum)
        lambdas = spectrum[:, 0]
        epsilons = spectrum[:, 1]
        lambdas_new = np.linspace(min(lambdas), max(lambdas), 100)
        epsilons_new = ldd.action_spectrum_interpolator(lambdas, epsilons, lambdas_new)
        c_points = [wavelength_to_rgb(l) for l in lambdas]
        c_line = wavelength_to_rgb(lambdas_new[np.argmax(epsilons_new)])
        ax.plot(lambdas_new, epsilons_new, c=c_line, label=ldd.name)
        ax.scatter(lambdas, epsilons, marker="o", s=50, color=c_points)
    title = (
        "Action/excitation spectra" if len(ldds) > 1 else f"Action/excitation spectrum"
    )
    ax.set(xlabel="λ (nm)", ylabel="ε", title=title)
    fig.legend()
    return fig, ax
