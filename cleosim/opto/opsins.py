"""TODO"""
from __future__ import annotations
from abc import ABC, abstractmethod

from brian2 import Synapses, NeuronGroup
from brian2.units import (
    mm2,
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
from brian2.units.allunits import radian
from brian2.core.base import BrianObjectException

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
# TODO: opsin model needs epsilon, way to take into account crosstalk
# -- partial current for non-peak wavelength
# should we set that here with all the other params? Or have it 
# separate and explicit?


class OpsinModel(ABC):
    """Base class for opsin model"""

    model: str
    """Basic Brian model equations string.
    
    Should contain a `rho_rel` term reflecting relative expression 
    levels. Will likely also contain special NeuronGroup-dependent
    symbols such as V_VAR_NAME to be replaced on injection in 
    :meth:`get_modified_model_for_ng`."""

    # TODO: opsin model needs epsilon, way to take into account crosstalk
    # -- partial current for non-peak wavelength
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