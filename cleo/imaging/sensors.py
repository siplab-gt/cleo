from __future__ import annotations

from attrs import define, field
from brian2 import np, Quantity, NeuronGroup, second, umolar, nmolar

from cleo.base import SynapseDevice
from cleo.light import LightDependent


@define(eq=False)
class Sensor(SynapseDevice):
    """Base class for sensors"""

    snr: float = field(kw_only=True)
    location: str = field(kw_only=True)
    """cytoplasm or membrane"""

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        """Returns a list of arrays in the order neuron groups/targets were received.

        Signals should be normalized to baseline of 0 and 1 corresponding
        to an action potential peak."""
        pass

    @property
    def exc_spectrum(self) -> list[tuple[float, float]]:
        """Excitation spectrum, alias for :attr:`spectrum`"""
        return self.spectrum


@define(eq=False)
class BaseGECI(Sensor):
    on_pre: str = ""
    # Ca is free, intracellular calcium concentration
    cal_model: str = field(default="Ca = Ca_pre : molar", init=False)
    fluor_model: str = field(
        default="""
            CaB_active = Ca : 1
            dFF = 1 / (1 + (K_d / CaB_active) ** n_H) : 1
        """,
        init=False,
    )
    location: str = field(default="cytoplasm", init=False)

    K_d: Quantity = field(kw_only=True)
    """indicator dissociation constant (binding affinity) (molar)"""
    n_H: float = field(kw_only=True)
    """Hill coefficient for conversion from Ca2+ to ΔF/F"""
    dFF_max: float = field(kw_only=True)
    """amplitude of Hill equation for conversion from Ca2+ to ΔF/F,
    Fmax/F0. May be approximated from 'dynamic range' in literature Fmax/Fmin"""
    h_ca_amp: float = field(kw_only=True)
    """amplitude of double exponential kernel representing CaB binding/activation"""
    h_ca_tau_on: Quantity = field(kw_only=True)
    """CaB binding/activation time constant (sec)"""
    h_ca_tau_off: Quantity = field(kw_only=True)
    """CaB unbinding/deactivation time constant (sec)"""

    def __attrs_post_init__(self):
        self.model = self.cal_model + "\n" + self.fluor_model

    # @property
    # def model(self) -> str:
    #     return self.cal_model + self.fluor_model


@define(eq=False)
class LightGECI(LightDependent, BaseGECI):
    fluor_model: str = field(
        default="""
            strength = some_function(Irr_pre) : 1
            dFF = strength * ...""",
        init=False,
    )


@define(eq=False, slots=False)
class HasCalDynamicsModel:
    """Pieced together from Lütke et al., 2013; Helmchen and Tank, 2015;
    and Song et al., 2021. (`code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_)
    """

    on_pre: str = field(default="Ca += dCa_T / (1 + kappa_S + kappa_B)", init=False)
    # eq 8 in Lütke 2013
    cal_model: str = field(
        default="""
            dCa/dt = -gamma * (Ca - Ca_rest) / (1 + kappa_S + kappa_B) : mmolar
            kappa_B = B_T * K_d / (Ca + K_d)**2 : 1""",
        init=False,
    )

    Ca_rest: Quantity = field(kw_only=True)
    """resting Ca2+ concentration (molar)"""
    gamma: Quantity = field(kw_only=True)
    """clearance/extrusion rate (1/sec)"""
    B_T: Quantity = field(kw_only=True)
    """total indicator (buffer) concentration (molar)"""
    kappa_S: float = field(kw_only=True)
    """Ca2+ binding ratio"""
    dCa_T: Quantity = field(kw_only=True)
    """total Ca2+ concentration increase per spike (molar)"""

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        return {ng: syn["dFF"] for ng, syn in self.synapses.items()}


@define(eq=False)
class CalDynamicGECI(HasCalDynamicsModel, BaseGECI):
    pass


@define(eq=False)
class CalDynamicLightGECI(HasCalDynamicsModel, LightGECI):
    pass


def geci(light_dependent, pre_existing_cal, **kwparams):
    GECI = {
        (False, False): CalDynamicGECI,
        (True, False): CalDynamicLightGECI,
        (False, True): BaseGECI,
        (True, True): LightGECI,
    }[light_dependent, pre_existing_cal]

    return GECI(
        **kwparams,
    )


def _create_geci_fn(K_d, n_H, dFF_max):
    def geci_fn(
        light_dependent=False,
        pre_existing_cal=False,
        K_d=K_d,
        n_H=n_H,
        dFF_max=dFF_max,
        Ca_rest=50 * nmolar,
        kappa_S=110,
        gamma=292.3 / second,
        B_T=10 * umolar,
        dCa_T=7.6 * umolar,  # Lütke et al., 2013 and NAOMi code
        **kwparams,
    ):
        """Returns a GECI model with parameters given in
        `NAOMi's code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_
        (Song et al., 2021).

        Latter 5 params apply only when simulating the calcium dynamics.
        """
        return geci(
            light_dependent,
            pre_existing_cal,
            K_d=K_d,
            n_H=n_H,
            dFF_max=dFF_max,
            Ca_rest=Ca_rest,
            kappa_S=kappa_S,
            gamma=gamma,
            B_T=B_T,
            dCa_T=dCa_T,
            **kwparams,
        )

    return geci_fn


# from NAOMi:
gcamp6f = _create_geci_fn(290 * nmolar, 2.7, 25.2)
gcamp6s = _create_geci_fn(147 * nmolar, 2.45, 27.2)
gcamp3 = _create_geci_fn(287 * nmolar, 2.52, 12)
ogb1 = _create_geci_fn(250 * nmolar, 1, 14)
gcamp6rs09 = _create_geci_fn(520 * nmolar, 3.2, 25)
gcamp6rs06 = _create_geci_fn(320 * nmolar, 3, 15)
jgcamp7f = _create_geci_fn(174 * nmolar, 2.3, 30.2)
jgcamp7s = _create_geci_fn(68 * nmolar, 2.49, 40.4)
jgcamp7b = _create_geci_fn(82 * nmolar, 3.06, 22.1)
jgcamp7c = _create_geci_fn(298 * nmolar, 2.44, 145.6)
