from __future__ import annotations

from attrs import define, field
from brian2 import np, Synapses, NeuronGroup

from cleo.base import SynapseDevice
from cleo.light import LightReceptor


@define(eq=False)
class Sensor(SynapseDevice):
    """Base class for sensors"""

    snr: float = field(kw_only=True)
    location: str = field(kw_only=True)
    """cytoplasm or membrane"""
    light_receptor: LightReceptor = field(kw_only=True, default=None)

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
class GECI(Sensor):
    """Pieced together from Lütke et al., 2013; Helmchen and Tank, 2015;
    and Song et al., 2021. (`code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_))
    """

    on_pre: str = "Ca += ΔCa_T / (1 + κ_S + κ_B)"

    Ca_rest: Quantity = field(kw_only=True)
    """resting Ca2+ concentration (molar)"""
    γ: Quantity = field(kw_only=True)
    """clearance/extrusion rate (1/sec)"""
    B_T: Quantity = field(kw_only=True)
    """total indicator (buffer) concentration (molar)"""
    K_d: Quantity = field(kw_only=True)
    """indicator dissociation constant (molar)"""
    κ_S: float = field(kw_only=True)
    """Ca2+ binding ratio"""
    ΔCa_T: Quantity = field(kw_only=True)
    """total Ca2+ concentration increase per spike (molar)"""

    def get_state(self) -> dict[NeuronGroup, ndarray]:
        return {ng: syn["ΔF_F"] for ng, syn in self.synapses.items()}


# @define(eq=False)
# class LightDependentDynamicGECI(DynamicGECI, LightReceptor):
#     model: str = (
#         _dynamic_geci_model
#         + """
#         strength = some_function(Irr_pre): 1
#         dF_F = strength * ...
#     """
#     )


def geci(snr, light_dependent=False, pre_existing_cal=False, **kwparams):
    if pre_existing_cal:
        cal_model = """
            Ca = CA_VAR_NAME_post : molar"""
    else:
        cal_model = """
            # eq 8 in Lütke 2013
            # Ca is free, intracellular calcium concentration
            dCa/dt = -γ * (Ca - Ca_rest) / (1 + κ_S + κ_B)  : molar
            κ_B = B_T * K_d / (Ca + K_d)**2 : 1 """

    if light_dependent:
        fluor_model = """
            strength = some_function(Irr_pre): 1
            dF_F = strength * ..."""
        light_receptor = LightReceptor(...)
    else:
        fluor_model = """
            ΔF_F = ... : 1"""
        light_receptor = None

    return GECI(
        model=cal_model + fluor_model,
        on_pre=on_pre,
        snr=snr,
        location="cytoplasm",
        light_receptor=light_receptor,
        **kwparams,
    )


def jgcamp7f(light_dependent=False):
    GECI = LightDependentDynamicGECI if light_dependent else SimpleDynamicGECI
    return GECI(snr=2, location="cytoplasm")
