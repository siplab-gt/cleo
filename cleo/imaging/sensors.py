from __future__ import annotations

from attrs import define, field, fields_dict, asdict
from brian2 import np, Quantity, NeuronGroup, second, umolar, nmolar

from cleo.base import SynapseDevice
from cleo.light import LightDependent


@define(eq=False)
class Sensor(SynapseDevice):
    """Base class for sensors"""

    sigma_noise: float = field(kw_only=True)
    """standard deviation of Gaussian noise in ΔF/F measurement"""
    dFF_1AP: float = field(kw_only=True)
    """ΔF/F for 1 AP, only used for scope SNR cutoff"""
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
class CalciumModel:
    """Base class for how GECI computes calcium concentration

    Must provide variable Ca (molar) in model."""

    on_pre: str = field(default="", init=False)
    model: str


@define(eq=False)
class PreexistingCalcium(CalciumModel):
    """Calcium concentration is pre-existing in neuron model"""

    model: str = field(default="Ca = Ca_pre : mmolar", init=False)


@define(eq=False)
class DynamicCalcium(CalciumModel):
    """Pieced together from Lütke et al., 2013; Helmchen and Tank, 2015;
    and Song et al., 2021. (`code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_)
    """

    on_pre: str = field(default="Ca += dCa_T / (1 + kappa_S + kappa_B)", init=False)
    """from eq 9 in Lütke et al., 2013"""
    model: str = field(
        default="""
            dCa/dt = -gamma * (Ca - Ca_rest) / (1 + kappa_S + kappa_B) : mmolar (clock-driven)
            kappa_B = B_T * K_d / (Ca + K_d)**2 : 1""",
        init=False,
    )
    """from eq 8 in Lütke et al., 2013"""

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


@define(eq=False)
class CalBindingActivationModel:
    """Base class for modeling calcium binding/activation"""

    model: str


@define(eq=False)
class NullBindingActivation(CalBindingActivationModel):
    """Doesn't model binding/activation; i.e., goes straight from [Ca2+] to ΔF/F"""

    model: str = field(default="CaB_active = Ca", init=False)


@define(eq=False)
class DoubExpCalBindingActivation(CalBindingActivationModel):
    """Double exponential kernel convolution representing CaB binding/activation.

    Convolution is implemented via ODEs; see ``notebooks/double_exp_conv_as_ode.ipynb``
    for derivation.

    Some parameters found `here <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/check_cal_params.m?at=master#lines-90>`_.
    Parameters fit `here <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/MiscCode/fit_NAOMi_calcium.m?at=master#fit_NAOMi_calcium.m>`_.
    """

    on_pre: str = field(default="", init=False)
    model: str = field(
        default="""
            dCaB_active/dt = beta : mmolar (clock-driven)
            lam = 1/t_off + 1/t_on : 1/second
            kap = 1/t_off : 1/second
            dbeta/dt = (                    # should be M/s/s
                ca_amp * (lam - kap) * Ca   # M/s
                - (kap + lam) * beta        # M/s/s
                - kap * lam * CaB_active    # M/s/s
            ) : mmolar/second (clock-driven)
            """,
        init=False,
    )

    ca_amp: float = field(kw_only=True)
    """amplitude of double exponential kernel"""
    t_on: Quantity = field(kw_only=True)
    """CaB binding/activation time constant (sec)"""
    t_off: Quantity = field(kw_only=True)
    """CaB unbinding/deactivation time constant (sec)"""


@define(eq=False)
class ExcitationModel:
    """Defines ``exc_factor``"""

    model: str


@define(eq=False)
class NullExcitation(ExcitationModel):
    model: str = field(default="exc_factor = 1 : 1", init=False)


@define(eq=False)
class LightExcitation(ExcitationModel):
    model: str = field(default="exc_factor = some_function(Irr_pre) : 1", init=False)


@define(eq=False)
class GECI(Sensor):
    """GECI model based on Song et al., 2021, with interchangeable components.

    See :func:`geci` for a convenience function for creating GECI models.

    As a potentially simpler alternative for future work, see the phenomological S2F model
    from `Zhang et al., 2023 <https://www.nature.com/articles/s41586-023-05828-9>`_.
    While parameter count looks similar, at least they have parameters fit already, and directly
    to data, rather than to biophysical processes before the data.
    """

    location: str = field(default="cytoplasm", init=False)

    cal_model: CalciumModel = field(kw_only=True)
    bind_act_model: CalBindingActivationModel = field(kw_only=True)
    exc_model: ExcitationModel = field(kw_only=True)

    fluor_model: str = field(
        default="""
            dFF = exc_factor * rho_rel * dFF_max / (1 + (K_d / CaB_active) ** n_H) : 1
            rho_rel : 1
        """,
        init=False,
    )
    """Uses a Hill equation to convert from Ca2+ to ΔF/F, as in Song et al., 2021"""
    K_d: Quantity = field(kw_only=True)
    """indicator dissociation constant (binding affinity) (molar)"""
    n_H: float = field(kw_only=True)
    """Hill coefficient for conversion from Ca2+ to ΔF/F"""
    dFF_max: float = field(kw_only=True)
    """amplitude of Hill equation for conversion from Ca2+ to ΔF/F,
    Fmax/F0. May be approximated from 'dynamic range' in literature Fmax/Fmin"""

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        return {ng_name: syn.dFF for ng_name, syn in self.synapses.items()}

    def __attrs_post_init__(self):
        self.model = "\n".join(
            [
                self.cal_model.model,
                self.bind_act_model.model,
                self.exc_model.model,
                self.fluor_model,
            ]
        )

    @property
    def params(self) -> dict:
        """Returns a dictionary of parameters for the model"""
        params = asdict(self, recurse=False)
        # remove generic fields that are not parameters
        for field in fields_dict(Sensor):
            params.pop(field)
        # remove private attributes
        for key in list(params.keys()):
            if key.startswith("_"):
                params.pop(key)
        # add params from sub-models
        for model in [self.cal_model, self.bind_act_model, self.exc_model]:
            to_add = asdict(model, recurse=False)
            to_add.pop("model")
            params.update(to_add)
        return params

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio for 1 AP"""
        return self.dFF_1AP / self.sigma_noise


class LightDependentGECI(GECI, LightDependent):
    pass


def geci(light_dependent, doub_exp_conv, pre_existing_cal, **kwparams):
    ExcModel = LightExcitation if light_dependent else NullExcitation
    CalModel = PreexistingCalcium if pre_existing_cal else DynamicCalcium
    BAModel = DoubExpCalBindingActivation if doub_exp_conv else NullBindingActivation

    subset = lambda to_keep: {k: kwparams[k] for k in to_keep}

    def init_from_kwparams(model):
        kwparams_to_keep = {}
        for field_name, field in fields_dict(model).items():
            if field.init and field_name in kwparams:
                kwparams_to_keep[field_name] = kwparams.pop(field_name)
        return model(**kwparams_to_keep)

    GECIClass = LightDependentGECI if light_dependent else GECI

    return GECIClass(
        cal_model=init_from_kwparams(CalModel),
        exc_model=init_from_kwparams(ExcModel),
        bind_act_model=init_from_kwparams(BAModel),
        **kwparams,
    )


def _create_geci_fn(
    name,
    K_d,
    n_H,
    dFF_max,
    sigma_noise_rel,
    dFF_1AP_rel=None,
    ca_amp=None,
    t_on=None,
    t_off=None,
    extra_doc="",
):
    """convenience function for creating GECI model functions.

    K_d is in nM.
    ca_amp is actually 1/sec, to cancel out time in convolution
    t_on, t_off are in 1/100 sec, as in NAOMi's code.

    dFF_1AP and sigma_noise are relative to GCaMP6s, since noise values obtained indirectly
    from Zhang et al., 2023, supp table 1, which is all relative to GCaMP6s.
    We take ΔF/F / SNR for 1 AP to get a measure of noise.

    We could have done something similar with supp table 1 from Dana 2019,
    but we assume measurements are more accurate from the later paper (as the paper explicitly states).
    Thus, we only use Dana 2019 for the absolute GCaMP6s measurement to scale
    the relative values given in Zhang et al., 2023 and for indiciators not in Zhang 2023
    supp table 1 (GCaMP3).
    """
    gcamp6s_dFF_1AP_dana2019 = 13.3
    gcamp6s_snr_1AP_dana2019 = 4.4
    sigma_gcamp6s = gcamp6s_dFF_1AP_dana2019 / gcamp6s_snr_1AP_dana2019
    sigma_noise = sigma_noise_rel * sigma_gcamp6s
    if dFF_1AP_rel is not None:
        dFF_1AP = dFF_1AP_rel * gcamp6s_dFF_1AP_dana2019
    else:
        dFF_1AP = None

    if ca_amp:
        ca_amp = ca_amp / second
    if t_on:
        t_on = t_on * second / 100
    if t_off:
        t_off = t_off * second / 100

    def geci_fn(
        light_dependent=False,
        doub_exp_conv=True,
        pre_existing_cal=False,
        K_d=K_d * nmolar,
        n_H=n_H,
        dFF_max=dFF_max,
        sigma_noise=sigma_noise,
        dFF_1AP=dFF_1AP,
        ca_amp=ca_amp,
        t_on=t_on,
        t_off=t_off,
        Ca_rest=50 * nmolar,
        kappa_S=110,
        gamma=292.3 / second,
        B_T=10 * umolar,
        dCa_T=7.6 * umolar,  # Lütke et al., 2013 and NAOMi code
        **kwparams,
    ):
        """Returns a GECI model with parameters given in
        `NAOMi's code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_
        (Song et al., 2021) as well as Dana et al., 2019 and Zhang et al., 2023.

        Only those parameters used in chosen model components apply.
        If the default is ``None``, then we don't have it fit yet.
        """
        return geci(
            light_dependent,
            doub_exp_conv,
            pre_existing_cal,
            K_d=K_d,
            n_H=n_H,
            dFF_max=dFF_max,
            sigma_noise=sigma_noise,
            dFF_1AP=dFF_1AP,
            ca_amp=ca_amp,
            t_on=t_on,
            t_off=t_off,
            Ca_rest=Ca_rest,
            kappa_S=kappa_S,
            gamma=gamma,
            B_T=B_T,
            dCa_T=dCa_T,
            name=name,
            **kwparams,
        )

    geci_fn.__doc__ += "\n\n" + (" " * 8) + extra_doc

    globals()[name] = geci_fn


# from NAOMi:
#                      K_d, n_H, dFF_max, sigma_noise_rel, dFF_1AP_rel, ca_amp, t_on, t_off
_create_geci_fn("gcamp6f", 290, 2.7, 25.2, 1.24, 0.735, 76.1251, 0.8535, 98.6173)
_create_geci_fn("gcamp6s", 147, 2.45, 27.2, 1, 1, 54.6943, 0.4526, 68.5461)
# noise, dFF for GCaMP3 from Dana 2019, since not in Zhang 2023
_create_geci_fn(
    "gcamp3", 287, 2.52, 12, (3.9 / 2.1) / (13.3 / 4.4), 3.9 / 13.3, 0.05, 1, 1
)

# from NAOMi, but
# don't have double exponential convolution parameters for these:
_create_geci_fn("ogb1", 250, 1, 14, 1, extra_doc="*Don't know sigma_noise*")
_create_geci_fn("gcamp6rs09", 520, 3.2, 25, 1, extra_doc="*Don't know sigma_noise*")
_create_geci_fn("gcamp6rs06", 320, 3, 15, 1, extra_doc="*Don't know sigma_noise*")
_create_geci_fn("jgcamp7f", 174, 2.3, 30.2, 0.72)
_create_geci_fn("jgcamp7s", 68, 2.49, 40.4, 0.33)
_create_geci_fn("jgcamp7b", 82, 3.06, 22.1, 0.25)
_create_geci_fn("jgcamp7c", 298, 2.44, 145.6, 0.39)
