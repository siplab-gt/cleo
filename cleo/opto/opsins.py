"""Contains opsin models and default parameters"""
from __future__ import annotations

from typing import Any, Tuple

from attrs import define, field
from brian2 import (
    Synapses,
    Unit,
    check_units,
    get_unit,
    implementation,
    np,
)
from brian2.units import (
    Quantity,
    amp,
    cm2,
    mM,
    mm2,
    ms,
    msiemens,
    mV,
    nsiemens,
    pcoulomb,
    psiemens,
    second,
    umeter,
    volt,
)
from brian2.units.allunits import radian

from cleo.base import SynapseDevice
from cleo.light import LightDependent


# hacky MRO stuff: base class order is important here
@define(eq=False)
class Opsin(LightDependent, SynapseDevice):
    """Base class for opsin model.

    Requires that the neuron model has a current term
    (by default Iopto) which is assumed to be positive (unlike the
    convention in many opsin modeling papers, where the current is
    described as negative).

    We approximate dynamics under multiple wavelengths using a weighted sum
    of photon fluxes, where the ε factor indicates the activation
    relative to the peak-sensitivy wavelength for an equivalent number of photons
    (see Mager et al, 2018). This weighted sum is an approximation of a nonlinear
    peak-non-peak wavelength relation; see ``notebooks/multi_wavelength_model.ipynb``
    for details."""

    @property
    def action_spectrum(self):
        """Alias for ``light_receptor.spectrum``"""
        return self.spectrum


@define(eq=False)
class MarkovOpsin(Opsin):
    """Base class for Markov state models à la Evans et al., 2016"""

    required_vars: list[Tuple[str, Unit]] = field(
        factory=lambda: [("Iopto", amp), ("v", volt)],
        init=False,
    )


@implementation(
    "cython",
    """
    cdef double f_unless_x0(double f, double x, double f_when_x0):
        if x == 0:
            return f_when_x0
        else:
            return f
    """,
)
@check_units(f=1, x=volt, f_when_0=1, result=1)
def f_unless_x0(f, x, f_when_x0):
    f[x == 0] = f_when_x0
    return f


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
        # v1/v0 when v-E == 0 via l'Hopital's rule
        fv = f_unless_x0(
            (1 - exp(-(V_VAR_NAME_post-E)/v0)) / ((V_VAR_NAME_post-E)/v1),
            V_VAR_NAME_post - E,
            v1/v0
        ) : 1

        IOPTO_VAR_NAME_post = -g0*fphi*fv*(V_VAR_NAME_post-E)*rho_rel : ampere (summed)
        rho_rel : 1""",
    )

    extra_namespace: dict[str, Any] = field(
        init=False, factory=lambda: {"f_unless_x0": f_unless_x0}
    )

    def init_syn_vars(self, opto_syn: Synapses) -> None:
        for varname, value in {"C1": 1, "O1": 0, "O2": 0}.items():
            setattr(opto_syn, varname, value)


@define(eq=False)
class BansalFourStateOpsin(MarkovOpsin):
    """4-state model from Bansal et al. 2020.

    The difference from the PyRhO model is that there is no voltage dependence.

    rho_rel is channel density relative to standard model fit;
    modifying it post-injection allows for heterogeneous opsin expression.

    IOPTO_VAR_NAME and V_VAR_NAME are substituted on injection.
    """

    Gd1: Quantity = 0.066 / ms
    Gd2: Quantity = 0.01 / ms
    Gr0: Quantity = 3.33e-4 / ms
    g0: Quantity = 3.2 * nsiemens
    phim: Quantity = 1e16 / mm2 / second  # *photon, not in Brian2
    k1: Quantity = 0.4 / ms
    k2: Quantity = 0.12 / ms
    Gf0: Quantity = 0.018 / ms
    Gb0: Quantity = 0.008 / ms
    kf: Quantity = 0.01 / ms
    kb: Quantity = 0.008 / ms
    gamma: Quantity = 0.05
    p: Quantity = 1
    q: Quantity = 1
    E: Quantity = 0 * mV

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

        IOPTO_VAR_NAME_post = -g0*fphi*(V_VAR_NAME_post-E)*rho_rel : ampere (summed)
        rho_rel : 1""",
    )

    def init_syn_vars(self, opto_syn: Synapses) -> None:
        for varname, value in {"C1": 1, "O1": 0, "O2": 0}.items():
            setattr(opto_syn, varname, value)


@define(eq=False)
class BansalThreeStatePump(MarkovOpsin):
    """3-state model from `Bansal et al. 2020 <https://iopscience.iop.org/article/10.1088/2057-1976/ab90a1>`_.
    Defaults are for eNpHR3.0.

    rho_rel is channel density relative to standard model fit;
    modifying it post-injection allows for heterogeneous opsin expression.

    IOPTO_VAR_NAME and V_VAR_NAME are substituted on injection.
    """

    Gd: Quantity = 0.25 / ms
    Gr: Quantity = 0.05 / ms
    ka: Quantity = 1 / ms
    p: Quantity = 0.7
    q: Quantity = 0.1
    phim: Quantity = 1.2e18 / mm2 / second  # photons
    E: Quantity = -400 * mV
    g0: Quantity = 22.34 * nsiemens
    a: Quantity = 0.02e-2 * mM / pcoulomb
    b: float = 12
    vartheta_max: Quantity = 5 * mM / second
    kd: Quantity = 16 * mM
    # Sukhdev Roy said they used 2.3 nS for the 'photocurrent simulation' and 2.3 mS/cm² for the 'neuronal simulation', whatever that means
    # g_Cl: Quantity = 2.3 * msiemens / cm2 * (30090 * umeter**2)  # surface area
    g_Cl: Quantity = 2.3 * nsiemens
    Cl_out: Quantity = 124 * mM
    Psi0: Quantity = 4.4286 * mM / second
    E_Cl0: Quantity = -70 * mV
    vmin: Quantity = -400 * mV
    """Needed to avoid jumps in [Cl_in] for EIF neurons"""
    vmax: Quantity = 50 * mV
    """Needed to avoid jumps in [Cl_in] for EIF neurons"""
    model: str = field(
        init=False,
        default="""
        dP0/dt = Gr*P6 - Ga*P0 : 1 (clock-driven)
        dP4/dt = Ga*P0 - Gd*P4 : 1 (clock-driven)
        P6 = 1 - P0 - P4 : 1

        Theta = int(phi_pre > 0*phi_pre) : 1
        Hp = Theta * phi_pre**p/(phi_pre**p + phim**p) : 1
        Ga = ka*Hp : hertz

        fphi = P4 : 1
        dCl_in/dt = a*(I_i + b*I_Cl_leak) : mmolar (clock-driven)
        E_Cl = -26.67*mV * log(Cl_out/Cl_in) : volt
        I_Cl_leak = g_Cl * (E_Cl0 - E_Cl) : ampere

        Psi = vartheta_max*Cl_out / (kd + Cl_out) / Psi0 : 1
        I_i = g0*fphi*Psi*rho_rel * (clip(V_VAR_NAME_post, vmin, vmax)-E) : ampere

        IOPTO_VAR_NAME_post = -(I_i + I_Cl_leak) : ampere (summed)
        rho_rel : 1""",
    )

    def init_syn_vars(self, opto_syn: Synapses) -> None:
        opto_syn.P0 = 1
        opto_syn.P4 = 0
        # P6 automatically set since it's 1 - P0 - P4
        RToverF = 26.67 * mV
        # need to remove and add back on units for log/exp to work
        opto_syn.Cl_in = np.exp(self.E_Cl0 / RToverF + np.log(self.Cl_out / mM)) * mM


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
