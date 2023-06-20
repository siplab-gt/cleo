"""A bunch of fitted opsin models."""
from typing import Callable

from attrs import field, define
from brian2.units import (
    nsiemens,
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

from cleo.opto.opsins import (
    FourStateOpsin,
    linear_interp_from_data,
    BansalFourStateOpsin,
)


@define(eq=False)
class ChR2_4S(FourStateOpsin):
    """A 4-state ChR2 model.

    Params taken from try.projectpyrho.org's default 4-state configuration.

    Action spectrum from `Nagel et al., 2003, Fig. 4a
    <https://www.pnas.org/doi/full/10.1073/pnas.1936192100>`_,
    using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    g0 = 114000 * psiemens
    gamma = 0.00742
    phim = 2.33e17 / mm2 / second  # *photon, not in Brian2
    k1 = 4.15 / ms
    k2 = 0.868 / ms
    p = 0.833
    Gf0 = 0.0373 / ms
    kf = 0.0581 / ms
    Gb0 = 0.0161 / ms
    kb = 0.063 / ms
    q = 1.94
    Gd1 = 0.105 / ms
    Gd2 = 0.0138 / ms
    Gr0 = 0.00033 / ms
    E = 0 * mV
    v0 = 43 * mV
    v1 = 17.1 * mV
    name = "ChR2"
    epsilon = linear_interp_from_data(
        [400, 422, 460, 470, 473, 500, 520, 540, 560],
        [0.34, 0.65, 0.96, 1, 1, 0.57, 0.22, 0.06, 0.01],
    )


@define(eq=False)
class VfChrimson_4S(BansalFourStateOpsin):
    """A 4-state Vf-Chrimson model.

    Params given in Bansal et al., 2020.
    Action spectra from `Mager et al., 2018, Supp. Fig. 1a
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-04146-3/MediaObjects/41467_2018_4146_MOESM1_ESM.docx>`_.
    """

    Gd1 = 0.37 / ms
    Gd2 = 0.175 / ms
    Gr0 = 6.67e-7 / ms
    g0 = 17.5 * nsiemens
    phim = 1.5e16 / mm2 / second  # *photon, not in Brian2
    k1 = 3 / ms
    k2 = 0.2 / ms
    Gf0 = 0.02 / ms
    Gb0 = 3.2e-3 / ms
    kf = 0.01 / ms
    kb = 0.01 / ms
    gamma = 0.05
    p = 1
    q = 1
    E = 0 * mV
    name = "Vf-Chrimson"
    epsilon = field(
        factory=lambda: linear_interp_from_data(
            [470, 490, 510, 530, 550, 570, 590, 593, 610, 630],
            [0.34, 0.51, 0.71, 0.75, 0.86, 1, 1, 1, 0.8, 0.48],
        )
    )


@define(eq=False)
class Chrimson_4S(BansalFourStateOpsin):
    """A 4-state Chrimson model.

    Params given in Bansal et al., 2020.
    Action spectra from `Mager et al., 2018, Supp. Fig. 1a
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-04146-3/MediaObjects/41467_2018_4146_MOESM1_ESM.docx>`_,
    using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    Gd1 = 0.041 / ms
    Gd2 = 0.01 / ms
    Gr0 = 6.67e-7 / ms
    g0 = 22 * nsiemens
    phim = 0.2e15 / mm2 / second  # *photon, not in Brian2
    k1 = 0.05 / ms
    k2 = 0.5 / ms
    Gf0 = 0.001 / ms
    Gb0 = 0.01 / ms
    kf = 0.01 / ms
    kb = 0.004 / ms
    gamma = 0.05
    p = 1
    q = 1
    E = 0 * mV
    name = "Chrimson"
    epsilon = field(
        default=linear_interp_from_data(
            [470, 490, 510, 530, 550, 570, 590, 593, 610, 630],
            [0.31, 0.47, 0.69, 0.75, 0.88, 0.97, 1, 1, 0.88, 0.55],
        )
    )