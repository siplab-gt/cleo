"""A bunch of fitted opsin models."""

from brian2.units import (
    mM,
    mm2,
    ms,
    mV,
    nsiemens,
    pcoulomb,
    psiemens,
    second,
)

from cleo.light.light_dependence import equal_photon_flux_spectrum, plot_spectra
from cleo.opto.opsins import (
    BansalFourStateOpsin,
    BansalThreeStatePump,
    FourStateOpsin,
)


def chr2_4s() -> FourStateOpsin:
    """Returns a 4-state ChR2 model.

    Params taken from try.projectpyrho.org's default 4-state configuration.
    Action spectrum from `Nagel et al., 2003, Fig. 4a
    <https://www.pnas.org/doi/full/10.1073/pnas.1936192100>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.

    Parameters can be changed after initialization but *before injection*.
    """
    return FourStateOpsin(
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
        spectrum=[
            (400, 0.34),
            (422, 0.65),
            (460, 0.96),
            (470, 1),
            (473, 1),
            (500, 0.57),
            (520, 0.22),
            (540, 0.06),
            (560, 0.01),
        ],
    )


def chr2_b4s() -> BansalFourStateOpsin:
    """Returns a 4-state ChR2 model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Nagel et al., 2003, Fig. 4a
    <https://www.pnas.org/doi/full/10.1073/pnas.1936192100>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.

    Parameters can be changed after initialization but *before injection*.
    """
    return BansalFourStateOpsin(
        Gd1=0.066 / ms,
        Gd2=0.01 / ms,
        Gr0=3.33e-4 / ms,
        g0=3.2 * nsiemens,
        phim=1e16 / mm2 / second,  # *photon, not in Brian2
        k1=0.4 / ms,
        k2=0.12 / ms,
        Gf0=0.018 / ms,
        Gb0=0.008 / ms,
        kf=0.01 / ms,
        kb=0.008 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=0 * mV,
        name="ChR2",
        spectrum=[
            (400, 0.34),
            (422, 0.65),
            (460, 0.96),
            (470, 1),
            (473, 1),
            (500, 0.57),
            (520, 0.22),
            (540, 0.06),
            (560, 0.01),
        ],
    )


def chr2_h134r_4s() -> BansalFourStateOpsin:
    """Returns a 4-state ChR2(H134R) model.

    Params given in Bansal et al., 2020.
    Action spectrum is same as for :func:`~chr2_4s`, but blue-shifted 20 nm
    (I cannot find it directly in the literature).

    Parameters can be changed after initialization but *before injection*.
    """

    return BansalFourStateOpsin(
        Gd1=0.045 / ms,
        Gd2=0.01 / ms,
        Gr0=1e-4 / ms,
        g0=37.2 * nsiemens,
        phim=1e16 / mm2 / second,  # *photon, not in Brian2
        k1=0.3 / ms,
        k2=0.12 / ms,
        Gf0=0.014 / ms,
        Gb0=0.01 / ms,
        kf=0.01 / ms,
        kb=0.01 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=0 * mV,
        name="ChR2(H134R)",
        spectrum=[
            (380, 0.34),
            (402, 0.65),
            (440, 0.96),
            (450, 1),
            (480, 0.57),
            (500, 0.22),
            (520, 0.06),
            (540, 0.01),
        ],
    )


def vfchrimson_4s() -> BansalFourStateOpsin:
    """Returns a 4-state vf-Chrimson model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Mager et al., 2018, Supp. Fig. 1a
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-04146-3/MediaObjects/41467_2018_4146_MOESM1_ESM.docx>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.

    Parameters can be changed after initialization but *before injection*.
    """
    return BansalFourStateOpsin(
        Gd1=0.37 / ms,
        Gd2=0.175 / ms,
        Gr0=6.67e-7 / ms,
        g0=17.5 * nsiemens,
        phim=1.5e16 / mm2 / second,
        k1=3 / ms,
        k2=0.2 / ms,
        Gf0=0.02 / ms,
        Gb0=3.2e-3 / ms,
        kf=0.01 / ms,
        kb=0.01 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=0 * mV,
        name="vf-Chrimson",
        spectrum=equal_photon_flux_spectrum(
            [
                (470, 0.34),
                (490, 0.51),
                (510, 0.71),
                (530, 0.75),
                (550, 0.86),
                (570, 1),
                (590, 1),
                (610, 0.8),
                (630, 0.48),
            ]
        ),
    )


def chrimson_4s() -> BansalFourStateOpsin:
    """Returns a 4-state Chrimson model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Mager et al., 2018, Supp. Fig. 1a
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-04146-3/MediaObjects/41467_2018_4146_MOESM1_ESM.docx>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.

    Parameters can be changed after initialization but *before injection*.
    """
    return BansalFourStateOpsin(
        Gd1=0.041 / ms,
        Gd2=0.01 / ms,
        Gr0=6.67e-7 / ms,
        g0=22 * nsiemens,
        phim=0.2e15 / mm2 / second,
        k1=0.05 / ms,
        k2=0.5 / ms,
        Gf0=0.001 / ms,
        Gb0=0.01 / ms,
        kf=0.01 / ms,
        kb=0.004 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=0 * mV,
        name="Chrimson",
        spectrum=equal_photon_flux_spectrum(
            [
                (470, 0.31),
                (490, 0.47),
                (510, 0.69),
                (530, 0.75),
                (550, 0.88),
                (570, 0.97),
                (590, 1),
                (610, 0.88),
                (630, 0.55),
            ]
        ),
    )


def gtacr2_4s() -> BansalFourStateOpsin:
    """Returns a 4-state model of GtACR2, an anion channel.

    Params given in Bansal et al., 2020.
    Action spectra from `Govorunova et al., 2015, Fig. 1f
    <https://www.science.org/doi/10.1126/science.aaa7484#F1>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.

    Parameters can be changed after initialization but *before injection*.
    """
    return BansalFourStateOpsin(
        Gd1=0.017 / ms,
        Gd2=0.01 / ms,
        Gr0=5.8e-4 / ms,
        g0=44 * nsiemens,
        phim=2e17 / mm2 / second,
        k1=40 / ms,
        k2=20 / ms,
        Gf0=0.001 / ms,
        Gb0=0.003 / ms,
        kf=0.001 / ms,
        kb=0.005 / ms,
        gamma=0.05,
        p=1,
        q=0.1,
        E=-69.5 * mV,
        name="GtACR2",
        spectrum=[
            (400, 0.40),
            (410, 0.49),
            (420, 0.56),
            (430, 0.65),
            (440, 0.82),
            (450, 0.88),
            (460, 0.88),
            (470, 1.0),
            (480, 0.91),
            (490, 0.67),
            (500, 0.41),
            (510, 0.21),
            (520, 0.12),
            (530, 0.06),
            (540, 0.02),
            (550, 0.00),
            (560, 0.00),
        ],
    )


def enphr3_3s():
    """Returns a 3-state model of eNpHR3, a chloride pump.

    Params given in Bansal et al., 2020.
    Action spectrum from `Gradinaru et al., 2010 <https://doi.org/10.1016/j.cell.2010.02.037>`_,
    Figure 3F,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalThreeStatePump(
        Gd=0.025 / ms,
        Gr=0.05 / ms,
        ka=1 / ms,
        p=0.7,
        q=0.1,
        phim=1.2e18 / mm2 / second,
        E=-400 * mV,
        g0=22.34 * nsiemens,
        a=0.02e-2 * mM / pcoulomb,
        b=12,
        name="eNpHR3.0",
        spectrum=[
            (390, 0.162),
            (405, 0.239),
            (430, 0.255),
            (445, 0.255),
            (470, 0.371),
            (495, 0.554),
            (520, 0.716),
            (542.5, 0.840),
            (560, 0.930),
            (590, 1),
            (630, 0.385),
        ],
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_spectra(
        chr2_4s(),
        chr2_h134r_4s(),
        vfchrimson_4s(),
        chrimson_4s(),
        gtacr2_4s(),
        enphr3_3s(),
        extrapolate=True,
    )
    plt.show()
