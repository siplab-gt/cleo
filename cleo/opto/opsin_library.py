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


ONE_P_TWO_P_RATIO = 270382.2996938772


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
            (800, 0.34 / ONE_P_TWO_P_RATIO),
            (844, 0.65 / ONE_P_TWO_P_RATIO),
            (920, 0.96 / ONE_P_TWO_P_RATIO),
            (940, 1 / ONE_P_TWO_P_RATIO),
            (946, 1 / ONE_P_TWO_P_RATIO),
            (1000, 0.57 / ONE_P_TWO_P_RATIO),
            (1040, 0.22 / ONE_P_TWO_P_RATIO),
            (1080, 0.06 / ONE_P_TWO_P_RATIO),
            (1120, 0.01 / ONE_P_TWO_P_RATIO),
        ],
    )


def ReaChR_4s() -> BansalFourStateOpsin:  # include this if it's a 4 state
    """Returns a 4-state ReaChR model.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Krause et al., 2017, Fig. 1b
    <https://doi.org/10.1016/j.bpj.2017.02.001>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    return BansalFourStateOpsin(
        Gd1=7.7e-3 / ms,
        Gd2=1.25e-3 / ms,
        Gr0=3.33e-5 / ms,
        g0=14.28 * nsiemens,
        phim=5e17 / mm2 / second,
        k1=1.2 / ms,
        k2=0.01 / ms,
        Gf0=0.0005 / ms,
        Gb0=0.0005 / ms,
        kf=0.012 / ms,
        kb=0.001 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=7 * mV,
        name="ReaChR",
        spectrum=[
            (479, 0.67),
            (390, 0.30),
            (401, 0.27),
            (411, 0.30),
            (421, 0.33),
            (431, 0.37),
            (441, 0.41),
            (451, 0.46),
            (461, 0.53),
            (472, 0.58),
            (490, 0.76),
            (500, 0.86),
            (511, 0.94),
            (520, 0.97),
            (531, 1),
            (540, 0.98),
            (549, 0.95),
            (559, 0.86),
            (571, 0.71),
            (580, 0.52),
            (590, 0.36),
        ],
    )


def ChrimsonR_4s() -> BansalFourStateOpsin:
    """Returns a 4-state CsChrimsonR model.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Cheong, Soon Keen et al., 2018, Fig. 2,
    <doi:10.1371/journal.pone.0194947>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalFourStateOpsin(
        Gd1=0.067 / ms,
        Gd2=0.01 / ms,
        Gr0=0.5e-3 / ms,
        g0=12.25 * nsiemens,
        phim=20e17 / mm2 / second,
        k1=6 / ms,
        k2=0.2 / ms,
        Gf0=0.02 / ms,
        Gb0=0.05 / ms,
        kf=0.1 / ms,
        kb=0.001 / ms,
        gamma=0.05,
        p=0.6,
        q=1,
        E=0 * mV,
        name="ChrimsonR",
        spectrum=[
            (386, 0.13),
            (398, 0.15),
            (406, 0.15),
            (423, 0.16),
            (439, 0.17),
            (446, 0.18),
            (453, 0.20),
            (460, 0.23),
            (467, 0.25),
            (479, 0.31),
            (496, 0.40),
            (507, 0.46),
            (518, 0.53),
            (522, 0.56),
            (528, 0.60),
            (532, 0.63),
            (535, 0.67),
            (538, 0.71),
            (546, 0.79),
            (556, 0.85),
            (569, 0.91),
            (580, 0.95),
            (590, 1),
            (599, 0.94),
            (605, 0.90),
            (610, 0.87),
            (619, 0.80),
            (624, 0.76),
            (629, 0.69),
            (632, 0.64),
            (634, 0.60),
            (637, 0.56),
            (643, 0.47),
            (639, 0.52),
            (647, 0.40),
            (652, 0.32),
            (660, 0.24),
            (668, 0.16),
            (678, 0.09),
            (686, 0.06),
        ],
    )


def CsChrimson_4s() -> BansalFourStateOpsin:
    """Returns a 4-state CsChrimson model.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Kim, Seungsoo, et al., 2015, Fig. 6.3a
    <https://doi.org/10.1007/978-3-319-12913-6_6>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalFourStateOpsin(
        Gd1=0.033 / ms,
        Gd2=0.017 / ms,
        Gr0=5e-6 / ms,
        g0=18.48 * nsiemens,
        phim=6e16 / mm2 / second,
        k1=3 / ms,
        k2=0.04 / ms,
        Gf0=0.005 / ms,
        Gb0=0.01 / ms,
        kf=0.01 / ms,
        kb=0.6 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=-10 * mV,
        name="CsChrimson",
        spectrum=[
            (402, 0.16),
            (419, 0.18),
            (435, 0.20),
            (449, 0.23),
            (466, 0.30),
            (482, 0.36),
            (497, 0.46),
            (514, 0.54),
            (529, 0.63),
            (545, 0.75),
            (561, 0.85),
            (576, 0.94),
            (590, 1),
            (591, 1.01),
            (607, 0.94),
            (622, 0.77),
            (638, 0.51),
            (652, 0.30),
            (670, 0.13),
            (684, 0.04),
        ],
    )


def bReaChES_4s() -> BansalFourStateOpsin:  # 3 state?
    """Returns a 4-state bReaChES model.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Bansal, H., Pyari, G. & Roy, S., 2024, Fig. 1c
    <https://doi.org/10.1038/s41598-024-62558-2>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalFourStateOpsin(
        Gd1=0.025 / ms,
        Gd2=0.01 / ms,
        Gr0=3.3e-5 / ms,
        g0=36.5 * nsiemens,
        phim=6e15 / mm2 / second,
        k1=0.4 / ms,
        k2=0.01 / ms,
        Gf0=0.002 / ms,
        Gb0=0.002 / ms,
        kf=0.01 / ms,
        kb=0.04 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=10 * mV,
        name="bReaChES",
        spectrum=[
            (390, 0.47),
            (410, 0.66),
            (430, 0.80),
            (450, 0.89),
            (470, 0.95),
            (490, 0.96),
            (510, 0.95),
            (530, 0.94),
            (550, 0.95),
            (570, 1),
            (590, 0.84),
            (611, 0.34),
            (631, 0.09),
            (651, 0.02),
        ],
    )


def ChRmine_4s() -> BansalFourStateOpsin:
    """Returns a 4-state ChRmine model.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Bansal, H., Pyari, G. & Roy, S., 2024, Fig. 1c
    <https://doi.org/10.1038/s41598-024-62558-2>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalFourStateOpsin(
        Gd1=0.02 / ms,
        Gd2=0.013 / ms,
        Gr0=5.9e-4 / ms,
        g0=110 * nsiemens,
        phim=2.1e15 / mm2 / second,
        k1=0.2 / ms,
        k2=0.01 / ms,
        Gf0=0.0027 / ms,
        Gb0=0.0005 / ms,
        kf=0.001 / ms,
        kb=0 / ms,
        gamma=0.05,
        p=0.8,
        q=1,
        E=5.64 * mV,
        name="ChRmine",
        spectrum=[
            (390, 0.37),
            (410, 0.55),
            (430, 0.72),
            (450, 0.85),
            (470, 0.93),
            (491, 0.99),
            (510, 1),
            (531, 0.98),
            (551, 0.94),
            (570, 0.86),
            (591, 0.70),
            (610, 0.48),
            (630, 0.26),
            (650, 0.10),
        ],
    )


# these numbers seem wrong based off of bansal table 1? nvm its from ibtronerueoscience
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
            (800, 0.34 / ONE_P_TWO_P_RATIO),
            (844, 0.65 / ONE_P_TWO_P_RATIO),
            (920, 0.96 / ONE_P_TWO_P_RATIO),
            (940, 1 / ONE_P_TWO_P_RATIO),
            (946, 1 / ONE_P_TWO_P_RATIO),
            (1000, 0.57 / ONE_P_TWO_P_RATIO),
            (1040, 0.22 / ONE_P_TWO_P_RATIO),
            (1080, 0.06 / ONE_P_TWO_P_RATIO),
            (1120, 0.01 / ONE_P_TWO_P_RATIO),
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
            # TODO: 40 nm shift?
            (760, 0.34 / ONE_P_TWO_P_RATIO),
            (804, 0.65 / ONE_P_TWO_P_RATIO),
            (880, 0.96 / ONE_P_TWO_P_RATIO),
            (900, 1 / ONE_P_TWO_P_RATIO),
            (960, 0.57 / ONE_P_TWO_P_RATIO),
            (1000, 0.22 / ONE_P_TWO_P_RATIO),
            (1040, 0.06 / ONE_P_TWO_P_RATIO),
            (1080, 0.01 / ONE_P_TWO_P_RATIO),
        ],
    )


# ibroneuroscience


def ChETA_4s() -> BansalFourStateOpsin:
    """Returns a 4-state ChETA model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Gunaydin, L., Yizhar, O., Berndt, A. et al., 2010, Fig. 2b
    <https://doi.org/10.1038/nn.2495>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    return BansalFourStateOpsin(
        Gd1=0.21 / ms,
        Gd2=0.02 / ms,
        Gr0=2e-4 / ms,
        g0=21.5 * nsiemens,
        phim=1e16 / mm2 / second,
        k1=3 / ms,
        k2=1 / ms,
        Gf0=0.02 / ms,
        Gb0=0.015 / ms,
        kf=0.01 / ms,
        kb=0.01 / ms,
        gamma=0.05,
        p=1,
        q=1,
        E=0 * mV,
        name="ChETA",
        spectrum=[
            (445, 0.36),
            (452, 0.47),
            (460, 0.61),
            (468, 0.77),
            (478, 0.89),
            (490, 1),  # <https://www.addgene.org/guides/optogenetics/>
            (500, 1),
            (514, 0.89),
            (530, 0.70),
            (550, 0.42),
            (576, 0.12),
        ],
    )


def Chronos_4s() -> BansalFourStateOpsin:
    """Returns a 4-state Chronos model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Soor, Navjeevan S., et al., 2019, Fig. 1b
    <https://doi.org/10.1088/1361-6463/aaf944>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    return BansalFourStateOpsin(
        Gd1=0.278 / ms,
        Gd2=0.01 / ms,
        Gr0=1.2e-3 / ms,
        g0=39 * nsiemens,
        phim=0.8e16 / mm2 / second,
        k1=1.8 / ms,
        k2=0.01 / ms,
        Gf0=0.05 / ms,
        Gb0=0.08 / ms,
        kf=0.1 / ms,
        kb=0.01 / ms,
        gamma=0.05,
        p=0.8,
        q=0.9,
        E=0 * mV,
        name="Chronos",
        spectrum=[
            (388, 0.18),
            (404, 0.21),
            (419, 0.30),
            (431, 0.41),
            (439, 0.50),
            (447, 0.57),
            (457, 0.67),
            (465, 0.74),
            (472, 0.80),
            (480, 0.86),
            (490, 0.92),
            (498, 1),
            (516, 0.84),
            (507, 0.90),
            (528, 0.68),
            (521, 0.76),
            (533, 0.59),
            (538, 0.49),
            (544, 0.40),
            (548, 0.32),
            (554, 0.23),
            (560, 0.14),
            (576, 0.05),
            (570, 0.08),
            (591, 0.01),
        ],
    )


def CheRiff_4s() -> BansalFourStateOpsin:
    """Returns a 4-state CheRiff model.

    Params given in Bansal et al., 2020.
    Action spectrum from `Hochbaum, Daniel R et al., 2014, Fig. 2a
    <https://doi.org/10.1038/nmeth.3000>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """

    return BansalFourStateOpsin(
        Gd1=0.063 / ms,
        Gd2=2.5e-3 / ms,
        Gr0=5e-4 / ms,
        g0=40 * nsiemens,
        phim=0.7e16 / mm2 / second,
        k1=1.4 / ms,
        k2=0.001 / ms,
        Gf0=0.01 / ms,
        Gb0=0.02 / ms,
        kf=0.02 / ms,
        kb=0.1 / ms,
        gamma=0.05,
        p=0.9,
        q=1,
        E=0 * mV,
        name="CheRiff",
        spectrum=[
            (386, 0.30),
            (403, 0.47),
            (419, 0.67),
            (435, 0.84),
            (450, 0.97),
            (466, 1),
            (483, 0.95),
            (498, 0.63),
            (515, 0.29),
            (529, 0.09),
            (546, 0.02),
            (560, 0.00),
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
                (940, 0.34 / ONE_P_TWO_P_RATIO),
                (980, 0.51 / ONE_P_TWO_P_RATIO),
                (1020, 0.71 / ONE_P_TWO_P_RATIO),
                (1060, 0.75 / ONE_P_TWO_P_RATIO),
                (1100, 0.86 / ONE_P_TWO_P_RATIO),
                (1140, 1 / ONE_P_TWO_P_RATIO),
                (1180, 1 / ONE_P_TWO_P_RATIO),
                (1220, 0.8 / ONE_P_TWO_P_RATIO),
                (1260, 0.48 / ONE_P_TWO_P_RATIO),
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
                (940, 0.31 / ONE_P_TWO_P_RATIO),
                (980, 0.47 / ONE_P_TWO_P_RATIO),
                (1020, 0.69 / ONE_P_TWO_P_RATIO),
                (1060, 0.75 / ONE_P_TWO_P_RATIO),
                (1100, 0.88 / ONE_P_TWO_P_RATIO),
                (1140, 0.97 / ONE_P_TWO_P_RATIO),
                (1180, 1 / ONE_P_TWO_P_RATIO),
                (1220, 0.88 / ONE_P_TWO_P_RATIO),
                (1260, 0.55 / ONE_P_TWO_P_RATIO),
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
            (800, 0.4 / ONE_P_TWO_P_RATIO),
            (820, 0.49 / ONE_P_TWO_P_RATIO),
            (840, 0.56 / ONE_P_TWO_P_RATIO),
            (860, 0.65 / ONE_P_TWO_P_RATIO),
            (880, 0.82 / ONE_P_TWO_P_RATIO),
            (900, 0.88 / ONE_P_TWO_P_RATIO),
            (920, 0.88 / ONE_P_TWO_P_RATIO),
            (940, 1.0 / ONE_P_TWO_P_RATIO),
            (960, 0.91 / ONE_P_TWO_P_RATIO),
            (980, 0.67 / ONE_P_TWO_P_RATIO),
            (1000, 0.41 / ONE_P_TWO_P_RATIO),
            (1020, 0.21 / ONE_P_TWO_P_RATIO),
            (1040, 0.12 / ONE_P_TWO_P_RATIO),
            (1060, 0.06 / ONE_P_TWO_P_RATIO),
            (1080, 0.02 / ONE_P_TWO_P_RATIO),
            (1100, 0.0 / ONE_P_TWO_P_RATIO),
            (1120, 0.0 / ONE_P_TWO_P_RATIO),
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
            (780, 0.162 / ONE_P_TWO_P_RATIO),
            (810, 0.239 / ONE_P_TWO_P_RATIO),
            (860, 0.255 / ONE_P_TWO_P_RATIO),
            (890, 0.255 / ONE_P_TWO_P_RATIO),
            (940, 0.371 / ONE_P_TWO_P_RATIO),
            (990, 0.554 / ONE_P_TWO_P_RATIO),
            (1040, 0.716 / ONE_P_TWO_P_RATIO),
            (1085.0, 0.84 / ONE_P_TWO_P_RATIO),
            (1120, 0.93 / ONE_P_TWO_P_RATIO),
            (1180, 1 / ONE_P_TWO_P_RATIO),
            (1260, 0.385 / ONE_P_TWO_P_RATIO),
        ],
    )


# nphr ? 4 state or 3 state? CHECK FOR ALL...turns out they are pumps so they are three state anyway


def nphr_4s():
    """Returns a 4-state model of NpHR, a chloride pump.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Bamberg et al., 1984, Fig. 3,
    <https://pubs.acs.org/doi/pdf/10.1021/bi00320a050>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalThreeStatePump(
        Gd=0.1099 / ms,
        Gr=0.05 / ms,
        ka=0.005 / ms,
        p=0.5,
        q=0.2,
        phim=1.5e18 / mm2 / second,
        E=-400 * mV,
        g0=17.7 * nsiemens,
        a=0.02e-2 * mM / pcoulomb,
        b=5,
        name="NpHR",
        spectrum=[
            (410, 0.194),
            (420, 0.176),
            (440, 0.199),
            (470, 0.259),
            (480, 0.346),
            (490, 0.429),
            (505, 0.502),
            (510, 0.574),
            (520, 0.667),
            (530, 0.799),
            (550, 0.844),
            (565, 0.903),
            (590, 0.947),
            (578, 1.0),
            (610, 0.853),
            (620, 0.619),
            (640, 0.457),
            (650, 0.299),
        ],
    )


def jaws_4s():
    """Returns a 4-state model of Jaws, a chloride pump.

    Params given in Bansal et al., 2020. [Table 1]
    Action spectrum from `Chuong, Amy S et al., 2014, Fig. 1d
    <https://doi.org/10.1038/nn.3752>`_,
    extracted using `Plot Digitizer <https://plotdigitizer.com/>`_.
    """
    return BansalThreeStatePump(  # table 1 params
        Gd=0.167 / ms,
        Gr=0.05 / ms,
        ka=1 / ms,
        p=0.8,
        q=1,
        phim=0.95e18 / mm2 / second,
        E=-400 * mV,
        g0=12.6 * nsiemens,
        a=0.02e-2 * mM / pcoulomb,
        b=6.5,
        name="Jaws",
        spectrum=[
            (402, 0.29),
            (419, 0.29),
            (434, 0.32),
            (450, 0.35),
            (466, 0.43),
            (482, 0.49),
            (497, 0.55),
            (514, 0.59),
            (529, 0.69),
            (545, 0.80),
            (560, 0.91),
            (577, 0.98),
            (592, 1),
            (600, 1),  # says in article
            (608, 0.94),
            (624, 0.81),
            (639, 0.54),
            (654, 0.26),
            (671, 0.09),
            (685, 0.02),
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
