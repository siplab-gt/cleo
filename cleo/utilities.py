"""Assorted utilities for developers."""
import warnings
from collections.abc import MutableMapping

import brian2 as b2
import neo
import quantities as pq
from brian2 import Quantity, np, second
from brian2.equations.equations import (
    DIFFERENTIAL_EQUATION,
    PARAMETER,
    SUBEXPRESSION,
    Equations,
)
from brian2.groups.group import get_dtype
from matplotlib import pyplot as plt

rng = np.random.default_rng()
"""supposed to be the central random number generator, but not yet used everywhere"""


def times_are_regular(times):
    if len(times) < 2:
        return False
    return np.allclose(np.diff(times), times[1] - times[0])


def analog_signal(t_ms, values_no_unit, units="") -> neo.core.basesignal.BaseSignal:
    if times_are_regular(t_ms):
        return neo.AnalogSignal(
            values_no_unit,
            t_start=t_ms[0] * pq.ms,
            units=units,
            sampling_period=(t_ms[1] - t_ms[0]) * pq.ms,
        )
    else:
        return neo.IrregularlySampledSignal(
            t_ms * pq.ms,
            values_no_unit,
            units=units,
        )


def add_to_neo_segment(
    segment: neo.core.Segment, *objects: neo.core.dataobject.DataObject
):
    """Conveniently adds multiple objects to a segment.

    Taken from :class:`neo.core.group.Group`."""
    container_lookup = {
        cls_name: getattr(segment, container_name)
        for cls_name, container_name in zip(
            segment._child_objects, segment._child_containers
        )
    }

    for obj in objects:
        cls = obj.__class__
        if hasattr(cls, "proxy_for"):
            cls = cls.proxy_for
        container = container_lookup[cls.__name__]
        container.append(obj)


def normalize_coords(coords: Quantity) -> Quantity:
    """Normalize coordinates to unit vectors."""
    return coords / np.linalg.norm(coords, axis=-1, keepdims=True)


def get_orth_vectors_for_V(V):
    """For nx3 block of row vectors V, return nx3 W1, W2 orthogonal
    vector blocks"""
    V = V.reshape((-1, 3, 1))
    n = V.shape[0]
    V = np.repeat(V, 3, axis=-1)
    assert V.shape == (n, 3, 3)
    q, r = np.linalg.qr(V)
    # get two vectors orthogonal to v from QR decomp
    W1, W2 = q[..., 1], q[..., 2]
    assert W1.shape == W2.shape == (n, 3)
    return W1.squeeze(), W2.squeeze()


def xyz_from_rθz(rs, thetas, zs, xyz_start, xyz_end):
    """Convert from cylindrical to Cartesian coordinates."""
    # not using np.linalg.norm because it strips units
    m = xyz_start.reshape((-1, 3)).shape[0]
    n = len(rs)

    cyl_length = np.sqrt(np.sum((xyz_end - xyz_start) ** 2, axis=-1, keepdims=True))
    assert cyl_length.shape in [(m, 1), (1,)]
    c = (xyz_end - xyz_start) / cyl_length  # unit vector in direction of cylinder
    # in case cyl_length is 0, producing nans
    assert c.shape in [(m, 3), (3,)]
    if c.shape == (m, 3):
        assert cyl_length.shape == (m, 1)
        c[cyl_length.ravel() == 0] = [0, 0, 1]
    elif c.shape == (3,):
        c[:] = [0, 0, 1]

    r1, r2 = get_orth_vectors_for_V(c)

    def r_unit_vecs(thetas):
        cosines = np.reshape(np.cos(thetas), (len(thetas), 1))
        sines = np.reshape(np.sin(thetas), (len(thetas), 1))
        # add axis for broadcasting so result is nx3
        cosines = np.cos(thetas)[..., np.newaxis]
        sines = np.sin(thetas)[..., np.newaxis]
        return r1.reshape((m, 1, 3)) * cosines + r2.reshape((m, 1, 3)) * sines

    coords = (
        xyz_start.reshape((m, 1, 3))
        + c.reshape((m, 1, 3)) * zs.reshape((1, n, 1))
        + rs.reshape((1, n, 1)) * r_unit_vecs(thetas)
    )
    assert coords.shape == (m, n, 3)
    return coords[..., 0], coords[..., 1], coords[..., 2]
    # coords = coords.reshape((m * n), 3)
    # return coords[:, 0], coords[:, 1], coords[:, 2]


def uniform_cylinder_rθz(n, rmax, zmax):
    """uniformly fills a cylinder with radius rmax and height zmax with n points.

    Does so by generating a Fibonacci spiral cylinder by rotating around axis
    and up and down cylinder simultaneously, using different angle steps."""
    indices = np.arange(0, n) + 0.5
    rs = rmax * np.sqrt(indices / n)
    golden_angle = np.pi * (1 + np.sqrt(5))
    thetas = golden_angle * indices
    # using sqrt(2) instead of golden ratio here so
    # the two angles don't coincide
    phis = 2 * np.pi * np.sqrt(2) * indices
    zs = zmax * (1 + np.sin(phis)) / 2

    return rs, thetas, zs


def modify_model_with_eqs(neuron_group, eqs_to_add):
    """Adapted from _create_variables() from neurongroup.py from Brian2
    source code v2.3.0.2
    """
    if type(eqs_to_add) == str:
        eqs_to_add = Equations(eqs_to_add)
    neuron_group.equations += eqs_to_add

    variables = neuron_group.variables

    dtype = {}
    if isinstance(dtype, MutableMapping):
        dtype["lastspike"] = neuron_group._clock.variables["t"].dtype

    for eq in eqs_to_add.values():
        dtype = get_dtype(eq, dtype)  # {} appears to be the default
        if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
            if "linked" in eq.flags:
                # 'linked' cannot be combined with other flags
                if not len(eq.flags) == 1:
                    raise SyntaxError(
                        ('The "linked" flag cannot be ' "combined with other flags")
                    )
                neuron_group._linked_variables.add(eq.varname)
            else:
                constant = "constant" in eq.flags
                shared = "shared" in eq.flags
                size = 1 if shared else neuron_group._N
                variables.add_array(
                    eq.varname,
                    size=size,
                    dimensions=eq.dim,
                    dtype=dtype,
                    constant=constant,
                    scalar=shared,
                )
        elif eq.type == SUBEXPRESSION:
            neuron_group.variables.add_subexpression(
                eq.varname,
                dimensions=eq.dim,
                expr=str(eq.expr),
                dtype=dtype,
                scalar="shared" in eq.flags,
            )
        else:
            raise AssertionError("Unknown type of equation: " + eq.eq_type)

    # Add the conditional-write attribute for variables with the
    # "unless refractory" flag
    if neuron_group._refractory is not False:
        for eq in neuron_group.equations.values():
            if eq.type == DIFFERENTIAL_EQUATION and "unless refractory" in eq.flags:
                not_refractory_var = neuron_group.variables["not_refractory"]
                var = neuron_group.variables[eq.varname]
                var.set_conditional_write(not_refractory_var)

    # Stochastic variables
    for xi in eqs_to_add.stochastic_variables:
        try:
            neuron_group.variables.add_auxiliary_variable(
                xi, dimensions=(second**-0.5).dim
            )
        except KeyError as ex:
            warnings.warn(
                "Adding a stochastic variable to a neuron group that already"
                " has a variable with the same name."
            )

    # Check scalar subexpressions
    for eq in neuron_group.equations.values():
        if eq.type == SUBEXPRESSION and "shared" in eq.flags:
            var = neuron_group.variables[eq.varname]
            for identifier in var.identifiers:
                if identifier in neuron_group.variables:
                    if not neuron_group.variables[identifier].scalar:
                        raise SyntaxError(
                            (
                                "Shared subexpression %s refers "
                                "to non-shared variable %s."
                            )
                            % (eq.varname, identifier)
                        )


def wavelength_to_rgb(wavelength_nm, gamma=0.8) -> tuple[float, float, float]:
    """taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """
    wavelength = wavelength_nm
    if wavelength < 380:
        wavelength = 380.0
    if wavelength > 750:
        wavelength = 750.0
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)


def brian_safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def style_plots_for_docs(dark=True):
    # some hacky workaround for params not being updated until after first plot
    f = plt.figure()
    plt.plot()
    plt.close(f)

    if dark:
        plt.style.use("dark_background")
        for obj in ["figure", "axes", "savefig"]:
            plt.rc(obj, facecolor="131416")  # color of Furo dark background
    else:
        plt.style.use("default")
    plt.rc("savefig", transparent=False)
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("font", **{"sans-serif": "Open Sans"})


def style_plots_for_paper(fontscale=5 / 6):
    """
    fontscale=5/6 goes from default paper font size of 9.6 down to 8
    """
    # some hacky workaround for params not being updated until after first plot
    f = plt.figure()
    plt.plot()
    plt.close(f)

    try:
        import seaborn as sns

        sns.set_context("paper", font_scale=fontscale)
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")
        warnings.warn("Seaborn not found, using matplotlib style, ignoring fontscale")
    plt.rc("savefig", transparent=True, bbox="tight", dpi=300)
    plt.rc("svg", fonttype="none")
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("font", **{"sans-serif": "Open Sans"})


def unit_safe_append(q1: Quantity, q2: Quantity, axis=0):
    if not b2.have_same_dimensions(q1, q2):
        raise ValueError("Dimensions must match")
    if isinstance(q1, Quantity):
        assert isinstance(q2, Quantity)
        unit = q1.get_best_unit()
        return np.append(q1 / unit, q2 / unit, axis=axis) * unit
    else:
        return np.append(q1, q2, axis=axis)
