"""Assorted utilities for developers."""
from collections.abc import MutableMapping
from math import ceil, floor

from scipy import linalg
import numpy as np

from brian2 import second
from brian2.groups.group import get_dtype
from brian2.equations.equations import (
    Equations,
    DIFFERENTIAL_EQUATION,
    SUBEXPRESSION,
    PARAMETER,
)


def get_orth_vectors_for_v(v):
    """Returns w1, w2 as 1x3 row vectors"""
    q, r = linalg.qr(
        np.column_stack([v, v, v])
    )  # get two vectors orthogonal to v from QR decomp
    return q[:, 1], q[:, 2]


def xyz_from_rθz(rs, thetas, zs, xyz_start, xyz_end):
    """Convert from cylindrical to Cartesian coordinates."""
    # not using np.linalg.norm because it strips units
    cyl_length = np.sqrt(np.sum(np.subtract(xyz_end, xyz_start) ** 2))
    c = (xyz_end - xyz_start) / cyl_length  # unit vector in direction of cylinder

    r1, r2 = get_orth_vectors_for_v(c)

    def r_unit_vecs(thetas):
        cosines = np.reshape(np.cos(thetas), (len(thetas), 1))
        sines = np.reshape(np.sin(thetas), (len(thetas), 1))
        # add axis for broadcasting so result is nx3
        cosines = np.cos(thetas)[..., np.newaxis]
        sines = np.sin(thetas)[..., np.newaxis]
        return r1 * cosines + r2 * sines

    coords = (
        xyz_start + c * zs[..., np.newaxis] + rs[..., np.newaxis] * r_unit_vecs(thetas)
    )
    return coords[:, 0], coords[:, 1], coords[:, 2]


def uniform_cylinder_rθz(n, rmax, zmax):
    # generate Fibonacci spiral cylinder by rotating around axis
    # and up and down cylinder simultaneously, using different angles
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
    for xi in neuron_group.equations.stochastic_variables:
        neuron_group.variables.add_auxiliary_variable(
            xi, dimensions=(second**-0.5).dim
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


def wavelength_to_rgb(wavelength_nm, gamma=0.8):
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
