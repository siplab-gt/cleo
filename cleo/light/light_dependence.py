from __future__ import annotations
from typing import Callable, Tuple
import warnings

from attrs import define, field, asdict, fields_dict
from brian2 import (
    np,
    Synapses,
    Function,
    NeuronGroup,
    Unit,
    BrianObjectException,
    get_unit,
    Equations,
    implementation,
    check_units,
)
from nptyping import NDArray
from brian2.units import (
    mm,
    mm2,
    nmeter,
    Quantity,
    second,
    ms,
    second,
    psiemens,
    nsiemens,
    mV,
    volt,
    amp,
    mM,
)
from brian2.units.allunits import radian
import numpy as np
from scipy.interpolate import CubicSpline

from cleo.base import InterfaceDevice, SynapseDevice
from cleo.coords import assign_coords
from cleo.utilities import wavelength_to_rgb


def linear_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return np.interp(lambda_new_nm, lambdas_nm, epsilons)


def cubic_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return CubicSpline(lambdas_nm, epsilons)(lambda_new_nm)


# hacky MRO stuff...multiple inheritance only works because slots=False,
# and must be placed *before* SynapseDevice to work right
@define(eq=False, slots=False)
class LightDependent:
    """Mix-in class for opsin and indicator. TODO

    We approximate dynamics under multiple wavelengths using a weighted sum
    of photon fluxes, where the ε factor indicates the activation
    relative to the peak-sensitivity wavelength for an equivalent number of photons
    (see Mager et al, 2018). This weighted sum is an approximation of a nonlinear
    peak-non-peak wavelength relation; see ``notebooks/multi_wavelength_model.ipynb``
    for details."""

    spectrum: list[tuple[float, float]] = field(factory=lambda: [(-1e10, 1), (1e10, 1)])
    """List of (wavelength, epsilon) tuples representing the action (opsin) or
    excitation (indicator) spectrum."""

    spectrum_interpolator: Callable = field(default=cubic_interpolator, repr=False)
    """Function of signature (lambdas_nm, epsilons, lambda_new_nm) that interpolates
    the action spectrum data and returns :math:`\\varepsilon \\in [0,1]` for the new
    wavelength."""

    @property
    def light_agg_ngs(self):
        return self.source_ngs

    def _get_source_for_synapse(
        self, target_ng: NeuronGroup, i_targets: list[int]
    ) -> Tuple[NeuronGroup, list[int]]:
        # create light aggregator neurons
        light_agg_ng = NeuronGroup(
            len(i_targets),
            model="""
            phi : 1/second/meter**2
            Irr : watt/meter**2
            """,
            name=f"light_agg_{self.name}_{target_ng.name}",
        )
        assign_coords(
            light_agg_ng,
            target_ng.x[i_targets] / mm,
            target_ng.y[i_targets] / mm,
            target_ng.z[i_targets] / mm,
            unit=mm,
        )
        return light_agg_ng, list(range(len(i_targets)))

    def epsilon(self, lambda_new) -> float:
        """Returns the epsilon value for a given lambda (in nm)
        representing the relative sensitivity of the opsin to that wavelength."""
        action_spectrum = np.array(self.spectrum)
        lambdas = action_spectrum[:, 0]
        epsilons = action_spectrum[:, 1]
        if lambda_new < min(lambdas) or lambda_new > max(lambdas):
            warnings.warn(
                f"λ = {lambda_new} nm is outside the range of the action spectrum data"
                f" for {self.name}. Assuming ε = 0."
            )
            return 0
        return self.spectrum_interpolator(lambdas, epsilons, lambda_new)


def plot_spectra(*ldds: LightDependent):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for ldd in ldds:
        spectrum = np.array(ldd.spectrum)
        lambdas = spectrum[:, 0]
        epsilons = spectrum[:, 1]
        lambdas_new = np.linspace(min(lambdas), max(lambdas), 100)
        epsilons_new = ldd.spectrum_interpolator(lambdas, epsilons, lambdas_new)
        c_points = [wavelength_to_rgb(l) for l in lambdas]
        c_line = wavelength_to_rgb(lambdas_new[np.argmax(epsilons_new)])
        ax.plot(lambdas_new, epsilons_new, c=c_line, label=ldd.name)
        ax.scatter(lambdas, epsilons, marker="o", s=50, color=c_points)
    title = (
        "Action/excitation spectra" if len(ldds) > 1 else f"Action/excitation spectrum"
    )
    ax.set(xlabel="λ (nm)", ylabel="ε", title=title)
    fig.legend()
    return fig, ax
