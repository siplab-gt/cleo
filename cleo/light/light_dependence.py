from __future__ import annotations

import warnings
from typing import Callable, Tuple

from attrs import define, field
from brian2 import NeuronGroup, mm, np
from scipy.interpolate import CubicSpline

from cleo.coords import assign_xyz
from cleo.utilities import brian_safe_name, wavelength_to_rgb


def linear_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return np.interp(lambda_new_nm, lambdas_nm, epsilons)


def cubic_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return CubicSpline(lambdas_nm, epsilons)(lambda_new_nm)


# hacky MRO stuff...multiple inheritance only works because slots=False,
# and must be placed *before* SynapseDevice to work right
@define(eq=False, slots=False)
class LightDependent:
    """Mix-in class for opsin and light-dependent indicator.
    Light-dependent devices are connected to light sources (and vice-versa)
    on injection via the registry.

    We approximate dynamics under multiple wavelengths using a weighted sum
    of photon fluxes, where the ε factor indicates the activation
    relative to the peak-sensitivity wavelength for a equivalent power, which
    most papers report. When they report the action spectrum for equivalent
    photon flux instead (see Mager et al, 2018), use :func:`equal_photon_flux_spectrum`.
    This weighted sum is an approximation of a nonlinear
    peak-non-peak wavelength relation; see ``notebooks/multi_wavelength_model.ipynb``
    for details."""

    spectrum: list[tuple[float, float]] = field()
    """List of (wavelength, epsilon) tuples representing the action (opsin) or
    excitation (indicator) spectrum."""

    @spectrum.default
    def _default_spectrum(self):
        warnings.warn(
            f"No spectrum provided for light-dependent device {self.name}."
            " Assuming ε = 1 for all λ."
        )
        return [(-1e10, 1), (1e10, 1)]

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
            name=f"light_agg_{brian_safe_name(self.name)}_{target_ng.name}",
        )
        assign_xyz(
            light_agg_ng,
            target_ng.x[i_targets] / mm,
            target_ng.y[i_targets] / mm,
            target_ng.z[i_targets] / mm,
            unit=mm,
        )
        return light_agg_ng, list(range(len(i_targets)))

    def epsilon(self, lambda_new) -> float:
        """Returns the :math:`\\varepsilon` value for a given lambda (in nm)
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
        eps_new = self.spectrum_interpolator(lambdas, epsilons, lambda_new)
        if eps_new < 0:
            warnings.warn(f"ε = {eps_new} < 0 for {self.name}. Setting ε = 0.")
            eps_new = 0
        return eps_new


def equal_photon_flux_spectrum(
    spectrum: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Converts an equival photon flux spectrum to an equal power density spectrum."""
    spectrum = np.array(spectrum)
    lambdas = spectrum[:, 0]
    eps_phi = spectrum[:, 1]
    eps_Irr = eps_phi / lambdas
    eps_Irr /= np.max(eps_Irr)
    return list(zip(lambdas, eps_Irr))


def plot_spectra(*ldds: LightDependent) -> tuple[plt.Figure, plt.Axes]:
    """Plots the action/excitation spectra for multiple light-dependent devices."""
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
