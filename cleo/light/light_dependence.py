from __future__ import annotations

import warnings
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from attrs import define, field
from brian2 import NeuronGroup, mm, np
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    interp1d,
)

from cleo.coords import assign_xyz
from cleo.utilities import brian_safe_name, wavelength_to_rgb


def linear_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    # return np.interp(lambda_new_nm, lambdas_nm, epsilons)
    return interp1d(lambdas_nm, epsilons, fill_value="extrapolate")(lambda_new_nm)


def cubic_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return CubicSpline(lambdas_nm, epsilons)(lambda_new_nm)


def pchip_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return PchipInterpolator(lambdas_nm, epsilons)(lambda_new_nm)


def makima_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    intrp = Akima1DInterpolator(lambdas_nm, np.log(epsilons))
    intrp.extrapolate = True
    intrp.method = "makima"
    return intrp(lambda_new_nm)


def _log_(fn, lambdas_nm, epsilons, lambda_new_nm):
    assert isinstance(epsilons, np.ndarray)
    epsilons[epsilons == 0] = 1e-10
    return np.exp(fn(lambdas_nm, np.log(epsilons), lambda_new_nm))


def log_linear_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return _log_(linear_interpolator, lambdas_nm, epsilons, lambda_new_nm)


def log_pchip_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return _log_(pchip_interpolator, lambdas_nm, epsilons, lambda_new_nm)


def log_makima_interpolator(lambdas_nm, epsilons, lambda_new_nm):
    return _log_(makima_interpolator, lambdas_nm, epsilons, lambda_new_nm)


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

    spectrum_interpolator: Callable = field(default=log_pchip_interpolator, repr=False)
    """Function of signature (lambdas_nm, epsilons, lambda_new_nm) that interpolates
    the action spectrum data and returns :math:`\\varepsilon \\in [0,1]` for the new
    wavelength."""
    extrapolate: bool = False
    """Whether or not to attempt to extrapolate (using :attr:`spectrum_interpolator`)
    outside of the provided excitation/action spectrum."""

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
        lambdas, epsilons = np.array(self.spectrum).T
        eps_new = self.spectrum_interpolator(lambdas, epsilons, lambda_new)

        # out of data range
        if lambda_new < min(lambdas) or lambda_new > max(lambdas):
            if not self.extrapolate:
                warnings.warn(
                    f"λ = {lambda_new} nm is outside the range of the action spectrum data"
                    f" for {self.name} and extrapolate=False. Assuming ε = 0."
                )
                return 0
            elif self.extrapolate:
                warnings.warn(
                    f"λ = {lambda_new} nm is outside the range of the action spectrum data"
                    f" for {self.name}. Extrapolating: ε = {eps_new:.3f}."
                )
        if eps_new < 0:
            warnings.warn(f"ε = {eps_new} < 0 for {self.name}. Setting ε = 0.")
            eps_new = 0
        if eps_new > 1:
            warnings.warn(f"ε = {eps_new} > 1 for {self.name}. Setting ε = 1.")
            eps_new = 1
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


def plot_spectra(
    *ldds: LightDependent, extrapolate=False
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the action/excitation spectra for multiple light-dependent devices."""
    import matplotlib.pyplot as plt

    if extrapolate:
        all_lambdas = [np.array(ldd.spectrum)[:, 0] for ldd in ldds]
        lambda_min = min([lambdas.min() for lambdas in all_lambdas])
        lambda_max = max([lambdas.max() for lambdas in all_lambdas])

    fig, ax = plt.subplots()
    for ldd in ldds:
        lambdas, epsilons = np.array(ldd.spectrum).T
        if not extrapolate:
            lambda_min = np.min(lambdas)
            lambda_max = np.max(lambdas)
        lambdas_new = np.linspace(lambda_min, lambda_max, 100)
        epsilons_new = ldd.spectrum_interpolator(lambdas, epsilons, lambdas_new)
        c_points = [wavelength_to_rgb(l) for l in lambdas]
        c_line = wavelength_to_rgb(lambdas[np.argmax(epsilons)])
        ax.plot(lambdas_new, epsilons_new, c=c_line, label=ldd.name)
        ax.scatter(lambdas, epsilons, marker="o", s=50, color=c_points)
    title = (
        "Action/excitation spectra" if len(ldds) > 1 else f"Action/excitation spectrum"
    )
    ax.set(xlabel="λ (nm)", ylabel="ε", title=title)
    fig.legend()
    return fig, ax
