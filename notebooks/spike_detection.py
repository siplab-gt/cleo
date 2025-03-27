# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "openai==1.68.2",
#     "scipy==1.15.2",
# ]
# ///

import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # r_thresh_slider = mo.ui.slider(
    #     10, 200, 10, 50, label="r_threshold (μm)", show_value=True
    # )
    # r_thresh_slider
    param_sliders = mo.ui.dictionary(
        {
            param: mo.ui.slider(
                start, stop, step, default, label=param, show_value=True
            )
            for param, start, stop, step, default in [
                ("r_half_detection", 10, 200, 10, 100),
                ("threshold_sigma", 1, 10, 1, 4),
                ("spike_amplitude_cv", 0, 1, 0.1, 0.2),
                ("r0", 0, 40, 1, 5),
            ]
        }
    )
    param_sliders.vstack()
    return (param_sliders,)


@app.cell(hide_code=True)
def _(param_sliders):
    params = {key: slider.value for key, slider in param_sliders.items()}
    return (params,)


@app.cell
def _(params):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm


    def spike_amplitude(radius):
        return (
            params["threshold_sigma"]
            * (params["r_half_detection"] + params["r0"])
            / (radius + params["r0"])
        )


    _r = np.linspace(5, 400, 100)
    _amp = spike_amplitude(_r)
    _sigma_ap = _amp * params["spike_amplitude_cv"]
    _sigma_noise = 1
    _sigma = _sigma_ap + _sigma_noise
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(_r, _amp, label="average spike amplitude")
    axs[0].fill_between(
        _r,
        _amp - 2 * _sigma,
        _amp + 2 * _sigma,
        alpha=0.3,
        label=r"$\pm 2 (\sigma_\text{AP} + \sigma_\text{noise})$",
    )
    axs[0].fill_between(
        _r,
        _amp - 2 * _sigma_noise,
        _amp + 2 * _sigma_noise,
        alpha=0.3,
        label=r"$\pm 2 \sigma_\text{noise}$",
    )
    axs[0].axhline(params["threshold_sigma"], ls="--", label="detection threshold")
    axs[0].legend()
    axs[0].set(
        ylabel="measured spike amplitude\n(normalized by $\\sigma_\\text{noise}$)"
    )

    _p_dtct = norm.sf(params["threshold_sigma"], loc=_amp, scale=_sigma)
    axs[1].plot(_r, _p_dtct)
    axs[1].axvline(
        params["r_half_detection"], ls="--", label=r"$r_\text{half detection}$"
    )
    axs[1].set(xlabel="r (μm)", ylabel="p(detection)")
    axs[1].legend()
    fig
    return axs, fig, norm, np, plt, spike_amplitude


if __name__ == "__main__":
    app.run()
