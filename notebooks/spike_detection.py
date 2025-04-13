# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "allensdk==2.16.2",
#     "marimo",
#     "matplotlib==3.10.1",
#     "nbformat==5.10.4",
#     "numpy==1.23.5",
#     "openai==1.68.2",
#     "polars==1.27.1",
#     "pyarrow==19.0.1",
#     "scipy==1.10.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(
    width="medium",
    app_title="Spike detection methodology",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Spike detection methodology""")
    return


@app.cell
def _(sampled_cells):
    sampled_cells.to_pandas()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To simulate spike detection in a real experiment, we consider background noise with standard deviation $\sigma_\text{noise}=1$ as the anchor parameter all other parameters are based on.
        As for the other parameters:

        - `r_noise_floor`: the radius at which the average extracellular action potential amplitude and the noise standard deviation are equal, i.e., $A(r_\text{noise floor}) = \sigma_\text{noise}=1$. What would reasonable values look like?
            - [Henze et al. (2000)](http://www.physiology.org/doi/10.1152/jn.2000.84.1.390) experimentally measure that the amplitude drops to ~0 around 140 μm
            - 200 μm is the largest radius described in [Pettersen et al., 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2186261/). They also state distances up to 100 μm are most important for spike detection, beyond which "it is difficult to extract spikes from the background noise level" (citing Buzsaki lab papers such as Henze et al., 2000)
            - [Cohen and Miles (2000)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2269886/) describe 80 μm as the distance where amplitude roughly equals noise.
            - [Somogyvari et al. (2005)](https://www.sciencedirect.com/science/article/pii/S0165027005001093) found the amplitude decayed to ~0 around 200-300 μm.
            - All of this points to 80 μm as a sensible default value.
        - `threshold_sigma`: the threshold above which an extracellular potential is classified as an action potential, as a multiple of $\sigma_\text{noise}$. Values in real experiments typically range from 3 to 6.
        - `spike_amplitude_cv`: The coefficient of variation of spike amplitudes, i.e., $\sigma_A/\mu_A$.
            - Can't find any good cited values. From randomly chosen mouse neurons in the [Allen Cell Type data](https://celltypes.brain-map.org/), it looks like 0-0.15 is a typical range, with the median around 0.05. See [below](#estimating-amplitude-cv-from-data). 
        - `r0`: A small distance added to $r$ before computing the amplitude to avoid division by 0 for the power law decay. It also makes some physical sense as the minimum distance from the current source it is possible to place an electrode, 5 μm being reasonable as the radius of a typical soma.
        - `amp_decay_power`: The power with which the amplitude decays, i.e., $A'(r) = r^{- \texttt{amp\_decay\_power}}$, where $A'(r)$ is the unshifted, unscaled amplitude function.

        Putting these parameters together, the final form of the amplitude function is

        $$A(r) = \frac{A'(r + r_0)}{A'(r_\text{noise floor} + r_0)} \qquad A'(r) = \frac{1}{r^{\texttt{amp\_decay\_power}}}$$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    param_sliders = mo.ui.dictionary(
        {
            param: mo.ui.slider(
                start, stop, step, default, label=f"`{param}`", show_value=True
            )
            for param, start, stop, step, default in [
                ("r_noise_floor", 40, 200, 10, 80),
                ("threshold_sigma", 1, 10, 1, 4),
                ("spike_amplitude_cv", 0, 0.3, 0.01, 0.05),
                ("r0", 0, 20, 1, 5),
                ("amp_decay_power", 0, 4, 0.2, 2),
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


    def spike_amp_unscaled(r):
        return r ** -params["amp_decay_power"]


    def spike_amplitude(radius):
        return spike_amp_unscaled(radius + params["r0"]) / spike_amp_unscaled(
            params["r_noise_floor"] + params["r0"]
        )


    rr = np.linspace(5, 300, 100)
    _amp = spike_amplitude(rr)
    _sigma_ap = _amp * params["spike_amplitude_cv"]
    _sigma_noise = 1
    _sigma = _sigma_ap + _sigma_noise
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(rr, _amp, label="average spike amplitude")
    axs[0].fill_between(
        rr,
        _amp - 2 * _sigma,
        _amp + 2 * _sigma,
        alpha=0.3,
        label=r"$\pm 2 (\sigma_\text{AP} + \sigma_\text{noise})$",
    )
    axs[0].fill_between(
        rr,
        _amp - 2 * _sigma_noise,
        _amp + 2 * _sigma_noise,
        alpha=0.3,
        label=r"$\pm 2 \sigma_\text{noise}$",
    )
    axs[0].axhline(params["threshold_sigma"], ls="--", label="detection threshold")
    axs[0].legend()
    axs[0].set(
        ylabel="measured spike amplitude\n(normalized by $\\sigma_\\text{noise}$)",
        ylim=[0, 11],
    )

    p_dtct = norm.sf(params["threshold_sigma"], loc=_amp, scale=_sigma)
    axs[1].plot(rr, p_dtct)
    # axs[1].axvline(
    #     params["r_half_detection"], ls="--", label=r"$r_\text{half detection}$"
    # )
    axs[1].set(xlabel="r (μm)", ylabel="p(detection)")
    fig
    return (
        axs,
        fig,
        norm,
        np,
        p_dtct,
        plt,
        rr,
        spike_amp_unscaled,
        spike_amplitude,
    )


@app.cell(hide_code=True)
def _(mo, r_half_detection):
    mo.md(
        rf"""
        ## Relating to previous methodology

        How does this relate to Cleo's previous versions, where we specified `r_perfect_detection` and `r_half_detection`?
        Well, there is no longer any guarantee of perfect detection due to noise.
        As for 50% detection?
        That would be where the amplitude meets the detection threshold.
        Given our current settings of `threshold_sigma`, `spike_amplitude_cv`, and `r0`, we get

        `r_half_detection = {r_half_detection:.2f} μm`
        """
    )
    return


@app.cell
def _(np, p_dtct, rr):
    r_half_detection = rr[np.where(p_dtct < 0.5)[0][0]]
    return (r_half_detection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Estimating amplitude CV from data

        Since I can't find good cited values for this, let's estimate from the [Allen Cell Types database](https://celltypes.brain-map.org/).
        Following [their documentation](https://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html).
        """
    )
    return


@app.cell
def _():
    import polars as pl
    from allensdk.core.cell_types_cache import CellTypesCache
    from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
    from allensdk.api.queries.cell_types_api import CellTypesApi

    # Instantiate the CellTypesCache instance.  The manifest_file argument
    # tells it where to store the manifest, which is a JSON file that tracks
    # file paths.  If you supply a relative path (like this), it will go
    # into your current working directory
    ctc = CellTypesCache(manifest_file="~/.cache/cell_types/manifest.json")
    cells = pl.DataFrame(ctc.get_cells())
    cells
    return (
        CellTypesApi,
        CellTypesCache,
        EphysSweepFeatureExtractor,
        cells,
        ctc,
        pl,
    )


@app.cell(hide_code=True)
def _(mo, n_cells):
    mo.md(
        rf"""We're going to randomly sample {n_cells} mouse cells, reporter-positive so we can tell their cell types:"""
    )
    return


@app.cell
def _(cells, pl):
    n_cells = 20
    sampled_cells = cells.filter(
        (pl.col("species") == "Mus musculus")
        & (pl.col("reporter_status") == "positive")
    ).sample(n=n_cells, seed=18810929)
    cell_ids = sampled_cells["id"]
    sampled_cells
    return cell_ids, n_cells, sampled_cells


@app.cell
def _(ctc, pl):
    # @mo.persistent_cache
    def get_sweep_for_cell(cell_id):
        data_set = ctc.get_ephys_data(cell_id)
        # print("sweep metadata:")
        # print(ctc.get_ephys_sweeps(cell_id)[sweep_number - 1])
        sweep_numbers = data_set.get_experiment_sweep_numbers()

        sweeps = pl.DataFrame(ctc.get_ephys_sweeps(cell_id))
        # get sweep with most spikes
        sweep_number = sweeps.sort(by="num_spikes", descending=False)[
            -1, "sweep_number"
        ]

        print(f"{cell_id=}")
        print(f"{sweep_number=}")
        return sweep_number
    return (get_sweep_for_cell,)


@app.cell
def _(
    EphysSweepFeatureExtractor,
    cell_ids,
    ctc,
    get_sweep_for_cell,
    mo,
    np,
    plt,
):
    @mo.persistent_cache
    def estimate_cv(cell_id):
        sweep_number = get_sweep_for_cell(cell_id)
        data_set = ctc.get_ephys_data(cell_id)
        sweep_data = data_set.get_sweep(sweep_number)

        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0 : index_range[1] + 1]  # in A
        v = sweep_data["response"][0 : index_range[1] + 1]  # in V
        i *= 1e12  # to pA
        v *= 1e3  # to mV

        sampling_rate = sweep_data["sampling_rate"]  # in Hz
        t = np.arange(0, len(v)) * (1.0 / sampling_rate)

        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(t, v)
        axes[1].plot(t, i)
        axes[0].set_ylabel("mV")
        axes[1].set_ylabel("pA")
        axes[1].set_xlabel("seconds")

        sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i)
        sweep_ext.process_spikes()
        t_spk = sweep_ext.spike_feature("peak_t")
        v_spk = sweep_ext.spike_feature("peak_v")
        v_thresh = sweep_ext.spike_feature("threshold_v")
        axes[0].plot(t_spk, v_spk, "r.")

        plt.show()
        # using threshold to peak as amplitude
        amp = v_spk - v_thresh
        return np.std(amp) / np.abs(np.mean(amp))


    # estimate_cv(464212183, 102)
    cvs = []
    for cell_id in cell_ids:
        cvs.append(estimate_cv(cell_id))
    return cell_id, cvs, estimate_cv


@app.cell
def _(cvs, pl, sampled_cells):
    sampled_cells_with_cvs = sampled_cells.with_columns(pl.DataFrame({"cv": cvs}))
    sampled_cells_with_cvs
    return (sampled_cells_with_cvs,)


@app.cell
def _(sampled_cells_with_cvs):
    import seaborn as sns

    sns.displot(
        sampled_cells_with_cvs.to_pandas(),
        x="cv",
        multiple="stack",
        kde=True,
        hue="dendrite_type",
    )
    return (sns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It appears that under 0.15 is most common, and that spiny dendrite (≈excitatory) cells have very low CVs compared to aspiny dendrite (≈inhibitory) cells.
        Let's use the median to prescribe a default:
        """
    )
    return


@app.cell
def _(cvs, np):
    np.median(cvs)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
