# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "brian2==2.8.0.4",
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==1.26.4",
#     "seaborn==0.13.2",
#     "setuptools==80.3.1",
#     "tklfp==0.2.1",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Reproducing original Teleńczuk et al., 2020 demo
        If we did this right, we should see similar output to the [original demo](https://zenodo.org/record/3866253#.Ydyk6GDMKUk) using spikes from a Brunel-Wang E-I network.

        We obtain that demo output from `telenzcuk20_demo.py`, which is taken straight from the original paper and modified with a few corrections (explained in the docstring there).
        You'll need to run that file first to produce `data/telenczuk20_demo.npz`.

        Our goal here is to check that the `tklfp` package and Cleo's wrapper around it `TKLFPSignal` both reproduce Teleńczuk et al., 2020's results.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Read data""")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(np):
    data = np.load("data/telenczuk20_demo.npz")

    # # magic command not supported in marimo; please file an issue to add support
    # # %load_ext autoreload
    # # '%autoreload 2' command supported automatically in marimo

    N = 5000  # nb of cells to consider
    Ne = 4000  # nb of excitatory cells
    Ni = 1000  # nb of inhibitory cells

    tmin = 9000  # min time (to skip)
    tmax = 1000  # max time

    # # we read in the data the same as the original code
    # dtype = {"names": ["cellid", "time"], "formats": ["i4", "f8"]}
    # inh_cells = np.loadtxt("brunel_inh.txt", dtype=dtype)
    # exc_cells = np.loadtxt("brunel_exc.txt", dtype=dtype)

    # # adjust time and convert to ms
    # inh_cells["time"] = inh_cells["time"] * 1000 - tmin
    # exc_cells["time"] = exc_cells["time"] * 1000 - tmin
    # # for inhibitory cells, ids start from Ne
    # inh_cells["cellid"] += Ne
    data.keys()
    return N, Ne, data, tmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Spatial data:""")
    return


@app.cell
def _(N, data, np):
    xmax = ymax = 0.2
    X, Y = data["x"], data["y"]
    Z = np.zeros(N)  # record with all cells at soma layer
    return X, Y, Z, xmax, ymax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Calculate using `tklfp` package""")
    return


@app.cell
def _(N, Ne, X, Y, Z, data, np, tmax, xmax, ymax):
    from tklfp import TKLFP

    # create a vector representing cell type
    is_excitatory = np.arange(N) < Ne
    tklfp = TKLFP(X, Y, Z, is_excitatory, elec_coords_mm=[[xmax / 2, ymax / 2, 0]])

    dt = 0.1  # time resolution
    npts = int(tmax / dt)
    t_eval_ms = np.arange(npts) * dt

    lfp = tklfp.compute(data["i_spk"], data["t_spk"], t_eval_ms)
    return lfp, t_eval_ms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Calculate using Cleo's `TKLFPSignal`""")
    return


@app.cell
def _():
    import cleo
    import brian2.only as b2

    b2.prefs.codegen.target = "numpy"
    return b2, cleo


@app.cell
def _(N, Ne, X, Y, Z, b2, cleo, data, tmax, xmax, ymax):
    # wish I could just subgroup SGG, but not supported
    # ng = b2.SpikeGeneratorGroup(N, data["i_spk"], data["t_spk"] * b2.ms)
    from_exc = data["i_spk"] < Ne
    ng_e = b2.SpikeGeneratorGroup(
        N, data["i_spk"][from_exc], data["t_spk"][from_exc] * b2.ms
    )
    ng_i = b2.SpikeGeneratorGroup(
        N, data["i_spk"][~from_exc] - Ne, data["t_spk"][~from_exc] * b2.ms
    )
    for var in ["x", "y", "z"]:
        ng_e.add_attribute(var)
        ng_i.add_attribute(var)

    ng_e.x = X[:Ne] * b2.mm
    ng_e.y = Y[:Ne] * b2.mm
    ng_e.z = Z[:Ne] * b2.mm
    ng_i.x = X[Ne:] * b2.mm
    ng_i.y = Y[Ne:] * b2.mm
    ng_i.z = Z[Ne:] * b2.mm

    sim = cleo.CLSimulator(b2.Network(ng_e, ng_i))
    sim.set_io_processor(cleo.ioproc.RecordOnlyProcessor(b2.ms))
    tklfp_signal = cleo.ephys.TKLFPSignal()
    probe = cleo.ephys.Probe(
        [xmax / 2, ymax / 2, 0] * b2.mm, signals=[tklfp_signal]
    )
    sim.inject(probe, ng_e, tklfp_type="exc")
    sim.inject(probe, ng_i, tklfp_type="inh")
    sim.run(tmax * b2.ms)
    return from_exc, tklfp_signal


@app.cell
def _(tklfp_signal):
    tklfp_signal.lfp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot raster and LFP""")
    return


@app.cell
def _(
    b2,
    cleo_paper_style,
    data,
    from_exc,
    lfp,
    plt,
    t_eval_ms,
    tklfp_signal,
    tmax,
):
    with cleo_paper_style():
        Nstp = 10  # step cell to draw
        tick_size = 2

        # plt.style.use("seaborn-v0_8-paper")
        fig, axes = plt.subplots(
            2, 1, figsize=(6.25, 4), sharex=True, layout="constrained"
        )

        axes[0].plot(
            # exc_cells[::Nstp]["time"],
            # exc_cells[::Nstp]["cellid"],
            data["t_spk"][from_exc][::Nstp],
            data["i_spk"][from_exc][::Nstp],
            ".",
            c="xkcd:tomato",
            ms=tick_size,
            rasterized=True,
        )
        axes[0].plot(
            data["t_spk"][~from_exc][::Nstp],
            data["i_spk"][~from_exc][::Nstp],
            # inh_cells[::Nstp]["time"],
            # inh_cells[::Nstp]["cellid"],
            ".",
            c="xkcd:cerulean blue",
            ms=tick_size,
            rasterized=True,
        )

        axes[1].plot(
            data["time"], data["lfp"], c="k", lw=3, label="Teleńczuk et al., 2020"
        )
        axes[1].plot(t_eval_ms, lfp, c="#8000b4", lw=2, label="tklfp")
        axes[1].plot(
            tklfp_signal.t / b2.ms,
            tklfp_signal.lfp / b2.uvolt,
            c="#df87e1",
            lw=1,
            label="Cleo",
            ls="--",
        )
        axes[1].plot()
        axes[1].set_xlabel("time (ms)")
        axes[1].set_xlim(0, tmax)
        axes[1].legend()

        # prettify graph
        # axes[0].spines["top"].set_visible(False)
        # axes[0].spines["right"].set_visible(False)
        # axes[1].spines["top"].set_visible(False)
        # axes[1].spines["right"].set_visible(False)

        axes[0].set(ylabel="neuron index")
        axes[1].set(ylabel="LFP (μV)")
        fig.savefig("img/fig/tklfp-verification.pdf", transparent=True, dpi=300)
    return


@app.cell
def _(cleo, plt):
    class cleo_paper_style:
        def __enter__(self):
            self.original_rcParams = plt.rcParams.copy()
            plt.rcdefaults()
            plt.rcParams["axes.facecolor"] = "white"
            plt.rcParams["figure.facecolor"] = "white"
            cleo.utilities.style_plots_for_paper()

        def __exit__(self, exc_type, exc_value, traceback):
            plt.rc("savefig", transparent=False)
            plt.show()
            plt.rcParams.update(self.original_rcParams)
    return (cleo_paper_style,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
