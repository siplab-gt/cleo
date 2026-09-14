# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "brian2==2.9.0",
#     "cleosim==0.18.1",
#     "marimo",
#     "matplotlib==3.10.3",
#     "seaborn==0.13.2",
#     "setuptools==80.9.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Ephys figure

    Adapted from the "Electrode recording" tutorial.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Preamble""")
    return


@app.cell
def _():
    import os

    import brian2.only as b2
    from brian2 import np
    import matplotlib.pyplot as plt
    import cleo
    import cleo.utilities
    from cleo import ephys

    # the default cython compilation target isn't worth it for
    # this trivial example
    b2.prefs.codegen.target = "numpy"
    seed = 18810929
    b2.seed(seed)
    np.random.seed(seed)
    cleo.utilities.set_seed(seed)

    cleo.utilities.style_plots_for_paper()

    # colors
    c = {
        "light": "#df87e1",
        "main": "#C500CC",
        "dark": "#8000B4",
        "exc": "#d6755e",
        "inh": "#056eee",
        "accent": "#36827F",
    }

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return b2, c, cleo, ephys, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Network setup""")
    return


@app.cell
def _(b2, c, cleo):
    N = 1000
    n_e = int(N * 0.8)
    n_i = int(N * 0.2)
    n_ext = 500

    neurons = b2.NeuronGroup(
        N,
        "dv/dt = -v / (10*ms) : 1",
        threshold="v > 1",
        reset="v = 0",
        refractory=2 * b2.ms,
    )
    ext_input = b2.PoissonGroup(n_ext, 24 * b2.Hz, name="ext_input")
    cleo.coords.assign_coords_rand_rect_prism(
        neurons, xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), zlim=(0.55, 0.9)
    )
    # need to create subgroups after assigning coordinates
    exc = neurons[:n_e]
    inh = neurons[n_e:]

    w0 = 0.06
    syn_exc = b2.Synapses(
        exc,
        neurons,
        f"w = {w0} : 1",
        on_pre="v_post += w",
        name="syn_exc",
        delay=1.5 * b2.ms,
    )
    syn_exc.connect(p=0.1)
    syn_inh = b2.Synapses(
        inh,
        neurons,
        f"w = -4*{w0} : 1",
        on_pre="v_post += w",
        name="syn_inh",
        delay=1.5 * b2.ms,
    )
    syn_inh.connect(p=0.1)
    syn_ext = b2.Synapses(
        ext_input, neurons, "w = .05 : 1", on_pre="v_post += w", name="syn_ext"
    )
    syn_ext.connect(p=0.1)

    # we'll monitor all spikes to compare with what we get on the electrode
    spike_mon = b2.SpikeMonitor(neurons)

    net = b2.Network(
        [neurons, exc, inh, syn_exc, syn_inh, ext_input, syn_ext, spike_mon]
    )
    sim = cleo.CLSimulator(net)
    cleo.viz.plot(
        exc, inh, colors=[c["exc"], c["inh"]], scatterargs={"alpha": 0.6}
    )
    return exc, inh, n_e, sim, spike_mon, syn_exc, syn_ext, syn_inh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Probe setup""")
    return


@app.cell
def _(b2, ephys):
    _coords = ephys.linear_shank_coords(
        1 * b2.mm, 32, start_location=(0, 0, 0.2) * b2.mm
    )
    probe = ephys.Probe(_coords, save_history=True)
    tklfp = ephys.TKLFPSignal()
    rwslfp = ephys.RWSLFPSignalFromSpikes()
    mua = ephys.MultiUnitActivity()
    ss = ephys.SortedSpiking()
    probe.add_signals(mua, ss, tklfp, rwslfp)
    return mua, probe, rwslfp, ss, tklfp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Measurement/detection and collision probability panels""")
    return


@app.cell
def _(b2, c, cleo_paper_style, mua, np, plt, ss):
    def meas_coll_plots():
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(4.6, 1.9), width_ratios=[1.8, 1], layout="constrained"
        )

        rlim = (0, 125)
        rr = np.linspace(*rlim, 100) * b2.um
        snr = mua.snr_by_distance(rr)
        ax1.plot(rr / b2.um, snr, c="k", label="SNR = $A_{\\mu s}/\\sigma_b$")
        ax1.axhline(
            mua.threshold_sigma, color="black", linestyle="--", label="threshold"
        )

        sigma_b = 1
        # ax1.hlines(1, 0, mua.r_noise_floor / b2.um, color='gray', linestyle='--', label='noise floor')
        # ax1.vlines(mua.r_noise_floor / b2.um, 0, 1, color='gray', linestyle='--')
        ax1.axhline(sigma_b, color="gray", linestyle="--", label="noise floor")

        sigma_ap = snr * mua.spike_amplitude_cv
        sigma_tot = np.sqrt(sigma_b**2 + sigma_ap**2)
        ax1.fill_between(
            rr / b2.um,
            snr - 2 * sigma_tot,
            snr + 2 * sigma_tot,
            alpha=0.3,
            label=r"$\pm 2 \sigma_{b+\text{AP}}$",
            color="gray",
        )
        ax1.fill_between(
            rr / b2.um,
            snr - 2 * sigma_ap,
            snr + 2 * sigma_ap,
            alpha=0.3,
            label=r"$\pm 2 \sigma_\text{AP}$",
            color=c["accent"],
        )

        ax1.set(
            xlabel="r (μm)",
            ylabel="SNR",
            title="EAP amplitude and detection probability",
            ylim=(0, 2 * mua.threshold_sigma),
            xlim=rlim,
        )
        # ax1.legend(loc='lower left', fontsize='small')

        # detection probabily twinx
        c_pd = c["dark"]
        ax1twin = ax1.twinx()
        ax1twin.set_ylabel(r"$p_\text{detect}$")
        p_dtct = mua.recall_by_distance(rr)
        ax1twin.plot(rr / b2.um, p_dtct, c=c_pd, label=r"$p_\text{detect}$")
        ax1twin.set(ylim=(-0.01, 1.01))
        # ax1twin.legend(loc='upper right', fontsize='small')
        fig.legend(
            fontsize="small",
            loc="upper right",
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=ax1.transAxes,
        )
        ax1twin.tick_params(axis="y", colors=c_pd)
        ax1twin.yaxis.label.set_color(c_pd)
        ax1twin.spines["right"].set_visible(True)
        ax1twin.spines["right"].set_color(c_pd)

        # Collision probability plot
        ############################
        t = np.linspace(0, 2) * b2.ms
        ax2.plot(t / b2.ms, mua.collision_prob_fn(t), c="k", label="multi-unit")
        ax2.plot(t / b2.ms, ss.collision_prob_fn(t), c="gray", label="sorted")
        ax2.set(
            xlabel=r"$t_\text{spike 2} - t_\text{spike 1}$ (ms)",
            ylabel=r"$p_\text{collision}$",
            title="Collision probability",
        )
        ax2.legend(fontsize="small")

        return fig


    with cleo_paper_style():
        meas_coll_plots().savefig("img/fig/ephys-meas-coll.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Collision probability panel""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3D network plot panel""")
    return


@app.cell
def _(c, cleo, cleo_paper_style, exc, inh, probe):
    with cleo_paper_style():
        fig, ax = cleo.viz.plot(
            exc,
            inh,
            colors=[c["exc"], c["inh"]],
            zlim=(0, 1200),
            devices=[(probe, {"size": 5})],
            scatterargs={"alpha": 0.3, "s": 2},
            figsize=(2, 3),
        )

        handles, labels = ax.get_legend_handles_labels()
        labels = ['exc', 'inh', 'probe']
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25))
        ax.set_xticks([-200, 200])
        ax.set_yticks([-200, 200])
        ax.set_zticks([0, 500, 1000])

        fig.savefig("img/fig/ephys-3d.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Specifying signals to record

    This looks right, but we need to specify what signals we want to pick up with our electrode.
    Here we'll explain the different options for spike and LFP recording.
    """
    )
    return


@app.cell
def _(b2, cleo, exc, inh, n_e, probe, sim, syn_exc, syn_ext, syn_inh):
    sim.set_io_processor(cleo.ioproc.RecordOnlyProcessor(sample_period=1 * b2.ms))
    sim.inject(
        probe,
        exc,
        tklfp_type="exc",
        ampa_syns=[syn_exc[f"j < {n_e}"], syn_ext[f"j < {n_e}"]],
        gaba_syns=[syn_inh[f"j < {n_e}"]],
    )
    sim.inject(probe, inh, tklfp_type="inh")
    sim_ready = True
    return (sim_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simulation and results

    Now we'll run the simulation:
    """
    )
    return


@app.cell
def _(b2, sim, sim_ready):
    if sim_ready:
        sim.run(250 * b2.ms)
    sim_done = True
    return (sim_done,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    And plot the output of the four signals we've recorded.
    We'll compare our recorded spikes to the ground truth for reference:
    """
    )
    return


@app.cell
def _(
    b2,
    c,
    cleo_paper_style,
    exc,
    mua,
    n_e,
    np,
    plt,
    probe,
    sim_done,
    spike_mon,
    ss,
):
    if not sim_done:
        print("sim not done")
    with cleo_paper_style():
        _fig, _axs = plt.subplots(
            3, 1, sharex=True, layout="constrained", figsize=(3, 3)
        )
        spikes_are_exc = spike_mon.i < n_e
        i_sorted_is_exc = np.array([ng == exc for ng, i in ss.i_ng_by_i_sorted])
        sorted_spikes_are_exc = i_sorted_is_exc[ss.i]
        for celltype, i_all, i_srt in [
            ("exc", spikes_are_exc, sorted_spikes_are_exc),
            ("inh", ~spikes_are_exc, ~sorted_spikes_are_exc),
        ]:
            _axs[0].plot(
                spike_mon.t[i_all] / b2.ms,
                spike_mon.i[i_all],
                ".",
                c=c[celltype],
                rasterized=True,
                label=celltype,
                ms=0.1,
            )
            _axs[1].plot(
                ss.t[i_srt] / b2.ms,
                ss.i[i_srt],
                ".",
                c=c[celltype],
                label=celltype,
                rasterized=True,
                ms=2,
            )
        # _axs[0].legend()
        _axs[0].set(ylabel="neuron #", title="ground-truth spikes")
        _axs[1].set(title="sorted spikes", ylabel="sorted unit #")
        _axs[2].plot(
            mua.t / b2.ms, mua.i, ".", color="xkcd:charcoal", rasterized=True, ms=1
        )
        _axs[2].set(
            title="multi-unit activity",
            ylabel="channel #",
            xlabel="time (ms)",
            ylim=[-0.5, probe.n - 0.5],
        )

        _fig.suptitle('Spikes from simulated network:')
        _fig.savefig('img/fig/ephys-spikes.svg')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    TKLFP supposedly outputs a value with an absolute scale in terms of μV, though it is quite high compared to $\pm0.1$ μV scale of RWSLFP as given in Mazzoni, Lindén et al., 2015.
    RWSLFP outputs unnormalized LFP instead of this $\pm0.1$ μV range to sidestep the complications of normalizing in a causal, stepwise manner.
    """
    )
    return


@app.cell
def _(b2, c, cleo_paper_style, np, plt, probe, rwslfp, sim_done, tklfp):
    if not sim_done:
        print("sim not done")
    from matplotlib.colors import LinearSegmentedColormap

    with cleo_paper_style():
        _fig, _axs = plt.subplots(
            1, 2, figsize=(2.9, 5), sharey=False, layout="constrained"
        )
        for _ax, lfp, _title in [
            (_axs[0], tklfp.lfp / b2.uvolt, "TKLFP"),
            (_axs[1], rwslfp.lfp, "RWSLFP"),
        ]:
            channel_offsets = -np.abs(np.quantile(lfp, 0.9)) * np.arange(probe.n)
            lfp2plot = lfp + channel_offsets
            _ax.plot(lfp2plot, color="xkcd:charcoal", lw=1)
            _ax.set(yticks=channel_offsets, xlabel="t (ms)", title=_title)
            extent = (0, 250, lfp2plot.min(), lfp2plot.max())
            cmap = LinearSegmentedColormap.from_list(
                "lfp", [c["accent"], "white", c["main"]]
            )
            im = _ax.imshow(
                lfp.T,
                aspect="auto",
                cmap=cmap,
                extent=extent,
                vmin=-np.max(np.abs(lfp)),
                vmax=np.max(np.abs(lfp)),
            )
        _fig.colorbar(im, aspect=40, label="LFP (a.u.)", ticks=[])
        _axs[0].set(ylabel="channel #", yticklabels=range(1, 33))
        _axs[1].set(yticklabels=[])
        _fig.suptitle("LFP proxies from simulated spikes:")
        _fig.savefig("img/fig/ephys-lfp.svg")
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
