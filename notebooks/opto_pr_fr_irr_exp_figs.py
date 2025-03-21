# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "bayesian-optimization==2.0.3",
#     "brian2==2.6.0",
#     "cleosim==0.17.0",
#     "marimo",
#     "matplotlib==3.10.1",
#     "nbformat==5.10.4",
#     "pandas==2.2.3",
#     "seaborn==0.13.2",
#     "setuptools==77.0.1",
#     "watchdog==6.0.0",
# ]
# [tool.marimo.runtime]
# watcher_on_save = "autorun"
# ///

import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Firing vs. stimulation frequency for different irradiance and expression levels

        ## Source data
        Here we'll replicate figure 4 of Foutz et al., 2012, relating firing rate to stimulation frequency for (a) a range of irradiances and (b) a range of channel densities. Stimulation is a train of 5-ms pulses.

        Original figure (left) and replotted, interpolated data (right)
        """
    )
    return


@app.cell(hide_code=True)
def _(interp_figs, mo, plt):
    import matplotlib.image as mpimg

    # plotting like this since since markdown can only display images
    # in a public/ folder
    img = mpimg.imread("img/orig/foutz12_4.jpg")
    plt.imshow(img)
    plt.gca().set(xticks=[], yticks=[])
    mo.hstack([plt.gcf(), mo.vstack(interp_figs)])
    return img, mpimg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Imports, parameters, etc.""")
    return


@app.cell(hide_code=True)
def _():
    import bayes_opt as bo
    import brian2 as b2
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from brian2 import mm2, ms, mV, mwatt, namp, np
    from opto_val import adex, lif

    import cleo

    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = 0.2 * b2.ms
    cleo.utilities.style_plots_for_paper()
    return (
        adex,
        b2,
        bo,
        cleo,
        lif,
        mV,
        matplotlib,
        mm2,
        ms,
        mwatt,
        namp,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(ms, np):
    rho_rel_conds = [1, 1, 1, 1.5, 0.75]
    Irr_factor_conds = [1.4, 1.2, 1, 1.2, 1.2]
    n_rates = 20
    pulse_width = 5 * ms
    pulse_rates = np.linspace(0.1, 200, n_rates)
    sim_len_s = 0.4
    return (
        Irr_factor_conds,
        n_rates,
        pulse_rates,
        pulse_width,
        rho_rel_conds,
        sim_len_s,
    )


@app.cell
def rates_data_path():
    def rates_data_path(model_str):
        return f"data/pr_fr_irr_exp_{model_str}.csv"
    return (rates_data_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Get interpolated rates
        We need the original data at the same pulse rate values as we'll be simulating to compute an objective function
        """
    )
    return


@app.cell
def _(foutz12_data_combined, np, pd, pulse_rates):
    def _interp_group(group):
        return {
            "pulse_rate": pulse_rates,
            "firing_rate": np.interp(
                pulse_rates, group["pulse_rate"], group["firing_rate"]
            ),
        }

    interp_rates = (
        foutz12_data_combined.groupby(["rho_rel", "Irr0/Irr0_thres"])
        .apply(_interp_group, include_groups=False)
        .apply(pd.Series)
    )
    interp_rates
    return (interp_rates,)


@app.cell
def _(interp_rates, pd, plot_fr_2panel):
    interp_rates_tidy = interp_rates.apply(pd.Series.explode).reset_index()
    interp_rates_tidy["name"] = "MCHH_Markov"
    interp_rates_tidy = interp_rates_tidy.astype(
        {"pulse_rate": float, "firing_rate": float}
    )
    interp_rates_tidy
    interp_figs = plot_fr_2panel(
        interp_rates_tidy,
        palette1="blend:#df87e1,#8000b4",
        palette2="blend:#69fff8,#36827F",
        figheight=2,
        figwidth=2,
    )
    return interp_figs, interp_rates_tidy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Simulations
        We use Bayesian optimization to fit neuron models.
        In each iteration we must first find the threshold light input required to make a neuron spike with a 5 ms pulse.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    model_types = [
        "LIF_simple",
        "LIF_Markov",
        "AdEx_simple",
        "AdEx_Markov",
    ]
    run_checkboxes = mo.ui.dictionary(
        {model_str: mo.ui.checkbox(False, label=model_str) for model_str in model_types}
    )
    mo.vstack(["(Re-)run simulations?", run_checkboxes.hstack(justify="space-around")])
    return model_types, run_checkboxes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Initial param settings to probe [from Neuronal Dynamics](https://neuronaldynamics.epfl.ch/online/Ch6.S2.html#:~:text=Table%C2%A06.1%3A%20Exemplar%20parameters):"""
    )
    return


@app.cell(hide_code=True)
def _(pd):
    _data = {
        "Type": [
            "Tonic",
            "Adapting",
            "Init. burst",
            "Bursting",
            "Irregular",
            "Transient",
            "Delayed",
        ],
        # needs to be in alphabetical order for bayes_opt, apparently
        "a": [0.0, 0.0, 0.5, -0.5, -0.5, 1.0, -1.0],
        "b": [60, 5.0, 7.0, 7.0, 7.0, 10, 10],
        "tau_m": [20, 20, 5.0, 5.0, 9.9, 10, 5.0],
        "tau_w": [30.0, 100, 100, 100, 100, 100, 100],
        "v_reset": [-55, -55, -51, -46, -46, -60, -60],
    }

    # Creating the DataFrame
    ndx_params = pd.DataFrame(_data)
    ndx_params
    return (ndx_params,)


@app.cell
def _():
    return


@app.cell
def _():
    debug = True

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    return debug, dprint


@app.cell
def _(
    Irr_factor_conds,
    adex,
    b2,
    bo,
    cleo,
    debug,
    dprint,
    interp_rates_tidy,
    lif,
    mo,
    n_rates,
    ndx_params,
    np,
    pd,
    plt,
    pulse_rates,
    pulse_width,
    rho_rel_conds,
    sim_len_s,
):
    @mo.persistent_cache
    def optimal_rates(model_str):
        neuron_type, opsin_type = model_str.split("_")
        # neurons are divided into 5 segments for 5 different Irr/rho settings
        # in each segment, each neuron receives a different pulse rate
        # so for n_rates=20, it's pulse rates 1-20, 1-20, 1-20, 1-20, 1-20
        n = 5 * n_rates
        if neuron_type == "LIF":
            NG = lif
            params2opt = ["tau_m", "v_reset"]
        elif neuron_type == "AdEx":
            NG = adex
            params2opt = ["tau_m", "a", "tau_w", "b", "v_reset"]
        ng = NG(n, model_str)
        cleo.coords.assign_coords(ng, [0, 0, 0] * b2.mm)
        spmon = b2.SpikeMonitor(ng, record=True)
        if opsin_type == "simple":
            opsin = cleo.opto.ProportionalCurrentOpsin(
                I_per_Irr=b2.namp / (b2.mwatt / b2.mm2)
            )
        elif opsin_type == "Markov":
            opsin = cleo.opto.chr2_4s()

        sim = cleo.CLSimulator(b2.Network(ng, spmon))
        rho_rel = np.repeat(rho_rel_conds, n_rates)
        sim.inject(opsin, ng, rho_rel=rho_rel)

        opsin_ng = opsin.source_ngs[ng.name]
        dprint(opsin.synapses[ng.name].equations)
        dprint(opsin.synapses[ng.name].rho_rel)
        cleo.utilities.modify_model_with_eqs(
            opsin_ng,
            """Irr_factor : 1
            pulse_rate : hertz""",
        )
        dprint(opsin_ng.equations)

        # modify phi by Irr_factor
        opsin_ng.Irr_factor = np.repeat(Irr_factor_conds, n_rates)
        dprint(opsin_ng.Irr_factor)

        opsin_ng.pulse_rate = np.tile(pulse_rates, 5) * b2.Hz
        dprint(opsin_ng.pulse_rate)
        # set up opsin run_regularly for different rates
        opsin_ng.run_regularly(
            "phi = phi_thresh * Irr_factor * int((t % (1 / pulse_rate)) < pulse_width)",
            dt=b2.defaultclock.dt,
        )
        sim.network.store()

        @mo.cache
        def tune_phi_thresh(tau_m, precision=1e10):
            """include tau_m and R in args for caching purposes"""
            i2record = (
                2 * n_rates
            )  # should be neuron with Irr_ratio = rho_rel = 1 and low pulse rate
            search_min, search_max = (1e14, 1e20)
            while (
                search_max - search_min > precision
            ):  # get down to {precision} mW/mm2 margin
                phi_curr = (search_min + search_max) / 2
                opsin_ng.namespace["phi_thresh"] = phi_curr / b2.mm2 / b2.second
                sim.run(pulse_width)
                sim.run(2 * pulse_width)  # wait 10 ms to make sure only 1 spike
                # state monitor plots show phi is as expected
                if spmon.count[i2record] > 0:  # spiked
                    search_max = phi_curr
                else:
                    search_min = phi_curr
                sim.network.restore()
            dprint(f"phi_thresh={phi_curr} / mm2 / s")
            return phi_curr / b2.mm2 / b2.second

        def simulate_rates(params):
            # add units to params
            for name, unit in [
                ("tau_m", b2.ms),
                ("a", b2.nsiemens),
                ("tau_w", b2.ms),
                ("b", b2.pamp),
                ("v_reset", b2.mV),
            ]:
                if name in params:
                    params[name] *= unit

            ng.namespace |= params
            tune_phi_thresh(params["tau_m"])
            sim.run(sim_len_s * b2.second)

            rates = pd.DataFrame(
                {
                    # need to multiply by 1 to go from Brian VariableView to plain numpy array
                    "rho_rel": opsin.synapses[ng.name].rho_rel * 1,
                    "Irr0/Irr0_thres": opsin_ng.Irr_factor * 1,
                    "pulse_rate": opsin_ng.pulse_rate / b2.Hz,
                    "firing_rate": spmon.count / sim_len_s,
                    "name": model_str,
                }
            )
            if debug:
                plt.plot(spmon.t, spmon.i, ".")
                plt.show()
            # dprint(rates)
            sim.network.restore()
            return rates

        def eval_params(**params):
            rates = simulate_rates(params)
            merged_df = interp_rates_tidy.merge(
                rates, on=["pulse_rate", "Irr0/Irr0_thres", "rho_rel"]
            )
            assert len(merged_df) == len(rates) == len(interp_rates_tidy)
            return -(
                (merged_df["firing_rate_x"] - merged_df["firing_rate_y"]) ** 2
            ).sum()

        optimizer = bo.BayesianOptimization(
            f=eval_params,
            pbounds={
                "tau_m": (0, 50),  # ms
                "a": (-5, 5),  # * b2.nsiemens,
                "tau_w": (0, 200),  # * b2.ms,
                "b": (0, 200),  # * b2.pamp,
                "v_reset": (-90, -30),  # * b2.mV,
            },
            verbose=2,
            random_state=16320829,
        )

        # probe from Neuronal Dynamics preset values
        for index, row in ndx_params.iterrows():
            del row["Type"]
            optimizer.probe(
                params=row,
                lazy=True,
            )
        optimizer.maximize(init_points=0, n_iter=7)

        best_rates = simulate_rates(optimizer.max["params"])

        dprint(f"{optimizer.max=}")
        return best_rates, optimizer.max["params"]
    return (optimal_rates,)


@app.cell
def _(
    foutz12_data_combined,
    model_types,
    optimal_rates,
    pd,
    rates_data_path,
    run_checkboxes,
):
    results_dfs = []
    for model_str in model_types:
        if run_checkboxes[model_str].value:
            print(f"running simulations for {model_str}")
            _rates_df, _params2opt = optimal_rates(model_str)
            results_dfs.append(_rates_df)
        else:
            try:
                _rates_df = pd.read_csv(rates_data_path(model_str))
                results_dfs.append(_rates_df)
                print(f"loaded saved data for {model_str}")
            except FileNotFoundError:
                print(f"skipping {model_str}")
    combined_data = pd.concat(results_dfs + [foutz12_data_combined])
    combined_data
    return combined_data, model_str, results_dfs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Load source data

        We prepare a pandas dataframe to store data in tidy format:
        """
    )
    return


@app.cell
def _():
    columns = ["name", "pulse_rate", "firing_rate", "Irr0/Irr0_thres", "rho_rel"]
    return (columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next we load the original data for comparison. First, from the top panel, showing data for different irradiance levels."""
    )
    return


@app.cell
def _(pd):
    wpd_data_irr = pd.read_csv("data/foutz12_4a_wpd.csv")
    wpd_data_irr
    return (wpd_data_irr,)


@app.cell
def _(columns, pd, wpd_data_irr):
    foutz12_data_irr = pd.DataFrame(columns=columns)

    for _i_col in [0, 2, 4]:
        for _i_row in range(1, len(wpd_data_irr)):
            if pd.isna(wpd_data_irr.iloc[_i_row, _i_col]):
                continue
            foutz12_data_irr.loc[len(foutz12_data_irr)] = [
                "MCHH_Markov",
                wpd_data_irr.iloc[_i_row, _i_col],
                wpd_data_irr.iloc[_i_row, _i_col + 1],
                wpd_data_irr.columns[_i_col],
                1,
            ]

    foutz12_data_irr
    return (foutz12_data_irr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now for the bottom panel with varying levels of opsin expression:""")
    return


@app.cell
def _(pd):
    wpd_data_exp = pd.read_csv("data/foutz12_4b_wpd.csv")
    wpd_data_exp.head()
    return (wpd_data_exp,)


@app.cell
def _(columns, foutz12_data_irr, pd, wpd_data_exp):
    foutz12_data_exp = pd.DataFrame(columns=columns)
    for _i_col in [0, 2, 4]:
        for _i_row in range(1, len(wpd_data_exp)):
            if pd.isna(wpd_data_exp.iloc[_i_row, _i_col]):
                continue
            foutz12_data_exp.loc[len(foutz12_data_exp)] = [
                "MCHH_Markov",
                wpd_data_exp.iloc[_i_row, _i_col],
                wpd_data_exp.iloc[_i_row, _i_col + 1],
                1.2,
                wpd_data_exp.columns[_i_col],
            ]
    foutz12_data_combined = pd.concat([foutz12_data_irr, foutz12_data_exp]).astype(
        {
            "name": "string",
            "pulse_rate": float,
            "firing_rate": float,
            "Irr0/Irr0_thres": float,
            "rho_rel": float,
        }
    )
    foutz12_data_combined
    return foutz12_data_combined, foutz12_data_exp


@app.cell
def _():
    # from math import sin, tau
    # from brian2 import Hz, ms, second, defaultclock

    # rerun_sim = prompt_rerun_button.value

    # class PulseController(cleo.ioproc.LatencyIOProcessor):
    #     def process(self, state, t_samp_ms):
    #         t = t_samp_ms * ms
    #         t_peak = 1 / (4 * pulse_rate)
    #         t_pulse_start = t_peak - 2.5 * ms
    #         sin_thres = sin(tau * pulse_rate * t_pulse_start)
    #         stim_on = int(np.sin(tau * self.pulse_rate * t) >= sin_thres)
    #         out = {}
    #         for _ng, fiber in fibers.items():
    #             out[fiber.name] = stim_on * Irr0_thres[_ng.name]
    #         return (out, t_samp_ms)

    # ctrl = PulseController(sample_period=defaultclock.dt)
    # sim.set_io_processor(ctrl)
    # if rerun_sim:
    #     for pulse_rate in pulse_rates * Hz:
    #         sim.reset()
    #         ctrl.pulse_rate = pulse_rate
    #         duration_s = 0.4
    #         sim.run(duration_s * second)
    #         for ng_name, mon in spike_mons.items():
    #             df_2 = pd.concat(
    #                 [
    #                     df_1,
    #                     pd.DataFrame(
    #                         {
    #                             "name": ng_name,
    #                             "pulse_rate": pulse_rate / Hz,
    #                             "firing_rate": np.array(mon.count_) / duration_s,
    #                             "Irr0/Irr0_thres": Irr0_ratio,
    #                             "rho_rel": rho_rel,
    #                         }
    #                     ),
    #                 ]
    #             )
    #     df_2.to_csv("data/pr_fr_irr_exp.csv", index=False)
    # df_2 = pd.read_csv("data/pr_fr_irr_exp.csv")
    # df_2.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting""")
    return


@app.cell(hide_code=True)
def _(pulse_rates, sns):
    def plot_fr_2panel(data, palette1, palette2, **figargs):
        kwargs = {
            "kind": "line",
            "col": "name",
            "x": "pulse_rate",
            "y": "firing_rate",
            "legend": True,
        }
        g_irr = sns.relplot(
            data=data[data.rho_rel == 1],
            hue="Irr0/Irr0_thres",
            palette=palette1,
            **kwargs,
        )
        g_exp = sns.relplot(
            data=data[data["Irr0/Irr0_thres"] == 1.2],
            hue="rho_rel",
            palette=palette2,
            **kwargs,
        )
        g_irr.set_axis_labels(x_var="")
        g_irr.set_titles(col_template="{col_name}")
        g_exp.set_titles(col_template="")
        for g in (g_irr, g_exp):
            g.set(xlim=(0, pulse_rates[-1]), ylim=(0, pulse_rates[-1]))
            g.legend.remove()
            g.fig.legend(
                handles=g.legend.legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0),
                ncol=3,
            )
            g.fig.set_figwidth(3.0)
            g.fig.set_figheight(1.5)
            g.fig.tight_layout()
            g.fig.set(**figargs)
        return (g_irr, g_exp)
    return (plot_fr_2panel,)


@app.cell
def _(combined_data, mo, plot_fr_2panel):
    g_irr_main, g_exp_main = plot_fr_2panel(
        combined_data[
            ~combined_data.name.str.contains("LIF_Markov|AdEx_simple", regex=True)
        ],
        palette1="blend:#df87e1,#8000b4",
        palette2="blend:#69fff8,#36827F",
    )
    g_irr_main.fig.savefig("img/fig/opto_pr_fr_irr.svg", bbox_inches="tight")
    g_exp_main.fig.savefig("img/fig/opto_pr_fr_exp.svg", bbox_inches="tight")
    mo.hstack([g_irr_main, g_exp_main])
    return g_exp_main, g_irr_main


@app.cell
def _(combined_data, mo, plot_fr_2panel):
    g_irr_supp, g_exp_supp = plot_fr_2panel(
        combined_data[
            combined_data.name.str.contains("LIF_Markov|AdEx_simple", regex=True)
        ],
        palette1="blend:#df87e1,#8000b4",
        palette2="blend:#69fff8,#36827F",
    )
    g_irr_supp.fig.savefig("img/fig/opto_pr_fr_irr_supp.svg")
    g_exp_supp.fig.savefig("img/fig/opto_pr_fr_exp_supp.svg")
    mo.hstack([g_irr_supp, g_exp_supp])
    return g_exp_supp, g_irr_supp


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # neuron_params = {
    #     "a": 0.0 * nsiemens,
    #     "b": 60 * pamp,
    #     "E_L": -70 * mV,
    #     "tau_m": 20 * ms,
    #     "R": 500 * Mohm,
    #     "theta": -50 * mV,
    #     "v_reset": -55 * mV,
    #     "tau_w": 30 * ms,
    #     "Delta_T": 2 * mV,
    # }

    # def lif(n, name="LIF"):
    #     ng = NeuronGroup(
    #         n,
    #         """dv/dt = (-(v - E_L) + R*Iopto) / tau_m : volt
    #         Iopto: amp
    #         """,
    #         threshold="v>=theta",
    #         reset="v=E_L",
    #         refractory=2 * ms,
    #         namespace=neuron_params,
    #         name=name,
    #     )
    #     ng.v = neuron_params["E_L"]
    #     return ng

    # def adex(n, name="AdEx"):
    #     ng = NeuronGroup(
    #         n,
    #         """dv/dt = (-(v - E_L) + Delta_T*exp((v-theta)/Delta_T) + R*(Iopto-w)) / tau_m : volt
    #         dw/dt = (a*(v-E_L) - w) / tau_w : amp
    #         Iopto : amp""",
    #         threshold="v>=30*mV",
    #         reset="v=v_reset; w+=b",
    #         namespace=neuron_params,
    #         name=name,
    #     )
    #     ng.v = neuron_params["E_L"]
    #     return ng

    # def Iopto_gain_from_factor(factor):
    #     return (
    #         factor
    #         * (neuron_params["theta"] - neuron_params["E_L"])
    #         / (neuron_params["R"])
    #     )

    # def get_Irr0_thres(
    #     pulse_widths,
    #     distance_mm,
    #     ng,
    #     gain_factor,
    #     precision=1,
    #     simple_opto=False,
    #     target="cython",
    # ):
    #     prefs.codegen.target = target
    #     mon = SpikeMonitor(ng, record=False)

    #     assign_coords_rand_rect_prism(
    #         ng,
    #         xlim=(0, 0),
    #         ylim=(0, 0),
    #         zlim=(distance_mm, distance_mm),
    #     )

    #     net = Network(mon, ng)
    #     sim = CLSimulator(net)

    #     if simple_opto:
    #         opsin = ProportionalCurrentOpsin(
    #             I_per_Irr=Iopto_gain_from_factor(gain_factor)
    #         )
    #     else:
    #         opsin = chr2_4s()
    #     sim.inject(opsin, ng)

    #     fiber = Light(light_model=fiber473nm())
    #     sim.inject(fiber, ng)

    #     sim.network.store()
    #     Irr0_thres = []
    #     for pw in pulse_widths:
    #         search_min, search_max = (0, 10000) * mwatt / mm2
    #         while (
    #             search_max - search_min > precision
    #         ):  # get down to {precision} mW/mm2 margin
    #             sim.network.restore()
    #             Irr0_curr = (search_min + search_max) / 2
    #             fiber.update(Irr0_curr)
    #             sim.run(pw * ms)
    #             fiber.update(0)
    #             sim.run(10 * ms)  # wait 10 ms to make sure only 1 spike
    #             if mon.count > 0:  # spiked
    #                 search_max = Irr0_curr
    #             else:
    #                 search_min = Irr0_curr
    #         Irr0_thres.append(Irr0_curr)

    #     return Irr0_thres
    return


if __name__ == "__main__":
    app.run()
