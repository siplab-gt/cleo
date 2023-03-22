from brian2 import (
    nsiemens,
    pamp,
    mV,
    ms,
    Mohm,
    NeuronGroup,
    SpikeMonitor,
    Network,
    prefs,
)
from cleo import CLSimulator
from cleo.opto import *
from cleo.coords import assign_coords_rand_rect_prism


neuron_params = {
    "a": 0.0 * nsiemens,
    "b": 60 * pamp,
    "E_L": -70 * mV,
    "tau_m": 20 * ms,
    "R": 500 * Mohm,
    "theta": -50 * mV,
    "v_reset": -55 * mV,
    "tau_w": 30 * ms,
    "Delta_T": 2 * mV,
}


def lif(n, name="LIF"):
    ng = NeuronGroup(
        n,
        """dv/dt = (-(v - E_L) + R*Iopto) / tau_m : volt
        Iopto: amp
        """,
        threshold="v>=theta",
        reset="v=E_L",
        refractory=2 * ms,
        namespace=neuron_params,
        name=name,
    )
    ng.v = neuron_params["E_L"]
    return ng


def adex(n, name="AdEx"):
    ng = NeuronGroup(
        n,
        """dv/dt = (-(v - E_L) + Delta_T*exp((v-theta)/Delta_T) + R*(Iopto-w)) / tau_m : volt
        dw/dt = (a*(v-E_L) - w) / tau_w : amp
        Iopto : amp""",
        threshold="v>=30*mV",
        reset="v=v_reset; w+=b",
        namespace=neuron_params,
        name=name,
    )
    ng.v = neuron_params["E_L"]
    return ng


def Iopto_gain_from_factor(factor):
    return (
        factor * (neuron_params["theta"] - neuron_params["E_L"]) / (neuron_params["R"])
    )


def get_Irr0_thres(
    pulse_widths,
    distance_mm,
    ng,
    gain_factor,
    precision=1,
    simple_opto=False,
    target="cython",
):
    prefs.codegen.target = target
    mon = SpikeMonitor(ng, record=False)

    assign_coords_rand_rect_prism(
        ng,
        xlim=(0, 0),
        ylim=(0, 0),
        zlim=(distance_mm, distance_mm),
    )

    net = Network(mon, ng)
    sim = CLSimulator(net)

    if simple_opto:
        opto = Light(
            name="opto",
            opsin_model=ProportionalCurrentModel(
                # use 240*(thresh-E_L) factor from tutorial
                Iopto_per_mW_per_mm2=Iopto_gain_from_factor(gain_factor)
            ),
            light_model_params=fiber473nm,
            location=(0, 0, 0) * mm,
        )
    else:
        opto = Light(
            name="opto",
            opsin_model=FourStateModel(ChR2_four_state),
            light_model_params=fiber473nm,
            location=(0, 0, 0) * mm,
        )
    sim.inject_stimulator(opto, ng)

    sim.network.store()
    Irr0_thres = []
    for pw in pulse_widths:
        search_min, search_max = (0, 10000)
        while (
            search_max - search_min > precision
        ):  # get down to {precision} mW/mm2 margin
            sim.network.restore()
            Irr0_curr = (search_min + search_max) / 2
            opto.update(Irr0_curr)
            sim.run(pw * ms)
            opto.update(0)
            sim.run(10 * ms)  # wait 10 ms to make sure only 1 spike
            if mon.count > 0:  # spiked
                search_max = Irr0_curr
            else:
                search_min = Irr0_curr
        Irr0_thres.append(Irr0_curr)

    return Irr0_thres
