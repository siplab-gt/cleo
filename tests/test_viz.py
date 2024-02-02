import itertools

import pytest
from brian2 import Network, NeuronGroup, mm, ms

import cleo
from cleo import CLSimulator
from cleo.coords import assign_xyz
from cleo.ephys import Probe
from cleo.light import Light, fiber473nm
from cleo.opto import chr2_4s
from cleo.viz import VideoVisualizer


@pytest.mark.slow
def test_VideoVisualizer():
    # need valid spiking neuron + opto for video to record
    ng = NeuronGroup(
        1,
        """v : volt
        Iopto : amp""",
        threshold="v > 1 * volt",
        reset="v = 0 * volt",
    )
    assign_xyz(ng, 0, 0, 0)
    light = Light(light_model=fiber473nm(), max_Irr0_mW_per_mm2=20)
    # opsin = chr2_4s()
    probe = Probe([(0, 0, 0.1)] * mm)

    sim = CLSimulator(Network(ng)).inject(light, ng).inject(probe, ng)

    vv = VideoVisualizer()
    sim.inject(vv, ng)

    sim.run(2 * ms)
    plotargs = {
        "colors": ["xkcd:emerald"],
        "xlim": (-0.2, 0.2),
        "ylim": (-0.2, 0.2),
        "zlim": (0, 0.8),
        "scatterargs": {"s": 20},  # to adjust neuron marker size
    }
    ani = vv.generate_Animation(plotargs)


@pytest.mark.slow
def test_plot_sim():
    ng = NeuronGroup(
        1,
        """v : volt
        Iopto : amp""",
        threshold="v > 1 * volt",
        reset="v = 0 * volt",
    )
    assign_xyz(ng, 0, 0, 0)
    light = Light(light_model=fiber473nm(), max_Irr0_mW_per_mm2=20)
    # opsin = chr2_4s()
    probe = Probe([(0, 0, 0.1)] * mm)

    sim = CLSimulator(Network(ng)).inject(light, ng).inject(probe, ng)

    vv = VideoVisualizer()
    sim.inject(vv, ng)

    ngs_sim_devs_npoints = itertools.product(
        [[ng], []],
        [None, sim],
        [[light, probe], [light], []],
        [None, 1, 100, 1000],
    )

    for ngs, sim_param, devices, n_points in ngs_sim_devs_npoints:
        if n_points:
            light_kwargs = {"n_points_per_source": n_points}
        else:
            light_kwargs = {}
        cleo.viz.plot(
            *ngs, sim=sim_param, devices=[(dev, light_kwargs) for dev in devices]
        )
