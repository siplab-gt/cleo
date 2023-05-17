from brian2 import NeuronGroup, Network, ms, mm
import pytest

from cleo import CLSimulator
from cleo.viz import VideoVisualizer
from cleo.opto import (
    Light,
    FourStateOpsin,
    ChR2_four_state,
    fiber473nm,
)
from cleo.coords import assign_coords
from cleo.ephys import Probe


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
    assign_coords(ng, 0, 0, 0)
    opto = Light(
        "opto",
        opsin_model=ChR2_four_state(),
        light_model=fiber473nm(),
        max_Irr0_mW_per_mm2=20,
    )
    probe = Probe("probe", [(0, 0, 0.1)] * mm)

    sim = CLSimulator(Network(ng)).inject(opto, ng).inject(probe, ng)

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
