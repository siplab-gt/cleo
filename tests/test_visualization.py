from brian2 import NeuronGroup, Network, ms, mm
import pytest

from cleosim import CLSimulator
from cleosim.visualization import VideoVisualizer
from cleosim.opto import (
    OptogeneticIntervention,
    four_state,
    ChR2_four_state,
    default_blue,
)
from cleosim.coordinates import assign_coords_rand_rect_prism
from cleosim.electrodes import Probe


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
    assign_coords_rand_rect_prism(ng, (0, 0), (0, 0), (0, 0))
    opto = OptogeneticIntervention(
        "opto", four_state, ChR2_four_state, default_blue, max_Irr0_mW_per_mm2=20
    )
    probe = Probe("probe", [(0, 0, 0.1)] * mm)

    sim = CLSimulator(Network(ng))
    sim.inject_stimulator(opto, ng)
    sim.inject_recorder(probe, ng)

    vv = VideoVisualizer("vv")
    sim.inject_device(vv, ng)

    sim.run(2 * ms)
    ani = vv.generate_Animation()
