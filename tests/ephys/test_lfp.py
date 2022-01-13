"""Tests for ephys.lfp module"""
import pytest
from brian2 import mm, Hz, ms, prefs, meter
from brian2.core.network import Network
import numpy as np
from brian2.groups.neurongroup import NeuronGroup
from brian2.input.poissongroup import PoissonGroup
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup

import cleosim
from cleosim.base import CLSimulator
from cleosim.ephys.electrodes import get_1D_probe_coords
from cleosim.ephys.lfp import TKLFPSignal
from cleosim.ephys import ElectrodeGroup
from cleosim.coordinates import assign_coords_rand_rect_prism


def inh():
    ipg = PoissonGroup(20, np.linspace(100, 500, 20) * Hz)
    assign_coords_rand_rect_prism(ipg, (-0.2, 0.2), (-2, 0.2), (0.75, 0.85))
    return ipg, "inh"


def exc():
    epg = PoissonGroup(80, np.linspace(100, 500, 80) * Hz)
    assign_coords_rand_rect_prism(epg, (-0.2, 0.2), (-2, 0.2), (0.75, 0.85))
    return epg, "exc"


def low_exc():
    epg = PoissonGroup(60, np.linspace(100, 500, 60) * Hz)
    assign_coords_rand_rect_prism(epg, (-0.2, 0.2), (-2, 0.2), (0.75, 0.85))
    return epg, "exc"


def high_exc():
    epg = PoissonGroup(800, np.linspace(100, 500, 800) * Hz)
    assign_coords_rand_rect_prism(epg, (-0.2, 0.2), (-2, 0.2), (0.75, 0.85))
    return epg, "exc"


@pytest.mark.parametrize(
    "groups_and_types,signal_positive",
    [
        ([inh], [1, 0, 1, 0]),
        ([exc], [0, 1, 1, 0]),
        # I thought inhibition should be able to overcome excitation on
        # the second position (.4 mm), since the I/E amp is 5 but E only 
        # outnumbers I 4:1. Excitation wins though, I assume because
        # of the larger spread of the E uLFP kernel.
        ([exc, inh], [0, 1, 1, 0]),
        # so lower excitation should let I dominate at .4 um but E at .8 (1st pos)
        ([low_exc, inh], [0, 0, 1, 0]),
        ([high_exc, inh], [0, 1, 1, 0]),
    ],
    ids=("inh", "exc", "tot", "tot_low_exc", "tot_high_exc"),
)
def test_TKLFPSignal(groups_and_types, signal_positive):
    prefs.codegen.target = "numpy"
    np.random.seed(1945)
    # since parametrize passes function, not return value
    groups_and_types = [gt() for gt in groups_and_types]
    net = Network(*[gt[0] for gt in groups_and_types])
    sim = CLSimulator(net)

    tklfp = TKLFPSignal("tklfp")
    # One probe in middle and another further out.
    # Here we put coords for two probes in one EG.
    # Alternatively you could create two separate EGs
    contact_coords = np.concatenate(
        # In the paper, z=0 corresponds to stratum pyramidale.
        # Here, z=0 is the surface and str pyr is at z=.8mm,
        # meaning a depth of .8mm
        (
            get_1D_probe_coords(1.2 * mm, 4, (0, 0, 0) * mm) / mm,
            get_1D_probe_coords(1.2 * mm, 4, (0.3, 0.3, 0) * mm) / mm,
        )
    )
    # need to strip and reattach units since concatenate isn't unit-safe
    contact_coords = contact_coords * mm
    eg = ElectrodeGroup("eg", contact_coords, signals=[tklfp])
    for group, tklfp_type in groups_and_types:
        sim.inject_recorder(eg, group, tklfp_type=tklfp_type, sampling_period_ms=1)

    # doesn't specify tklfp_type:
    with pytest.raises(Exception):
        sim.inject_recorder(eg, group, sampling_period_ms=1)
    # doesn't specify sampling period:
    with pytest.raises(Exception):
        sim.inject_recorder(eg, group, tklfp_type="inh")

    sim.run(20 * ms)

    y = tklfp.get_state()
    # signal should be stronger in closer probe (first 4 contacts)
    assert all(np.abs(y[:4]) >= np.abs(y[4:]))

    # check signal is positive or negative as expected
    print(y)
    assert np.all((y[:4] > 0) == signal_positive)
    # for second probe as well:
    assert np.all((y[4:] > 0) == signal_positive)
