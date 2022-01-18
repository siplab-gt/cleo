"""Tests for ephys.lfp module"""
import pytest
from brian2 import mm, Hz, ms, Network, seed
import numpy as np
from brian2.input.poissongroup import PoissonGroup

import cleosim
from cleosim.base import CLSimulator
from cleosim.electrodes import linear_shank_coords, TKLFPSignal, ElectrodeGroup
from cleosim.coordinates import assign_coords_rand_rect_prism
from cleosim.electrodes.probes import concat_coords


def _groups_types_ei(n_e, n_i):
    generators = []
    if n_e > 0:

        def gen():
            epg = PoissonGroup(n_e, np.linspace(100, 500, n_e) * Hz)
            assign_coords_rand_rect_prism(epg, (-0.2, 0.2), (-2, 0.05), (0.75, 0.85))
            # out.append((ipg, "exc"))
            return (epg, "exc")

        generators.append(gen)
    if n_i > 0:

        def gen():
            ipg = PoissonGroup(n_i, np.linspace(100, 500, n_i) * Hz)
            assign_coords_rand_rect_prism(ipg, (-0.2, 0.2), (-2, 0.05), (0.75, 0.85))
            # out.append((ipg, "inh"))
            return (ipg, "inh")

        generators.append(gen)
    return generators


@pytest.mark.parametrize(
    "groups_and_types,signal_positive",
    [
        # sig_pos is for .8, .4, 0, -.4 mm with respect to the neuron
        # or 0, .4, .8, 1.2 mm in cleosim depth coordinates
        (_groups_types_ei(0, 100), [1, 0, 1, 0]),
        (_groups_types_ei(100, 0), [0, 1, 1, 0]),
        # lower excitation should let I dominate
        (_groups_types_ei(20, 80), [1, 0, 1, 0]),
        # higher excitation should let E dominate
        (_groups_types_ei(200, 20), [0, 1, 1, 0]),
        # we won't test a normal E-I balance: too unpredictable
    ],
    ids=("inh", "exc", "tot_low_exc", "tot_high_exc"),
)
def test_TKLFPSignal(groups_and_types, signal_positive, rand_seed):
    """Can run multiple times with different seeds from command line
    with --seed [num. seeds]"""
    np.random.seed(rand_seed)
    seed(rand_seed)
    # since parametrize passes function, not return value
    groups_and_types = [gt() for gt in groups_and_types]
    net = Network(*[gt[0] for gt in groups_and_types])
    sim = CLSimulator(net)

    tklfp = TKLFPSignal("tklfp")
    # One probe in middle and another further out.
    # Here we put coords for two probes in one EG.
    # Alternatively you could create two separate EGs
    contact_coords = concat_coords(
        # In the paper, z=0 corresponds to stratum pyramidale.
        # Here, z=0 is the surface and str pyr is at z=.8mm,
        # meaning a depth of .8mm
        linear_shank_coords(1.2 * mm, 4, (0, 0, 0) * mm),
        linear_shank_coords(1.2 * mm, 4, (0.2, 0.2, 0) * mm),
    )
    eg = ElectrodeGroup("eg", contact_coords, signals=[tklfp])
    for group, tklfp_type in groups_and_types:
        sim.inject_recorder(eg, group, tklfp_type=tklfp_type, sampling_period_ms=1)

    # doesn't specify tklfp_type:
    with pytest.raises(Exception):
        sim.inject_recorder(eg, group, sampling_period_ms=1)
    # doesn't specify sampling period:
    with pytest.raises(Exception):
        sim.inject_recorder(eg, group, tklfp_type="inh")

    sim.run(30 * ms)

    lfp = tklfp.get_state()
    # signal should be stronger in closer probe (first 4 contacts)
    assert all(np.abs(lfp[:4]) >= np.abs(lfp[4:]))
    # sign should be same in both probes
    assert all((lfp[:4] > 0) == (lfp[4:] > 0))

    # check signal is positive or negative as expected
    assert np.all((lfp[:4] > 0) == signal_positive)
    # for second probe as well:
    assert np.all((lfp[4:] > 0) == signal_positive)
