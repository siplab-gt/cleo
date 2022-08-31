"""Tests for ephys.lfp module"""
import pytest
from brian2 import mm, Hz, ms, Network, seed, SpikeGeneratorGroup
import numpy as np
from brian2.input.poissongroup import PoissonGroup
from tklfp import TKLFP

from cleo import CLSimulator
from cleo.processing import RecordOnlyProcessor
from cleo.ephys import linear_shank_coords, concat_coords, TKLFPSignal, Probe
from cleo.coords import assign_coords_rand_rect_prism, assign_coords


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
        # or 0, .4, .8, 1.2 mm in cleo depth coordinates
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

    tklfp = TKLFPSignal("tklfp", save_history=True)
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
    probe = Probe("probe", contact_coords, signals=[tklfp])
    for group, tklfp_type in groups_and_types:
        sim.inject_recorder(probe, group, tklfp_type=tklfp_type, sample_period_ms=1)

    # doesn't specify tklfp_type:
    with pytest.raises(Exception):
        sim.inject_recorder(probe, group, sample_period_ms=1)
    # doesn't specify sampling period:
    with pytest.raises(Exception):
        sim.inject_recorder(probe, group, tklfp_type="inh")

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

    # reset should clear buffer, zeroing out the signal
    assert tklfp.lfp_uV.shape == (1, 8)
    sim.reset()
    assert tklfp.lfp_uV.shape == (0, 8)
    lfp_reset = tklfp.get_state()
    assert np.all(lfp_reset == 0)
    assert tklfp.lfp_uV.shape == (1, 8)


def test_TKLFPSignal_out_of_range():
    # make sure logic works when neuron groups are ignored because they're too far away
    n = 5
    pgs = []
    for i in range(4):
        pg = PoissonGroup(n, 500 * Hz)
        assign_coords(pg, 5 * i, 5 * i, 5 * i)  # ranging from close (origin) to far
        pgs.append(pg)
    net = Network(*pgs)
    sim = CLSimulator(net)
    tklfp = TKLFPSignal("tklfp")
    probe = Probe(
        "probe", [[0, 0, 0], [5, 5, 5]] * mm, signals=[tklfp]
    )  # contacts at origin and 5,5,5
    sim.inject_recorder(probe, *pgs, tklfp_type="exc", sample_period_ms=1)
    sim.run(30 * ms)
    lfp = tklfp.get_state()
    assert lfp.shape == (2,)
    assert not np.all(lfp == 0)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1783, 1865, 1918, 1945])
@pytest.mark.parametrize("is_exc", [True, False])
def test_TKLFP_orientation(seed, is_exc):
    # here we'll just compare TKLFPSignal's output to TKLFP. Should
    # have done this for the other tests
    rng = np.random.default_rng(seed)
    n_nrns = 5
    n_elec = 4
    # random network setup and spikes
    c_mm = 2 * rng.random((n_nrns, 3)) - 1
    orientation = 2 * rng.random((n_nrns, 3)) - 1
    elec_coords = (2 * rng.random((n_elec, 3)) - 1) * mm
    # random spikes during first 40 ms
    n_spk = n_nrns * 4
    i_spk = rng.choice(range(n_nrns), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * ms
    sgg = SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # hack for assign_coords to work
    assign_coords(sgg, c_mm[:, 0], c_mm[:, 1], c_mm[:, 2])

    # cleo setup
    sim = CLSimulator(Network(sgg))
    tklfp_signal = TKLFPSignal("tklfp", save_history=True)
    probe = Probe("probe", elec_coords, [tklfp_signal])
    samp_period = 10 * ms
    sim.set_io_processor(RecordOnlyProcessor(samp_period / ms))  # record every 10 ms
    sim.inject_recorder(
        probe, sgg, tklfp_type="exc" if is_exc else "inh", orientation=orientation
    )

    sim.run(7 * samp_period)

    # tklfp
    elec_coords_mm_invert = elec_coords / mm
    elec_coords_mm_invert[:, 2] *= -1
    orientation_invert = orientation.copy()
    orientation_invert[:, 2] *= -1
    tklfp = TKLFP(
        c_mm[:, 0],
        c_mm[:, 1],
        -c_mm[:, 2],
        is_exc,
        elec_coords_mm_invert,
        orientation_invert,
    )
    tklfp_out = tklfp.compute(i_spk, t_spk / ms, sim.io_processor.t_samp_ms)

    # not super close--why? inverting z axis could introduce some floating point
    # differences. Also, TKLFPSignal ignores spikes with low contributions
    # BUT--after exploring a lower spike collection threshold and still seeing
    # the same phenomenon I think the biggest difference is that tklfp
    # computes it all post-hoc but TKLFPSignal can only compute using
    # a causal buffer of spikes
    assert np.allclose(tklfp_signal.lfp_uV, tklfp_out, atol=5e-3)
