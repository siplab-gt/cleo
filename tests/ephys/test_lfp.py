# """Tests for ephys.lfp module"""
import time
from itertools import product

import brian2.only as b2
import neo
import numpy as np
import pytest
import quantities as pq
import wslfp
from brian2 import Hz, Network, SpikeGeneratorGroup, mm, ms, seed, uvolt
from tklfp import TKLFP

import cleo
from cleo import CLSimulator
from cleo.coords import (
    assign_coords,
    assign_coords_rand_rect_prism,
    assign_xyz,
    concat_coords,
)
from cleo.ephys import (
    Probe,
    RWSLFPSignalFromPSCs,
    RWSLFPSignalFromSpikes,
    TKLFPSignal,
    linear_shank_coords,
)
from cleo.ioproc import RecordOnlyProcessor


def _groups_types_ei(n_e, n_i):
    generators = []
    if n_e > 0:

        def gen():
            epg = b2.PoissonGroup(n_e, np.linspace(100, 500, n_e) * Hz)
            assign_coords_rand_rect_prism(epg, (-0.2, 0.2), (-2, 0.05), (0.75, 0.85))
            # out.append((ipg, "exc"))
            return (epg, "exc")

        generators.append(gen)
    if n_i > 0:

        def gen():
            ipg = b2.PoissonGroup(n_i, np.linspace(100, 500, n_i) * Hz)
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
    with --seeds [num. seeds]"""
    np.random.seed(rand_seed)
    seed(rand_seed)
    # since parametrize passes function, not return value
    groups_and_types = [gt() for gt in groups_and_types]
    net = Network(*[gt[0] for gt in groups_and_types])
    sim = CLSimulator(net)

    tklfp = TKLFPSignal()
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
    probe = Probe(contact_coords, signals=[tklfp], save_history=True)
    for group, tklfp_type in groups_and_types:
        sim.inject(probe, group, tklfp_type=tklfp_type, sample_period_ms=1)

    # doesn't specify tklfp_type:
    with pytest.raises(Exception):
        sim.inject(probe, group, sample_period_ms=1)
    # doesn't specify sampling period:
    with pytest.raises(Exception):
        sim.inject(probe, group, tklfp_type="inh")

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
    assert tklfp.lfp.shape == (1, 8)
    sim.reset()
    assert tklfp.lfp.shape == (0, 8)
    lfp_reset = tklfp.get_state()
    assert np.all(lfp_reset == 0)
    assert tklfp.lfp.shape == (1, 8)


def test_TKLFPSignal_out_of_range():
    # make sure logic works when neuron groups are ignored because they're too far away
    n = 5
    pgs = []
    for i in range(4):
        pg = b2.PoissonGroup(n, 500 * Hz)
        assign_xyz(pg, 5 * i, 5 * i, 5 * i)  # ranging from close (origin) to far
        pgs.append(pg)
    net = Network(*pgs)
    sim = CLSimulator(net)
    tklfp = TKLFPSignal()
    probe = Probe(
        [[0, 0, 0], [5, 5, 5]] * mm, signals=[tklfp]
    )  # contacts at origin and 5,5,5
    sim.inject(probe, *pgs, tklfp_type="exc", sample_period_ms=1)
    sim.run(30 * ms)
    lfp = tklfp.get_state()
    assert lfp.shape == (2,)
    assert not np.all(lfp == 0)


@pytest.mark.slow
@pytest.mark.parametrize("is_exc", [True, False])
def test_TKLFP_orientation(rand_seed, is_exc):
    # here we'll just compare TKLFPSignal's output to TKLFP. Should
    # have done this for the other tests
    rng = np.random.default_rng(rand_seed)
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
    assign_xyz(sgg, c_mm[:, 0], c_mm[:, 1], c_mm[:, 2])

    # cleo setup
    sim = CLSimulator(Network(sgg))
    tklfp_signal = TKLFPSignal()
    probe = Probe(elec_coords, [tklfp_signal], save_history=True)
    samp_period = 10 * ms
    sim.set_io_processor(RecordOnlyProcessor(samp_period / ms))  # record every 10 ms
    sim.inject(
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
    assert np.allclose(tklfp_signal.lfp / uvolt, tklfp_out, atol=5e-3)


@pytest.mark.parametrize(
    "LFPSignal", [TKLFPSignal, RWSLFPSignalFromSpikes, RWSLFPSignalFromPSCs]
)
@pytest.mark.parametrize(
    "t,regular_samples", [(0, False), (1, False), (10, False), (10, True)]
)
@pytest.mark.parametrize("n_channels", [1, 4])
def test_lfp_signal_to_neo(LFPSignal, n_channels, t, regular_samples):
    sig = LFPSignal()
    probe = Probe(np.random.rand(n_channels, 3) * mm, [sig])
    if regular_samples:
        sig.t = np.arange(t) * ms
    else:
        sig.t_ms = np.sort(np.random.rand(t) * t)
    sig.lfp = np.random.rand(t, n_channels)
    neo_sig = sig.to_neo()

    if regular_samples and t > 1:
        assert type(neo_sig) == neo.AnalogSignal
    else:
        assert type(neo_sig) == neo.IrregularlySampledSignal

    assert np.all(
        neo_sig.array_annotations["x"] / pq.mm == sig.probe.coords[..., 0] / mm
    )
    assert np.all(
        neo_sig.array_annotations["y"] / pq.mm == sig.probe.coords[..., 1] / mm
    )
    assert np.all(
        neo_sig.array_annotations["z"] / pq.mm == sig.probe.coords[..., 2] / mm
    )
    assert np.all(neo_sig.array_annotations["i_channel"] == np.arange(probe.n))
    assert np.all(neo_sig.magnitude == sig.lfp)
    assert neo_sig.name == f"{sig.probe.name}.{sig.name}"


_base_param_options = [
    ("pop_agg", (True, False)),
    ("amp_func", (wslfp.aussel18, wslfp.mazzoni15_pop)),
    # ("wslfp_kwargs", ({}, {"tau_ampa_ms": 5})),
    ("wslfp_kwargs", ({}, {"tau_gaba_ms": 1})),
    ("wslfp_kwargs", ({}, {"alpha": 1})),
    (
        "wslfp_kwargs",
        ({"source_coords_are_somata": True}, {"source_coords_are_somata": False}),
    ),
    (
        "wslfp_kwargs",
        ({"source_dendrite_length_um": 250}, {"source_dendrite_length_um": 300}),
    ),
]


def test_RWSLFPSignalFromSpikes(rand_seed):
    rng = np.random.default_rng(rand_seed)
    b2.seed(rand_seed)
    n_exc = 16
    n_inh = 5
    n_elec = 4
    elec_coords = rng.uniform(-1, 1, (n_elec, 3)) * mm

    exc = b2.PoissonGroup(n_exc, 40 * Hz)
    assign_coords(exc, rng.uniform(-1, 1, (n_exc, 3)) * mm)
    inh = b2.PoissonGroup(n_inh, 40 * Hz)
    # just need synapses onto exc to test RWSLFP
    syn_e2e = b2.Synapses(exc, exc, "w : 1")
    syn_e2e.connect(p=0.2)
    syn_i2e = b2.Synapses(inh, exc, "w : 1")
    syn_i2e.connect(p=0.2)

    # cleo setup
    sim = CLSimulator(Network(exc, inh, syn_e2e, syn_i2e))
    sim.set_io_processor(RecordOnlyProcessor(sample_period_ms=10))

    def add_rwslfp_sig(
        pop_agg=True,
        amp_func=wslfp.mazzoni15_pop,
        ornt=[0, 0, -1],
        wslfp_kwargs={},
        tau1_ampa=2 * ms,
        tau2_ampa=0.4 * ms,
        tau1_gaba=5 * ms,
        tau2_gaba=0.25 * ms,
        syn_delay=1 * ms,
        homog_J=True,
        I_threshold=0.01,
    ):
        rwslfp_sig = RWSLFPSignalFromSpikes(
            pop_aggregate=pop_agg,
            amp_func=amp_func,
            tau1_ampa=tau1_ampa,
            tau2_ampa=tau2_ampa,
            tau1_gaba=tau1_gaba,
            tau2_gaba=tau2_gaba,
            syn_delay=syn_delay,
            I_threshold=I_threshold,
            wslfp_kwargs=wslfp_kwargs,
        )
        # separate probe for each signal since some injection kwargs need to be different per signal
        i = len(sim.recorders)
        probe = Probe(elec_coords, [rwslfp_sig], name=f"probe{i}")

        if homog_J:
            syn_e2e.w = 1
            syn_i2e.w = -n_exc / n_inh
        else:
            syn_e2e.w = rng.lognormal(size=len(syn_e2e))
            syn_i2e.w = -n_exc / n_inh * rng.lognormal(size=len(syn_i2e))

        sim.inject(
            probe, exc, orientation=ornt, ampa_syns=[syn_e2e], gaba_syns=[syn_i2e]
        )
        return rwslfp_sig

    signals_for_options = []
    param_options = [
        ("ornt", (rng.normal(size=(n_exc, 3)), (-0.2, -0.1, -0.3))),
        ("tau1_ampa", (2, 1) * ms),
        ("tau2_ampa", (0.4, 0.2) * ms),
        ("tau1_gaba", (5, 6) * ms),
        ("tau2_gaba", (0.25, 0.2) * ms),
        ("syn_delay", (1, 2) * ms),
        ("homog_J", (True, False)),
        ("I_threshold", (0.1, 0.001)),
    ]
    for param_name, param_vals in _base_param_options + param_options:
        signals_to_compare = []
        for val in param_vals:
            # store result of each different value for each param
            signals_to_compare.append(add_rwslfp_sig(**{param_name: val}))
        signals_for_options.append((param_name, signals_to_compare))
    sim.run(100 * ms)
    # each parameter change should change the resulting signal:
    assert len(signals_for_options) == len(_base_param_options) + len(param_options)
    for param, signals in signals_for_options:
        assert len(signals) > 1
        for i, sig1 in enumerate(signals):
            for sig2 in signals[i + 1 :]:
                assert not np.allclose(
                    sig1.lfp, sig2.lfp
                ), f"{param} variation not yielding different results"


@pytest.mark.parametrize("samp_period_ms", [1, 1.4])
def test_RWSLFPSignalFromPSCs(rand_seed, samp_period_ms):
    rng = np.random.default_rng(rand_seed)
    b2.seed(rand_seed)
    n_exc = 16
    n_elec = 4
    elec_coords = rng.uniform(-0.5, 0.5, (n_elec, 3)) * mm

    exc = b2.NeuronGroup(
        n_exc,
        """dIampa1/dt = xi_1 / sqrt(dt) : 1
           dIampa2/dt = xi_2 / sqrt(dt) : 1
           dIgaba1/dt = xi_3 / sqrt(dt) : 1
           dIgaba2/dt = xi_4 / sqrt(dt) : 1
        """,
    )
    assign_coords(exc, rng.uniform(-0.5, 0.5, (n_exc, 3)) * mm)

    # cleo setup
    sim = CLSimulator(Network(exc))
    sim.set_io_processor(RecordOnlyProcessor(samp_period_ms))

    def add_rwslfp_sig(
        pop_agg=True,
        amp_func=wslfp.mazzoni15_pop,
        ornt=[0, 0, -1],
        wslfp_kwargs={},
        Iampa_var_names=["Iampa1"],
        Igaba_var_names=["Igaba1"],
        name=None,
    ):
        rwslfp_sig = RWSLFPSignalFromPSCs(
            pop_aggregate=pop_agg,
            amp_func=amp_func,
            wslfp_kwargs=wslfp_kwargs,
        )
        if name:
            rwslfp_sig.name = name
        # separate probe for each signal since some injection kwargs need to be different per signal
        i = len(sim.recorders)
        probe = Probe(elec_coords, [rwslfp_sig], name=f"probe{i}")

        sim.inject(
            probe,
            exc,
            orientation=ornt,
            Iampa_var_names=Iampa_var_names,
            Igaba_var_names=Igaba_var_names,
        )
        return rwslfp_sig

    signals_for_options = []
    param_options = [
        ("ornt", (rng.normal(size=(n_exc, 3)), (-0.2, -0.1, -0.3))),
        ("Iampa_var_names", (["Iampa1"], ["Iampa1", "Iampa2"])),
        ("Igaba_var_names", (["Igaba1"], ["Igaba1", "Igaba2"])),
    ]
    for param_name, param_vals in _base_param_options + param_options:
        signals = []
        for val in param_vals:
            # store result of each different value for each param
            signals.append(
                add_rwslfp_sig(**{param_name: val}, name=f"{param_name}_{val}")
            )
        signals_for_options.append((param_name, signals))
    sim.run(30 * ms)
    assert not np.allclose(exc.Iampa1_, exc.Iampa2_)
    # each parameter change should change the resulting signal:
    assert len(signals_for_options) == len(_base_param_options) + len(param_options)
    for param, signals in signals_for_options:
        assert len(signals) > 1
        for i, sig1 in enumerate(signals):
            for sig2 in signals[i + 1 :]:
                assert not np.allclose(
                    sig1.lfp, sig2.lfp
                ), f"{sig1.name} and {sig2.name} not yielding different results"


def test_psc_buffer():
    sig = RWSLFPSignalFromPSCs()
    t_buf = [1, 2, 4]  # skipped 3 for some reason; should be [2, 3, 4]
    n_src = 4
    I_buf = np.arange(3)[..., None] + np.arange(n_src)
    print(I_buf)
    t_eval = 2
    with pytest.warns(match="buffer is unexpected"):
        assert np.all(
            sig._curr_from_buffer(t_buf, I_buf, t_eval, n_sources=n_src)
            == np.arange(1, 5)
        )
        assert np.all(
            sig._curr_from_buffer([1, 3, 4], I_buf, t_eval, n_sources=n_src)
            == np.arange(0.5, 4.5)
        )

    assert np.all(sig._curr_from_buffer([3, 4, 5], I_buf, t_eval, n_sources=n_src) == 0)
    assert np.all(
        sig._curr_from_buffer([-1, 0, 1], I_buf, t_eval, n_sources=n_src) == 0
    )


def test_psc_buf_len():
    sig = RWSLFPSignalFromPSCs()
    # args are buf_width, sample_period
    assert sig._buf_len(0, 1) == 1
    assert sig._buf_len(0, 1.5) == 1
    assert sig._buf_len(0, 12.13) == 1
    assert sig._buf_len(1, 1) == 2
    assert sig._buf_len(1, 0.5) == 3
    assert sig._buf_len(2, 1) == 3
    assert sig._buf_len(6, 1) == 7
    assert sig._buf_len(6, 1.2) == 6
    assert sig._buf_len(6, 1.4) == 6
    assert sig._buf_len(6, 1.5) == 5
    assert sig._buf_len(6, 1.6) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-xs", "--lf"])
