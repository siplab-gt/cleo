"""Tests for cleo/processing/__init__.py"""
from typing import Any, Tuple

import pytest
from brian2 import Hz, Network, NeuronGroup, Quantity, ms, np

import cleo
from cleo.ioproc import LatencyIOProcessor


class MyLIOP(LatencyIOProcessor):
    def __init__(self, sample_period, **kwargs):
        super().__init__(sample_period, **kwargs)
        self.delay = 1.199 * ms
        self.count_no_input = 0
        self.count = 0

    def process(self, state_dict: dict, t_samp: Quantity):
        self.count += 1
        try:
            in_signal = state_dict["in"]
            out = in_signal + t_samp / ms
            return {"out": out}, t_samp + self.delay
        except KeyError:
            self.count_no_input += 1
            return {}, t_samp

    def reset(self):
        self.count_no_input = 0
        self.count = 0


def _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs):
    expected_out = [{} if out is None else {"out": out} for out in outputs]
    for i in range(len(t)):
        assert myLIOP.is_sampling_now(t[i]) == sampling[i]
        if myLIOP.is_sampling_now(t[i]):
            myLIOP.put_state({"in": inputs[i]}, t[i])
        assert myLIOP.get_stim_values(t[i]) == expected_out[i]
    assert myLIOP.count == np.sum(sampling)


def test_LatencyIOProcessor_fixed_serial():
    myLIOP = MyLIOP(1 * ms, sampling="fixed", processing="serial")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4] * ms
    sampling = [True, True, False, False, True, False, False]
    inputs = [42, 66, -1, -1, 1847, -1, -1]
    outputs = [None, None, 42, None, None, None, 67]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)
    assert len(myLIOP.out_buffer) > 0
    myLIOP._base_reset()
    myLIOP.reset()
    assert len(myLIOP.out_buffer) == 0
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_fixed_parallel():
    myLIOP = MyLIOP(1 * ms, sampling="fixed", processing="parallel")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4] * ms
    sampling = [True, True, False, False, True, False, False]
    inputs = [42, 66, -1, -1, 1847, -1, -1]
    outputs = [None, None, 42, None, None, 67, None]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)
    assert len(myLIOP.out_buffer) > 0
    myLIOP._base_reset()
    myLIOP.reset()
    assert len(myLIOP.out_buffer) == 0
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_wait_serial():
    myLIOP = MyLIOP(1 * ms, sampling="when idle", processing="serial")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4] * ms
    sampling = [True, False, True, False, False, False, True]
    inputs = [42, -1, 66, -1, -1, -1, 1847]
    outputs = [None, None, 42, None, None, None, 67.2]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)
    assert len(myLIOP.out_buffer) > 0
    myLIOP._base_reset()
    myLIOP.reset()
    assert len(myLIOP.out_buffer) == 0
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_wait_parallel():
    """This combination doesn't make much sense
    as noted in the class docstring. It behaves just as wait_serial
    because waiting results in only one sample at a time being
    processed."""

    myLIOP = MyLIOP(1 * ms, sampling="when idle", processing="parallel")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4] * ms
    sampling = [True, False, True, False, False, False, True]
    inputs = [42, -1, 66, -1, -1, -1, 1847]
    outputs = [None, None, 42, None, None, None, 67.2]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)
    assert len(myLIOP.out_buffer) > 0
    myLIOP._base_reset()
    myLIOP.reset()
    assert len(myLIOP.out_buffer) == 0
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_sim_LIOP_reset():
    sim = cleo.CLSimulator(Network())
    liop = MyLIOP(1 * ms)
    sim.set_io_processor(liop)
    sim.run(10 * ms)
    assert liop.count == liop.count_no_input == 10
    liop.out_buffer.append(({}, 0))
    assert len(liop.out_buffer) > 0
    sim.reset()
    assert liop.count == liop.count_no_input == 0
    assert len(liop.out_buffer) == 0


class SampleCounter(cleo.IOProcessor):
    """Just count samples"""

    def is_sampling_now(self, t_query) -> np.bool:
        """from LatencyIOProcessor"""
        resid_ms = np.round((t_query % self.sample_period) / ms, 6)
        if np.isclose(resid_ms, 0) or np.isclose(
            resid_ms, np.round(self.sample_period / ms, 6)
        ):
            return True

    def __init__(self):
        self.count = 0
        self.sample_period = 1 * ms
        self.latest_ctrl_signal = {}

    def put_state(self, state_dict: dict, t_samp: Quantity):
        self.count += 1
        return ({}, t_samp)

    def get_ctrl_signals(self, t_query: np.float) -> dict:
        return {}


@pytest.mark.slow
def test_no_skip_sampling():
    sc = SampleCounter()
    net = Network()
    sim = cleo.CLSimulator(net)
    sim.set_io_processor(sc)
    nsamp = 3000
    sim.run(nsamp * ms)
    assert sc.count == nsamp


def test_no_skip_sampling_short():
    net = Network()
    sim = cleo.CLSimulator(net)
    Tsamp = 0.2 * ms
    liop = MyLIOP(Tsamp)
    sim.set_io_processor(liop)
    nsamp = 20
    sim.run(nsamp * Tsamp)
    assert liop.count_no_input == nsamp


class WaveformController(LatencyIOProcessor):
    def process(self, state_dict, t):
        return {"steady": t / ms, "time-varying": t / ms + 1}, t + 3 * ms

    def preprocess_ctrl_signals(
        self, latest_ctrl_signals: dict, t_query: Quantity
    ) -> dict:
        out = {}
        # (sample_time_ms+1) * whether query time is even
        out["time-varying"] = latest_ctrl_signals.get("time-varying", 0) * int(
            (t_query / ms) % 2 == 0
        )
        return out


def test_intersample_waveform():
    ctrlr = WaveformController(sample_period=2 * ms)
    trange = np.arange(0, 10) * ms
    exp_outs = [
        {"time-varying": 0},  # t=0
        {"time-varying": 0},
        {"time-varying": 0},
        # t_query=3, t_sample=0
        {"steady": 0, "time-varying": 0},
        # t_query=4, t_sample=0
        {"time-varying": 1},
        # t_query=5, t_sample=2
        {"steady": 2, "time-varying": 0},
        # t_query=6, t_sample=2
        {"time-varying": 3},
        # t_query=7, t_sample=4
        {"steady": 4, "time-varying": 0},
        # t_query=8, t_sample=4
        {"time-varying": 5},
        # t_query=9, t_sample=6
        {"steady": 6, "time-varying": 0},
        # t_query=10, t_sample=6
        {"time-varying": 7},
    ]
    for t, exp_out in zip(trange, exp_outs):
        if ctrlr.is_sampling_now(t):
            ctrlr.put_state({}, t)
        assert ctrlr.get_stim_values(t) == exp_out
