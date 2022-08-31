"""Tests for cleo/processing/__init__.py"""
from typing import Any, Tuple

from brian2 import Network, PoissonGroup, ms, Hz

from cleo import CLSimulator
from cleo.processing import LatencyIOProcessor, ProcessingBlock, ConstantDelay


class MyProcessingBlock(ProcessingBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output(self, input: Any, **kwargs) -> Any:
        measurement_time = kwargs["measurement_time"]
        return input + measurement_time


def test_ProcessingBlock():
    const_delay = 5
    my_block = MyProcessingBlock(delay=ConstantDelay(const_delay), save_history=True)

    # blank history to start
    assert len(my_block.t_in_ms) == 0
    assert len(my_block.t_out_ms) == 0
    assert len(my_block.values) == 0

    meas_time = [1, 8]
    in_time = [2, 9]
    inputs = [42, -1]
    outputs = [a + b for a, b in zip(inputs, meas_time)]
    out_times = [t + const_delay for t in in_time]

    for i in [0, 1]:
        out, out_time = my_block.process(
            inputs[i],
            t_in_ms=in_time[i],
            measurement_time=meas_time[i],
        )
        # process with extra arg
        assert out == outputs[i]
        # delay
        assert out_time == out_times[i]
        # save history
        assert len(my_block.t_in_ms) == i + 1
        assert len(my_block.t_out_ms) == i + 1
        assert len(my_block.values) == i + 1


class MyLIOP(LatencyIOProcessor):
    def __init__(self, sample_period_ms, **kwargs):
        super().__init__(sample_period_ms, **kwargs)
        self.delay = 1.199
        self.component = MyProcessingBlock(delay=ConstantDelay(self.delay))

    def process(self, state_dict: dict, sample_time_ms: float) -> Tuple[dict, float]:
        input = state_dict["in"]
        out, out_t = self.component.process(
            input, sample_time_ms, measurement_time=sample_time_ms
        )
        return {"out": out}, out_t


def _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs):
    for i in range(len(t)):
        assert myLIOP.is_sampling_now(t[i]) == sampling[i]
        if myLIOP.is_sampling_now(t[i]):
            myLIOP.put_state({"in": inputs[i]}, t[i])
        expected_out = [out if out is None else {"out": out} for out in outputs]
        assert myLIOP.get_ctrl_signal(t[i]) == expected_out[i]


def test_LatencyIOProcessor_fixed_serial():
    myLIOP = MyLIOP(1, sampling="fixed", processing="serial")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4]
    sampling = [True, True, False, False, True, False, False]
    inputs = [42, 66, -1, -1, 1847, -1, -1]
    outputs = [None, None, 42, None, None, None, 67]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_fixed_parallel():
    myLIOP = MyLIOP(1, sampling="fixed", processing="parallel")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4]
    sampling = [True, True, False, False, True, False, False]
    inputs = [42, 66, -1, -1, 1847, -1, -1]
    outputs = [None, None, 42, None, None, 67, None]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_wait_serial():
    myLIOP = MyLIOP(1, sampling="when idle", processing="serial")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4]
    sampling = [True, False, True, False, False, False, True]
    inputs = [42, -1, 66, -1, -1, -1, 1847]
    outputs = [None, None, 42, None, None, None, 67.2]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


def test_LatencyIOProcessor_wait_parallel():
    """This combination doesn't make much sense
    as noted in the class docstring. It behaves just as wait_serial
    because waiting results in only one sample at a time being
    processed."""

    myLIOP = MyLIOP(1, sampling="when idle", processing="parallel")
    t = [0, 1, 1.2, 1.3, 2, 2.3, 2.4]
    sampling = [True, False, True, False, False, False, True]
    inputs = [42, -1, 66, -1, -1, -1, 1847]
    outputs = [None, None, 42, None, None, None, 67.2]  # input + measurement_time
    _test_LatencyIOProcessor(myLIOP, t, sampling, inputs, outputs)


class SampleCounter(LatencyIOProcessor):
    """Just count samples"""

    def __init__(self, sample_period_ms, **kwargs):
        super().__init__(sample_period_ms, **kwargs)
        self.count = 0

    def process(self, state_dict: dict, sample_time_ms: float) -> Tuple[dict, float]:
        self.count += 1
        return ({}, sample_time_ms)


def test_no_skip_sampling():
    sc = SampleCounter(1)
    net = Network(PoissonGroup(1, 100 * Hz))
    sim = CLSimulator(net)
    sim.set_io_processor(sc)
    sim.run(150 * ms)
    assert sc.count == 150
