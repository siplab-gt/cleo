"""Tests for base module"""

import pytest
from brian2 import (
    NeuronGroup,
    Network,
    Synapses,
    PopulationRateMonitor,
    ms,
    BrianObjectException,
)

from cleo import CLSimulator, IOProcessor, Recorder, Stimulator


class MyStim(Stimulator):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup):
        external_input = NeuronGroup(1, "v = -70 : 1")
        syn = Synapses(external_input, neuron_group)
        self.brian_objects.update([syn, external_input])


class MyRec(Recorder):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup):
        self.mon = PopulationRateMonitor(neuron_group)
        self.brian_objects.add(self.mon)

    def get_state(self):
        return -1


@pytest.fixture
def neurons():
    return NeuronGroup(1, "v = -70 : 1", threshold="v > -50")


@pytest.fixture
def sim(neurons):
    net = Network(neurons)
    return CLSimulator(net)


def test_stimulator(sim, neurons):
    my_stim = MyStim(42, name="my_stim", save_history=True)
    sim.inject(my_stim, neurons)
    assert sim.stimulators["my_stim"] == my_stim

    assert len(my_stim.brian_objects) == 2
    assert all([obj in sim.network.objects for obj in my_stim.brian_objects])

    assert my_stim.value == 42
    my_stim.update(43)
    assert my_stim.value == 43
    assert len(my_stim.values) == len(my_stim.t_ms) == 1  # from update

    sim.update_stimulators({"my_stim": 42})
    assert my_stim.value == 42
    assert len(my_stim.values) == len(my_stim.t_ms) == 2  # from 2 updates

    my_stim.reset()
    assert len(my_stim.values) == len(my_stim.t_ms) == 0  # init

    neurons2 = NeuronGroup(1, "v = -70*mV : volt")
    with pytest.raises(Exception):  # neuron2 not in network
        sim.inject(my_stim, neurons2)


def test_recorder(sim, neurons):
    my_rec = MyRec(name="my_rec")
    sim.inject(my_rec, neurons)
    assert sim.recorders["my_rec"] == my_rec

    assert len(my_rec.brian_objects) == 1
    assert all([obj in sim.network.objects for obj in my_rec.brian_objects])

    assert my_rec.get_state() == -1
    assert sim.get_state() == {"my_rec": -1}


class MyProcLoop(IOProcessor):
    def put_state(self, state_dict: dict, time):
        mock_processing = {-1: "expected"}
        self.my_stim_out = mock_processing[state_dict["my_rec"]]

    def get_ctrl_signal(self, time) -> dict:
        return {"my_stim": self.my_stim_out}

    def is_sampling_now(self, time) -> bool:
        return True


def test_io_processor_in_sim(sim, neurons):
    my_rec = MyRec(name="my_rec")
    sim.inject(my_rec, neurons)
    my_stim = MyStim(42, name="my_stim")
    sim.inject(my_stim, neurons)

    sim.set_io_processor(MyProcLoop())
    sim.run(0.1 * ms)
    assert my_stim.value == "expected"


def test_namespace_level():
    test_v = -5
    ng = NeuronGroup(1, "v = -70 + test_v: 1")
    sim = CLSimulator(Network(ng))
    sim.run(0.1 * ms)
    with pytest.raises(BrianObjectException):
        sim.run(0.1 * ms, level=0)
    with pytest.raises(BrianObjectException):
        sim.run(0.1 * ms, level=2)
