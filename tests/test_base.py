"""Tests for base module"""

import neo
import pytest
from brian2 import (
    BrianObjectException,
    Network,
    NeuronGroup,
    PopulationRateMonitor,
    Synapses,
    ms,
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
    assert len(my_stim.values) == len(my_stim.t/ms) == 2  # from update

    sim.update_stimulators({"my_stim": 42})
    assert my_stim.value == 42
    assert len(my_stim.values) == len(my_stim.t/ms) == 3  # from 2 updates

    my_stim.reset()
    assert len(my_stim.values) == len(my_stim.t/ms) == 1  # init

    neurons2 = NeuronGroup(1, "v = -70*mV : volt")
    with pytest.raises(Exception):  # neuron2 not in network
        sim.inject(my_stim, neurons2)

    # ok to inject same device again
    sim.inject(my_stim, neurons)
    # but not another device with same name
    my_stim2 = MyStim(name="my_stim")
    with pytest.raises(ValueError):
        sim.inject(my_stim2, neurons)
    my_stim2.name = "my_stim2"
    sim.inject(my_stim2, neurons)
    assert my_stim2 in sim.stimulators.values()


def test_recorder(sim, neurons):
    my_rec = MyRec(name="my_rec")
    sim.inject(my_rec, neurons)
    assert sim.recorders["my_rec"] == my_rec

    assert len(my_rec.brian_objects) == 1
    assert all([obj in sim.network.objects for obj in my_rec.brian_objects])

    assert my_rec.get_state() == -1
    assert sim.get_state() == {"my_rec": -1}

    # ok to inject same device again
    sim.inject(my_rec, neurons)
    # but not another device with same name
    my_rec2 = MyRec(name="my_rec")
    with pytest.raises(ValueError):
        sim.inject(my_rec2, neurons)
    my_rec2.name = "my_rec2"
    sim.inject(my_rec2, neurons)
    assert my_rec2 in sim.recorders.values()


class MyProcLoop(IOProcessor):
    def put_state(self, state_dict: dict, time):
        mock_processing = {-1: "expected"}
        self.my_stim_out = mock_processing[state_dict["my_rec"]]

    def get_ctrl_signals(self, time) -> dict:
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


def test_sim_to_neo():
    ng = NeuronGroup(1, "v = -70 : 1", threshold="v > -50")
    sim = CLSimulator(Network(ng))

    n_stims = 2
    for i in range(n_stims):
        stim = MyStim(name=f"stim{i}")
        sim.inject(stim, ng)
    rec = MyRec(name="rec")
    sim.inject(rec, ng)

    sim_neo = sim.to_neo()
    assert type(sim_neo) == neo.core.Block
    assert len(sim_neo.segments) == 1
    assert len(sim_neo.data_children_recur) == n_stims


def test_stim_to_neo():
    stim1 = MyStim(name="stim1")
    stim1.t = [0, 1, 2]*ms
    stim1.values = [1, 2, 3]
    stim1_neo = stim1.to_neo()
    assert stim1_neo.name == stim1.name
    assert type(stim1_neo) == neo.core.AnalogSignal

    stim2 = MyStim(name="stim2")
    stim2.t = [0, 1, 4]*ms
    stim2.values = [1, 2, 3]
    stim2_neo = stim2.to_neo()
    assert stim2_neo.name == stim2.name
    assert type(stim2_neo) == neo.core.IrregularlySampledSignal


if __name__ == "__main__":
    pytest.main(["-s", __file__])
