"""Tests for base module"""

import pytest
from brian2 import NeuronGroup, Network, collect, Synapses, PopulationRateMonitor

from clocsim import CLOCSimulator, InterfaceDevice, Recorder, Stimulator

class MyStim(Stimulator):
    def connect_to_neuron_group(self, neuron_group: NeuronGroup):
        external_input = NeuronGroup(1, 'v = -70 : 1')
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
    return NeuronGroup(1, 'v = -70 : 1', threshold='v > -50')

@pytest.fixture
def sim(neurons):
    net = Network(neurons)
    return CLOCSimulator(net)


def test_stimulator(sim, neurons):
    my_stim = MyStim('my_stim', 42)
    sim.inject_stimulator(my_stim, neurons)
    assert sim.stimulators['my_stim'] == my_stim

    assert len(my_stim.brian_objects) == 2
    assert all([obj in sim.network.objects for obj in my_stim.brian_objects])

    assert my_stim.value == 42
    my_stim.update(43)
    assert my_stim.value == 43


def test_recorder(sim, neurons):
    my_rec = MyRec('my_rec')
    sim.inject_recorder(my_rec, neurons)
    assert sim.recorders['my_rec'] == my_rec

    assert len(my_rec.brian_objects) == 1
    assert all([obj in sim.network.objects for obj in my_rec.brian_objects])

    assert my_rec.get_state() == -1
    assert sim.get_state() == {'my_rec': -1}