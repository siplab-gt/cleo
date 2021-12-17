"""Tests for stimulators/__init__.py"""

import pytest
from brian2 import NeuronGroup

from cleosim.stimulators import StateVariableSetter


@pytest.fixture
def neurons():
    ng = NeuronGroup(1, "dv/dt = -70 - v: 1", reset="v = -70", threshold="v>-50")
    ng.v = -70
    return ng


neurons2 = neurons


def test_StateVariableSetter(neurons, neurons2):
    sv_stim = StateVariableSetter("sv_stim", "v", 1, start_value=-1)
    assert neurons.v == -70
    assert neurons2.v == -70

    sv_stim.connect_to_neuron_group(neurons)
    assert neurons.v == -1
    assert neurons2.v == -70

    sv_stim.connect_to_neuron_group(neurons2)
    sv_stim.update(43)
    assert neurons.v == 43
    assert neurons2.v == 43