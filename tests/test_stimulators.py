"""Tests for stimulators/__init__.py"""

import pytest
from brian2 import NeuronGroup

from cleo.stimulators import StateVariableSetter


@pytest.fixture
def neurons():
    ng = NeuronGroup(1, "dv/dt = -70 - v: 1", reset="v = -70", threshold="v>-50")
    ng.v = -70
    return ng


neurons2 = neurons


def test_StateVariableSetter(neurons, neurons2):
    sv_stim = StateVariableSetter(variable_to_ctrl="v", unit=1, default_value=-1)
    assert neurons.v == -70
    assert neurons2.v == -70

    with pytest.raises(AttributeError):
        # tries to access sim.network.t but it doesn't have sim
        sv_stim.connect_to_neuron_group(neurons)
    sv_stim.save_history = False
    sv_stim.connect_to_neuron_group(neurons)
    assert neurons.v == -1
    assert neurons2.v == -70

    sv_stim.connect_to_neuron_group(neurons2)
    sv_stim.update(43)
    assert neurons.v == 43
    assert neurons2.v == 43
