"""Tests for electrodes module"""
from brian2 import NeuronGroup
from cleosim.ephys import ElectrodeGroup

def test_ElectrodeGroup():
    eg = ElectrodeGroup("eg", [0, 0, 0])
    assert eg.n == 1

def test_probe_coords():
    pass

def test_array_coords():
    pass

