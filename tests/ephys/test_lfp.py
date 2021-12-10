"""Tests for ephys.lfp module"""

from brian2.groups.neurongroup import NeuronGroup
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup

import cleosim
from cleosim.ephys.lfp import TelenczukLFP


def test_TelenczukLFP():
    tlfp = TelenczukLFP('tlfp')
    ng = SpikeGeneratorGroup(16, [], [])
    cleosim.coordinates.assign_coords_rand_rect_prism()
    tlfp.connect_to_neuron_group()