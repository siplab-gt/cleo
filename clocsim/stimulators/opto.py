from brian2 import NeuronGroup, nm

from .base import Stimulator

class OptogeneticIntervention(Stimulator):
    '''
    Requires neurons to have 3D spatial coordinates already assigned.
    Will add the necessary equations to the neurons for the optogenetic model.
    '''
    def __init__(self, name, delivery='fiber', opsin='ChR2', wavelength=473*nm,
            location=(0,0,0), direction=(0,0,1), model_states=4):
        super().__init__(name)
        self.location = location
        self.direction = direction
        if delivery not in ['fiber']:
            raise NotImplementedError
        if model_states not in [4]:
            raise NotImplementedError

        self.value = 0

    def connect_to_neurons(self, neuron_group):
        opto_syn = Synapses(neuron_group, neuron_group,
                model='''
                
                ''')

    def update(self, ctrl_signal):
        self.value = ctrl_signal