from typing import Tuple

from brian2 import (NeuronGroup, nmeter, second, psiemens, photon,
        usecond, farad, umeter, ms, mm, mm2, mV, Synapses, Equations)
import numpy as np

from . import Stimulator

# from PyRhO: Evans et al. 2016
# assuming this model is defined on "synapses" influencing a post-synaptic
# target population
four_state = '''
    dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1
    dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gr)*O1 : 1
    dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1
    dC2/dt = Gd2*O2 - (Gr0+Ga2)*C2 : 1

    C1+O1+O2+C2 = 1

    Ga1 = k1*phi**p/(phi**p + phim**p) : 1
    Gf = kf*phi**q/(phi**q + phim**q) + Gf0 : 1
    Gb = kb*phi**q/(phi**q + phim**q) + Gb0 : 1
    Ga2 = k2*phi**p/(phi**p + phim**p) : 1

    fphi = O1 + gamma*O2 : 1
    fv = (1 - exp(-(v_post-E)/v0)) / ((v_post-E)/v1) : 1

    Iopto_post = g0*fphi*fv*(v_post-E)
'''

# from try.projectpyrho.org's default 4-state params
ChR2_4s = {
    'g0': 114000*psiemens, 
    'gamma': 0.00742,
    'phim': 2.33e17/mm2/second,  # *photon, not in Brian2
    'k1': 4.15/ms,
    'k2': 0.868/ms,
    'p': 0.833,
    'Gf0': 0.0373/ms,
    'kf': 0.0581/ms,
    'Gb0': 0.0161/ms,
    'kb': 0.063/ms,
    'q': 1.94,
    'Gd1': 0.105/ms,
    'Gd2': 0.0138/ms,
    'Gr0': 0.00033/ms,
    'E': 0*mV,
    'v0': 43*mV,
    'v1': 17.1*mV
}

# from Foutz et al. 2012
default_fiber = {
    'R0': 0.1*mm,  # optical fiber radius
    'NAfib': 0.37  # optical fiber numerical aperture
}

# from Foutz et al. 2012
default_tissue = {
    'K': 7.37/mm,  # absorbance coefficient
    'S': 0.125/mm,  # scattering coefficient
    'ntis': 1.36  # tissue index of refraction
}

class OptogeneticIntervention(Stimulator):
    '''
    Requires neurons to have 3D spatial coordinates already assigned.
    Will add the necessary equations to the neurons for the optogenetic model.
    '''
    def __init__(self, name, opsin_model:str, opsin_params:dict, wavelength,
            light_source:dict=default_fiber, tissue:dict=default_tissue,
            location:Tuple[float, float, float]=(0,0,0), 
            direction:Tuple[float, float, float]=(0,0,1)):
        super().__init__(name)
        self.opsin_model = Equations(opsin_model, **opsin_params)
        self.light_source = light_source
        self.tissue = tissue
        self.location = location
        self.direction = direction

        self.value = 0

    def _Foutz12_transmittance(r, z):
        '''Foutz et al. 2012 transmittance model: Gaussian cone+Kubelka-Munk'''
        theta_div = np.arcsin(self.light_source['NAfib'] / self.tissue['ntis'])
        R = self.light_source['R0'] + z*np.tan()
        return G*C*M

    def connect_to_neurons(self, neuron_group):
        # calculate z, r coordinates
        r = ...
        z = ...
        light_model = '''
            Irr = Irr0*T : watt/meter2
            Irr0 : watt/meter2
            T : 1
            phi = ?
        '''
        opto_syn = Synapses(neuron_group, neuron_group,
                model=self.opsin_model+light_model)

    def update(self, ctrl_signal):
        self.value = ctrl_signal