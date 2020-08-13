from typing import Tuple

from brian2 import (NeuronGroup, Synapses, Equations)
from brian2.units import *
from brian2.units.allunits import meter2
import brian2.units.unitsafefunctions as usf
import numpy as np

from . import Stimulator

# from PyRhO: Evans et al. 2016
# assuming this model is defined on "synapses" influencing a post-synaptic
# target population. rho_rel is channel density relative to standard model fit,
# allowing for heterogeneous opsin expression.
four_state = '''
    dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1
    dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gr)*O1 : 1
    dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1
    dC2/dt = Gd2*O2 - (Gr0+Ga2)*C2 : 1
    Ga1 = k1*phi**p/(phi**p + phim**p) : 1
    Gf = kf*phi**q/(phi**q + phim**q) + Gf0 : 1
    Gb = kb*phi**q/(phi**q + phim**q) + Gb0 : 1
    Ga2 = k2*phi**p/(phi**p + phim**p) : 1

    fphi = O1 + gamma*O2 : 1
    fv = (1 - exp(-(v_post-E)/v0)) / ((v_post-E)/v1) : 1

    Iopto_post = g0*fphi*fv*(v_post-E)*rho_rel : ampere
    rho_rel = 1 : 1
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
            p_expression:float=1, location:Tuple[float, float, float]=(0,0,0)*mm,
            direction:Tuple[float, float, float]=(0,0,1)):
        '''
        direction: (x,y,z) tuple representing direction light is pointing
        '''
        super().__init__(name)
        self.opsin_model = Equations(opsin_model, **opsin_params)
        self.wavelength = wavelength
        self.light_source = light_source
        self.tissue = tissue
        self.p_expression = p_expression
        self.location = location
        # direction unit vector
        self.dir_uvec = (direction/np.linalg.norm(direction)).reshape((3,1))

    def _Foutz12_transmittance(self, r, z):
        '''Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation'''
        # divergence half-angle of cone
        theta_div = np.arcsin(self.light_source['NAfib'] / self.tissue['ntis'])
        Rz = self.light_source['R0'] + z*np.tan(theta_div)  # radius as light spreads
        C = (self.light_source['R0']/Rz)**2
        G = 1/np.sqrt(2*np.pi) * np.exp(-2*(r/Rz)**2)
        
        S = self.tissue['S']
        a = 1 + self.tissue['K']/S
        b = np.sqrt(a**2 - 1)
        M = b / (a*np.sinh(b*S*np.sqrt(r**2+z**2)) + b*np.cosh(b*S*np.sqrt(r**2+z**2)))
        T = G*C*M
        # should be 0 for negative z
        T[z<0] = 0
        return T

    def connect_to_neurons(self, neuron_group):
        # calculate z, r coordinates
        coords = np.vstack([neuron_group.x, neuron_group.y, neuron_group.z]).T
        z = coords @ self.dir_uvec
        axial_points = z @ self.dir_uvec.T
        r = coords - axial_points

        light_model = Equations('''
            Irr = Irr0*T : watt/meter2
            Irr0 : watt/meter2
            T : 1
            phi = Irr / Ephoton : 1/second/meter2''',
            # Ephoton = h*c/lambda
            Ephoton = 6.63e-34*meter2*kgram/second * 2.998e8*meter/second \
                / self.wavelength
        )

        self.opto_syn = Synapses(neuron_group, neuron_group,
                model=self.opsin_model+light_model)
        # calculate transmittance coefficient for each point
        self.opto_syn.T = self._Foutz12_transmittance(r, z)

        self.opto_syn.connect(j='i', p=self.p_expression)
        self.brian_objects.add(self.opto_syn)

    def add_self_to_plot(self, ax, axis_scale_unit):
        # show light with point field, assigning r and z coordinates
        # to all points
        xlim=ax.get_xlim(); ylim=ax.get_ylim(); zlim=ax.get_zlim()
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        z = np.linspace(zlim[0], zlim[1], 100)
        x, y, z = np.meshgrid(x, y, z)

        coords = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        coords = coords*axis_scale_unit
        rel_coords = coords - self.location  # relative to fiber location
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        zc = usf.dot(rel_coords, self.dir_uvec)  # distance along cylinder axis
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt( np.sum( (rel_coords-usf.dot(zc, self.dir_uvec.T))**2, axis=1 ) )
        r = r.reshape((-1, 1))

        T = self._Foutz12_transmittance(r, zc)
        # filter out points with <0.001 transmittance to make plotting faster
        plot_threshold = 0.0001
        idx_to_plot = T[:,0] >= plot_threshold
        x = x.flatten()[idx_to_plot]; y = y.flatten()[idx_to_plot]; z = z.flatten()[idx_to_plot]
        T = T[idx_to_plot, 0]
        ax.scatter(x, y, z, c=T, cmap=self._alpha_cmap_for_wavelength(), marker='.',
                edgecolors='face')

    def update(self, ctrl_signal):
        self.opto_syn.Irr0 = ctrl_signal
    
    def _alpha_cmap_for_wavelength(self):
        from matplotlib import colors
        from ..utilities import wavelength_to_rgb
        c = wavelength_to_rgb(self.wavelength/nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 1)
        return colors.LinearSegmentedColormap.from_list(
            'incr_alpha', [(0, c_clear), (1, c_opaque)]
        )

        