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
    dC1/dt = Gd1*O1 + Gr0*C2 - Ga1*C1 : 1 (clock-driven)
    dO1/dt = Ga1*C1 + Gb*O2 - (Gd1+Gf)*O1 : 1 (clock-driven)
    dO2/dt = Ga2*C2 + Gf*O1 - (Gd2+Gb)*O2 : 1 (clock-driven)
    C2 = 1 - C1 - O1 - O2 : 1
    # dC2/dt = Gd2*O2 - (Gr0+Ga2)*C2 : 1 (clock-driven)

    Theta = int(phi > 0*phi) : 1
    Hp = Theta * phi**p/(phi**p + phim**p) : 1
    Ga1 = k1*Hp : hertz
    Ga2 = k2*Hp : hertz
    Hq = Theta * phi**q/(phi**q + phim**q) : 1
    Gf = kf*Hq + Gf0 : hertz
    Gb = kb*Hq + Gb0 : hertz

    fphi = O1 + gamma*O2 : 1
    fv = (1 - exp(-(v_post-E)/v0)) / ((v_post-E)/v1) : 1

    Iopto_post = g0*fphi*fv*(v_post-E)*rho_rel : ampere (summed)
    rho_rel = 1 : 1
'''

# from try.projectpyrho.org's default 4-state params
ChR2_four_state = {
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
default_blue = {
    'R0': 0.1*mm,  # optical fiber radius
    'NAfib': 0.37,  # optical fiber numerical aperture
    'wavelength': 473*nmeter,
    # NOTE: the following depend on wavelength and tissue properties and thus would be different for another wavelength
    # 'K': 7.37/mm,  # absorbance coefficient
    'K': 0.125/mm,  # absorbance coefficient
    # 'S': 0.125/mm,  # scattering coefficient
    'S': 7.37/mm,  # scattering coefficient
    'ntis': 1.36  # tissue index of refraction
}

class OptogeneticIntervention(Stimulator):
    '''
    Requires neurons to have 3D spatial coordinates already assigned.
    Will add the necessary equations to the neurons for the optogenetic model.
    '''
    def __init__(self, name, opsin_model:str, opsin_params:dict,
            light_model_params:dict,
            p_expression:float=1, location:Tuple[float, float, float]=(0,0,0)*mm,
            direction:Tuple[float, float, float]=(0,0,1)):
        '''
        direction: (x,y,z) tuple representing direction light is pointing
        '''
        super().__init__(name)
        self.opsin_model = Equations(opsin_model, **opsin_params)
        self.light_model_params = light_model_params
        self.p_expression = p_expression
        self.location = location
        # direction unit vector
        self.dir_uvec = (direction/np.linalg.norm(direction)).reshape((3,1))

    def _Foutz12_transmittance(self, r, z, scatter=True, spread=True, gaussian=True):
        '''Foutz et al. 2012 transmittance model: Gaussian cone with Kubelka-Munk propagation'''

        if spread:
            # divergence half-angle of cone
            theta_div = np.arcsin(self.light_model_params['NAfib'] / 
                    self.light_model_params['ntis'])
            Rz = self.light_model_params['R0'] + z*np.tan(theta_div)  # radius as light spreads("apparent radius" from original code)
            C = (self.light_model_params['R0']/Rz)**2
        else:
            Rz = self.light_model_params['R0']  # "apparent radius"
            C = 1

        if gaussian:
            G = 1/np.sqrt(2*np.pi) * np.exp(-2*(r/Rz)**2)
        else:
            G = 1
        
        def kubelka_munk(dist):
            S = self.light_model_params['S']
            a = 1 + self.light_model_params['K']/S
            b = np.sqrt(a**2 - 1)
            dist = np.sqrt(r**2 + z**2)
            return b / (a*np.sinh(b*S*dist) + b*np.cosh(b*S*dist))
        M = kubelka_munk(np.sqrt(r**2+z**2)) if scatter else 1

        # cone of light shouldn't extend backwards
        T = G*C*M
        T[z<0] = 0
        return T


    def get_rz_for_xyz(self, x, y, z):
        '''Assumes x, y, z already have units'''
        def flatten_if_needed(var):
            if len(var.shape) != 1:
                return var.flatten()
            else: return var
        # have to add unit back on since it's stripped by vstack
        coords = np.vstack([flatten_if_needed(x), flatten_if_needed(y), flatten_if_needed(z)]).T*meter
        rel_coords = coords - self.location  # relative to fiber location
        # must use brian2's dot function for matrix multiply to preserve
        # units correctly.
        zc = usf.dot(rel_coords, self.dir_uvec)  # distance along cylinder axis
        # just need length (norm) of radius vectors
        # not using np.linalg.norm because it strips units
        r = np.sqrt( np.sum( (rel_coords-usf.dot(zc, self.dir_uvec.T))**2, axis=1 ) )
        r = r.reshape((-1, 1))
        return r, zc


    def connect_to_neurons(self, neuron_group):
        r, z = self.get_rz_for_xyz(neuron_group.x, neuron_group.y, neuron_group.z)

        # Ephoton = h*c/lambda
        E_photon = 6.63e-34*meter2*kgram/second * 2.998e8*meter/second \
            / self.light_model_params['wavelength']

        light_model = Equations('''
            Irr = Irr0*T : watt/meter**2
            Irr0 : watt/meter**2 
            T : 1
            phi = Irr / Ephoton : 1/second/meter**2
            Ephoton = E_photon : joule''',
            E_photon=E_photon
        )

        self.opto_syn = Synapses(neuron_group,
                model=self.opsin_model+light_model)
        self.opto_syn.connect(j='i', p=self.p_expression)
        for k, v in {'C1':1, 'O1':0, 'O2':0}.items():
            setattr(self.opto_syn, k, v)
        # calculate transmittance coefficient for each point
        self.opto_syn.T = self._Foutz12_transmittance(r, z).flatten()
        self.brian_objects.add(self.opto_syn)

    def add_self_to_plot(self, ax, axis_scale_unit):
        # show light with point field, assigning r and z coordinates
        # to all points
        xlim=ax.get_xlim(); ylim=ax.get_ylim(); zlim=ax.get_zlim()
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        z = np.linspace(zlim[0], zlim[1], 100)
        x, y, z = np.meshgrid(x, y, z)*axis_scale_unit

        r, zc = self.get_rz_for_xyz(x, y, z)
        T = self._Foutz12_transmittance(r, zc)
        # filter out points with <0.001 transmittance to make plotting faster
        plot_threshold = 0.0001
        idx_to_plot = T[:,0] >= plot_threshold
        x = x.flatten()[idx_to_plot]; y = y.flatten()[idx_to_plot]; z = z.flatten()[idx_to_plot]
        T = T[idx_to_plot, 0]
        ax.scatter(x/axis_scale_unit, y/axis_scale_unit, z/axis_scale_unit, c=T, cmap=self._alpha_cmap_for_wavelength(), marker='.',
                edgecolors='face')

    def update(self, ctrl_signal):
        self.opto_syn.Irr0 = ctrl_signal
    
    def _alpha_cmap_for_wavelength(self):
        from matplotlib import colors
        from ..utilities import wavelength_to_rgb
        c = wavelength_to_rgb(self.light_model_params['wavelength']/nmeter)
        c_clear = (*c, 0)
        c_opaque = (*c, 1)
        return colors.LinearSegmentedColormap.from_list(
            'incr_alpha', [(0, c_clear), (1, c_opaque)]
        )

        