from brian2 import nsiemens, pamp, mV, ms, Mohm, NeuronGroup

neuron_params = {
    'a': 0.0*nsiemens, 'b': 60*pamp, 'E_L': -70*mV, 'tau_m': 20*ms,
    'R': 500*Mohm, 'theta': -50*mV, 'v_reset': -55*mV, 'tau_w': 30*ms,
    'Delta_T': 2*mV
}

def lif(n, name='LIF'):
    ng = NeuronGroup(n,
        '''dv/dt = (-(v - E_L) + R*Iopto) / tau_m : volt
        Iopto: amp
        ''',
        threshold='v>=theta',
        reset='v=E_L',
        refractory=2*ms,
        namespace=neuron_params,
        name=name
    )
    ng.v = neuron_params["E_L"]
    return ng

def adex(n, name='AdEx'):
    ng = NeuronGroup(n,
        '''dv/dt = (-(v - E_L) + Delta_T*exp((v-theta)/Delta_T) + R*(w+Iopto)) / tau_m : volt
        dw/dt = (a*(v-E_L) - w) / tau_w : amp
        Iopto : amp''',
        threshold='v>=30*mV',
        reset='v=v_reset; w+=b',
        namespace=neuron_params,
        name=name
    )
    ng.v = neuron_params["E_L"]
    return ng
