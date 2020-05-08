from brian2 import *
from clocsim.base import CLOCSimulator


## a group to record from, receiving Poisson input
recording_group = NeuronGroup(100, '''
        dv/dt = -v / tau : volt
        tau: second
''')
recording_group.tau = 10*ms

input_group = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)

S = Synapses(input_group, recording_group, on_pre='v+=0.1*mV')
S.connect(j='i')

## a group to control
control_group = NeuronGroup(100, '''dv/dt = -v / tau : volt
                                    tau: second''')
control_group.tau = 11*ms

mon = SpikeMonitor(control_group)


## run simulation
net = Network(collect())
net.run(1*s)