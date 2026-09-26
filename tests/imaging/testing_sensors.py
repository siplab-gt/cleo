# import sensors
from cleo.imaging import sensors, scope
from cleo.base import SynapseDevice
from cleo.light import LightDependent, Light, LightModel
# import Dictionaries
from brian2 import nmolar, np, second, umolar, prefs
import brian2 as b2
import cleo
from cleo.light.two_photon import tp_light_from_scope
import matplotlib.pyplot as plt

b2.prefs.codegen.target='numpy'

def create_mini_simulation(n_neurons, radius, x, y, z):
    model = '''
        dv/dt = 0/ms : 1
        Irr : watt/meter**2
        phi : 1/second/meter**2
        soma_radius : meter'''
    ng = b2.NeuronGroup(n_neurons, model=model, threshold='v > 1', reset='v=0', name='my_neuron')
    ng.soma_radius=radius
    ng.add_attribute('x')
    ng.add_attribute('y')
    ng.add_attribute('z')
    ng.x = (np.arange(n_neurons) * 5 + x) * b2.um
    ng.y = np.ones(n_neurons) * y * b2.um
    ng.z = np.ones(n_neurons) * z * b2.um
    sim = cleo.CLSimulator(b2.Network(ng))
    return sim, ng

def test_light_intensity():
    radius_val = 15 * b2.um
    sim, ng = create_mini_simulation(5, radius_val, 2, 0, 100)
    gcamp = sensors.load_geci_dataframe('gcamp3')
    my_scope = scope.Scope(sensor=gcamp, img_width = 200 * b2.um, focus_depth = 0 * b2.um, location = (0, 0, 100) * b2.um)
    sim.inject(my_scope, ng)
    targets_found = sum(len(i_t) for i_t in my_scope.i_targets_per_injct)
    assert targets_found > 0, f"Scope at {my_scope.location} found 0 neurons. Check the coordinates"
    tp_light = tp_light_from_scope(my_scope, wavelength=500 * b2.nmeter)
    sim.inject(tp_light, ng)
    my_scope.inject_sensor_for_targets()
    try:
        sim.run(10 * b2.ms)
        state = my_scope.get_state()
        assert state.shape == (targets_found,), f"Expected {targets_found} neurons instead found {state.shape[0]}"
    except KeyError:
        print("soma_radius could not be resolved by the scope.")

def create_extremes_mini_simulation(n_neurons, radius, x, y, z, z_extreme):
    model = '''
        dv/dt = 0/ms : 1
        Irr : watt/meter**2
        phi : 1/second/meter**2
        soma_radius : meter'''
    ng = b2.NeuronGroup(n_neurons, model=model, threshold='v > 1', reset='v=0', name='my_neuron')
    ng.soma_radius=radius
    ng.add_attribute('x')
    ng.add_attribute('y')
    ng.add_attribute('z')
    ng.x = (np.arange(n_neurons) * 5 + x) * b2.um
    ng.y = np.ones(n_neurons) * y * b2.um
    ng.z = np.ones(n_neurons) * z * b2.um
    ng.z[-1] = z_extreme * b2.um
    sim = cleo.CLSimulator(b2.Network(ng))
    return sim, ng

def test_extreme_light_intensity():
    radius_val = 15 * b2.um
    sim, ng = create_extremes_mini_simulation(5, radius_val, 2, 0, 100, 300)
    gcamp = sensors.load_geci_dataframe('gcamp3')
    my_scope = scope.Scope(sensor=gcamp, img_width = 200 * b2.um, focus_depth = 0 * b2.um, location = (0, 0, 100) * b2.um, snr_cutoff = 2.0)
    sim.inject(my_scope, ng)
    targets_found = sum(len(i_t) for i_t in my_scope.i_targets_per_injct)
    assert targets_found > 0, f"Scope at {my_scope.location} found 0 neurons. Check the coordinates"
    tp_light = tp_light_from_scope(my_scope, wavelength=500 * b2.nmeter)
    sim.inject(tp_light, ng)
    my_scope.inject_sensor_for_targets()
    try:
        sim.run(10 * b2.ms)
        state = my_scope.get_state()
        assert state.shape == (targets_found,), f"Expected {targets_found} neurons instead found {state.shape[0]}"
    except KeyError:
        print("soma_radius could not be resolved by the scope.")

def sensitivity_test():
    radius_val = 15 * b2.um
    sim, ng = create_mini_simulation(5, radius_val, 2, 0, 100)
    gcamp = sensors.load_geci_dataframe('gcamp3')
    my_scope = scope.Scope(sensor=gcamp, img_width = 200 * b2.um, focus_depth = 0 * b2.um, location = (0, 0, 100) * b2.um,
                            save_history=True)
    sim.inject(my_scope, ng)
    my_scope.inject_sensor_for_targets()
    tp_light = tp_light_from_scope(my_scope, wavelength=500 * b2.nmeter)
    sim.inject(tp_light, ng)
    tp_light.intensity = 0 * b2.watt/b2.meter ** 2
    dFF_list = []
    t_list = []

    for _ in range(50):
        sim.run(1 * b2.ms)
        dFF_list.append(my_scope.get_state())
        t_list.append(sim.network.t/b2.ms)

    tp_light.intensity = 20 * b2.watt/b2.meter ** 2
    for _ in range(50):
        sim.run(1 * b2.ms)
        dFF_list.append(my_scope.get_state())
        t_list.append(sim.network.t/b2.ms)

    dFF_history = np.array(dFF_list)
    time_history = np.array(t_list)

    if (dFF_history.size > 0):
        Plot_Generator(time_history, dFF_history)
    else:
        print(f"Error: No data recorded in dFF_history")

def Plot_Generator(time_history, dFF_history):
    for i in range(len(dFF_history[0])):
        plt.plot(time_history, dFF_history[:, i].T, label=f"Neuron Group {i + 1}")
    plt.xlabel('Time in (ms)')
    plt.ylabel('dFF')
    plt.title('dFF over time for different neurons')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_light_intensity()
    test_extreme_light_intensity()
    sensitivity_test()