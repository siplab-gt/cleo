import numpy as np
from brian2 import SpikeMonitor, NeuronGroup, ms
from scipy.optimize import curve_fit, minimize, Bounds
from sklearn.linear_model import LinearRegression

class S2FGECI:
    def __init__(self, Fm, F0, tau_r, tau_d1, tau_d2, r, time_window, spectrum):
        self.spike_monitors = {}
        self.Fm = Fm
        self.F0 = F0
        self.tau_r = tau_r
        self.tau_d1 = tau_d1
        self.tau_d2 = tau_d2
        self.r = r
        self.time_window = time_window
        self.spectrum = spectrum

    def connect_to_neuron_group(self, neuron_group: NeuronGroup):
        """
        Connects a neuron group to the S2F model by setting up a SpikeMonitor.

        Parameters:
        - neuron_group: The NeuronGroup to monitor for spikes.
        """
        try:
            # Create a SpikeMonitor for the neuron group
            spike_monitor = SpikeMonitor(neuron_group)
            
            # Store the SpikeMonitor in a dictionary keyed by the neuron group's name
            self.spike_monitors[neuron_group.name] = spike_monitor
            
            # Log the successful connection
            print(f"Connected neuron group '{neuron_group.name}' with SpikeMonitor.")
        except Exception as e:
            # Log any errors encountered during setup
            print(f"Error connecting neuron group '{neuron_group.name}': {e}")

    def get_state(self, current_time):
        fluorescence_output = {}
        for group_name, spike_monitor in self.spike_monitors.items():
            spike_times = spike_monitor.t
            neuron_indices = spike_monitor.i

            # Filter spikes within the time window
            recent_spikes_mask = spike_times > (current_time - self.time_window * ms)
            recent_spike_indices = neuron_indices[recent_spikes_mask]

            # Compute ΔF/F for each neuron
            dff = self.compute_dff(recent_spike_indices, spike_monitor.source.N)
            fluorescence_output[group_name] = dff

        return fluorescence_output

    def compute_dff(self, spike_indices, num_neurons):
        # Initialize ΔF/F array
        dff = np.zeros(num_neurons)
        # Count spikes for each neuron
        spike_counts = np.bincount(spike_indices, minlength=num_neurons)
        # Convert spike counts to ΔF/F
        dff = self.F0 + (self.Fm - self.F0) * spike_counts
        return dff

    def apply_light_dependency(self, fluorescence, wavelength):
        # Use the epsilon function to adjust fluorescence based on light conditions
        sensitivity = self.epsilon(wavelength)
        adjusted_fluorescence = fluorescence * sensitivity
        return adjusted_fluorescence

    def epsilon(self, lambda_new):
        # Interpolate the action spectrum to find the relative sensitivity
        action_spectrum = np.array(self.spectrum)
        lambdas = action_spectrum[:, 0]
        epsilons = action_spectrum[:, 1]
        if lambda_new < min(lambdas) or lambda_new > max(lambdas):
            return 0
        eps_new = np.interp(lambda_new, lambdas, epsilons)
        return max(0, eps_new)  # Ensure ε is non-negative

# Example usage
neuron_group = NeuronGroup(100, 'dv/dt = -v / (10*ms) : 1')  # Example neuron group
s2f_model = S2FGECI(Fm=1.0, F0=0.1, tau_r=0.01, tau_d1=0.1, tau_d2=0.2, r=0.5, time_window=100, spectrum=[(400, 0.1), (500, 0.5), (600, 0.9)])
s2f_model.connect_to_neuron_group(neuron_group)

# Simulate and get state
current_time = 1000 * ms  # Example current time
fluorescence_state = s2f_model.get_state(current_time)
print(fluorescence_state)
