from brian2 import NeuronGroup, Synapses, second, umolar, nmolar, ms
from attrs import define, field
import numpy as np
from scipy.special import expit

from cleo.base import SynapseDevice
from cleo.light import LightDependent

from brian2.units import meter

nm = meter * 1e-9

@define(eq=False, slots=False)
class S2FSensor(SynapseDevice):
    rise: float = field(kw_only=True)
    decay1: float = field(kw_only=True)
    decay2: float = field(kw_only=True, default=0)
    r: float = field(kw_only=True, default=0)
    Fm: float = field(kw_only=True, default=1.0)
    Ca0: float = field(kw_only=True, default=0.0)
    beta: float = field(kw_only=True, default=1.0)
    F0: float = field(kw_only=True, default=0.0)
    sigma_noise: float = field(kw_only=True, default=0.0)
    spike_monitors: dict[NeuronGroup, Synapses] = field(factory=dict, init=False)

    def spike_to_calcium(self, spike_times, ca_times):
        tau_r = self.rise
        tau_d1 = self.decay1
        tau_d2 = self.decay2
        r = self.r
        ca_trace = np.zeros_like(ca_times, dtype=float)

        for spk in spike_times:
            ca_trace_tmp = (
                r * np.exp(-(ca_times - spk) / tau_d1) + np.exp(-(ca_times - spk) / tau_d2)
            ) * (1 - np.exp(-(ca_times - spk) / tau_r))
            ca_trace_tmp[ca_times <= spk] = 0  # Ignore calcium changes before the spike
            ca_trace += ca_trace_tmp

        # Add Gaussian noise
        noise = np.random.normal(0, self.sigma_noise, size=ca_trace.shape)
        ca_trace += noise
        ca_trace[ca_trace < 0] = 0  # Truncate negative values
        return ca_trace

    def sigmoid_response(self, ca_trace):
        return self.Fm / (1 + np.exp(-self.beta * (ca_trace - self.Ca0))) + self.F0

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
            
            # Add the SpikeMonitor to the simulation's objects if necessary
            # This ensures the monitor is updated during the simulation
            if hasattr(self, 'brian_objects'):
                self.brian_objects.add(spike_monitor)
            
            # Log the successful connection
            print(f"Connected neuron group '{neuron_group.name}' with SpikeMonitor.")
        except Exception as e:
            # Log any errors encountered during setup
            print(f"Error connecting neuron group '{neuron_group.name}': {e}")


@define(eq=False, slots=False)
class S2FLightDependentGECI(S2FSensor, LightDependent):
    spectrum: list[tuple[float, float]] = field(kw_only=True)

    def get_response(self, spike_times, ca_times, neuron_coords):
        ca_trace = self.spike_to_calcium(spike_times, ca_times)
        light_intensity = self.light_agg_ngs.transmittance(neuron_coords)
        epsilon = self.epsilon(self.light_source.wavelength / nm)
        adjusted_light_intensity = light_intensity * epsilon
        return self.sigmoid_response(ca_trace) * adjusted_light_intensity


@define(eq=False, slots=False)
class S2FLightIndependentGECI(S2FSensor):
    def get_response(self, spike_times, ca_times):
        ca_trace = self.spike_to_calcium(spike_times, ca_times)
        return self.sigmoid_response(ca_trace)


@define(eq=False, slots=False)
class S2FModel:
    sensor: S2FSensor = field(kw_only=True)
    neuron_group: NeuronGroup = field(kw_only=True)

    def init_syn_vars(self, syn: Synapses) -> None:
        self.sensor.init_syn_vars(syn)

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        fluorescence_output = {}

        for neuron_group, spike_monitor in self.sensor.spike_monitors.items():
            spike_times = spike_monitor.t
            neuron_indices = spike_monitor.i
            
            # Define the time range in ms
            ca_times = self.sensor.sim.network.t / ms
            
            group_fluorescence = np.zeros((neuron_group.N, len(ca_times)))
            
            for neuron_idx in np.unique(neuron_indices):
                neuron_spike_times = spike_times[neuron_indices == neuron_idx]
                group_fluorescence[neuron_idx] = self.sensor.get_response(neuron_spike_times, ca_times)
            
            fluorescence_output[neuron_group] = group_fluorescence

        return fluorescence_output


# Define specific S2F GECI functions
def _create_s2f_geci_fn(
    name,
    rise,
    decay1,
    decay2,
    r,
    Fm,
    Ca0,
    beta,
    F0,
    sigma_noise,
):
    def s2f_geci_fn(light_dependent=False, light_source=None, spectrum=[]):
        if light_dependent:
            return S2FLightDependentGECI(
                rise=rise,
                decay1=decay1,
                decay2=decay2,
                r=r,
                Fm=Fm,
                Ca0=Ca0,
                beta=beta,
                F0=F0,
                sigma_noise=sigma_noise,
                spectrum=spectrum,
                light_source=light_source,
            )
        else:
            return S2FLightIndependentGECI(
                rise=rise,
                decay1=decay1,
                decay2=decay2,
                r=r,
                Fm=Fm,
                Ca0=Ca0,
                beta=beta,
                F0=F0,
                sigma_noise=sigma_noise,
            )
    globals()[name] = s2f_geci_fn


# Define specific S2F GECI functions based on the chart
_create_s2f_geci_fn("jGCaMP8f", 1.85, 34.07, 263.70, 0.48, 6.104380, 4.170575, 0.390533, -1.001000, 0)
_create_s2f_geci_fn("jGCaMP8m", 2.46, 41.64, 245.80, 0.28, 7.454645, 2.691117, 0.360008, -2.050880, 0)
_create_s2f_geci_fn("jGCaMP8s", 5.65, 86.26, 465.45, 0.19, 7.455792, 1.282417, 0.343721, -2.919320, 0)
_create_s2f_geci_fn("jGCaMP7f", 16.21, 95.27, 398.22, 0.24, 6.841247, 5.562159, 0.423212, -0.593480, 0)
_create_s2f_geci_fn("XCaMP-Gf", 13.93, 99.38, 312.85, 0.20, 2.363793, 3.936075, 0.471668, -0.319370, 0)
_create_s2f_geci_fn("GCaMP6s", 50.81, 1702.21, 0.00, 0.00, 3.334000, 3.142000, 1.332000, -0.049982, 0)
_create_s2f_geci_fn("GCaMP6s-TG", 133.01, 1262.78, 0.00, 0.00, 3.596000, 3.303000, 2.897000, -0.000251, 0)
_create_s2f_geci_fn("GCaMP6f", 9.98, 682.58, 0.00, 0.00, 1.905000, 3.197000, 1.410000, -0.020769, 0)
_create_s2f_geci_fn("GCaMP6f-TG", 20.82, 629.74, 0.00, 0.00, 2.818000, 5.821000, 1.046000, -0.006377, 0)
