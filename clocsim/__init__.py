"""Contains class definitions for essential, base classes."""

from abc import ABC, abstractmethod

from brian2 import NeuronGroup, Network, NetworkOperation, defaultclock
from numpy import rec


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

class InterfaceDevice(ABC):
    """Base class for devices to be injected into the network."""

    def __init__(self, name):
        self.name = name
        self.brian_objects = set()

    @abstractmethod
    def connect_to_neuron_group(self, neuron_group: NeuronGroup):
        """Connect device to given `neuron_group`.

        If your device introduces any objects which Brian must
        keep track of, such as a NeuronGroup, Synapses, or Monitor,
        make sure to add these to `self.brian_objects`.

        Parameters
        ----------
        neuron_group : NeuronGroup
        """
        pass


class ControlLoop(ABC):
    """Abstract class for implementing signal processing and control.

    This must be implemented by the user with their desired closed-loop
    use case, though most users will find the :func:`~control_loop:DelayControlLoop`
    class more useful, since delay handling is already defined.
    """

    @abstractmethod
    def is_sampling_now(self, time) -> bool:
        """Whether the `ControlLoop` will take a sample at this timestep.

        Parameters
        ----------
        time : Brian 2 temporal Unit
            Current timestep.

        Returns
        -------
        bool
        """
        pass

    @abstractmethod
    def put_state(self, state_dict: dict, time):
        """Deliver network state to the control loop.

        Parameters
        ----------
        state_dict : dict
            A dictionary of recorder measurements, as returned by
            :func:`~base.CLOCSimulator.get_state()`
        time : brian2 temporal Unit
            The current simulation timestep. Essential for simulating
            control latency and for time-varying control.
        """
        pass

    # The output should be a dictionary of {stimulator_name: value, ...}
    @abstractmethod
    def get_ctrl_signal(self, time) -> dict:
        """Get per-stimulator control signal from the control loop.

        Parameters
        ----------
        time : Brian 2 temporal Unit
            Current timestep

        Returns
        -------
        dict
            A {`stimulator_name`: `value`} dictionary for updating stimulators.
        """
        pass


class Recorder(InterfaceDevice):
    """Device for taking measurements of the network."""

    @abstractmethod
    def get_state(self):
        """Return current measurement."""
        pass


class Stimulator(InterfaceDevice):
    """Device for manipulating the network."""

    def __init__(self, name, start_value):
        super().__init__(name)
        self.value = start_value

    def update(self, ctrl_signal):
        """Set the stimulator value.

        By default this simply sets `value` to `ctrl_signal`.
        You will want to implement this method if your
        stimulator requires additional logic.

        Parameters
        ----------
        ctrl_signal : any
            The value the stimulator is to take.
        """
        self.value = ctrl_signal


class CLOCSimulator:
    """Integrates simulation components and runs."""

    def __init__(self, brian_network: Network):
        self.network = brian_network
        self.stimulators = {}
        self.recorders = {}
        self.controller = None

    def _inject_device(self, device: InterfaceDevice, *neuron_groups):
        if len(neuron_groups) == 0:
            raise Exception('Injecting stimulator for no neuron groups '
                            'is meaningless.')
        for ng in neuron_groups:
            device.connect_to_neuron_group(ng)
        for brian_object in device.brian_objects:
            self.network.add(brian_object)


    def inject_stimulator(self, stimulator: Stimulator, *neuron_groups):
        """Inject stimulator into given neuron groups.

        `connect_to_neuron_group(group)` is called for each `group`.
        
        Parameters
        ----------
        stimulator : Stimulator
        """
        self._inject_device(stimulator, *neuron_groups)
        self.stimulators[stimulator.name] = stimulator

    def inject_recorder(self, recorder: Recorder, *neuron_groups):
        """Inject recorder into given neuron groups.

        `connect_to_neuron_group(group)` is called for each `group`.

        Parameters
        ----------
        recorder : Recorder
        """
        self._inject_device(recorder, *neuron_groups)
        self.recorders[recorder.name] = recorder

    def get_state(self):
        """Return current recorder measurements.

        Returns
        -------
        dict
            A dictionary of `name`: `state` pairs for
            all recorders in the simulator.
        """
        state = {}
        for name, recorder in self.recorders.items():
            state[name] = recorder.get_state()
        return state

    def update_stimulators(self, ctrl_signals):
        """Update stimulators with output from control loop.

        Parameters
        ----------
        ctrl_signals : dict
            {`stimulator_name`: `ctrl_signal`} dictionary with values
            to update each stimulator.
        """
        if ctrl_signals is None:
            return
        for name, signal in ctrl_signals.items():
            self.stimulators[name].update(signal)

    def set_control_loop(self, control_loop, communication_period=None):
        """Set simulator control loop.

        Parameters
        ----------
        control_loop : ControlLoop
        """

        def communicate_with_ctrl_loop(t):
            if control_loop.is_sampling_now(t):
                control_loop.put_state(self.get_state(), t)
            ctrl_signal = control_loop.get_ctrl_signal(t)
            self.update_stimulators(ctrl_signal)
        # communication should be at every timestep. The ControlLoop
        # decides when to sample and deliver results.
        self.network.add(NetworkOperation(communicate_with_ctrl_loop, dt=defaultclock.dt))

    def run(self, duration):
        """Run simulation.

        Parameters
        ----------
        duration : brian2 temporal Unit
            Length of simulation
        """
        self.network.run(duration)
