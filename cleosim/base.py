"""Contains definitions for essential, base classes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from brian2 import (
    NeuronGroup,
    Subgroup,
    Network,
    NetworkOperation,
    defaultclock,
    ms,
    Unit,
    Quantity,
)
from matplotlib import pyplot as plt


class InterfaceDevice(ABC):
    """Base class for devices to be injected into the network"""

    name: str
    """Unique identifier for device.
    Used as a key in output/input dicts
    """
    brian_objects: set
    """All the Brian objects added to the network by this device.
    Must be kept up-to-date in :meth:`connect_to_neuron_group` and
    other functions so that those objects can be automatically added
    to the network when the device is injected.
    """
    sim: CLSimulator
    """The simulator the device is injected into
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name : str
            Unique identifier for the device.
        """
        self.name = name
        self.brian_objects = set()

    @abstractmethod
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Connect device to given `neuron_group`.

        If your device introduces any objects which Brian must
        keep track of, such as a NeuronGroup, Synapses, or Monitor,
        make sure to add these to `self.brian_objects`.

        Parameters
        ----------
        neuron_group : NeuronGroup
        **kwparams : optional, passed from `inject_recorder` or
            `inject_stimulator`
        """
        pass

    def add_self_to_plot(self, ax: plt.Axes, axis_scale_unit: Unit) -> None:
        """Add device to an existing plot

        Should only be called by :func:`~cleosim.coordinates.plot_neuron_positions`.

        Parameters
        ----------
        ax : plt.Axes
            The existing matplotlib Axes object
        axis_scale_unit : Unit
            The unit used to label axes and define chart limits
        """
        pass


class IOProcessor(ABC):
    """Abstract class for implementing sampling, signal processing and control

    This must be implemented by the user with their desired closed-loop
    use case, though most users will find the :func:`~processing.LatencyIOProcessor`
    class more useful, since delay handling is already defined.
    """

    sample_period_ms: float

    @abstractmethod
    def is_sampling_now(self, time) -> bool:
        """Whether the `IOProcessor` will take a sample at this timestep.

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
    def put_state(self, state_dict: dict, time) -> None:
        """Deliver network state to the :class:`IOProcessor`.

        Parameters
        ----------
        state_dict : dict
            A dictionary of recorder measurements, as returned by
            :func:`~cleosim.CLSimulator.get_state()`
        time : brian2 temporal Unit
            The current simulation timestep. Essential for simulating
            control latency and for time-varying control.
        """
        pass

    @abstractmethod
    def get_ctrl_signal(self, time) -> dict:
        """Get per-stimulator control signal from the :class:`IOProcessor`.

        Parameters
        ----------
        time : Brian 2 temporal Unit
            Current timestep

        Returns
        -------
        dict
            A {'stimulator_name': value} dictionary for updating stimulators.
        """
        pass

    def reset(self, **kwargs) -> None:
        pass


class Recorder(InterfaceDevice):
    """Device for taking measurements of the network."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return current measurement."""
        pass

    def reset(self, **kwargs) -> None:
        """Reset the recording device to a neutral state"""
        pass


class Stimulator(InterfaceDevice):
    """Device for manipulating the network"""

    def __init__(self, name: str, start_value) -> None:
        """
        Parameters
        ----------
        name : str
            Unique device name used in :meth:`CLSimulator.update_stimulators`
        start_value : any
            The stimulator's default value
        """
        super().__init__(name)
        self.value = start_value

    def update(self, ctrl_signal) -> None:
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

    def reset(self, **kwargs) -> None:
        """Reset the stimulator device to a neutral state"""
        pass


class CLSimulator:
    """The centerpiece of cleosim. Integrates simulation components and runs."""

    io_processor: IOProcessor
    network: Network
    recorders = "set[Recorder]"
    stimulators = "set[Stimulator]"
    _processing_net_op: NetworkOperation
    _net_store_name: str = "cleosim default"

    def __init__(self, brian_network: Network) -> None:
        """
        Parameters
        ----------
        brian_network : Network
            The Brian network forming the core model
        """
        self.network = brian_network
        self.stimulators = {}
        self.recorders = {}
        self.io_processor = None
        self._processing_net_op = None

    def _inject_device(
        self, device: InterfaceDevice, *neuron_groups, **kwparams
    ) -> None:
        if len(neuron_groups) == 0:
            raise Exception("Injecting stimulator for no neuron groups is meaningless.")
        for ng in neuron_groups:
            if type(ng) == NeuronGroup:
                if ng not in self.network.objects:
                    raise Exception(
                        f"Attempted to connect device {device.name} to neuron group {ng.name}, which is not part of the simulator's network."
                    )
            elif type(ng) == Subgroup:
                # must look at sorted_objects because ng.source is unhashable
                if ng.source not in self.network.sorted_objects:
                    raise Exception(
                        f"Attempted to connect device {device.name} to neuron group {ng.source.name}, which is not part of the simulator's network."
                    )
            device.sim = self
            device.connect_to_neuron_group(ng, **kwparams)
        for brian_object in device.brian_objects:
            self.network.add(brian_object)
        self.network.store(self._net_store_name)

    def inject_stimulator(
        self, stimulator: Stimulator, *neuron_groups: NeuronGroup, **kwparams
    ) -> None:
        """Inject stimulator into given neuron groups.

        :meth:`Stimulator.connect_to_neuron_group` is called for each `group`.

        Parameters
        ----------
        stimulator : Stimulator
            The stimulator to inject
        *neuron_groups : NeuronGroup
            The groups to inject the stimulator into
        **kwparams : any
            Passed on to :meth:`Stimulator.connect_to_neuron_group` function.
            Necessary for parameters that can vary by neuron group, such
            as opsin expression levels.
        """
        self._inject_device(stimulator, *neuron_groups, **kwparams)
        self.stimulators[stimulator.name] = stimulator

    def inject_recorder(
        self, recorder: Recorder, *neuron_groups: NeuronGroup, **kwparams
    ) -> None:
        """Inject recorder into given neuron groups.

        :meth:`Recorder.connect_to_neuron_group` is called for each `group`.

        Parameters
        ----------
        recorder : Recorder
            The recorder to inject into the simulation
        *neuron_groups : NeuronGroup
            The groups to inject the recorder into
        **kwparams : any
            Passed on to :meth:`Recorder.connect_to_neuron_group` function.
            Necessary for parameters that can vary by neuron group, such
            as inhibitory/excitatory cell type
        """
        self._inject_device(recorder, *neuron_groups, **kwparams)
        self.recorders[recorder.name] = recorder

    def get_state(self) -> dict:
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

    def update_stimulators(self, ctrl_signals) -> None:
        """Update stimulators with output from the :class:`IOProcessor`

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

    def set_io_processor(self, io_processor, communication_period=None) -> None:
        """Set simulator IO processor

        Will replace any previous IOProcessor so there is only one at a time.
        A Brian NetworkOperation is created to govern communication between
        the Network and the IOProcessor.

        Parameters
        ----------
        io_processor : IOProcessor
        """
        self.io_processor = io_processor
        # remove previous NetworkOperation
        if self._processing_net_op is not None:
            self.network.remove(self._processing_net_op)
            self._processing_net_op = None

        if io_processor is None:
            return

        def communicate_with_io_proc(t):
            if io_processor.is_sampling_now(t / ms):
                io_processor.put_state(self.get_state(), t / ms)
            ctrl_signal = io_processor.get_ctrl_signal(t / ms)
            self.update_stimulators(ctrl_signal)

        # communication should be at every timestep. The IOProcessor
        # decides when to sample and deliver results.
        if communication_period is None:
            communication_period = defaultclock.dt
        self._processing_net_op = NetworkOperation(
            communicate_with_io_proc, dt=communication_period
        )
        self.network.add(self._processing_net_op)
        self.network.store(self._net_store_name)

    def run(self, duration: Quantity, **kwparams) -> None:
        """Run simulation.

        Parameters
        ----------
        duration : brian2 temporal Quantity
            Length of simulation
        **kwparams : additional arguments passed to brian2.run()
            level has a default value of 1
        """
        level = kwparams.get("level", 1)
        kwparams["level"] = level
        self.network.run(duration, **kwparams)

    def reset(self, **kwargs):
        """Reset the simulator to a neutral state

        Restores the Brian Network to where it was when the
        CLSimulator was last modified (last injection, IOProcessor change).
        Calls reset() on stimulators, recorders, and IOProcessor.
        """
        # kwargs passed to stimulators, recorders, and io_processor reset
        self.network.restore(self._net_store_name)
        for stim in self.stimulators.values():
            stim.reset(**kwargs)
        for rec in self.recorders.values():
            rec.reset(**kwargs)
        if self.io_processor is not None:
            self.io_processor.reset(**kwargs)
