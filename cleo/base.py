"""Contains definitions for essential, base classes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable
import datetime

from attrs import define, field
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.artist import Artist
import neo

import cleo.utilities


class NeoExportable(ABC):
    """Mixin class for classes that can be exported to Neo objects"""

    @abstractmethod
    def to_neo(self) -> neo.core.BaseNeo:
        """Return a neo.core.AnalogSignal object with the device's data

        Returns
        -------
        neo.core.BaseNeo
            Neo object representing exported data
        """
        pass


@define(eq=False)
class InterfaceDevice(ABC):
    """Base class for devices to be injected into the network"""

    brian_objects: set = field(factory=set, init=False)
    """All the Brian objects added to the network by this device.
    Must be kept up-to-date in :meth:`connect_to_neuron_group` and
    other functions so that those objects can be automatically added
    to the network when the device is injected.
    """
    sim: CLSimulator = field(init=False, default=None)
    """The simulator the device is injected into """

    name: str = field(kw_only=True)
    """Unique identifier for device, used in sampling, plotting, etc.
    Name of the class by default."""

    @name.default
    def _default_name(self) -> str:
        return self.__class__.__name__

    def init_for_simulator(self, simulator: CLSimulator) -> None:
        """Initialize device for simulator on initial injection

        This function is called only the first time a device is
        injected into a simulator and performs any operations
        that are independent of the individual neuron groups it
        is connected to.

        Parameters
        ----------
        simulator : CLSimulator
            simulator being injected into
        """
        pass

    def reset(self, **kwargs) -> None:
        """Reset the device to a neutral state"""
        pass

    @abstractmethod
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Connect device to given `neuron_group`.

        If your device introduces any objects which Brian must
        keep track of, such as a NeuronGroup, Synapses, or Monitor,
        make sure to add these to `self.brian_objects`.

        Parameters
        ----------
        neuron_group : NeuronGroup
        **kwparams : optional, passed from `inject` or
            `inject`
        """
        pass

    def add_self_to_plot(
        self, ax: Axes3D, axis_scale_unit: Unit, **kwargs
    ) -> list[Artist]:
        """Add device to an existing plot

        Should only be called by :func:`~cleo.viz.plot`.

        Parameters
        ----------
        ax : Axes3D
            The existing matplotlib Axes object
        axis_scale_unit : Unit
            The unit used to label axes and define chart limits
        **kwargs : optional

        Returns
        -------
        list[Artist]
            A list of artists used to render the device. Needed for use
            in conjunction with :class:`~cleo.viz.VideoVisualizer`.
        """
        return []

    def update_artists(self, artists: list[Artist], *args, **kwargs) -> list[Artist]:
        """Update the artists used to render the device

        Used to set the artists' state at every frame of a video visualization.
        The current state would be passed in `*args` or `**kwargs`

        Parameters
        ----------
        artists : list[Artist]
            the artists used to render the device originally, i.e.,
            which were returned from the first :meth:`add_self_to_plot` call.

        Returns
        -------
        list[Artist]
            The artists that were actually updated. Needed for efficient
            blit rendering, where only updated artists are re-rendered.
        """
        return []


class IOProcessor(ABC):
    """Abstract class for implementing sampling, signal processing and control

    This must be implemented by the user with their desired closed-loop
    use case, though most users will find the :func:`~processing.LatencyIOProcessor`
    class more useful, since delay handling is already defined.
    """

    sample_period_ms: float
    """Determines how frequently the processor takes samples"""

    @abstractmethod
    def is_sampling_now(self, time) -> bool:
        """Determines whether the processor will take a sample at this timestep.

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
            :func:`~cleo.CLSimulator.get_state()`
        time : brian2 temporal Unit
            The current simulation timestep. Essential for simulating
            control latency and for time-varying control.
        """
        pass

    @abstractmethod
    def get_ctrl_signal(self, time) -> dict:
        """Get per-stimulator control signal from the :class:`~cleo.IOProcessor`.

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


@define(eq=False)
class Recorder(InterfaceDevice):
    """Device for taking measurements of the network."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return current measurement."""
        pass


@define(eq=False)
class Stimulator(InterfaceDevice, NeoExportable):
    """Device for manipulating the network"""

    value: Any = field(init=False, default=None)
    """The current value of the stimulator device"""
    default_value: Any = 0
    """The default value of the device---used on initialization and on :meth:`~reset`"""
    t_ms: list[float] = field(factory=list, init=False, repr=False)
    """Times stimulator was updated, stored if :attr:`save_history`"""
    values: list[Any] = field(factory=list, init=False, repr=False)
    """Values taken by the stimulator at each :meth:`~update` call, 
    stored if :attr:`save_history`"""
    save_history: bool = True
    """Determines whether :attr:`t_ms` and :attr:`values` are recorded"""

    def __attrs_post_init__(self):
        self.value = self.default_value

    def _init_saved_vars(self):
        if self.save_history:
            self.t_ms = []
            self.values = []

    def update(self, ctrl_signal) -> None:
        """Set the stimulator value.

        By default this simply sets `value` to `ctrl_signal`.
        You will want to implement this method if your
        stimulator requires additional logic. Use super.update(self, value)
        to preserve the self.value attribute logic

        Parameters
        ----------
        ctrl_signal : any
            The value the stimulator is to take.
        """
        self.value = ctrl_signal
        if self.save_history:
            self.t_ms.append(self.sim.network.t / ms)
            self.values.append(self.value)

    def reset(self, **kwargs) -> None:
        """Reset the stimulator device to a neutral state"""
        self.value = self.default_value
        self._init_saved_vars()

    def to_neo(self):
        signal = cleo.utilities.analog_signal(self.t_ms, self.values, "dimensionless")
        signal.name = self.name
        signal.description = "Exported from Cleo stimulator device"
        signal.annotate(export_datetime=datetime.datetime.now())
        return signal


@define(eq=False)
class CLSimulator(NeoExportable):
    """The centerpiece of cleo. Integrates simulation components and runs."""

    network: Network = field(repr=False)
    """The Brian network forming the core model"""
    io_processor: IOProcessor = field(default=None, init=False)
    recorders: dict[str, Recorder] = field(factory=dict, init=False, repr=False)
    stimulators: dict[str, Stimulator] = field(factory=dict, init=False, repr=False)
    devices: set[InterfaceDevice] = field(factory=set, init=False)
    _processing_net_op: NetworkOperation = field(default=None, init=False, repr=False)
    _net_store_name: str = field(default="cleo default", init=False, repr=False)

    def inject(
        self, device: InterfaceDevice, *neuron_groups: NeuronGroup, **kwparams: Any
    ) -> CLSimulator:
        """Inject InterfaceDevice into the network, connecting to specified neurons.

        Calls :meth:`~InterfaceDevice.connect_to_neuron_group` for each group with
        kwparams and adds the device's :attr:`~InterfaceDevice.brian_objects`
        to the simulator's :attr:`network`.

        Parameters
        ----------
        device : InterfaceDevice
            Device to inject

        Returns
        -------
        CLSimulator
            self
        """
        if len(neuron_groups) == 0:
            raise Exception("Injecting stimulator for no neuron groups is meaningless.")
        for ng in neuron_groups:
            if type(ng) == NeuronGroup:
                if ng not in self.network.objects:
                    raise Exception(
                        f"Attempted to connect device {device.name} to neuron group "
                        f"{ng.name}, which is not part of the simulator's network."
                    )
            elif type(ng) == Subgroup:
                # must look at sorted_objects because ng.source is unhashable
                if ng.source not in self.network.sorted_objects:
                    raise Exception(
                        f"Attempted to connect device {device.name} to neuron group "
                        f"{ng.source.name}, which is not part of the simulator's network."
                    )
            if device.sim not in [None, self]:
                raise Exception(
                    f"Attempted to inject device {device.name} into {self}, "
                    f"but it was previously injected into {device.sim}. "
                    "Each device can only be injected into one CLSimulator."
                )
            if device.sim is None:
                device.sim = self
                device.init_for_simulator(self)
            device.connect_to_neuron_group(ng, **kwparams)
        for brian_object in device.brian_objects:
            if brian_object not in self.network.objects:
                self.network.add(brian_object)
        self.network.store(self._net_store_name)
        if isinstance(device, Recorder):
            self.recorders[device.name] = device
        if isinstance(device, Stimulator):
            self.stimulators[device.name] = device
        self.devices.add(device)
        return self

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

    def set_io_processor(self, io_processor, communication_period=None) -> CLSimulator:
        """Set simulator IO processor

        Will replace any previous IOProcessor so there is only one at a time.
        A Brian NetworkOperation is created to govern communication between
        the Network and the IOProcessor.

        Parameters
        ----------
        io_processor : IOProcessor

        Returns
        -------
        CLSimulator
            self
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
        return self

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
        Calls reset() on devices and IOProcessor.
        """
        # kwargs passed to stimulators, recorders, and io_processor reset
        self.network.restore(self._net_store_name)
        for device in self.devices:
            device.reset(**kwargs)
        if self.io_processor is not None:
            self.io_processor.reset(**kwargs)

    def to_neo(self) -> neo.core.Block:
        block = neo.Block(
            description="Exported from Cleo simulation",
        )
        block.annotate(export_datetime=datetime.datetime.now())
        seg = neo.Segment()
        block.segments.append(seg)
        for device in self.devices:
            if not isinstance(device, NeoExportable):
                continue
            dev_neo = device.to_neo()
            if isinstance(dev_neo, neo.core.Group):
                data_objects = dev_neo.data_children_recur
                block.groups.append(dev_neo)
            elif isinstance(dev_neo, neo.core.dataobject.DataObject):
                data_objects = [dev_neo]
            cleo.utilities.add_to_neo_segment(seg, *data_objects)
        return block
