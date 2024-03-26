"""Contains definitions for essential, base classes."""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Any, Tuple

import neo
from attrs import asdict, define, field, fields_dict
from brian2 import (
    BrianObjectException,
    Equations,
    Network,
    NetworkOperation,
    NeuronGroup,
    Quantity,
    Subgroup,
    Synapses,
    Unit,
    defaultclock,
    ms,
    np,
)
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D

from cleo.registry import registry_for_sim
from cleo.utilities import add_to_neo_segment, analog_signal, brian_safe_name


class NeoExportable(ABC):
    """Mixin class for classes that can be exported to Neo objects"""

    @abstractmethod
    def to_neo(self) -> neo.core.BaseNeo:
        """Return a Neo signal object with the device's data

        Returns
        -------
        neo.core.BaseNeo
            Neo object representing exported data
        """
        pass


@define(eq=False)
class InterfaceDevice(ABC):
    """Base class for devices to be injected into the network"""

    brian_objects: set = field(factory=set, init=False, repr=False)
    """All the Brian objects added to the network by this device.
    Must be kept up-to-date in :meth:`connect_to_neuron_group` and
    other functions so that those objects can be automatically added
    to the network when the device is injected.
    """
    sim: CLSimulator = field(init=False, default=None, repr=False)
    """The simulator the device is injected into """
    name: str = field(kw_only=True)
    """Identifier for device, used in sampling, plotting, etc.
    Name of the class by default. Must be unique among recorders and stimulators"""
    save_history: bool = field(default=True, kw_only=True)
    """Determines whether times and inputs/outputs are recorded.
    True by default.
    
    For stimulators, this is when :meth:`~Stimulator.update` is called.
    For recorders, it is when :meth:`~Recorder.get_state` is called."""

    @name.default
    def _default_name(self) -> str:
        return self.__class__.__name__

    def init_for_simulator(self, simulator: CLSimulator) -> None:
        """Initialize device for simulator on initial injection.

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
        make sure to add these to :attr:`~cleo.InterfaceDevice.brian_objects`.

        Parameters
        ----------
        neuron_group : NeuronGroup
        **kwparams : optional
            Passed from `inject`
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


@define
class IOProcessor(ABC):
    """Abstract class for implementing sampling, signal processing and control

    This must be implemented by the user with their desired closed-loop
    use case, though most users will find the :func:`~processing.LatencyIOProcessor`
    class more useful, since delay handling is already defined.
    """

    sample_period_ms: float = 1
    """Determines how frequently the processor takes samples"""

    latest_ctrl_signal: dict = field(factory=dict, init=False, repr=False)
    """The most recent control signal returned by :meth:`get_ctrl_signals`"""

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
    def put_state(self, state_dict: dict, sample_time_ms: float) -> None:
        """Deliver network state to the :class:`IOProcessor`.

        Parameters
        ----------
        state_dict : dict
            A dictionary of recorder measurements, as returned by
            :func:`~cleo.CLSimulator.get_state()`
        sample_time_ms: float
            The current simulation timestep. Essential for simulating
            control latency and for time-varying control.
        """
        pass

    @abstractmethod
    def get_ctrl_signals(self, query_time_ms: float) -> dict:
        """Get per-stimulator control signal from the :class:`~cleo.IOProcessor`.

        Parameters
        ----------
        query_time_ms : float
            Current simulation time.

        Returns
        -------
        dict
            A {'stimulator_name': ctrl_signal} dictionary for updating stimulators.
        """
        pass

    def get_stim_values(self, query_time_ms: float) -> dict:
        ctrl_signals = self.get_ctrl_signals(query_time_ms)
        self.latest_ctrl_signal.update(ctrl_signals)
        stim_value_conversions = self.preprocess_ctrl_signals(
            self.latest_ctrl_signal, query_time_ms
        )
        return ctrl_signals | stim_value_conversions

    def preprocess_ctrl_signals(
        self, latest_ctrl_signals: dict, query_time_ms: float
    ) -> dict:
        """Preprocess control signals as needed to control stimulator waveforms between samples.

        I.e., if a control signal defines the frequency of a periodic light stimulus, this
        function computes the current intensity given the latest frequency and the current
        time. This is called immediately after :meth:`get_ctrl_signals` and on every timestep
        to update the stimulator waveform between samples.

        This only needs to be implemented when a stimulus that varies between samples is desired.
        Otherwise, the control signal returned by :meth:`get_ctrl_signals` is used directly.
        If not all stimulators need this functionality, only return a dict for those that do.
        The original, unprocessed control signal is used for the others.

        Parameters
        ----------
        query_time_ms : float
            Current simulation time.

        Returns
        -------
        dict
            A {'stimulator_name': value} dictionary for updating stimulators.
        """
        return {}

    def get_intersample_ctrl_signal(self, query_time_ms: float) -> dict:
        """Get per-stimulator control signal between samples. I.e., for implementing
        a time-varying waveform based on parameters from the last sample.
        Such parameters will need to be stored in the :class:`~cleo.IOProcessor`."""
        return {}

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
    """Times stimulator was updated, stored if :attr:`~cleo.InterfaceDevice.save_history`"""
    values: list[Any] = field(factory=list, init=False, repr=False)
    """Values taken by the stimulator at each :meth:`~update` call, 
    stored if :attr:`~cleo.InterfaceDevice.save_history`"""

    def __attrs_post_init__(self):
        self.value = self.default_value
        self._init_saved_vars()

    def _init_saved_vars(self):
        if self.save_history:
            if self.sim:
                t0 = self.sim.network.t / ms
            else:
                t0 = 0
            self.t_ms = [t0]
            self.values = [self.value]

    def update(self, ctrl_signal) -> None:
        """Set the stimulator value.

        By default this sets :attr:`value` to ``ctrl_signal`` and updates saved
        times and values. You will want to implement this method if your stimulator
        requires additional logic. Use ``super.update(self, value)``
        to preserve the ``self.value`` and ``save_history`` logic

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
        signal = analog_signal(self.t_ms, self.values, "dimensionless")
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
            if (
                device.name in self.recorders
                and device is not self.recorders[device.name]
            ):
                raise ValueError(
                    f"Another Recorder with name {device.name} has already been injected"
                )
            self.recorders[device.name] = device
        if isinstance(device, Stimulator):
            if (
                device.name in self.stimulators
                and device is not self.stimulators[device.name]
            ):
                raise ValueError(
                    f"Another Stimulator with name {device.name} has already been injected"
                )
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

    def update_stimulators(self, stim_values: dict[str, Any]) -> None:
        """Update stimulators with output from the :class:`IOProcessor`

        Parameters
        ----------
        stim_values : dict
            {`stimulator_name`: `stim_value`} dictionary with values
            to update each stimulator.
        """
        for name, value in stim_values.items():
            self.stimulators[name].update(value)

    def set_io_processor(
        self, io_processor: IOProcessor, communication_period=None
    ) -> CLSimulator:
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
            # assuming no one will have timesteps shorter than nanoseconds...
            now_ms = round(t / ms, 6)
            if io_processor.is_sampling_now(now_ms):
                io_processor.put_state(self.get_state(), now_ms)
            stim_values = io_processor.get_stim_values(now_ms)
            self.update_stimulators(stim_values)

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
        Calls reset() on :attr:`devices` and :attr:`io_processor`.
        """
        # kwargs passed to stimulators, recorders, and io_processor reset
        self.network.restore(self._net_store_name)
        for device in self.devices:
            device.reset(**kwargs)
        if self.io_processor is not None:
            self.io_processor.reset(**kwargs)

    def to_neo(self) -> neo.core.Block:
        """Exports simulator data to a Neo Block

        Returns
        -------
        neo.core.Block
            Neo Block containing signals representing each device's data
        """
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
            add_to_neo_segment(seg, *data_objects)
        return block


@define(eq=False)
class SynapseDevice(InterfaceDevice):
    """Base class for devices that record from/stimulate neurons via
    a Synapses object with device-specific model. Used for opsin and
    indicator classes"""

    model: str = field(init=False)
    """Basic Brian model equations string.
    
    Should contain a `rho_rel` term reflecting relative expression 
    levels. Will likely also contain special NeuronGroup-dependent
    symbols such as V_VAR_NAME to be replaced on injection in 
    :meth:`modify_model_and_params_for_ng`."""

    on_pre: str = field(init=False, default="")
    """Model string for :class:`brian2.synapses.synapses.Synapses` reacting to spikes."""

    synapses: dict[str, Synapses] = field(factory=dict, init=False, repr=False)
    """Stores the synapse objects implementing the model, connecting from source
    (light aggregator neurons or the target group itself) to target neuron groups,
    of form ``{target_ng.name: synapses}``."""

    source_ngs: dict[str, NeuronGroup] = field(factory=dict, init=False, repr=False)
    """``{target_ng.name: source_ng}`` dict of source neuron groups.
    
    The source is the target itself by default or light aggregator neurons for
    :class:`~cleo.light.LightDependent`."""

    per_ng_unit_replacements: list[Tuple[str, str]] = field(
        factory=list, init=False, repr=False
    )
    """List of (UNIT_NAME, neuron_group_specific_unit_name) tuples to be substituted
    in the model string on injection and before checking required variables."""

    required_vars: list[Tuple[str, Unit]] = field(factory=list, init=False, repr=False)
    """Default names of state variables required in the neuron group,
    along with units, e.g., [('Iopto', amp)].
    
    It is assumed that non-default values can be passed in on injection
    as a keyword argument ``[default_name]_var_name=[non_default_name]``
    and that these are found in the model string as 
    ``[DEFAULT_NAME]_VAR_NAME`` before replacement."""

    extra_namespace: dict = field(factory=dict, repr=False)
    """Additional items (beyond parameters) to be added to the opto synapse namespace"""

    def _get_source_for_synapse(
        self, target_ng: NeuronGroup, i_targets: list[int]
    ) -> Tuple[NeuronGroup, list[int]]:
        """Get the source neuron group and indices of source neurons.

        Parameters
        ----------
        ng : NeuronGroup
            The target neuron group.
        i_targets : list[int]
            The indices of the target neurons in the target neuron group.

        Returns
        -------
        Tuple[NeuronGroup, list[int]]
            A tuple containing the source neuron group and indices to use in Synapses
        """
        # by default the source is the target group itself
        return target_ng, i_targets

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        """Transfect neuron group with device.

        Parameters
        ----------
        neuron_group : NeuronGroup
            The neuron group to transform

        Keyword args
        ------------
        p_expression : float
            Probability (0 <= p <= 1) that a given neuron in the group
            will express the protein. 1 by default.
        i_targets : array-like
            Indices of neurons in the group to transfect. recommended for efficiency
            when stimulating or imaging a small subset of the group.
            Incompatible with ``p_expression``.
        rho_rel : float
            The expression level, relative to the standard model fit,
            of the protein. 1 by default. For heterogeneous expression,
            this would have to be modified in the light-dependent synapse
            post-injection, e.g., ``opsin.syns["neuron_group_name"].rho_rel = ...``
        [default_name]_var_name : str
            See :attr:`~required_vars`. Allows for custom variable names.
        """
        if neuron_group.name in self.source_ngs:
            assert neuron_group.name in self.synapses
            raise ValueError(
                f"{self.__class__.__name__} {self.name} already connected to neuron group"
                f" {neuron_group.name}"
            )

        # get modified synapse model string (i.e., with names/units specified)
        mod_syn_model, mod_syn_params = self.modify_model_and_params_for_ng(
            neuron_group, kwparams
        )

        # handle p_expression
        if "p_expression" in kwparams:
            if "i_targets" in kwparams:
                raise ValueError("p_expression and i_targets are incompatible")
            p_expression = kwparams.get("p_expression", 1)
            expr_bool = np.random.rand(neuron_group.N) < p_expression
            i_targets = np.where(expr_bool)[0]
        elif "i_targets" in kwparams:
            i_targets = kwparams["i_targets"]
        else:
            i_targets = list(range(neuron_group.N))
        if len(i_targets) == 0:
            return

        source_ng, i_sources = self._get_source_for_synapse(neuron_group, i_targets)

        syn = Synapses(
            source_ng,
            neuron_group,
            model=mod_syn_model,
            on_pre=self.on_pre,
            namespace=mod_syn_params,
            name=f"syn_{brian_safe_name(self.name)}_{neuron_group.name}",
        )
        syn.namespace.update(self.extra_namespace)
        syn.connect(i=i_sources, j=i_targets)
        self.init_syn_vars(syn)
        # relative protein density
        syn.rho_rel = kwparams.get("rho_rel", 1)

        # store at the end, after all checks have passed
        self.source_ngs[neuron_group.name] = source_ng
        self.brian_objects.add(source_ng)
        self.synapses[neuron_group.name] = syn
        self.brian_objects.add(syn)

        registry = registry_for_sim(self.sim)
        registry.register(self, neuron_group)

    def modify_model_and_params_for_ng(
        self, neuron_group: NeuronGroup, injct_params: dict
    ) -> Tuple[Equations, dict]:
        """Adapt model for given neuron group on injection

        This enables the specification of variable names
        differently for each neuron group, allowing for custom names
        and avoiding conflicts.

        Parameters
        ----------
        neuron_group : NeuronGroup
            NeuronGroup this opsin model is being connected to
        injct_params : dict
            kwargs passed in on injection, could contain variable
            names to plug into the model

        Keyword Args
        ------------
        model : str, optional
            Model to start with, by default that defined for the class.
            This allows for prior string manipulations before it can be
            parsed as an `Equations` object.

        Returns
        -------
        Equations, dict
            A tuple containing an Equations object
            and a parameter dictionary, constructed from :attr:`~model`
            and :attr:`~params`, respectively, with modified names for use
            in :attr:`~cleo.opto.OptogeneticIntervention.synapses`
        """
        model = self.model

        # perform unit substitutions
        for unit_name, neuron_group_unit_name in self.per_ng_unit_replacements:
            model = model.replace(unit_name, neuron_group_unit_name)

        # check required variables/units and replace placeholder names
        for default_name, unit in self.required_vars:
            var_name = injct_params.get(f"{default_name}_var_name", default_name)
            if var_name not in neuron_group.variables or not neuron_group.variables[
                var_name
            ].unit.has_same_dimensions(unit):
                raise BrianObjectException(
                    (
                        f"{var_name} : {unit.name} needed in the model of NeuronGroup "
                        f"{neuron_group.name} to connect Opsin {self.name}."
                    ),
                    neuron_group,
                )
            # opsin synapse model needs modified names
            to_replace = f"{default_name}_var_name".upper()
            model = model.replace(to_replace, var_name)

        # Synapse variable and parameter names cannot be the same as any
        # neuron group variable name
        return self._fix_name_conflicts(model, neuron_group)

    @property
    def params(self) -> dict:
        """Returns a dictionary of parameters for the model"""
        params = asdict(self, recurse=False)
        # remove generic fields that are not parameters
        # assume we only want fields in the last class in the
        # hierarchy
        parent_class = type(self).__mro__[1]
        for field in fields_dict(parent_class):
            params.pop(field)
        # remove private attributes
        for key in list(params.keys()):
            if key.startswith("_"):
                params.pop(key)
        return params

    def _fix_name_conflicts(
        self, modified_model: str, neuron_group: NeuronGroup
    ) -> Tuple[Equations, dict]:
        modified_params = self.params.copy()
        rename = lambda x: f"{x}_syn"

        # get variables to rename
        opsin_eqs = Equations(modified_model)
        substitutions = {}
        for var in opsin_eqs.names:
            if var in neuron_group.variables:
                substitutions[var] = rename(var)

        # and parameters
        for param in self.params.keys():
            if param in neuron_group.variables:
                substitutions[param] = rename(param)
                modified_params[rename(param)] = modified_params[param]
                del modified_params[param]

        mod_opsin_eqs = opsin_eqs.substitute(**substitutions)
        return mod_opsin_eqs, modified_params

    def reset(self, **kwargs):
        for opto_syn in self.synapses.values():
            self.init_syn_vars(opto_syn)

    def init_syn_vars(self, syn: Synapses) -> None:
        """Initializes appropriate variables in Synapses implementing the model

        Also called on :meth:`reset`.

        Parameters
        ----------
        syn : Synapses
            The synapses object implementing this model
        """
        pass
