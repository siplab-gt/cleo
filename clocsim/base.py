from brian2 import NeuronGroup, Network, ms, NetworkOperation
from abc import ABC, abstractmethod

class InterfaceDevice(ABC):
    def __init__(self):
        self.name = None
        self.brian_objects = set()
        pass

    @abstractmethod
    def connect_to_neurons(self, neuron_group: NeuronGroup):
        pass


class ControlLoop(ABC):
    @abstractmethod
    def put_state(self, state_dict: dict, time):
        pass

    # The output should be a dictionary of {stimulator_name: value, ...}
    @abstractmethod
    def get_ctrl_signal(self, time) -> dict:
        pass


class Recorder(InterfaceDevice):
    def __init__(self):
        pass

    @abstractmethod
    def get_state(self):
        pass


class Stimulator(InterfaceDevice):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, ctrl_signal):
        pass


class CLOCSimulator:
    def __init__(self, brianNetwork: Network):
        self.network = brianNetwork
        self.stimulators = {}
        self.recorders = {}
        self.controller = None

    def inject_stimulator(self, stimulator: Stimulator, *neuron_groups):
        for ng in neuron_groups:
            stimulator.connect_to_neurons(ng)
        self.stimulators[stimulator.name] = stimulator
        for brian_object in stimulator.brian_objects:
            self.network.add(brian_object)

    def inject_recorder(self, recorder: Recorder, *neuron_groups):
        for ng in neuron_groups:
            recorder.connect_to_neurons(ng)
        self.recorders[recorder.name] = recorder
        for brian_object in recorder.brian_objects:
            self.network.add(brian_object)

    def get_state(self):
        state = {}
        for name, recorder in self.recorders.items():
            state[name] = recorder.get_state()
        return state

    def update_controllers(self, ctrl_signal):
        for name, signal in ctrl_signal.items():
            self.stimulators[name].update(signal)

    def set_control_loop(self, control_loop, sample_period=1*ms, poll_ctrl_period=0.1*ms):
        self.control_loop = control_loop

        def send_signal(t):
            control_loop.put_state(self.get_state(), t)

        def receive_signal(t):
            ctrl_signal = control_loop.get_ctrl_signal(t)
            self.update_controllers(ctrl_signal)

        self.network.add(NetworkOperation(send_signal, dt=sample_period))
        self.network.add(NetworkOperation(receive_signal, dt=poll_ctrl_period))



    def run(self, duration):
        self.network.run(duration)
