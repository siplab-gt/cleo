from brian2 import NeuronGroup, Network
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
    def get_control_signal(self, time) -> dict:
        pass


class Recorder(InterfaceDevice):
    def __init__(self):
        pass


class Stimulator(InterfaceDevice):
    def __init__(self):
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

    def set_control_loop(self, control_loop, control_period):
        self.controller = controller
        self.network.


    def run(self, duration):
        self.network.run(duration)
