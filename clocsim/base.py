from abc import ABC, abstractmethod

from brian2 import NeuronGroup, Network, ms, NetworkOperation, defaultclock


class InterfaceDevice(ABC):
    def __init__(self, name):
        self.name = name
        self.brian_objects = set()

    @abstractmethod
    def connect_to_neurons(self, neuron_group: NeuronGroup):
        pass


class ControlLoop(ABC):
    @abstractmethod
    def is_sampling_now(self, time) -> bool:
        pass

    @abstractmethod
    def put_state(self, state_dict: dict, time):
        pass

    # The output should be a dictionary of {stimulator_name: value, ...}
    @abstractmethod
    def get_ctrl_signal(self, time) -> dict:
        pass


class Recorder(InterfaceDevice):
    @abstractmethod
    def get_state(self):
        pass


class Stimulator(InterfaceDevice):
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
        if ctrl_signal is None:
            return
        for name, signal in ctrl_signal.items():
            self.stimulators[name].update(signal)

    def set_control_loop(self, control_loop, communication_period=None):
        self.control_loop = control_loop
        if communication_period is None:
            communication_period = defaultclock.dt

        def communicate_with_ctrl_loop(t):
            if control_loop.is_sampling_now(t):
                control_loop.put_state(self.get_state(), t)
            ctrl_signal = control_loop.get_ctrl_signal(t)
            self.update_controllers(ctrl_signal)

        self.network.add(NetworkOperation(communicate_with_ctrl_loop, dt=communication_period))



    def run(self, duration):
        self.network.run(duration)


if __name__ == "__main__":
    rec = Recorder()
    print(rec.brian_objects)