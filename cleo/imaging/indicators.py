from __future__ import annotations

from attrs import define, field
from brian2 import Synapses, NeuronGroup


@define(eq=False)
class Indicator:
    snr: float = field()
    location: str = field()
    """cytoplasm or membrane"""
    name: str = field()

    @name.default
    def _default_name(self):
        return self.__class__.__name__

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    brian_objects: set[Synapses] = field(factory=set, init=False)

    def connect_to_neuron_group(self, ng, i_targets, **kwparams):
        pass

    def get_state(self) -> list[np.ndarray]:
        """Returns a list of arrays in the order neuron groups/targets were received.

        Signals should be normalized to baseline of 0 and 1 corresponding
        to an action potential peak."""
        pass


def jgcamp7f():
    return Indicator(snr=2, location="cytoplasm")
