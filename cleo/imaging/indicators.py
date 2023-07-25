from __future__ import annotations

from attrs import define, field
from brian2 import Synapses, NeuronGroup

from cleo.light import LightDependentDevice


@define(eq=False)
class Indicator(LightDependentDevice):
    snr: float = field(kw_only=True)
    location: str = field(kw_only=True)
    """cytoplasm or membrane"""

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    def get_state(self) -> list[np.ndarray]:
        """Returns a list of arrays in the order neuron groups/targets were received.

        Signals should be normalized to baseline of 0 and 1 corresponding
        to an action potential peak."""
        pass


class F:
    pass


def jgcamp7f():
    return Indicator(snr=2, location="cytoplasm")
