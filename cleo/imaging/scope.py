from __future__ import annotations
from typing import Any

from attrs import define, field
from brian2 import NeuronGroup, mm, np, Quantity, um
from nptyping import NDArray

from cleo.base import Recorder
from cleo.imaging.noise import ImagingNoise
from cleo.imaging.indicators import Indicator


def target_neurons_in_plane(
    ng, scope_location, scope_direction, scope_focus_depth, scope_img_width
):
    """Returns a tuple of (i_targets, signal_strength)"""
    return [0], [1]


@define(eq=False)
class Scope(Recorder):
    indicator: Indicator = field()
    focus_depth: Quantity = field()
    img_width: Quantity = field()
    location: Quantity = [0, 0, 0] * mm
    direction: np.ndarray = np.array([0, 0, 1])
    noises: list[ImagingNoise] = field(factory=list)
    soma_radius: Quantity = field(default=10 * um, kw_only=True)

    neuron_groups: list[NeuronGroup] = field(factory=list, repr=False, init=False)
    i_targets_per_ng: list[NDArray[(Any,), int]] = field(
        factory=list, repr=False, init=False
    )
    signal_strengths_per_ng: list[NDArray[(Any,), float]] = field(
        factory=list, repr=False, init=False
    )

    def target_neurons_in_plane(self, ng, focus_depth: Quantity = None):
        focus_depth = focus_depth or self.focus_depth
        return target_neurons_in_plane(
            ng, self.location, self.direction, focus_depth, self.img_width
        )

    def get_state(self) -> NDArray[(Any,), float]:
        sig_strength = np.concatenate(self.signal_strengths_per_ng)
        out = self.indicator.get_state() * sig_strength
        for noise in self.noises:
            out += noise.compute(self.t_ms)
        return out

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        self.neuron_groups.append(neuron_group)
        focus_depth = kwparams.get("focus_depth", self.focus_depth)
        i_targets, sig_strength = self.target_neurons_in_plane(
            neuron_group, focus_depth
        )
        self.i_targets_per_ng.append(i_targets)
        self.signal_strengths_per_ng.append(sig_strength)

        for noise in self.noises:
            noise.init_for_ng(
                neuron_group,
                i_targets,
                self.location,
                self.direction,
                focus_depth,
                **kwparams,
            )

        self.indicator.init_for_ng(neuron_group, i_targets, **kwparams)
