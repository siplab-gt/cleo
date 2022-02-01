from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from operator import concat
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.core import numeric
import numpy.typing as npt
from brian2 import NeuronGroup, mm, Unit, Quantity, umeter

from cleosim.base import Recorder
from cleosim.utilities import get_orth_vectors_for_v


class Signal(ABC):
    name: str
    brian_objects: set
    probe: Probe

    def __init__(self, name: str) -> None:
        self.name = name
        self.brian_objects = set()
        self.probe = None

    def init_for_electrode_group(self, eg: Probe):
        if self.probe is not None and self.probe is not eg:
            raise ValueError(
                f"Signal {self.name} has already been initialized "
                f"for ElectrodeGroup {self.probe.name} "
                f"and cannot be used with another."
            )
        self.probe = eg

    @abstractmethod
    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        pass

    @abstractmethod
    def get_state(self) -> Any:
        pass

    def reset(self, **kwargs) -> None:
        pass


class Probe(Recorder):
    coords: Quantity
    signals: list[Signal]
    n: int

    def __init__(self, name: str, coords: Quantity, signals: Iterable[Signal] = []):
        super().__init__(name)
        self.coords = coords.reshape((-1, 3))
        if len(self.coords.shape) != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                "coords must be an n by 3 array (with unit) with x, y, and z"
                "coordinates for n contact locations."
            )
        self.n = len(self.coords)
        self.signals = []
        self.add_signals(*signals)

    def add_signals(self, *signals: Signal):
        for signal in signals:
            signal.init_for_electrode_group(self)
            self.signals.append(signal)

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams):
        for signal in self.signals:
            signal.connect_to_neuron_group(neuron_group, **kwparams)
            self.brian_objects.update(signal.brian_objects)

    def get_state(self):
        state_dict = {}
        for signal in self.signals:
            state_dict[signal.name] = signal.get_state()
        return state_dict

    def add_self_to_plot(self, ax: Axes3D, axis_scale_unit: Unit):
        marker = ax.scatter(
            self.xs / axis_scale_unit,
            self.ys / axis_scale_unit,
            self.zs / axis_scale_unit,
            marker="x",
            s=40,
            color="xkcd:dark gray",
            label=self.name,
            depthshade=False,
        )
        handles = ax.get_legend().legendHandles
        handles.append(marker)
        ax.legend(handles=handles)

    @property
    def xs(self):
        return self.coords[:, 0]

    @property
    def ys(self):
        return self.coords[:, 1]

    @property
    def zs(self):
        return self.coords[:, 2]

    def reset(self, **kwargs):
        for signal in self.signals:
            signal.reset()


def concat_coords(*coords: Quantity):
    out = np.vstack([c / mm for c in coords])
    return out * mm


def linear_shank_coords(
    array_length: Quantity,
    channel_count: int,
    start_location: Quantity = (0, 0, 0) * mm,
    direction: Tuple[float, float, float] = (0, 0, 1),
) -> npt.NDArray:
    dir_uvec = direction / np.linalg.norm(direction)
    end_location = start_location + array_length * dir_uvec
    return np.linspace(start_location, end_location, channel_count)


def tetrode_shank_coords(
    array_length: Quantity,
    tetrode_count: int,
    start_location: Quantity = (0, 0, 0) * mm,
    direction: Tuple[float, float, float] = (0, 0, 1),
    tetrode_width: Quantity = 25 * umeter,
) -> npt.NDArray:
    dir_uvec = direction / np.linalg.norm(direction)
    end_location = start_location + array_length * dir_uvec
    center_locs = np.linspace(start_location, end_location, tetrode_count)
    # need to add coords around the center locations
    # tetrode_width is the length of one side of the square, so the diagonals
    # are measured in width/sqrt(2)
    #    x      -dir*width/sqrt(2)
    # x  .  x   +/- orth*width/sqrt(2)
    #    x      +dir*width/sqrt(2)
    orth_uvec, _ = get_orth_vectors_for_v(dir_uvec)
    return np.repeat(center_locs, 4, axis=0) + tetrode_width / np.sqrt(2) * np.tile(
        np.vstack([-dir_uvec, -orth_uvec, orth_uvec, dir_uvec]), (tetrode_count, 1)
    )


def poly2_shank_coords(
    array_length: Quantity,
    channel_count: int,
    intercol_space: Quantity,
    start_location: Quantity = (0, 0, 0) * mm,
    direction: Tuple[float, float, float] = (0, 0, 1),
) -> npt.NDArray:
    dir_uvec = direction / np.linalg.norm(direction)
    end_location = start_location + array_length * dir_uvec
    out = np.linspace(start_location, end_location, channel_count)
    orth_uvec, _ = get_orth_vectors_for_v(dir_uvec)
    # place contacts on alternating sides of the central axis
    even_channels = np.arange(channel_count) % 2 == 0
    out[even_channels] += intercol_space / 2 * orth_uvec
    out[~even_channels] -= intercol_space / 2 * orth_uvec
    return out


def poly3_shank_coords(
    array_length: Quantity,
    channel_count: int,
    intercol_space: Quantity,
    start_location: Quantity = (0, 0, 0) * mm,
    direction: Tuple[float, float, float] = (0, 0, 1),
) -> npt.NDArray:
    # makes middle column longer if not even. Nothing fancier.
    # length measures middle column
    dir_uvec = direction / np.linalg.norm(direction)
    end_location = start_location + array_length * dir_uvec
    center_loc = start_location + array_length * dir_uvec / 2
    n_middle = channel_count // 3 + channel_count % 3
    n_side = int((channel_count - n_middle) / 2)

    middle = np.linspace(start_location, end_location, n_middle)

    spacing = array_length / n_middle
    side_length = n_side * spacing
    orth_uvec, _ = get_orth_vectors_for_v(dir_uvec)
    side = np.linspace(
        center_loc - dir_uvec * side_length / 2,
        center_loc + dir_uvec * side_length / 2,
        n_side,
    )
    side1 = side + orth_uvec * intercol_space
    side2 = side - orth_uvec * intercol_space
    out = concat_coords(middle, side1, side2)
    return out[out[:, 2].argsort()]  # sort to return superficial -> deep


def tile_coords(coords: Quantity, num_tiles: int, tile_vector: Quantity):
    num_coords = coords.shape[0]
    # num_tiles X 3
    offsets = np.linspace((0, 0, 0) * mm, tile_vector, num_tiles)
    # num_coords X num_tiles X 3
    out = np.tile(coords[:, np.newaxis, :], (1, num_tiles, 1)) + offsets
    return out.reshape((num_coords * num_tiles, 3), order="F")
