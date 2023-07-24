from __future__ import annotations
from typing import Any

from attrs import define, field
from brian2 import NeuronGroup, Unit, mm, np, Quantity, um, meter
import matplotlib as mpl
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from nptyping import NDArray

from cleo.base import Recorder

from cleo.imaging.indicators import Indicator
from cleo.coords import coords_from_ng
from cleo.utilities import normalize_coords


def target_neurons_in_plane(
    ng,
    scope_focus_depth,
    scope_img_width,
    scope_location=(0, 0, 0) * mm,
    scope_direction=(0, 0, 1),
    soma_radius=10 * um,
    indicator_location="cytoplasm",
):
    """Returns a tuple of (i_targets, sig_strength)"""
    ng_coords = coords_from_ng(ng)
    # Compute the normal vector and the center of the plane
    plane_normal = scope_direction
    plane_center = scope_location + plane_normal * scope_focus_depth
    # Compute the distance of each neuron from the plane
    perp_distances = np.abs(np.dot(ng_coords - plane_center, plane_normal))
    snr_focus_factor = np.ones(ng.N)
    # signal falloff with shrinking cross-section (or circumference)
    r_soma_visible = np.sqrt(soma_radius**2 - perp_distances**2)
    r_soma_visible[np.isnan(r_soma_visible)] = 0
    if indicator_location == "cytoplasm":
        snr_focus_factor *= (r_soma_visible / soma_radius) ** 2
    else:
        assert indicator_location == "membrane"
        snr_focus_factor *= r_soma_visible / soma_radius

    # get only neurons in view
    coords_on_plane = ng_coords - plane_normal * perp_distances[:, np.newaxis]
    plane_distances = np.sqrt(np.sum((coords_on_plane - plane_center) ** 2, axis=1))

    i_targets = np.flatnonzero(
        np.logical_and(snr_focus_factor > 0, plane_distances < scope_img_width / 2)
    )

    return i_targets, snr_focus_factor[i_targets]


@define(eq=False)
class Scope(Recorder):
    indicator: Indicator = field()
    focus_depth: Quantity = field()
    img_width: Quantity = field()
    location: Quantity = [0, 0, 0] * mm
    direction: NDArray[(3,), float] = field(
        default=(0, 0, 1), converter=normalize_coords
    )
    # noises: list[ImagingNoise] = field(factory=list)
    soma_radius: Quantity = field(default=10 * um)
    snr_cutoff: float = field(default=1)
    """applied only when not focus_depth is not None"""
    rand_seed: int = field(default=None, repr=False)
    dFF: list[NDArray[(Any,), float]] = field(factory=list, init=False, repr=False)

    neuron_groups: list[NeuronGroup] = field(factory=list, repr=False, init=False)
    i_targets_per_injct: list[NDArray[(Any,), int]] = field(
        factory=list, repr=False, init=False
    )
    snr_per_injct: list[NDArray[(Any,), float]] = field(
        factory=list, repr=False, init=False
    )

    _rng: np.random.Generator = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self._rng = np.random.default_rng(self.rand_seed)

    def target_neurons_in_plane(
        self, ng, focus_depth: Quantity = None, soma_radius: Quantity = None
    ) -> tuple[NDArray[(Any,), int], NDArray[(Any,), float]]:
        focus_depth = focus_depth or self.focus_depth
        soma_radius = soma_radius or self.soma_radius
        return target_neurons_in_plane(
            ng,
            focus_depth,
            self.img_width,
            self.location,
            self.direction,
            soma_radius,
            self.indicator.location,
        )

    def get_state(self) -> NDArray[(Any,), float]:
        snr = np.concatenate(self.snr_per_injct)
        signal = np.concatenate(self.indicator.get_state()) * snr / (1 + snr)
        std_noise = 1 / (1 + snr)
        noise = self._rng.normal(0, std_noise, len(signal))
        # for noise in self.noises:
        #     out += noise.compute(self.t_ms)
        return signal + noise

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        focus_depth = kwparams.get("focus_depth", self.focus_depth)
        soma_radius = kwparams.get("soma_radius", self.soma_radius)
        if focus_depth:
            if "i_targets" in kwparams or "snr" in kwparams:
                raise ValueError(
                    "i_targets and snr can not be specified while focus_depth",
                    " is not None",
                )
            i_targets, snr_focus_factor = self.target_neurons_in_plane(
                neuron_group, focus_depth, soma_radius
            )
            base_snr = kwparams.get("base_snr", self.indicator.snr)
            snr = snr_focus_factor * base_snr
            i_targets = i_targets[snr > self.snr_cutoff]
            snr = snr[snr > self.snr_cutoff]
        else:
            i_targets = kwparams.pop("i_targets", neuron_group.i_)
            snr = kwparams.get("snr", self.indicator.snr)
            _, snr = np.broadcast_arrays(i_targets, snr)
        assert len(i_targets) == len(snr)

        self.indicator.connect_to_neuron_group(neuron_group, i_targets, **kwparams)
        self.neuron_groups.append(neuron_group)
        self.i_targets_per_injct.append(i_targets)
        self.snr_per_injct.append(snr)

        # for noise in self.noises:
        #     noise.init_for_ng(
        #         neuron_group,
        #         i_targets,
        #         self.location,
        #         self.direction,
        #         focus_depth,
        #         **kwparams,
        #     )

    def add_self_to_plot(
        self, ax: Axes3D, axis_scale_unit: Unit, **kwargs
    ) -> list[Artist]:
        color = kwargs.pop("color", "xkcd:fluorescent green")
        snr = np.concatenate(self.snr_per_injct)
        coords = (
            np.concatenate(
                [
                    coords_from_ng(ng)[i_targets]
                    for ng, i_targets in zip(
                        self.neuron_groups, self.i_targets_per_injct
                    )
                ]
            )
            * meter
        )
        assert coords.shape == (len(snr), 3)
        assert len(snr) == len(coords)
        scope_marker = ax.quiver(
            self.location[0] / axis_scale_unit,
            self.location[1] / axis_scale_unit,
            self.location[2] / axis_scale_unit,
            self.direction[0],
            self.direction[1],
            self.direction[2],
            color=color,
            lw=5,
            label=self.name,
            pivot="tip",
            length=0.1,
            normalize=True,
        )

        # Define the center, normal vector, and radius
        normal = self.direction
        center = self.location + normal * self.focus_depth
        radius = self.img_width / 2

        # Generate the points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.sqrt(np.linspace(0, 1, 5))
        # angle away from z axis
        phi = np.arccos(normal[2])
        # angle relative to x axis
        theta0 = np.arctan2(normal[1], normal[0])
        x = center[0] + radius * np.cos(phi) * np.outer(np.cos(theta), r)
        y = center[1] + radius * np.cos(phi) * np.outer(np.sin(theta), r)
        z = center[2] + radius * np.sin(phi) * np.outer(-np.cos(theta - theta0), r)

        plane = ax.plot_surface(
            x / axis_scale_unit,
            y / axis_scale_unit,
            z / axis_scale_unit,
            color=color,
            alpha=0.3,
        )

        target_markers = ax.scatter(
            coords[:, 0] / axis_scale_unit,
            coords[:, 1] / axis_scale_unit,
            coords[:, 2] / axis_scale_unit,
            marker="^",
            # s=snr * 100,
            # alpha=self._rng.uniform(0, 1, len(snr)),
            c=color,
            s=5,
            # cmap="Greens",
            label=self.indicator.name,
            **kwargs,
        )
        color_rgba = target_markers.get_facecolor()
        color_rgba[:, :3] = 0.3 * color_rgba[:, :3]
        target_markers.set(color=color_rgba)
        handles = ax.get_legend().legendHandles

        patch = mpl.patches.Patch(color=color, label=self.name)
        handles.append(patch)
        ax.legend(handles=handles)
        return [scope_marker, target_markers, plane]
