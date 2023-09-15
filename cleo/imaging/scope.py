from __future__ import annotations
from typing import Any
import warnings

from attrs import define, field
from brian2 import NeuronGroup, Unit, mm, np, Quantity, um, meter, ms
import matplotlib as mpl
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from nptyping import NDArray

from cleo.base import Recorder

from cleo.imaging.sensors import Sensor
from cleo.coords import coords_from_ng
from cleo.utilities import normalize_coords, rng


def target_neurons_in_plane(
    ng,
    scope_focus_depth,
    scope_img_width,
    scope_location=(0, 0, 0) * mm,
    scope_direction=(0, 0, 1),
    soma_radius=10 * um,
    sensor_location="cytoplasm",
):
    """Returns a tuple of (i_targets, noise_focus_factor, coords_on_plane)"""
    ng_coords = coords_from_ng(ng)
    # Compute the normal vector and the center of the plane
    plane_normal = scope_direction
    plane_center = scope_location + plane_normal * scope_focus_depth
    # Compute the distance of each neuron from the plane
    perp_distances = np.abs(np.dot(ng_coords - plane_center, plane_normal))
    noise_focus_factor = np.ones(ng.N)
    # signal falloff with shrinking cross-section (or circumference)
    # ignore numpy warning
    with np.errstate(invalid="ignore"):
        r_soma_visible = np.sqrt(soma_radius**2 - perp_distances**2)
    r_soma_visible[np.isnan(r_soma_visible)] = 0
    if sensor_location == "cytoplasm":
        relative_num_pixels = (r_soma_visible / soma_radius) ** 2
    else:
        assert sensor_location == "membrane"
        relative_num_pixels = r_soma_visible / soma_radius
    with np.errstate(divide="ignore"):
        noise_focus_factor /= np.sqrt(relative_num_pixels)

    # get only neurons in view
    coords_on_plane = ng_coords - plane_normal * perp_distances[:, np.newaxis]
    plane_distances = np.sqrt(np.sum((coords_on_plane - plane_center) ** 2, axis=1))

    i_targets = np.flatnonzero(
        np.logical_and(r_soma_visible > 0, plane_distances < scope_img_width / 2)
    )

    return i_targets, noise_focus_factor[i_targets], coords_on_plane[i_targets]


@define(eq=False)
class Scope(Recorder):
    sensor: Sensor = field()
    img_width: Quantity = field()
    focus_depth: Quantity = None
    location: Quantity = [0, 0, 0] * mm
    direction: NDArray[(3,), float] = field(
        default=(0, 0, 1), converter=normalize_coords
    )
    soma_radius: Quantity = field(default=10 * um)
    snr_cutoff: float = field(default=1)
    """applied only when not focus_depth is not None"""
    rand_seed: int = field(default=None, repr=False)
    dFF: list[NDArray[(Any,), float]] = field(factory=list, init=False, repr=False)
    """ΔF/F from every call to :meth:`get_state`.
    Shape is (n_samples, n_ROIs). Stored if :attr:`~cleo.InterfaceDevice.save_history`"""
    t_ms: list[float] = field(factory=list, init=False, repr=False)
    """Times at which sensor traces are recorded, in ms, stored if
    :attr:`~cleo.InterfaceDevice.save_history`"""

    neuron_groups: list[NeuronGroup] = field(factory=list, repr=False, init=False)
    i_targets_per_injct: list[NDArray[(Any,), int]] = field(
        factory=list, repr=False, init=False
    )
    sigma_per_injct: list[NDArray[(Any,), float]] = field(
        factory=list, repr=False, init=False
    )
    focus_coords_per_injct: list[NDArray[(Any,), float]] = field(
        factory=list, repr=False, init=False
    )

    @property
    def n(self):
        return np.sum(len(i_t) for i_t in self.i_targets_per_injct)

    def _init_saved_vars(self):
        if self.save_history:
            self.t_ms = []
            self.dFF = []

    def _update_saved_vars(self, t_ms, dFF):
        if self.save_history:
            self.t_ms.append(t_ms)
            self.dFF.append(dFF)

    def __attrs_post_init__(self):
        self._init_saved_vars()

    def reset(self, **kwargs) -> None:
        self._init_saved_vars()

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
            self.sensor.location,
        )

    def get_state(self) -> NDArray[(Any,), float]:
        sigma_noise = np.concatenate(self.sigma_per_injct)
        signal = []
        signal_per_ng = self.sensor.get_state()
        # sensor has just one signal for neuron group, not storing
        # separately for each injection, so we'll recover that here
        n_prev_targets_for_ng = {}
        for ng, i_targets in zip(self.neuron_groups, self.i_targets_per_injct):
            subset_start = n_prev_targets_for_ng.get(ng, 0)
            subset_for_injct = slice(subset_start, subset_start + len(i_targets))
            signal.append(signal_per_ng[ng.name][subset_for_injct])
            n_prev_targets_for_ng[ng] = subset_start + len(i_targets)
        signal = np.concatenate(signal)
        noise = rng.normal(0, sigma_noise, len(signal))
        assert self.n == len(signal) == len(noise) == len(sigma_noise)

        state = signal + noise
        now_ms = self.sim.network.t / ms
        self._update_saved_vars(now_ms, state)
        return signal + noise

    def connect_to_neuron_group(self, neuron_group: NeuronGroup, **kwparams) -> None:
        focus_depth = kwparams.get("focus_depth", self.focus_depth)
        soma_radius = kwparams.get("soma_radius", self.soma_radius)
        if focus_depth:
            if "i_targets" in kwparams or "sigma_noise" in kwparams:
                raise ValueError(
                    "i_targets and sigma_noise can not be specified while focus_depth",
                    " is not None",
                )
            (
                i_targets,
                noise_focus_factor,
                focus_coords,
            ) = self.target_neurons_in_plane(neuron_group, focus_depth, soma_radius)
            base_sigma = kwparams.get("base_sigma", self.sensor.sigma_noise)
            sigma_noise = noise_focus_factor * base_sigma
            if self.sensor.dFF_1AP:
                snr = self.sensor.dFF_1AP / sigma_noise
                i_targets = i_targets[snr > self.snr_cutoff]
                sigma_noise = sigma_noise[snr > self.snr_cutoff]
                focus_coords = focus_coords[snr > self.snr_cutoff]
            else:
                warnings.warn(
                    f"SNR cutoff not used, since {self.sensor.name} does not have dFF_1AP defined."
                )
        else:
            i_targets = kwparams.pop("i_targets", neuron_group.i_)
            sigma_noise = kwparams.get("sigma_noise", self.sensor.sigma_noise)
            _, sigma_noise = np.broadcast_arrays(i_targets, sigma_noise)
            focus_coords = coords_from_ng(neuron_group)[i_targets]
        assert len(i_targets) == len(sigma_noise) == len(focus_coords)

        self.neuron_groups.append(neuron_group)
        self.i_targets_per_injct.append(i_targets)
        self.sigma_per_injct.append(sigma_noise)
        self.focus_coords_per_injct.append(focus_coords)

    def i_targets_for_neuron_group(self, neuron_group):
        """can handle multiple injections into same ng"""
        i_targets_for_ng = []
        for ng, i_targets in zip(self.neuron_groups, self.i_targets_per_injct):
            if ng is neuron_group:
                i_targets_for_ng.extend(i_targets)
        return i_targets_for_ng

    def inject_sensor_for_targets(self, **kwparams) -> None:
        for ng in set(self.neuron_groups):
            i_all_targets = self.i_targets_for_neuron_group(ng)
            self.sim.inject(self.sensor, ng, i_targets=i_all_targets, **kwparams)

    def add_self_to_plot(
        self, ax: Axes3D, axis_scale_unit: Unit, **kwargs
    ) -> list[Artist]:
        color = kwargs.pop("color", "xkcd:fluorescent green")
        snr = np.concatenate(self.sigma_per_injct)
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
            length=100 * um / axis_scale_unit,
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
            c=color,
            s=5,
            label=self.sensor.name,
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
