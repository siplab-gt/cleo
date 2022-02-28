"""Contains functions for assigning neuron coordinates and visualizing"""

from __future__ import annotations
from typing import Tuple

from brian2 import mm, meter, Unit
from brian2.groups.group import Group
from brian2.groups.neurongroup import NeuronGroup
from brian2.units.fundamentalunits import get_dimensions
import numpy as np

from cleosim.utilities import get_orth_vectors_for_v, modify_model_with_eqs


def assign_coords_grid_rect_prism(
    neuron_group: NeuronGroup,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    shape: Tuple[int, int, int],
    unit: Unit = mm,
) -> None:
    """Assign grid coordinates to neurons in a rectangular grid

    Parameters
    ----------
    neuron_group : NeuronGroup
        The neuron group to assign coordinates to
    xlim : Tuple[float, float]
        xmin, xmax, with no unit
    ylim : Tuple[float, float]
        ymin, ymax, with no unit
    zlim : Tuple[float, float]
        zmin, zmax with no unit
    shape : Tuple[int, int, int]
        n_x, n_y, n_z tuple representing the shape of the resulting grid
    unit : Unit, optional
        Brian unit determining what scale to use for coordinates, by default mm

    Raises
    ------
    ValueError
        When the shape is incompatible with the number of neurons in the group
    """
    _init_variables(neuron_group)
    num_grid_elements = np.product(shape)
    if num_grid_elements != len(neuron_group):
        raise ValueError(
            f"Number of elements specified in shape ({num_grid_elements}) "
            f"does not match the number of neurons in the group ({len(neuron_group)})."
        )

    x = np.linspace(xlim[0], xlim[1], shape[0])
    y = np.linspace(ylim[0], ylim[1], shape[1])
    z = np.linspace(zlim[0], zlim[1], shape[2])

    x, y, z = np.meshgrid(x, y, z)

    neuron_group.x = x.flatten() * unit
    neuron_group.y = y.flatten() * unit
    neuron_group.z = z.flatten() * unit


def assign_coords_rand_rect_prism(
    neuron_group: NeuronGroup,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    unit: Unit = mm,
) -> None:
    """Assign random coordinates to neurons within a rectangular prism

    Parameters
    ----------
    neuron_group : NeuronGroup
        neurons to assign coordinates to
    xlim : Tuple[float, float]
        xmin, xmax without unit
    ylim : Tuple[float, float]
        ymin, ymax without unit
    zlim : Tuple[float, float]
        zmin, zmax without unit
    unit : Unit, optional
        Brian unit to specify scale implied in limits, by default mm
    """
    _init_variables(neuron_group)
    x = (xlim[1] - xlim[0]) * np.random.random(len(neuron_group)) + xlim[0]
    y = (ylim[1] - ylim[0]) * np.random.random(len(neuron_group)) + ylim[0]
    z = (zlim[1] - zlim[0]) * np.random.random(len(neuron_group)) + zlim[0]

    neuron_group.x = x.flatten() * unit
    neuron_group.y = y.flatten() * unit
    neuron_group.z = z.flatten() * unit


def assign_coords_rand_cylinder(
    neuron_group: NeuronGroup,
    xyz_start: Tuple[float, float, float],
    xyz_end: Tuple[float, float, float],
    radius: float,
    unit: Unit = mm,
) -> None:
    """Assign random coordinates within a cylinder.

    *DON'T USE YET*: spacing isn't uniform, clustering around
    the center since a random height and radius is chosen for
    each neuron

    Parameters
    ----------
    neuron_group : NeuronGroup
        neurons to assign coordinates to
    xyz_start : Tuple[float, float, float]
        starting position of cylinder without unit
    xyz_end : Tuple[float, float, float]
        ending position of cylinder without unit
    radius : float
        radius of cylinder without unit
    unit : Unit, optional
        Brian unit to scale other params, by default mm
    """
    _init_variables(neuron_group)
    xyz_start = np.array(xyz_start)
    xyz_end = np.array(xyz_end)
    rs = radius * np.random.random(len(neuron_group))
    thetas = 2 * np.pi * np.random.random(len(neuron_group))
    cyl_length = np.linalg.norm(xyz_end - xyz_start)
    z_cyls = cyl_length * np.random.random(len(neuron_group))

    c = np.reshape(
        (xyz_end - xyz_start) / cyl_length, (-1, 1)
    )  # unit vector in direction of cylinder

    r1, r2 = get_orth_vectors_for_v(c)

    def r_unit_vecs(thetas):
        r1s = np.repeat(r1, len(thetas), 0)
        r2s = np.repeat(r2, len(thetas), 0)
        cosines = np.reshape(np.cos(thetas), (len(thetas), 1))
        sines = np.reshape(np.sin(thetas), (len(thetas), 1))
        return r1s * cosines + r2s * sines

    rs = np.reshape(rs, (len(rs), 1))
    points = np.reshape(xyz_start, (1, 3)) + (c * z_cyls).T + rs * r_unit_vecs(thetas)

    neuron_group.x = points[:, 0] * unit
    neuron_group.y = points[:, 1] * unit
    neuron_group.z = points[:, 2] * unit


def _init_variables(group: Group):
    for dim_name in ["x", "y", "z"]:
        if hasattr(group, dim_name):
            setattr(group, dim_name, 0 * mm)
        else:
            if type(group) == NeuronGroup:
                modify_model_with_eqs(group, f"{dim_name}: meter")
            elif issubclass(type(group), Group):
                group.variables.add_array(
                    dim_name,
                    size=group._N,
                    dimensions=get_dimensions(meter),
                    dtype=float,
                    constant=True,
                    scalar=False,
                )
            else:
                raise NotImplementedError(
                    "Coordinate assignment only implemented for brian2.Group objects"
                )
