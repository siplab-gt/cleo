from brian2 import np, um, NeuronGroup, Network, meter
import pytest

import cleo
from cleo.imaging import (
    Scope,
    # UniformGaussianNoise,
    jgcamp7f,
    target_neurons_in_plane,
    Indicator,
)
from cleo.coords import assign_coords, assign_coords_rand_rect_prism


def test_scope():
    scope = Scope(
        focus_depth=200 * um,
        img_width=500 * um,
        indicator=Indicator(location="cytoplasm", snr=2),
    )
    assert np.all(scope.direction == [0, 0, 1])
    assert np.all(scope.location == [0, 0, 0] * um)

    ng = NeuronGroup(100, "dv/dt = -v / (10*ms) : 1")
    assign_coords(ng, 0, 0, 0)
    ng.z[0:40] = 200 * um
    ng.z[40:75] = 300 * um
    ng.z[75:] = 400 * um

    sim = cleo.CLSimulator(Network(ng))

    # injection with depth defined equivalent to getting plane targets separately
    sim.inject(scope, ng)
    i_targets, snr_focus_factor = scope.target_neurons_in_plane(ng)
    assert np.all(scope.i_targets_per_injct[0] == i_targets)
    assert np.all(scope.snr_per_injct[0] == snr_focus_factor * scope.indicator.snr)
    i_targets, snr_focus_factor = target_neurons_in_plane(
        ng,
        scope.focus_depth,
        scope.img_width,
        scope.location,
        scope.direction,
        scope.soma_radius,
    )
    assert np.all(scope.i_targets_per_injct[0] == i_targets)
    assert np.all(scope.snr_per_injct[0] == snr_focus_factor * scope.indicator.snr)

    # got all neurons in first layer, max signal strength
    assert len(i_targets) == 40
    assert np.all(snr_focus_factor == 1)

    sim.inject(scope, ng, focus_depth=300 * um)
    assert len(scope.i_targets_per_injct[1]) == 35

    sim.inject(scope, ng, focus_depth=None, i_targets=range(75, 80), snr=0.5)
    assert len(scope.snr_per_injct[2]) == 5
    assert np.all(scope.snr_per_injct[2] == 0.5)

    custom_snrs = np.linspace(0.1, 0.5, 6)
    sim.inject(scope, ng, focus_depth=None, i_targets=range(80, 86), snr=custom_snrs)
    assert np.all(scope.snr_per_injct[3] == custom_snrs)

    with pytest.raises(ValueError):
        # can't specify both i_targets and focus_depth
        sim.inject(scope, ng, focus_depth=200 * um, i_targets=range(86, 90))


def test_target_neurons_in_plane(rand_seed):
    rng = np.random.default_rng(rand_seed)
    focus_depth, img_width = 200 * um, 200 * um
    ng = NeuronGroup(500, "dv/dt = -v / (10*ms) : 1")
    assign_coords_rand_rect_prism(ng, [-0.2, 0.2], [-0.2, 0.2], [0.1, 0.3])
    # snr_focus_factor decreases with distance from plane
    i_targets, snr_focus_factor = target_neurons_in_plane(ng, focus_depth, img_width)
    assert np.all(
        np.diff(snr_focus_factor[np.argsort(np.abs(ng.z[i_targets] - focus_depth))])
        <= 0
    )
    assert len(i_targets) < ng.N
    # below 0 not collected
    assert np.all(snr_focus_factor > 0)

    # snrff stays same along plane
    ng.z = focus_depth
    i_targets, snr_focus_factor = target_neurons_in_plane(ng, focus_depth, img_width)
    assert np.all(np.diff(snr_focus_factor) == 0)
    # neurons outside image not collected
    assert np.sum(np.sqrt(ng.x**2 + ng.y**2) < img_width / 2) == len(i_targets)

    # snrff increases with increasing soma radius
    ng.z = rng.uniform(focus_depth - 30 * um, focus_depth + 30 * um, ng.N) * meter
    i_targets, snr_focus_factor = target_neurons_in_plane(ng, focus_depth, img_width)
    i_targets_bigger, snrff_bigger = target_neurons_in_plane(
        ng, focus_depth, img_width, soma_radius=20 * um
    )
    assert len(i_targets_bigger) > len(i_targets)
    assert np.all(snrff_bigger[np.in1d(i_targets_bigger, i_targets)] > snr_focus_factor)

    # snrff stronger for membrane than cytoplasm
    # (counterintuitive since the overall SNR should be stronger,
    #  but the dropoff is faster for circumference than area of
    #  the cross-section of a sphere)
    i_targets_membrane, snrff_membrane = target_neurons_in_plane(
        ng, focus_depth, img_width, indicator_location="membrane"
    )
    assert np.all(
        snrff_membrane[np.in1d(i_targets_membrane, i_targets)] > snr_focus_factor
    )
    assert len(i_targets_membrane) == len(i_targets)  # since 0 is cutoff

    # TODO: random rotations


if __name__ == "__main__":
    test_scope()
    test_target_neurons_in_plane(134087)
