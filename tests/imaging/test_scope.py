from brian2 import np, um, NeuronGroup, Network

import cleo
from cleo.imaging import Scope, UniformGaussianNoise, jgcamp7f, target_neurons_in_plane
from cleo.coords import assign_coords


def test_scope():
    scope = Scope(
        focus_depth=200 * um,
        img_width=500 * um,
        # indicator=jgcamp7f(),
        noises=[UniformGaussianNoise(sigma=0.1)],
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
    i_targets, sig_strength = scope.target_neurons_in_plane(ng)
    assert np.all(scope.i_targets_per_ng[0] == i_targets)
    assert np.all(scope.signal_strengths_per_ng[0] == sig_strength)
    i_targets, sig_strength = target_neurons_in_plane(
        ng, scope.location, scope.direction, scope.focus_depth, scope.img_width
    )
    assert np.all(scope.i_targets_per_ng[0] == i_targets)
    assert np.all(scope.signal_strengths_per_ng[0] == sig_strength)

    # got all neurons in first layer, max signal strength
    assert len(i_targets) == 40
    assert np.all(sig_strength == 1)

    sim.inject(scope, focus_depth=300 * um)
    assert len(i_targets) == 35


def test_target_neurons_in_plane():
    # how to specify image height and width? Maybe just a circle 😏
    pass
