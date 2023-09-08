import brian2 as b2
from brian2 import np
import pytest

import cleo
from cleo.imaging import gcamp6f


def test_geci(rand_seed):
    rng = np.random.default_rng(rand_seed)
    # random spikes during first 40 ms
    n_nrns = 5
    n_spk = n_nrns * 2
    i_spk = rng.choice(range(n_nrns), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * b2.ms
    sgg = b2.SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # hack for assign_coords to work
    # cleo.coords.assign_coords(sgg, c_mm[:, 0], c_mm[:, 1], c_mm[:, 2])
    geci = gcamp6f()

    sim = cleo.CLSimulator(b2.Network(sgg))
    sim.inject(geci, sgg)
    c = geci.get_state()[sgg.name]
    assert np.shape(c) == (n_nrns,)
    assert np.all(c == 0)
    sim.run(50 * b2.ms)
    assert np.all(c > 0)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
