import brian2 as b2
from brian2 import np
import pytest
import cleo
from imaging.naomi import gcamp6f

@pytest.mark.slow
def test_geci(rand_seed):
    rng = np.random.default_rng(rand_seed)
    n_nrns = 5
    i_spike_bound = 3
    n_spk = n_nrns * 2
    i_spk = rng.choice(range(i_spike_bound), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * b2.ms
    sgg = b2.SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # hack for assign_coords to work
    geci_model = gcamp6f()

    sim = cleo.CLSimulator(b2.Network(sgg))
    sim.inject(geci_model, sgg)
    dFF = geci_model.get_state()[sgg.name]
    assert np.shape(dFF) == (n_nrns,)
    assert np.all(dFF == 0)  # resting state, since I correct to 0
    sim.run(50 * b2.ms)
    for i_spiking_nrn in range(i_spike_bound):
        for i_nonspiking_nrn in range(i_spike_bound, n_nrns):
            assert dFF[i_spiking_nrn] > dFF[i_nonspiking_nrn]

if __name__ == "__main__":
    test_geci(np.random.default_rng())
