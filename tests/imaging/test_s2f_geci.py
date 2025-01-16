import brian2 as b2
from brian2 import np
import pytest

import cleo
from cleo.imaging.s2f import S2FLightDependentGECI, S2FModel
from cleo.imaging.sensors import gcamp6f
from cleo.base import CLSimulator

@pytest.mark.slow
def test_geci(rand_seed):
    rng = np.random.default_rng(rand_seed)
    # random spikes during first 40 ms
    n_nrns = 5
    i_spike_bound = 3
    n_spk = n_nrns * 2
    i_spk = rng.choice(range(i_spike_bound), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * b2.ms
    sgg = b2.SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # hack for assign_coords to work
    geci = gcamp6f()

    sim = cleo.CLSimulator(b2.Network(sgg))
    sim.inject(geci, sgg)
    dFF = geci.get_state()[sgg.name]
    assert np.shape(dFF) == (n_nrns,)
    assert np.all(dFF == 0)  # resting state, since I correct to 0
    sim.run(50 * b2.ms)
    for i_spiking_nrn in range(i_spike_bound):
        for i_nonspiking_nrn in range(i_spike_bound, n_nrns):
            assert dFF[i_spiking_nrn] > dFF[i_nonspiking_nrn]

@pytest.mark.slow
def test_s2f_model(rand_seed):
    rng = np.random.default_rng(rand_seed)
    # Generate random spikes during first 40 ms
    n_nrns = 5
    i_spike_bound = 3
    n_spk = n_nrns * 2
    i_spk = rng.choice(range(i_spike_bound), n_spk)
    t_spk = 40 * rng.random((n_spk,)) * b2.ms
    sgg = b2.SpikeGeneratorGroup(n_nrns, i_spk, t_spk)
    sgg._N = n_nrns  # Optional: needed for coordinate assignment if applicable

    # Create the light-dependent S2F sensor
    spectrum = [(473, 1.0), (500, 0.8)]  # Example excitation spectrum
    s2f_sensor = S2FLightDependentGECI(
        rise=1.0, decay1=2.0, decay2=0.5, r=0.1,
        Fm=1.0, Ca0=0.5, beta=1.0, F0=0.0, sigma_noise=0.01,
        spectrum=spectrum
    )

    sim = CLSimulator(b2.Network(sgg))
    s2f_model = S2FModel(sensor=s2f_sensor, neuron_group=sgg)
    sim.inject(s2f_model, sgg)

    # Retrieve initial state
    dFF = s2f_model.get_state()[sgg]
    assert np.shape(dFF) == (n_nrns,)
    assert np.all(dFF == 0)  # Assuming a zero resting state or modify accordingly

    # Run the simulation
    sim.run(50 * b2.ms)

    # Post-simulation assertions
    for i_spiking_nrn in range(i_spike_bound):
        for i_nonspiking_nrn in range(i_spike_bound, n_nrns):
            assert dFF[i_spiking_nrn] > dFF[i_nonspiking_nrn]

if __name__ == "__main__":
    pytest.main(["-s", __file__])
