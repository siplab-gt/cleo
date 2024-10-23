import pytest
import numpy as np

# Define the parameters for different indicators
light_dependent_params = [
    {'rise': 1.85, 'decay1': 34.07, 'decay2': 263.70, 'r': 0.48, 'Fm': 6.104380, 'Ca0': 4.170575, 'beta': 0.390533, 'F0': -1.001000},  # jGCaMP8f
    {'rise': 2.46, 'decay1': 41.64, 'decay2': 245.80, 'r': 0.28, 'Fm': 7.454645, 'Ca0': 2.691117, 'beta': 0.360008, 'F0': -2.050880},  # jGCaMP8m
    {'rise': 5.65, 'decay1': 86.26, 'decay2': 465.45, 'r': 0.19, 'Fm': 7.455792, 'Ca0': 1.282417, 'beta': 0.343721, 'F0': -2.919320},  # jGCaMP8s
]

light_independent_params = [
    {'rise': 16.21, 'decay1': 95.27, 'decay2': 398.22, 'r': 0.24},  # jGCaMP7f
    {'rise': 50.81, 'decay1': 1702.21, 'decay2': 0.00, 'r': 0.00},  # GCaMP6s
]

@pytest.mark.parametrize("params", light_dependent_params)
def test_s2f_with_light_dependent_sensor(params):
    # Move the import here to avoid circular imports
    from imaging.s2f import S2F, LightDependentSensor

    sensor = LightDependentSensor(**params)
    s2f = S2F(sensor)

    spike_times = np.array([0, 50, 100])
    ca_times = np.arange(-100, 500)
    response = s2f.get_state(spike_times, ca_times)

    # Basic assertions to ensure the output is correct
    assert response is not None
    assert len(response) == len(ca_times)
    assert np.all(response >= 0)  # Ensure the response is non-negative

@pytest.mark.parametrize("params", light_independent_params)
def test_s2f_with_light_independent_sensor(params):
    # Move the import here to avoid circular imports
    from imaging.s2f import S2F, LightIndependentSensor

    sensor = LightIndependentSensor(**params)
    s2f = S2F(sensor)

    spike_times = np.array([0, 50, 100])
    ca_times = np.arange(-100, 500)
    response = s2f.get_state(spike_times, ca_times)

    # Basic assertions to ensure the output is correct
    assert response is not None
    assert len(response) == len(ca_times)
    assert np.all(response >= 0)  # Ensure the response is non-negative

if __name__ == "__main__":
    pytest.main(["-s", __file__])
