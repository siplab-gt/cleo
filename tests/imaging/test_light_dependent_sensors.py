import pytest
from brian2 import umolar
from cleo.imaging.sensors import (
    LightDependentGECI,
    LightExcitation,
    DynamicCalcium,
    NullBindingActivation,
)

def test_light_dependent_geci_is_correct_type():
    """LightDependentGECI should be successfully instantiated with arguments"""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )
    assert isinstance(sensor, LightDependentGECI)

def test_light_dependent_geci_uses_light_excitation():
    """LightDependentGECI should default to using the LightExcitation model"""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )
    assert isinstance(sensor.exc_model, LightExcitation)

def test_light_excitation_has_required_params():
    """LightExcitation model should contain all core Hill equation parameters"""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )
    assert hasattr(sensor.exc_model, 'baseline')
    assert hasattr(sensor.exc_model, 'A')
    assert hasattr(sensor.exc_model, 'n')
    assert hasattr(sensor.exc_model, 'ec50')
    assert hasattr(sensor.exc_model, 'k')


def test_snr_increases_with_laser_power():
    """SNR should increase as laser power increases."""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )

    laser_powers = [1.0, 5.0, 10.0]

    snrs = []

    for power in laser_powers:
        _, _, snr = sensor.compute_hill_raster_noise(
            laser_power=power,
            frequency=30.0,
            soma_radius=10.0,
            img_width=500.0,
            flyback_time=0.001,
        reference_power=10.0,
        reference_dwell=0.001,
        )
        snrs.append(snr)

    assert snrs[0] < snrs[1] < snrs[2]


def test_snr_increases_with_soma_radius():
    """SNR should increase as soma radius increases."""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )

    soma_radii = [5.0, 10.0, 20.0]
    snrs = []

    for radius in soma_radii:
        _, _, snr = sensor.compute_hill_raster_noise(
            laser_power=10.0,
            frequency=30.0,
            soma_radius=radius,
            img_width=500.0,
            flyback_time=0.001,
        reference_power=10.0,
        reference_dwell=0.001,
        )
        snrs.append(snr)

    assert snrs[0] < snrs[1] < snrs[2]

def test_snr_decreases_with_image_width():
    """SNR should decrease as image width increases."""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0
    )

    image_widths = [250.0, 500.0, 1000.0]
    snrs = []

    for width in image_widths:
        _, _, snr = sensor.compute_hill_raster_noise(
            laser_power=10.0,
            frequency=30.0,
            soma_radius=10.0,
            img_width=width,
            flyback_time=0.001,
        reference_power=10.0,
        reference_dwell=0.001,
        )
        snrs.append(snr)

    assert snrs[0] > snrs[1] > snrs[2]


def test_snr_decreases_with_frequency():
    """SNR should decrease as imaging frequency increases."""
    sensor = LightDependentGECI(
        name="test_sensor",
        sigma_noise=0.1,
        dFF_1AP=0.1,
        cal_model=DynamicCalcium(),
        bind_act_model=NullBindingActivation(),
        exc_model=LightExcitation(),
        K_d=1.0 * umolar,
        n_H=1.0,
        dFF_max=1.0,
    )

    frequencies = [10.0, 20.0, 40.0]
    snrs = []

    for frequency in frequencies:
        _, _, snr = sensor.compute_hill_raster_noise(
            laser_power=10.0,
            frequency=frequency,
            soma_radius=10.0,
            img_width=500.0,
            flyback_time=0.001,
            reference_power=10.0,
            reference_dwell=0.001,
        )
        snrs.append(snr)

    assert snrs[0] > snrs[1] > snrs[2]

if __name__ == "__main__":
    pytest.main(["-sx", __file__])
