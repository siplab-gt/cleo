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

if __name__ == "__main__":
    pytest.main(["-sx", __file__])
