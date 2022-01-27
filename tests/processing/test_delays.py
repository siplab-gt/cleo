"""Test basic Delay classes in processing.delays.py"""

import numpy as np
import pytest

from cleosim.processing import ConstantDelay, GaussianDelay


def test_ConstantDelay():
    assert ConstantDelay(1).compute() == 1
    assert ConstantDelay(5).compute() == 5


def test_GaussianDelay():
    np.random.seed(1820)
    assert GaussianDelay(1, 0.1).compute() == 0.9398680594942359
    assert GaussianDelay(1, 0.1).compute() == 0.9611997124217544
    with pytest.warns(UserWarning):
        assert GaussianDelay(-1, 0.01).compute() == 0
    assert GaussianDelay(5, 1).compute() == 3.9169426457170893
    assert GaussianDelay(5, 1).compute() == 4.636290455175261
