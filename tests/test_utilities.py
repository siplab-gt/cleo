import pytest

from cleo.utilities import rng, set_seed


def test_set_seed():
    seed = 42
    set_seed(seed)
    first_random_number = rng.random()
    print(rng.bit_generator.state)

    set_seed(seed)
    second_random_number = rng.random()

    assert first_random_number == second_random_number


if __name__ == "__main__":
    pytest.main(["-x", __file__])
