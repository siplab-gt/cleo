def pytest_addoption(parser):
    parser.addoption("--seeds", help="run N different random seeds")


def pytest_generate_tests(metafunc):
    if "rand_seed" in metafunc.fixturenames:
        seeds = metafunc.config.getoption("seeds")
        if seeds:
            metafunc.parametrize("rand_seed", range(int(seeds)))
        else:
            metafunc.parametrize("rand_seed", [421])
