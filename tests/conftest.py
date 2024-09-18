import brian2.only as b2
import matplotlib

b2.prefs.codegen.target = "numpy"  # to avoid cython overhead for short tests
matplotlib.use("Agg")  # to avoid GUI backend


def pytest_addoption(parser):
    parser.addoption("--seeds", help="run N different random seeds")


def pytest_generate_tests(metafunc):
    if "rand_seed" in metafunc.fixturenames:
        seeds = metafunc.config.getoption("seeds")
        if seeds:
            metafunc.parametrize("rand_seed", range(int(seeds)))
        else:
            metafunc.parametrize("rand_seed", [421])
