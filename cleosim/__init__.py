from abc import ABC, abstractmethod

from brian2 import NeuronGroup, Subgroup, Network, NetworkOperation, defaultclock, ms

# bring nested modules up to second level
from cleosim.base import *
from cleosim.processing import controllers, delays, observers

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
