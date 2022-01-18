# auto-import submodules
import cleosim.electrodes
import cleosim.opto
import cleosim.coordinates
import cleosim.stimulators
import cleosim.recorders
import cleosim.utilities
# bring nested modules up to second level
from cleosim.base import *
from cleosim.processing import controllers, delays, observers

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
