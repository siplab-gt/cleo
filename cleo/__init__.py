"""Contains core classes and functions for the Cleo package."""
from __future__ import annotations

# auto-import submodules
import cleo.coords
import cleo.ephys
import cleo.imaging
import cleo.ioproc
import cleo.opto
import cleo.recorders
import cleo.registry
import cleo.stimulators
import cleo.utilities
import cleo.viz
from cleo.base import (
    CLSimulator,
    InterfaceDevice,
    IOProcessor,
    NeoExportable,
    Recorder,
    Stimulator,
    SynapseDevice,
)
