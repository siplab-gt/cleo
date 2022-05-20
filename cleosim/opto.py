"""Contains opsin models, light sources, and some parameters"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any

from brian2 import Synapses, NeuronGroup
from brian2.units import (
    mm,
    mm2,
    nmeter,
    meter,
    kgram,
    Quantity,
    second,
    ms,
    second,
    psiemens,
    mV,
    volt,
    amp,
    mwatt,
)
from brian2.units.allunits import meter2, radian
import brian2.units.unitsafefunctions as usf
from brian2.core.base import BrianObjectException
import numpy as np

from cleosim.stimulators import Stimulator

