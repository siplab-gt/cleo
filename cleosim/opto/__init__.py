"""Contains opsin models, light sources, and some parameters"""
# TODO: import from light too
# import stuff here as in electrodes.__init__ so that the user
# can import directly from opto
from cleosim.opto.opsins import (
    FourStateModel,
    ProportionalCurrentModel,
    ChR2_four_state
)