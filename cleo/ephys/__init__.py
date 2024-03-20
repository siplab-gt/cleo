"""Contains probes, convenience functions for generating electrode array coordinates,
signals, spiking, and LFP"""
from cleo.ephys.lfp import (
    LFPSignalBase,
    RWSLFPSignalBase,
    RWSLFPSignalFromPSCs,
    RWSLFPSignalFromSpikes,
    TKLFPSignal,
)
from cleo.ephys.probes import (
    Probe,
    Signal,
    linear_shank_coords,
    poly2_shank_coords,
    poly3_shank_coords,
    tetrode_shank_coords,
    tile_coords,
)
from cleo.ephys.spiking import MultiUnitSpiking, SortedSpiking, Spiking
