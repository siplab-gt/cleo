"""Contains opsin models, light sources, and some parameters"""
from cleo.opto.registry import (
    lor_for_sim,
    LightOpsinRegistry,
)
from cleo.opto.opsins import (
    FourStateOpsin,
    ProportionalCurrentOpsin,
    ChR2_four_state,
    Opsin,
)
from cleo.opto.light import (
    Light,
    fiber473nm,
    FiberModel,
    LightModel,
)
