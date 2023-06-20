"""Contains opsin models, light sources, and some parameters"""
from cleo.opto.registry import (
    lor_for_sim,
    LightOpsinRegistry,
)
from cleo.opto.opsins import (
    FourStateOpsin,
    ProportionalCurrentOpsin,
    Opsin,
)
from cleo.opto.opsin_library import (
    ChR2_4S,
    VfChrimson_4S,
    Chrimson_4S,
)
from cleo.opto.light import (
    Light,
    fiber473nm,
    FiberModel,
    LightModel,
)
