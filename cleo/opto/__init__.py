"""Contains opsin models, light sources, and some parameters"""
from cleo.opto.registry import (
    lor_for_sim,
    LightOpsinRegistry,
)
from cleo.opto.opsins import (
    FourStateOpsin,
    ProportionalCurrentOpsin,
    Opsin,
    BansalFourStateOpsin,
    BansalThreeStatePump,
    linear_interpolator,
    cubic_interpolator,
    plot_action_spectra,
)
from cleo.opto.opsin_library import (
    chr2_4s,
    chr2_b4s,
    vfchrimson_4s,
    chrimson_4s,
    gtacr2_4s,
)
from cleo.opto.light import (
    Light,
    fiber473nm,
    FiberModel,
    LightModel,
)
