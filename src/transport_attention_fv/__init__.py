"""CFL-conditioned attention fluxes for conservative finite-volume solvers."""

from .equations import InviscidBurgers1D, ScalarBurgers2D, ShallowWater1D
from .models import (
    BurgersFaceAttention1D,
    BurgersFaceAttention2D,
    ShallowWaterFaceAttention1D,
)

__all__ = [
    "InviscidBurgers1D",
    "ScalarBurgers2D",
    "ShallowWater1D",
    "BurgersFaceAttention1D",
    "BurgersFaceAttention2D",
    "ShallowWaterFaceAttention1D",
]

__version__ = "0.1.0"
