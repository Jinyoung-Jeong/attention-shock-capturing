from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InviscidBurgers1D:
    """u_t + (u^2/2)_x = 0."""

    def flux(self, u: torch.Tensor) -> torch.Tensor:
        return 0.5 * u.square()

    def max_speed(self, u: torch.Tensor) -> torch.Tensor:
        return u.abs().amax()

    def lax_hopf_cell_average(
        self, u0: torch.Tensor, t: float, quadrature_points: int = 8
    ) -> torch.Tensor:
        """Periodic Lax--Hopf entropy solution represented by cell averages."""
        if t == 0:
            return u0.clone()
        nx = u0.numel()
        dx = 1.0 / nx
        centers = (torch.arange(nx, dtype=u0.dtype, device=u0.device) + 0.5) * dx
        offsets = (torch.arange(quadrature_points, dtype=u0.dtype, device=u0.device) + 0.5)
        offsets = (offsets / quadrature_points - 0.5) * dx
        samples = [self._lax_hopf_pointwise(u0, centers + offset, t) for offset in offsets]
        return torch.stack(samples).mean(0)

    @staticmethod
    def _lax_hopf_pointwise(u0: torch.Tensor, query: torch.Tensor, t: float) -> torch.Tensor:
        nx = u0.numel()
        dx = 1.0 / nx
        tiles = 5
        extended = u0.repeat(tiles)
        left = -2.0 + torch.arange(tiles * nx, dtype=u0.dtype, device=u0.device) * dx
        right = left + dx
        primitive = torch.zeros_like(extended)
        primitive[1:] = torch.cumsum(extended[:-1] * dx, 0)
        stationary = query[:, None] - extended[None] * t
        minimizer = torch.maximum(torch.minimum(stationary, right[None]), left[None])
        cost = primitive[None] + extended[None] * (minimizer - left[None])
        cost = cost + (query[:, None] - minimizer).square() / (2 * t)
        index = cost.argmin(1)
        optimum = minimizer[torch.arange(query.numel(), device=query.device), index]
        return (query - optimum) / t


@dataclass(frozen=True)
class ScalarBurgers2D:
    """u_t + (beta_x u^2/2)_x + (beta_y u^2/2)_y = 0."""

    beta_x: float = 1.0
    beta_y: float = 0.0

    def flux_x(self, u: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.beta_x * u.square()

    def flux_y(self, u: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.beta_y * u.square()

    def spectral_rate(self, u: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        return u.abs().amax() * (abs(self.beta_x) / dx + abs(self.beta_y) / dy)


@dataclass(frozen=True)
class ShallowWater1D:
    """Wet-bed, flat-bottom shallow-water system with U=(h,q)."""

    gravity: float = 1.0
    depth_floor: float = 1.0e-8

    def primitive(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = state[..., 0].clamp_min(self.depth_floor)
        return h, state[..., 1] / h

    def flux(self, state: torch.Tensor) -> torch.Tensor:
        h, velocity = self.primitive(state)
        q = state[..., 1]
        return torch.stack((q, q * velocity + 0.5 * self.gravity * h.square()), dim=-1)

    def wave_speeds(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, velocity = self.primitive(state)
        c = torch.sqrt(self.gravity * h)
        return velocity - c, velocity + c

    def max_speed(self, state: torch.Tensor) -> torch.Tensor:
        minus, plus = self.wave_speeds(state)
        return torch.maximum(minus.abs(), plus.abs()).amax()
