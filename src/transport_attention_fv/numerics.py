from __future__ import annotations

from collections.abc import Callable

import torch

from .equations import ShallowWater1D


def conservative_update_1d(state: torch.Tensor, face_flux: torch.Tensor, dt_dx: torch.Tensor | float) -> torch.Tensor:
    scale = torch.as_tensor(dt_dx, dtype=state.dtype, device=state.device)
    while scale.ndim < state.ndim:
        scale = scale.unsqueeze(-1)
    return state - scale * (face_flux - torch.roll(face_flux, 1, dims=-2 if state.ndim > 2 else -1))


def conservative_update_2d(
    state: torch.Tensor,
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
    dt_dx: torch.Tensor | float,
    dt_dy: torch.Tensor | float,
) -> torch.Tensor:
    sx = torch.as_tensor(dt_dx, dtype=state.dtype, device=state.device)
    sy = torch.as_tensor(dt_dy, dtype=state.dtype, device=state.device)
    while sx.ndim < state.ndim:
        sx = sx.unsqueeze(-1)
        sy = sy.unsqueeze(-1)
    return state - sx * (flux_x - torch.roll(flux_x, 1, dims=-1)) - sy * (
        flux_y - torch.roll(flux_y, 1, dims=-2)
    )


def burgers_godunov_flux(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    shock = left > right
    speed = 0.5 * (left + right)
    shock_flux = torch.where(speed >= 0, 0.5 * left.square(), 0.5 * right.square())
    rare_flux = torch.where(left >= 0, 0.5 * left.square(), torch.where(right <= 0, 0.5 * right.square(), 0.0))
    return torch.where(shock, shock_flux, rare_flux)


def _weno5_left(v: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    vm2, vm1, v0, vp1, vp2 = [torch.roll(v, s, dims=dim) for s in (2, 1, 0, -1, -2)]
    p0 = (2 * vm2 - 7 * vm1 + 11 * v0) / 6
    p1 = (-vm1 + 5 * v0 + 2 * vp1) / 6
    p2 = (2 * v0 + 5 * vp1 - vp2) / 6
    b0 = 13 / 12 * (vm2 - 2 * vm1 + v0).square() + 0.25 * (vm2 - 4 * vm1 + 3 * v0).square()
    b1 = 13 / 12 * (vm1 - 2 * v0 + vp1).square() + 0.25 * (vm1 - vp1).square()
    b2 = 13 / 12 * (v0 - 2 * vp1 + vp2).square() + 0.25 * (3 * v0 - 4 * vp1 + vp2).square()
    a0, a1, a2 = 0.1 / (eps + b0).square(), 0.6 / (eps + b1).square(), 0.3 / (eps + b2).square()
    total = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / total


def weno5_states(v: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    left = _weno5_left(v, dim)
    flipped = torch.flip(v, dims=(dim,))
    right_flipped = _weno5_left(flipped, dim)
    right = torch.roll(torch.flip(right_flipped, dims=(dim,)), -1, dims=dim)
    return left, right


def burgers_weno5_flux_1d(state: torch.Tensor) -> torch.Tensor:
    left, right = weno5_states(state, -1)
    return burgers_godunov_flux(left, right)


def _choose_smooth(candidates: list[torch.Tensor], indicators: list[torch.Tensor]) -> torch.Tensor:
    values = torch.stack(candidates, 0)
    choice = torch.stack(indicators, 0).argmin(0, keepdim=True)
    return torch.gather(values, 0, choice).squeeze(0)


def burgers_eno3_flux_1d(state: torch.Tensor) -> torch.Tensor:
    """Historical ENO-type minimum-Jiang-Shu-indicator variant (not recursive ENO).

    The name is retained for production-result compatibility; see
    docs/REPRODUCIBILITY.md for the separate paper ENO-3 convergence check.
    """
    im2, im1, i0, ip1, ip2, ip3 = [torch.roll(state, shift, -1) for shift in (2, 1, 0, -1, -2, -3)]
    q0 = im2 / 3 - 7 * im1 / 6 + 11 * i0 / 6
    q1 = -im1 / 6 + 5 * i0 / 6 + ip1 / 3
    q2 = i0 / 3 + 5 * ip1 / 6 - ip2 / 6
    b0 = 13 / 12 * (im2 - 2 * im1 + i0).square() + 0.25 * (im2 - 4 * im1 + 3 * i0).square()
    b1 = 13 / 12 * (im1 - 2 * i0 + ip1).square() + 0.25 * (im1 - ip1).square()
    b2 = 13 / 12 * (i0 - 2 * ip1 + ip2).square() + 0.25 * (3 * i0 - 4 * ip1 + ip2).square()
    left = _choose_smooth([q0, q1, q2], [b0, b1, b2])
    r0 = 11 * ip1 / 6 - 7 * ip2 / 6 + ip3 / 3
    r1 = i0 / 3 + 5 * ip1 / 6 - ip2 / 6
    r2 = -im1 / 6 + 5 * i0 / 6 + ip1 / 3
    rb0 = 13 / 12 * (ip3 - 2 * ip2 + ip1).square() + 0.25 * (ip3 - 4 * ip2 + 3 * ip1).square()
    rb1 = 13 / 12 * (ip2 - 2 * ip1 + i0).square() + 0.25 * (ip2 - i0).square()
    rb2 = 13 / 12 * (ip1 - 2 * i0 + im1).square() + 0.25 * (3 * ip1 - 4 * i0 + im1).square()
    right = _choose_smooth([r0, r1, r2], [rb0, rb1, rb2])
    return burgers_godunov_flux(left, right)


def burgers_rusanov_flux(left: torch.Tensor, right: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    beta = torch.as_tensor(beta, dtype=left.dtype, device=left.device)
    while beta.ndim < left.ndim:
        beta = beta.unsqueeze(-1)
    return 0.25 * beta * (left.square() + right.square()) - 0.5 * beta.abs() * torch.maximum(
        left.abs(), right.abs()
    ) * (right - left)


def burgers_weno5_rhs_2d(
    state: torch.Tensor, bx: torch.Tensor | float, by: torch.Tensor | float, dx: float, dy: float
) -> torch.Tensor:
    lx, rx = weno5_states(state, -1)
    ly, ry = weno5_states(state, -2)
    fx = burgers_rusanov_flux(lx, rx, bx)
    fy = burgers_rusanov_flux(ly, ry, by)
    return -(fx - torch.roll(fx, 1, -1)) / dx - (fy - torch.roll(fy, 1, -2)) / dy


def shallow_water_hll_flux(left: torch.Tensor, right: torch.Tensor, equation: ShallowWater1D) -> torch.Tensor:
    fl, fr = equation.flux(left), equation.flux(right)
    lm, lp = equation.wave_speeds(left)
    rm, rp = equation.wave_speeds(right)
    sl = torch.minimum(lm, rm).minimum(torch.zeros_like(lm))
    sr = torch.maximum(lp, rp).maximum(torch.zeros_like(lp))
    denom = (sr - sl).clamp_min(1.0e-12)
    middle = (sr[..., None] * fl - sl[..., None] * fr + (sl * sr)[..., None] * (right - left)) / denom[..., None]
    return torch.where((sl >= 0)[..., None], fl, torch.where((sr <= 0)[..., None], fr, middle))


def shallow_water_weno5_flux(state: torch.Tensor, equation: ShallowWater1D) -> torch.Tensor:
    components = []
    for component in range(2):
        left, right = weno5_states(state[..., component], -1)
        components.append((left, right))
    ul = torch.stack([item[0] for item in components], -1)
    ur = torch.stack([item[1] for item in components], -1)
    ul[..., 0].clamp_(min=equation.depth_floor)
    ur[..., 0].clamp_(min=equation.depth_floor)
    return shallow_water_hll_flux(ul, ur, equation)


def ssprk3_step(state: torch.Tensor, dt: float, rhs: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    y1 = state + dt * rhs(state)
    y2 = 0.75 * state + 0.25 * (y1 + dt * rhs(y1))
    return state / 3 + 2 * (y2 + dt * rhs(y2)) / 3


def integrate_to_times(
    initial: torch.Tensor,
    output_times: list[float],
    rhs: Callable[[torch.Tensor], torch.Tensor],
    stable_dt: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    state = initial.clone()
    outputs = [state.clone()]
    time = 0.0
    for target in output_times[1:]:
        while time < target - 1.0e-14:
            dt = min(stable_dt(state), target - time)
            state = ssprk3_step(state, dt, rhs)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Reference integration became non-finite")
            time += dt
        outputs.append(state.clone())
    return torch.stack(outputs)
