from __future__ import annotations

import math

import torch


def _periodic_overlap(left: torch.Tensor, right: torch.Tensor, start: float, width: float) -> torch.Tensor:
    """Exact overlap of periodic interval [start,start+width) with cells [left,right)."""
    result = torch.zeros_like(left)
    for shift in (-1.0, 0.0, 1.0):
        lo = max(start + shift, 0.0)
        hi = min(start + width + shift, 1.0)
        if hi > lo:
            result += (torch.minimum(right, torch.as_tensor(hi, dtype=right.dtype)) -
                       torch.maximum(left, torch.as_tensor(lo, dtype=left.dtype))).clamp_min(0)
    return result


def periodic_top_hats_1d(
    nx: int,
    generator: torch.Generator,
    count_range: tuple[int, int] = (2, 6),
    width_range: tuple[float, float] = (0.05, 0.25),
    amplitude_range: tuple[float, float] = (-1.5, 1.5),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    edges = torch.linspace(0, 1, nx + 1, dtype=dtype)
    left, right = edges[:-1], edges[1:]
    state = torch.zeros(nx, dtype=dtype)
    count = int(torch.randint(count_range[0], count_range[1] + 1, (1,), generator=generator))
    for _ in range(count):
        start = float(torch.rand((), generator=generator))
        width = width_range[0] + (width_range[1] - width_range[0]) * float(
            torch.rand((), generator=generator)
        )
        amplitude = amplitude_range[0] + (amplitude_range[1] - amplitude_range[0]) * float(
            torch.rand((), generator=generator)
        )
        state += amplitude * _periodic_overlap(left, right, start, width) * nx
    return state


def periodic_rectangles_2d(
    ny: int,
    nx: int,
    generator: torch.Generator,
    count_range: tuple[int, int] = (2, 6),
    width_range: tuple[float, float] = (0.05, 0.25),
    target_amplitude_range: tuple[float, float] = (0.5, 1.5),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    x_edges = torch.linspace(0, 1, nx + 1, dtype=dtype)
    y_edges = torch.linspace(0, 1, ny + 1, dtype=dtype)
    state = torch.zeros((ny, nx), dtype=dtype)
    count = int(torch.randint(count_range[0], count_range[1] + 1, (1,), generator=generator))
    for _ in range(count):
        x0, y0 = float(torch.rand((), generator=generator)), float(torch.rand((), generator=generator))
        wx = width_range[0] + (width_range[1] - width_range[0]) * float(torch.rand((), generator=generator))
        wy = width_range[0] + (width_range[1] - width_range[0]) * float(torch.rand((), generator=generator))
        amp = 2.0 * float(torch.rand((), generator=generator)) - 1.0
        ox = _periodic_overlap(x_edges[:-1], x_edges[1:], x0, wx) * nx
        oy = _periodic_overlap(y_edges[:-1], y_edges[1:], y0, wy) * ny
        state += amp * oy[:, None] * ox[None, :]
    target = target_amplitude_range[0] + (target_amplitude_range[1] - target_amplitude_range[0]) * float(
        torch.rand((), generator=generator)
    )
    return state * (target / state.abs().amax().clamp_min(1.0e-12))


def shallow_water_segments_1d(
    nx: int,
    generator: torch.Generator,
    count_range: tuple[int, int] = (2, 6),
    width_range: tuple[float, float] = (0.05, 0.25),
    depth_range: tuple[float, float] = (0.5, 1.5),
    velocity_range: tuple[float, float] = (-0.5, 0.5),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    h = torch.ones(nx, dtype=dtype)
    velocity = torch.zeros(nx, dtype=dtype)
    edges = torch.linspace(0, 1, nx + 1, dtype=dtype)
    count = int(torch.randint(count_range[0], count_range[1] + 1, (1,), generator=generator))
    for _ in range(count):
        start = float(torch.rand((), generator=generator))
        width = width_range[0] + (width_range[1] - width_range[0]) * float(torch.rand((), generator=generator))
        coverage = _periodic_overlap(edges[:-1], edges[1:], start, width) * nx
        depth_value = depth_range[0] + (depth_range[1] - depth_range[0]) * float(torch.rand((), generator=generator))
        velocity_value = velocity_range[0] + (velocity_range[1] - velocity_range[0]) * float(
            torch.rand((), generator=generator)
        )
        h = h * (1 - coverage) + depth_value * coverage
        velocity = velocity * (1 - coverage) + velocity_value * coverage
    return torch.stack((h, h * velocity), dim=-1)


def random_direction(generator: torch.Generator) -> tuple[float, float]:
    angle = 2 * math.pi * float(torch.rand((), generator=generator))
    bx, by = math.cos(angle), math.sin(angle)
    scale = abs(bx) + abs(by)
    return bx / scale, by / scale
