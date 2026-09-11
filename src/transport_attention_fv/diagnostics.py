from __future__ import annotations

import torch


def subcell_shock_position_1_to_0(state: torch.Tensor, expected: float) -> float:
    """Locate a 1-to-0 Burgers shock by interpolating its u=0.5 crossing."""
    nx = state.numel()
    dx = 1.0 / nx
    right = torch.roll(state, -1)
    centers = (torch.arange(nx, device=state.device, dtype=state.dtype) + 0.5) * dx
    distance = torch.remainder(centers - expected + 0.5, 1.0) - 0.5
    falling = state - right
    score = torch.where(distance.abs() <= 0.15, falling, torch.full_like(falling, -torch.inf))
    index = int(score.argmax())
    denominator = float((state[index] - right[index]).abs())
    fraction = 0.5 if denominator < 1.0e-12 else float(
        (state[index] - 0.5) / (state[index] - right[index])
    )
    return float((centers[index] + max(0.0, min(1.0, fraction)) * dx) % 1.0)


def burgers_rarefaction_cell_averages(
    nx: int, time: float, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Cell averages of the entropy fan for u_L=-1, u_R=1 centered at x=0.5."""
    edges = torch.linspace(0.0, 1.0, nx + 1, device=device, dtype=dtype) - 0.5

    def primitive(z: torch.Tensor) -> torch.Tensor:
        return torch.where(
            z <= -time,
            -z - 0.5 * time,
            torch.where(z >= time, z - 0.5 * time, z.square() / (2.0 * time)),
        )

    return (primitive(edges[1:]) - primitive(edges[:-1])) * nx


def attention_region_statistics(
    state: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    cfl: float,
    shock_fraction: float = 0.3,
    smooth_fraction: float = 0.02,
) -> list[dict]:
    """Summarize head-wise attention on oracle shock and smooth face masks.

    ``weights`` must have shape ``(batch, faces, candidates, heads)``.
    """
    if weights.ndim != 4 or weights.shape[:2] != state.shape:
        raise ValueError((state.shape, weights.shape))
    if weights.shape[-2] != offsets.numel():
        raise ValueError((weights.shape, offsets.shape))
    face_mean = 0.5 * (state + torch.roll(state, -1, -1))
    jump = (torch.roll(state, -1, -1) - state).abs()
    maximum_jump = jump.amax(dim=1, keepdim=True).clamp_min(1.0e-12)
    masks = {
        "shock": jump >= shock_fraction * maximum_jump,
        "smooth": jump <= smooth_fraction * maximum_jump,
    }
    probabilities = weights.clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.log()).sum(-2)
    left = weights[..., offsets < 0, :].sum(-2)
    right = weights[..., offsets > 0, :].sum(-2)
    asymmetry = (left - right).abs()
    direction = torch.sign(face_mean)[..., None]
    upwind_bias = torch.where(direction >= 0, left - right, right - left)
    center_mass = weights[..., offsets.abs() <= 0.5, :].sum(-2)
    far_mass = 1.0 - center_mass
    center_of_mass = (weights * offsets.view(1, 1, -1, 1)).sum(-2)
    upwind_com = -direction * center_of_mass
    moving = face_mean.abs() > 1.0e-8

    rows = []
    heads = weights.shape[-1]
    for region, region_mask in masks.items():
        for head in [*range(heads), "mean"]:
            def head_view(value: torch.Tensor) -> torch.Tensor:
                return value.mean(-1) if head == "mean" else value[..., int(head)]

            selected = region_mask
            selected_moving = selected & moving
            count = int(selected.sum())
            moving_count = int(selected_moving.sum())

            def selected_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
                chosen = head_view(value)[mask]
                return float(chosen.mean()) if chosen.numel() else float("nan")

            rows.append({
                "cfl": cfl,
                "region": region,
                "head": str(head),
                "faces": count,
                "moving_faces": moving_count,
                "attention_entropy": selected_mean(entropy, selected),
                "left_right_asymmetry": selected_mean(asymmetry, selected),
                "upwind_bias": selected_mean(upwind_bias, selected_moving),
                "center_mass": selected_mean(center_mass, selected),
                "far_mass": selected_mean(far_mass, selected),
                "absolute_center_of_mass": selected_mean(center_of_mass.abs(), selected),
                "upwind_center_of_mass": selected_mean(upwind_com, selected_moving),
            })
    return rows
