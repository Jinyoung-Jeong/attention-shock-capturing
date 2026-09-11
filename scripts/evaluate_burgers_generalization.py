#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from transport_attention_fv.equations import InviscidBurgers1D
from transport_attention_fv.initial_conditions import periodic_top_hats_1d
from transport_attention_fv.models import build_model
from transport_attention_fv.numerics import conservative_update_1d
from transport_attention_fv.utils import load_tensor_file, load_yaml, resolve_path


def _smooth_sine_cell_averages(nx: int, generator: torch.Generator) -> torch.Tensor:
    base = 0.75 + 0.25 * float(torch.rand((), generator=generator))
    amplitude = 0.2 + 0.3 * float(torch.rand((), generator=generator))
    phase = float(torch.rand((), generator=generator))
    edges = torch.linspace(0.0, 1.0, nx + 1, dtype=torch.float64)
    shifted = 2.0 * math.pi * (edges - phase)
    sine_average = (torch.cos(shifted[:-1]) - torch.cos(shifted[1:])) * nx / (2.0 * math.pi)
    return base + amplitude * sine_average


def _initial_condition(case: str, nx: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if case == "in_distribution_top_hats":
        return periodic_top_hats_1d(nx, generator)
    if case == "single_top_hat":
        return periodic_top_hats_1d(nx, generator, count_range=(1, 1))
    if case == "narrow_top_hats":
        return periodic_top_hats_1d(
            nx, generator, count_range=(2, 6), width_range=(0.02, 0.05)
        )
    if case == "smooth_sine_pre_shock":
        return _smooth_sine_cell_averages(nx, generator)
    raise ValueError(case)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference-only grid, IC-family, and horizon transfer for 1D Burgers"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--reference-grid", type=int, default=256)
    parser.add_argument("--reference-base-frames", type=int, default=80)
    parser.add_argument("--quadrature", type=int, default=16)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if config["problem"] != "burgers_1d":
        raise ValueError("This diagnostic is defined for burgers_1d")
    device = torch.device(args.device)
    run = resolve_path(args.config, config["output"]["directory"]) / "burgers_1d" / f"seed_{args.seed}"
    checkpoint = run / "checkpoint_best_large_step.pt"
    payload = load_tensor_file(checkpoint, map_location=device)
    model = build_model("burgers_1d", config["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    equation = InviscidBurgers1D()
    base_cfl = float(config["data"]["base_cfl"])
    cases = (
        "in_distribution_top_hats", "single_top_hat", "narrow_top_hats",
        "smooth_sine_pre_shock",
    )
    cfl_values = (0.4, 1.6)
    horizon_multipliers = (1, 2)
    sample_rows: list[dict] = []

    for case in cases:
        case_samples = 1 if case == "smooth_sine_pre_shock" else args.samples
        for nx in args.grid_sizes:
            for sample in range(case_samples):
                ic_seed = 41000 + sample
                initial = _initial_condition(case, nx, ic_seed).float().to(device)
                maximum_speed = float(initial.abs().amax().clamp_min(1.0e-6))
                for cfl in cfl_values:
                    dt = cfl / (nx * maximum_speed)
                    for horizon_multiplier in horizon_multipliers:
                        terminal_time = (
                            horizon_multiplier * args.reference_base_frames * base_cfl
                            / (args.reference_grid * maximum_speed)
                        )
                        steps_float = terminal_time / dt
                        steps = int(round(steps_float))
                        if abs(steps - steps_float) > 1.0e-8:
                            raise ValueError(
                                f"non-integer rollout: nx={nx}, cfl={cfl}, horizon={horizon_multiplier}"
                            )
                        scale = torch.tensor([dt * nx], device=device)
                        state = initial[None]
                        initial_mass = float(state.sum())
                        for _ in range(steps):
                            flux = model(state, scale)
                            state = conservative_update_1d(state, flux, scale)
                        reference = equation.lax_hopf_cell_average(
                            initial, terminal_time, args.quadrature
                        ).to(device)
                        difference = state[0] - reference
                        finite = bool(torch.isfinite(state).all()) and float(state.abs().max()) < 1.0e6
                        sample_rows.append({
                            "seed": args.seed,
                            "case": case,
                            "grid_size": nx,
                            "cfl": cfl,
                            "horizon_multiplier": horizon_multiplier,
                            "sample": sample,
                            "steps": steps,
                            "terminal_time": terminal_time,
                            "finite": finite,
                            "l1": float(difference.abs().mean()) if finite else math.inf,
                            "l2": float(difference.square().mean().sqrt()) if finite else math.inf,
                            "linf": float(difference.abs().max()) if finite else math.inf,
                            "conservation_error": abs(float(state.sum()) - initial_mass) if finite else math.inf,
                        })

    output = run / "generalization"
    _write_csv(output / "per_sample.csv", sample_rows)
    summary_rows = []
    keys = sorted({
        (row["case"], row["grid_size"], row["cfl"], row["horizon_multiplier"])
        for row in sample_rows
    })
    for case, nx, cfl, horizon in keys:
        group = [
            row for row in sample_rows
            if (row["case"], row["grid_size"], row["cfl"], row["horizon_multiplier"])
            == (case, nx, cfl, horizon)
        ]
        valid = [row for row in group if row["finite"]]
        summary_rows.append({
            "seed": args.seed,
            "case": case,
            "grid_size": nx,
            "cfl": cfl,
            "horizon_multiplier": horizon,
            "samples": len(group),
            "success_rate": len(valid) / len(group),
            "median_l1_successful": float(np.median([row["l1"] for row in valid])) if valid else math.inf,
            "median_l2_successful": float(np.median([row["l2"] for row in valid])) if valid else math.inf,
            "median_linf_successful": float(np.median([row["linf"] for row in valid])) if valid else math.inf,
            "mean_conservation_error": float(np.mean([row["conservation_error"] for row in valid])) if valid else math.inf,
        })
    _write_csv(output / "summary.csv", summary_rows)
    print(f"generalization diagnostics: {output}")


if __name__ == "__main__":
    main()
