#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transport_attention_fv.data import TrajectoryDataset
from transport_attention_fv.equations import ShallowWater1D
from transport_attention_fv.numerics import (
    burgers_weno5_rhs_2d,
    integrate_to_times,
    shallow_water_weno5_flux,
)
from transport_attention_fv.utils import load_yaml, resolve_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-time reference-grid self-convergence audit")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    problem = config["problem"]
    output_root = resolve_path(args.config, config["output"]["directory"]).parent
    output = args.output or output_root / "reference_audits" / f"{problem}.json"
    if problem == "burgers_1d":
        save_json(output, {
            "problem": problem,
            "reference": "periodic Lax--Hopf cell averages",
            "numerical_refinement": None,
        })
        return

    device = torch.device(args.device)
    grid = int(config["data"]["grid_size"])
    reference_cfl = float(config["data"]["reference_cfl"])
    split_results = {}
    for split in ("train", "test"):
        dataset = TrajectoryDataset(
            resolve_path(args.config, config["data"]["directory"]) / problem / f"{split}.pt"
        )
        stored_factor = int(config["data"].get(
            f"reference_factor_{split}", config["data"]["reference_factor"]
        ))
        audit_factor = 2 * stored_factor
        errors_l1, errors_l2 = [], []
        for index in range(min(args.samples, len(dataset))):
            item = dataset[index]
            initial = item["states"][0].double()
            terminal_time = float(item["dt_base"]) * int(config["data"]["base_frames"])
            if problem == "burgers_2d":
                high = initial.repeat_interleave(audit_factor, 0).repeat_interleave(audit_factor, 1).to(device)
                reference_grid = grid * audit_factor
                spacing = 1.0 / reference_grid
                bx, by = float(item["beta_x"]), float(item["beta_y"])
                rhs = lambda u: burgers_weno5_rhs_2d(u, bx, by, spacing, spacing)
                stable = lambda u: reference_cfl * spacing / float(u.abs().amax().clamp_min(1.0e-6))
                final = integrate_to_times(high, [0.0, terminal_time], rhs, stable)[-1]
                coarse = final.reshape(grid, audit_factor, grid, audit_factor).mean((1, 3)).cpu()
            else:
                equation = ShallowWater1D(gravity=float(config["data"].get("gravity", 1.0)))
                high = initial.repeat_interleave(audit_factor, 0).to(device)
                reference_grid = grid * audit_factor
                spacing = 1.0 / reference_grid

                def rhs(u: torch.Tensor) -> torch.Tensor:
                    face_flux = shallow_water_weno5_flux(u[None], equation)[0]
                    return -(face_flux - torch.roll(face_flux, 1, 0)) / spacing

                stable = lambda u: reference_cfl * spacing / float(equation.max_speed(u).clamp_min(1.0e-6))
                final = integrate_to_times(high, [0.0, terminal_time], rhs, stable)[-1]
                coarse = final.reshape(grid, audit_factor, 2).mean(1).cpu()
            difference = (coarse - item["states"][-1].double()).reshape(-1)
            errors_l1.append(float(difference.abs().mean()))
            errors_l2.append(float(difference.square().mean().sqrt()))
        split_results[split] = {
            "samples": len(errors_l1),
            "stored_reference_factor": stored_factor,
            "audit_reference_factor": audit_factor,
            "final_time_l1_per_sample": errors_l1,
            "final_time_l2_per_sample": errors_l2,
            "median_l1": float(torch.tensor(errors_l1).quantile(0.5)),
            "median_l2": float(torch.tensor(errors_l2).quantile(0.5)),
        }
    save_json(output, {
        "problem": problem,
        "interpretation": (
            "The train split measures label-fidelity error at the training reference factor; "
            "the test split measures the error floor of the reported evaluation reference."
        ),
        "splits": split_results,
    })
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
