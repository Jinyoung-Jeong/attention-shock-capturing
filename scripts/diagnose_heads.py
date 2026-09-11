#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from transport_attention_fv.data import TrajectoryDataset
from transport_attention_fv.models import BurgersFaceAttention1D, build_model
from transport_attention_fv.numerics import conservative_update_1d
from transport_attention_fv.utils import load_tensor_file, load_yaml, resolve_path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid(state: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(error) & torch.isfinite(state).reshape(state.shape[0], -1).all(1) & (
        state.reshape(state.shape[0], -1).abs().amax(1) < 1.0e6
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the functional contribution of each Burgers attention head"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfl", type=float, nargs="*", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if config["problem"] != "burgers_1d":
        raise ValueError("Head diagnostics are defined for the central 1D Burgers experiment")
    device = torch.device(args.device)
    root = resolve_path(args.config, config["output"]["directory"]) / "burgers_1d"
    run = root / f"seed_{args.seed}"
    checkpoint = run / "checkpoint_best_large_step.pt"
    payload = load_tensor_file(checkpoint, map_location=device)
    model = build_model("burgers_1d", config["model"]).to(device)
    if not isinstance(model, BurgersFaceAttention1D):
        raise TypeError("The requested checkpoint is not an attention model")
    model.load_state_dict(payload["model"])
    model.eval()

    dataset = TrajectoryDataset(
        resolve_path(args.config, config["data"]["directory"]) / "burgers_1d" / "test.pt"
    )
    loader = DataLoader(
        dataset, batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False, num_workers=0,
    )
    base_cfl = float(config["data"]["base_cfl"])
    final_frame = int(config["data"]["base_frames"])
    cfl_values = args.cfl or [
        min(float(value) for value in config["evaluation"]["cfl_values"]),
        float(config["evaluation"].get("diagnostic_cfl", 1.6)),
    ]
    conditions: list[tuple[str, int | None]] = [("learned", None)]
    for intervention in ("uniform_attention", "zero_context"):
        conditions.extend((intervention, head) for head in range(model.attention.heads))

    rows: list[dict] = []
    for cfl in cfl_values:
        stride = int(round(cfl / base_cfl))
        if abs(stride * base_cfl - cfl) > 1.0e-8 or final_frame % stride:
            raise ValueError(f"CFL {cfl} does not reach the common final frame")
        steps = final_frame // stride
        accumulators = {
            condition: {"errors": [], "flux_changes": [], "failures": 0}
            for condition in conditions
        }
        for raw in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
            initial = batch["states"][:, 0]
            target = batch["states"][:, final_frame]
            dt = batch["dt_base"] * stride
            scale = dt * initial.shape[-1]
            learned_flux = model(initial, scale)
            flux_norm = learned_flux.reshape(initial.shape[0], -1).square().mean(1).sqrt().clamp_min(1.0e-12)

            for condition in conditions:
                mode, head = condition
                state = initial.clone()
                first_flux = None
                for _ in range(steps):
                    flux = model(
                        state, scale,
                        head_index=head,
                        head_intervention="none" if mode == "learned" else mode,
                    )
                    if first_flux is None:
                        first_flux = flux
                    state = conservative_update_1d(state, flux, scale)
                error = (state - target).reshape(state.shape[0], -1).square().mean(1).sqrt()
                valid = _valid(state, error)
                store = accumulators[condition]
                store["failures"] += int((~valid).sum())
                store["errors"].extend(error[valid].cpu().tolist())
                relative_flux_change = (
                    (first_flux - learned_flux).reshape(initial.shape[0], -1).square().mean(1).sqrt()
                    / flux_norm
                )
                finite_flux = torch.isfinite(relative_flux_change)
                store["flux_changes"].extend(relative_flux_change[finite_flux].cpu().tolist())

        baseline_error = float(np.median(accumulators[("learned", None)]["errors"]))
        for mode, head in conditions:
            store = accumulators[(mode, head)]
            median_error = float(np.median(store["errors"])) if store["errors"] else math.inf
            successes = len(dataset) - int(store["failures"])
            rows.append({
                "seed": args.seed,
                "cfl": cfl,
                "sequential_steps": steps,
                "intervention": mode,
                "head": "all" if head is None else head,
                "samples": len(dataset),
                "successes": successes,
                "failures": int(store["failures"]),
                "success_rate": successes / len(dataset),
                "median_l2_successful": median_error,
                "relative_l2_change_from_learned": (
                    median_error / baseline_error - 1.0
                    if math.isfinite(median_error) and baseline_error > 0 else math.inf
                ),
                "median_relative_first_step_flux_change": float(np.median(store["flux_changes"])),
            })

    _write_csv(run / "diagnostics" / "head_interventions.csv", rows)
    print(f"head diagnostics: {run / 'diagnostics' / 'head_interventions.csv'}")


if __name__ == "__main__":
    main()
