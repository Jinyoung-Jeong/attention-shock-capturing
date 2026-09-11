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
from transport_attention_fv.diagnostics import (
    attention_region_statistics,
    burgers_rarefaction_cell_averages,
    subcell_shock_position_1_to_0,
)
from transport_attention_fv.models import build_model
from transport_attention_fv.training import advance_model
from transport_attention_fv.utils import load_tensor_file, load_yaml, resolve_path, save_json


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_or_none(value: torch.Tensor | float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", default=2027, type=int)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_yaml(args.config)
    problem = config["problem"]
    device = torch.device(args.device)
    run = resolve_path(args.config, config["output"]["directory"]) / problem / f"seed_{args.seed}"
    checkpoint = args.checkpoint or run / "checkpoint_best_large_step.pt"
    payload = load_tensor_file(checkpoint, map_location=device)
    model = build_model(problem, config["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = TrajectoryDataset(resolve_path(args.config, config["data"]["directory"]) / problem / "test.pt")
    loader = DataLoader(dataset, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False)
    base_cfl = float(config["data"]["base_cfl"])
    final_frame = int(config["data"]["base_frames"])
    cfl_values = [float(value) for value in config["evaluation"]["cfl_values"]]
    diagnostic_cfl = float(config["evaluation"].get("diagnostic_cfl", max(cfl_values)))
    stride = int(round(diagnostic_cfl / base_cfl))
    if abs(stride * base_cfl - diagnostic_cfl) > 1.0e-8 or final_frame % stride:
        raise ValueError(f"diagnostic CFL {diagnostic_cfl} does not reach the common final frame")
    steps = final_frame // stride
    output = args.output or run / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    intervention_rows = []
    for mode in ("learned", "uniform", "cfl_blind", "frozen_center"):
        errors, failures = [], 0
        for raw in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
            state = batch["states"][:, 0]
            dt = batch["dt_base"] * stride
            for _ in range(steps):
                state = advance_model(problem, model, state, dt, batch, intervention=mode)
            target = batch["states"][:, final_frame]
            error = (state - target).reshape(state.shape[0], -1).square().mean(1).sqrt()
            valid = torch.isfinite(error) & (state.reshape(state.shape[0], -1).abs().amax(1) < 1.0e6)
            if problem == "shallow_water_1d":
                valid &= state[..., 0].amin(1) > 0
            failures += int((~valid).sum())
            errors.extend(error[valid].cpu().tolist())
        successes = len(dataset) - failures
        intervention_rows.append({
            "intervention": mode,
            "samples": len(dataset),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(dataset),
            "median_l2_successful": float(np.median(errors)) if errors else float("inf"),
        })
    _write_csv(output / "interventions.csv", intervention_rows)

    if problem == "burgers_1d":
        nx = int(config["data"]["grid_size"])
        positions = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device)
        reach_rows = []
        attention_region_rows = []
        for cfl in cfl_values:
            local_stride = int(round(cfl / base_cfl))
            physical, inferred = [], []
            region_states, region_weights = [], []
            for raw in loader:
                state = raw["states"][:, 0].to(device)
                scale = raw["dt_base"].to(device) * local_stride * state.shape[-1]
                _, weights = model(state, scale, return_attention=True)
                region_states.append(state.cpu())
                region_weights.append(weights.cpu())
                mean_weights = weights.mean(-1)
                reach = -(mean_weights * positions).sum(-1)
                face_mean = 0.5 * (state + torch.roll(state, -1, -1))
                target_reach = scale[:, None] * face_mean
                jump = (torch.roll(state, -1, -1) - state).abs()
                maximum_jump = jump.amax(dim=1, keepdim=True).clamp_min(1.0e-12)
                smooth = jump <= 0.02 * maximum_jump
                moving = face_mean.abs() > 1.0e-8
                mask = smooth & moving
                physical.extend(target_reach[mask].cpu().tolist())
                inferred.extend(reach[mask].cpu().tolist())
            x, y = torch.tensor(physical, dtype=torch.float64), torch.tensor(inferred, dtype=torch.float64)
            x_centered, y_centered = x - x.mean(), y - y.mean()
            slope = (x_centered * y_centered).sum() / x_centered.square().sum().clamp_min(1.0e-15)
            intercept = y.mean() - slope * x.mean()
            correlation = (x_centered * y_centered).sum() / torch.sqrt(
                x_centered.square().sum() * y_centered.square().sum()
            ).clamp_min(1.0e-15)
            reach_rows.append({
                "cfl": cfl,
                "faces": int(x.numel()),
                "face_selection": "smooth faces with nonzero mean transport speed",
                "slope": _finite_or_none(slope),
                "intercept": _finite_or_none(intercept),
                "pearson_r": _finite_or_none(correlation),
                "mean_abs_transport_reach": float(x.abs().mean()),
                "mean_abs_attention_reach": float(y.abs().mean()),
                "max_abs_attention_reach": float(y.abs().max()),
                "fraction_abs_attention_reach_ge_2_25": float((y.abs() >= 2.25).double().mean()),
            })
            attention_region_rows.extend(attention_region_statistics(
                torch.cat(region_states), torch.cat(region_weights), positions.cpu(), cfl
            ))
        _write_csv(output / "reach_regression_by_cfl.csv", reach_rows)
        _write_csv(output / "attention_region_stats_by_cfl.csv", attention_region_rows)
        selected_reach = min(reach_rows, key=lambda row: abs(float(row["cfl"]) - diagnostic_cfl))
        save_json(output / "reach_regression.json", {
            "definition": "attention reach and mean-state transport reach, both in cell widths",
            **selected_reach,
        })

        constant_rows = []
        constant_states = torch.linspace(-1.5, 1.5, 61, device=device)
        for cfl in cfl_values:
            states = constant_states[:, None].expand(-1, nx)
            scale = torch.full((states.shape[0],), cfl, device=device)
            predicted = model(states, scale).mean(-1)
            physical_flux = 0.5 * constant_states.square()
            zero_index = int(constant_states.abs().argmin())
            gauge_offset = predicted[zero_index]
            flux_one_index = int((constant_states - 1.0).abs().argmin())
            learned_speed = predicted[flux_one_index] - gauge_offset
            for index, value in enumerate(constant_states):
                constant_rows.append({
                    "cfl": cfl,
                    "state": float(value),
                    "predicted_flux_mean": float(predicted[index]),
                    "physical_flux": float(physical_flux[index]),
                    "absolute_consistency_error": float(
                        (predicted[index] - physical_flux[index]).abs()
                    ),
                    "gauge_offset_from_zero_state": float(gauge_offset),
                    "gauge_corrected_consistency_error": float(
                        ((predicted[index] - gauge_offset) - physical_flux[index]).abs()
                    ),
                    "learned_rankine_hugoniot_speed_1_to_0": float(learned_speed),
                    "relative_shock_speed_bias_percent": float((learned_speed - 0.5) / 0.5 * 100.0),
                })
        _write_csv(output / "constant_state_consistency.csv", constant_rows)

        centers = (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) / nx
        initial = ((centers >= 0.25) & (centers < 0.75)).float()[None]
        terminal_time = final_frame * base_cfl / nx
        expected_position = (0.75 + 0.5 * terminal_time) % 1.0
        shock_rows = []
        rarefaction_rows = []
        for cfl in cfl_values:
            local_stride = int(round(cfl / base_cfl))
            if abs(local_stride * base_cfl - cfl) > 1.0e-8 or final_frame % local_stride:
                raise ValueError(f"CFL {cfl} does not reach the common diagnostic time")
            local_steps = final_frame // local_stride
            local_dt = torch.tensor([cfl / nx], device=device)
            local_batch = {"states": initial[:, None], "dt_base": local_dt}
            state = initial.clone()
            for _ in range(local_steps):
                state = advance_model(problem, model, state, local_dt, local_batch)
            measured_position = subcell_shock_position_1_to_0(state[0], expected_position)
            signed_error = float(
                torch.remainder(torch.tensor(measured_position - expected_position) + 0.5, 1.0) - 0.5
            )
            displacement = float(
                torch.remainder(torch.tensor(measured_position - 0.75) + 0.5, 1.0) - 0.5
            )
            shock_rows.append({
                "cfl": cfl,
                "steps": local_steps,
                "terminal_time": terminal_time,
                "expected_position": expected_position,
                "measured_position": measured_position,
                "signed_position_error": signed_error,
                "absolute_position_error_cells": abs(signed_error) * nx,
                "measured_speed": displacement / terminal_time,
                "mass_error": float((state.sum() - initial.sum()).abs()),
                "finite": bool(torch.isfinite(state).all()),
            })

            rarefaction_initial = torch.where(centers < 0.5, -1.0, 1.0)[None]
            rarefaction_batch = {"states": rarefaction_initial[:, None], "dt_base": local_dt}
            rarefaction_state = rarefaction_initial.clone()
            for _ in range(local_steps):
                rarefaction_state = advance_model(
                    problem, model, rarefaction_state, local_dt, rarefaction_batch
                )
            exact = burgers_rarefaction_cell_averages(
                nx, terminal_time, device, rarefaction_state.dtype
            )
            central = (centers - 0.5).abs() <= 0.2
            central_error = rarefaction_state[0, central] - exact[central]
            positive_jumps = torch.roll(rarefaction_state[0], -1) - rarefaction_state[0]
            rarefaction_rows.append({
                "cfl": cfl,
                "steps": local_steps,
                "terminal_time": terminal_time,
                "central_l1": float(central_error.abs().mean()),
                "central_l2": float(central_error.square().mean().sqrt()),
                "center_interface_jump": float(
                    rarefaction_state[0, nx // 2] - rarefaction_state[0, nx // 2 - 1]
                ),
                "max_positive_jump_in_central_window": float(positive_jumps[central].max()),
                "mass_error": float((rarefaction_state.sum() - rarefaction_initial.sum()).abs()),
                "finite": bool(torch.isfinite(rarefaction_state).all()),
            })
        _write_csv(output / "riemann_shock_speed.csv", shock_rows)
        _write_csv(output / "sonic_rarefaction.csv", rarefaction_rows)

    print(intervention_rows)


if __name__ == "__main__":
    main()
