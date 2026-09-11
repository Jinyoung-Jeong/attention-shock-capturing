from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import TrajectoryDataset
from .equations import ShallowWater1D
from .models import build_model
from .numerics import (
    burgers_eno3_flux_1d,
    burgers_godunov_flux,
    burgers_weno5_flux_1d,
    burgers_weno5_rhs_2d,
    conservative_update_1d,
    shallow_water_hll_flux,
    shallow_water_weno5_flux,
)
from .training import advance_model
from .utils import load_tensor_file, resolve_path, save_json


def _classical_fe(problem: str, method: str, state: torch.Tensor, dt: torch.Tensor, batch: dict, gravity: float) -> torch.Tensor:
    if problem == "burgers_1d":
        scale = dt * state.shape[-1]
        if method == "godunov":
            flux = burgers_godunov_flux(state, torch.roll(state, -1, -1))
        elif method == "eno3":
            flux = burgers_eno3_flux_1d(state)
        else:
            flux = burgers_weno5_flux_1d(state)
        return conservative_update_1d(state, flux, scale)
    if problem == "burgers_2d":
        dx = 1.0 / state.shape[-1]
        rhs = burgers_weno5_rhs_2d(state, batch["beta_x"], batch["beta_y"], dx, dx)
        return state + dt[:, None, None] * rhs
    if problem == "shallow_water_1d":
        equation = ShallowWater1D(gravity=gravity)
        scale = dt * state.shape[-2]
        if method == "hll":
            flux = shallow_water_hll_flux(state, torch.roll(state, -1, -2), equation)
        else:
            flux = shallow_water_weno5_flux(state, equation)
        return conservative_update_1d(state, flux, scale)
    raise ValueError(problem)


def _ssprk3(
    fe: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    y1 = fe(state)
    y2 = 0.75 * state + 0.25 * fe(y1)
    result = state / 3 + 2 * fe(y2) / 3
    return result, (y1, y2, result)


def _methods(problem: str) -> list[tuple[str, str, str]]:
    if problem == "burgers_1d":
        return [
            ("Godunov + Forward Euler", "godunov", "fe"),
            ("ENO-3 + Forward Euler", "eno3", "fe"),
            ("WENO-5 + Forward Euler", "weno5", "fe"),
            ("WENO-5 + SSP-RK3", "weno5", "rk3"),
        ]
    if problem == "burgers_2d":
        return [("WENO-5 + Forward Euler", "weno5", "fe"), ("WENO-5 + SSP-RK3", "weno5", "rk3")]
    return [("HLL + Forward Euler", "hll", "fe"), ("WENO-5 + Forward Euler", "weno5", "fe"),
            ("WENO-5 + SSP-RK3", "weno5", "rk3")]


def _instantaneous_cfl(
    problem: str, state: torch.Tensor, dt: torch.Tensor, batch: dict, gravity: float,
) -> torch.Tensor:
    if problem == "burgers_1d":
        return dt * state.shape[-1] * state.abs().amax(-1)
    if problem == "burgers_2d":
        sx, sy = dt * state.shape[-1], dt * state.shape[-2]
        rate = sx * batch["beta_x"].abs() + sy * batch["beta_y"].abs()
        return rate * state.abs().amax((-1, -2))
    equation = ShallowWater1D(gravity=gravity)
    minus, plus = equation.wave_speeds(state)
    speed = torch.maximum(minus.abs(), plus.abs()).amax(-1)
    return dt * state.shape[-2] * speed


def _conserved_totals(problem: str, state: torch.Tensor) -> torch.Tensor:
    """Unweighted cell sums, retaining independent conserved components."""
    if problem == "shallow_water_1d":
        return state.double().sum(dim=1)
    return state.double().sum(tuple(range(1, state.ndim))).unsqueeze(-1)


def _summarize(records: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["method"], r["cfl"]) for r in records})
    for method, cfl in keys:
        group = [r for r in records if r["method"] == method and r["cfl"] == cfl]
        valid = [r for r in group if not r["failed"]]
        depth_values = [r["depth_l2"] for r in valid if np.isfinite(r["depth_l2"])]
        discharge_values = [r["discharge_l2"] for r in valid if np.isfinite(r["discharge_l2"])]
        depth_conservation = [r["depth_conservation"] for r in valid
                              if np.isfinite(r.get("depth_conservation", np.nan))]
        discharge_conservation = [r["discharge_conservation"] for r in valid
                                  if np.isfinite(r.get("discharge_conservation", np.nan))]
        successes = len(valid)
        row = {
            "method": method, "cfl": cfl, "samples": len(group),
            "successes": successes,
            "failures": sum(r["failed"] for r in group),
            "success_rate": successes / len(group),
            "sequential_steps": int(group[0]["steps"]),
            "flux_evaluations": int(group[0]["flux_evaluations"]),
            "timing_samples": int(group[0]["timing_samples"]),
            "median_l1_successful": float(np.median([r["l1"] for r in valid])) if valid else float("inf"),
            "median_l2_successful": float(np.median([r["l2"] for r in valid])) if valid else float("inf"),
            "median_linf_successful": float(np.median([r["linf"] for r in valid])) if valid else float("inf"),
            "mean_conservation_error": float(np.mean([r["conservation"] for r in valid])) if valid else float("inf"),
            "mean_depth_conservation_error": float(np.mean(depth_conservation)) if depth_conservation else float("nan"),
            "mean_discharge_conservation_error": float(np.mean(discharge_conservation)) if discharge_conservation else float("nan"),
            "wall_clock_seconds": float(group[0]["wall_clock_seconds"]),
            "median_realized_cfl": float(np.median([r["realized_cfl"] for r in valid])) if valid else float("inf"),
            "max_realized_cfl": float(np.max([r["realized_cfl"] for r in valid])) if valid else float("inf"),
            "nonpositive_depth_count": int(sum(r["nonpositive_depth"] for r in group)),
            "median_depth_l2": float(np.median(depth_values)) if depth_values else float("nan"),
            "median_discharge_l2": float(np.median(discharge_values)) if discharge_values else float("nan"),
        }
        summary.append(row)
    return summary


@torch.no_grad()
def evaluate(
    config: dict, config_path: Path, checkpoint: Path, device: torch.device,
    output_dir: Path, include_baselines: bool = True, attention_intervention: str = "learned",
) -> None:
    problem = config["problem"]
    data_path = resolve_path(config_path, config["data"]["directory"]) / problem / "test.pt"
    dataset = TrajectoryDataset(data_path)
    batch_size = int(config["evaluation"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    payload = load_tensor_file(checkpoint, map_location=device)
    model = build_model(problem, config["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    base_cfl = float(config["data"]["base_cfl"])
    cfl_values = [float(value) for value in config["evaluation"]["cfl_values"]]
    final_frame = int(config["data"]["base_frames"])
    gravity = float(config["model"].get("gravity", 1.0))
    warmups = int(config["evaluation"].get("timing_warmup", 1))
    repeats = int(config["evaluation"].get("timing_repeats", 3))
    timing_samples = min(int(config["evaluation"].get("timing_samples", batch_size)), len(dataset))
    records: list[dict] = []
    for cfl in cfl_values:
        stride = int(round(cfl / base_cfl))
        if abs(stride * base_cfl - cfl) > 1.0e-8 or final_frame % stride:
            raise ValueError(f"CFL {cfl} is not an integer stride of base CFL {base_cfl}")
        steps = final_frame // stride
        attention_name = (
            str(config["evaluation"].get("method_label", "CFL-conditioned attention"))
            if attention_intervention == "learned"
            else {
                "cfl_blind": "Retrained ablation: CFL-blind",
                "uniform": "Retrained ablation: uniform attention",
            }[attention_intervention]
        )
        methods = [(attention_name, "attention", "fe")]
        if include_baselines:
            methods.extend(_methods(problem))
        for name, method, integrator in methods:
            def run_rollout(
                initial: torch.Tensor, batch: dict, dt: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                current = initial.clone()
                realized = _instantaneous_cfl(problem, current, dt, batch, gravity)
                minimum_depth = (
                    current[..., 0].amin(-1) if problem == "shallow_water_1d"
                    else torch.full_like(realized, float("inf"))
                )
                for _ in range(steps):
                    if method == "attention":
                        current = advance_model(
                            problem, model, current, dt, batch,
                            intervention=attention_intervention,
                        )
                    else:
                        fe = lambda u: _classical_fe(problem, method, u, dt, batch, gravity)
                        if integrator == "fe":
                            current = fe(current)
                            stages = (current,)
                        else:
                            current, stages = _ssprk3(fe, current)
                        for stage in stages:
                            realized = torch.maximum(
                                realized, _instantaneous_cfl(problem, stage, dt, batch, gravity)
                            )
                            if problem == "shallow_water_1d":
                                minimum_depth = torch.fmin(minimum_depth, stage[..., 0].amin(-1))
                        continue
                    realized = torch.maximum(realized, _instantaneous_cfl(problem, current, dt, batch, gravity))
                    if problem == "shallow_water_1d":
                        minimum_depth = torch.fmin(minimum_depth, current[..., 0].amin(-1))
                return current, realized, minimum_depth

            timing_loader = DataLoader(
                Subset(dataset, range(timing_samples)), batch_size=timing_samples,
                shuffle=False, num_workers=0,
            )
            timing_raw = next(iter(timing_loader))
            timing_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in timing_raw.items()
            }
            timing_initial = timing_batch["states"][:, 0].clone()
            timing_dt = timing_batch["dt_base"] * stride
            for _ in range(warmups):
                _ = run_rollout(timing_initial, timing_batch, timing_dt)
            timings = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = run_rollout(timing_initial, timing_batch, timing_dt)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            elapsed = float(np.median(timings))

            offset = 0
            for raw in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
                initial = batch["states"][:, 0].clone()
                initial_sum = _conserved_totals(problem, initial)
                dt = batch["dt_base"] * stride
                state, realized_cfl, minimum_depth = run_rollout(initial, batch, dt)
                target = batch["states"][:, final_frame]
                flat_error = (state - target).reshape(state.shape[0], -1)
                final_sum = _conserved_totals(problem, state)
                conservation = (final_sum - initial_sum).abs()
                for local in range(state.shape[0]):
                    error = flat_error[local]
                    failed = not bool(torch.isfinite(error).all()) or float(state[local].abs().amax()) > 1.0e6
                    nonpositive_depth = 0
                    if problem == "shallow_water_1d":
                        nonpositive_depth = int(float(minimum_depth[local]) <= 0)
                        failed |= bool(nonpositive_depth)
                        depth_error = state[local, :, 0] - target[local, :, 0]
                        discharge_error = state[local, :, 1] - target[local, :, 1]
                    else:
                        depth_error = discharge_error = None
                    records.append(
                        {
                            "sample": offset + local, "method": name, "cfl": cfl, "steps": steps,
                            "flux_evaluations": steps * (3 if integrator == "rk3" else 1), "failed": int(failed),
                            "timing_samples": timing_samples,
                            "l1": float(error.abs().mean()) if not failed else float("nan"),
                            "l2": float(error.square().mean().sqrt()) if not failed else float("nan"),
                            "linf": float(error.abs().amax()) if not failed else float("nan"),
                            "conservation": float(conservation[local].max()) if not failed else float("nan"),
                            "depth_conservation": float(conservation[local, 0]) if not failed and problem == "shallow_water_1d" else float("nan"),
                            "discharge_conservation": float(conservation[local, 1]) if not failed and problem == "shallow_water_1d" else float("nan"),
                            "wall_clock_seconds": elapsed,
                            "realized_cfl": float(realized_cfl[local]) if not failed else float("nan"),
                            "nonpositive_depth": nonpositive_depth,
                            "depth_l2": float(depth_error.square().mean().sqrt()) if not failed and depth_error is not None else float("nan"),
                            "discharge_l2": float(discharge_error.square().mean().sqrt()) if not failed and discharge_error is not None else float("nan"),
                        }
                    )
                offset += state.shape[0]
            print(f"evaluated {problem}: {name}, CFL={cfl}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = _summarize(records)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    save_json(output_dir / "metadata.json", {
        "checkpoint": checkpoint.name, "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_role": "best_large_step", "include_baselines": include_baselines,
        "attention_intervention": attention_intervention,
        "conservation_definition": "Unweighted cell-sum error; maximum over independent components. Shallow-water h and q errors are also reported separately.",
    })
