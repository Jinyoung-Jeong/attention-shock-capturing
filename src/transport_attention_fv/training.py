from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .models import build_model
from .numerics import conservative_update_1d, conservative_update_2d
from .utils import load_tensor_file, resolve_path, runtime_manifest, save_json, seed_everything


def advance_model(problem: str, model: nn.Module, state: torch.Tensor, dt: torch.Tensor, batch: dict, intervention: str = "learned") -> torch.Tensor:
    if problem == "burgers_1d":
        scale = dt * state.shape[-1]
        flux = model(state, scale, intervention=intervention)
        return conservative_update_1d(state, flux, scale)
    if problem == "burgers_2d":
        nx, ny = state.shape[-1], state.shape[-2]
        sx, sy = dt * nx, dt * ny
        fx, fy = model(state, sx, sy, batch["beta_x"], batch["beta_y"], intervention=intervention)
        return conservative_update_2d(state, fx, fy, sx, sy)
    if problem == "shallow_water_1d":
        scale = dt * state.shape[-2]
        flux = model(state, scale, intervention=intervention)
        return conservative_update_1d(state, flux, scale)
    raise ValueError(problem)


def gradient_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    coefficient: float,
    component_scales: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    scaled_prediction, scaled_target = prediction, target
    if component_scales is not None:
        scales = torch.as_tensor(component_scales, dtype=target.dtype, device=target.device)
        if target.shape[-1] != scales.numel():
            raise ValueError(f"component_scales={component_scales} do not match target shape {target.shape}")
        scaled_prediction = prediction / scales
        scaled_target = target / scales
    if scaled_target.ndim == 3 and scaled_target.shape[-1] != 2:  # scalar field on a 2D grid
        gradient = (scaled_target - torch.roll(scaled_target, 1, -1)).abs()
        gradient = gradient + (scaled_target - torch.roll(scaled_target, 1, -2)).abs()
    else:
        cell_dim = -2 if scaled_target.shape[-1] == 2 and scaled_target.ndim >= 3 else -1
        gradient = (scaled_target - torch.roll(scaled_target, 1, cell_dim)).abs()
    if scaled_target.shape[-1] == 2 and scaled_target.ndim >= 3:
        gradient = gradient.mean(-1, keepdim=True)
    weight = 1 + coefficient * gradient / gradient.mean().clamp_min(1.0e-8)
    return (weight * (scaled_prediction - scaled_target).square()).mean()


def rollout_loss(
    problem: str,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    stride: int,
    rollout_steps: int,
    gradient_weight: float,
    intervention: str = "learned",
    component_scales: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    trajectories = batch["states"]
    available = trajectories.shape[1] - 1 - rollout_steps * stride
    start = int(torch.randint(0, max(available + 1, 1), (1,)))
    state = trajectories[:, start]
    dt = batch["dt_base"] * stride
    losses = []
    for step in range(1, rollout_steps + 1):
        state = advance_model(problem, model, state, dt, batch, intervention=intervention)
        target = trajectories[:, start + step * stride]
        losses.append(gradient_weighted_mse(
            state, target, gradient_weight, component_scales=component_scales
        ))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_metrics(
    problem: str, model: nn.Module, loader: DataLoader, stride: int, steps: int,
    device: torch.device, intervention: str = "learned",
    component_scales: list[float] | tuple[float, ...] | None = None,
) -> dict[str, float]:
    errors, failures = [], 0
    for raw in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
        state = batch["states"][:, 0]
        dt = batch["dt_base"] * stride
        for _ in range(steps):
            state = advance_model(problem, model, state, dt, batch, intervention=intervention)
        target = batch["states"][:, steps * stride]
        difference = state - target
        if component_scales is not None:
            scales = torch.as_tensor(component_scales, dtype=state.dtype, device=state.device)
            difference = difference / scales
        reduce_dims = tuple(range(1, state.ndim))
        sample_error = difference.square().mean(reduce_dims).sqrt()
        valid = torch.isfinite(sample_error)
        if problem == "shallow_water_1d":
            valid &= state[..., 0].amin(-1) > 0
        failures += int((~valid).sum())
        errors.extend(sample_error[valid].cpu().tolist())
    median = float(torch.tensor(errors).median()) if errors else math.inf
    return {"failures": failures, "samples": len(loader.dataset), "median_l2": median}


def _rollout_curriculum(epoch: int, epochs: int, minimum: int, maximum: int) -> int:
    if epochs <= 1:
        return maximum
    fraction = epoch / (epochs - 1)
    return min(maximum, minimum + int(round(fraction * (maximum - minimum))))


def train(
    config: dict[str, Any], config_path: Path, seed: int, device: torch.device,
    output_dir: Path, intervention: str = "learned",
) -> None:
    seed_everything(seed, deterministic=bool(config["training"].get("deterministic", False)))
    problem = config["problem"]
    data_root = resolve_path(config_path, config["data"]["directory"]) / problem
    train_data = TrajectoryDataset(data_root / "train.pt")
    val_data = TrajectoryDataset(data_root / "validation.pt")
    training = config["training"]
    batch_size = int(training["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(problem, config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(training["minimum_learning_rate"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "runtime.json", {
        **runtime_manifest(), "seed": seed, "problem": problem, "training_intervention": intervention,
    })
    log_path = output_dir / "training.csv"
    best_large = (math.inf, math.inf)
    best_mixed = math.inf
    skipped = 0
    start_epoch = 0
    latest_path = output_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        resume = load_tensor_file(latest_path, map_location=device)
        if (resume.get("seed") != seed or resume.get("problem") != problem
                or resume.get("training_intervention", "learned") != intervention):
            raise ValueError(f"Checkpoint identity mismatch: {latest_path}")
        if resume.get("config") != config:
            raise ValueError(
                f"Checkpoint protocol mismatch: {latest_path}. "
                "Move the prior run aside; do not resume it under a changed configuration."
            )
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume["epoch"])
        if start_epoch >= epochs:
            print(f"Already complete: {latest_path} ({start_epoch}/{epochs})", flush=True)
            return
        if (output_dir / "checkpoint_best_large_step.pt").exists():
            prior = load_tensor_file(output_dir / "checkpoint_best_large_step.pt", map_location="cpu")
            metric = prior["validation_large"]
            best_large = (metric["failures"], metric["median_l2"])
        if (output_dir / "checkpoint_best_mixed.pt").exists():
            prior = load_tensor_file(output_dir / "checkpoint_best_mixed.pt", map_location="cpu")
            best_mixed = float(prior["validation_mixed_l2"])
    new_log = not log_path.exists() or start_epoch == 0
    mode = "w" if new_log else "a"
    with log_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "rollout", "loss", "val_failures", "val_l2", "lr", "skipped"])
        if new_log:
            writer.writeheader()
        for epoch in range(start_epoch, epochs):
            model.train()
            rollout = _rollout_curriculum(epoch, epochs, int(training["rollout_min"]), int(training["rollout_max"]))
            total = 0.0
            updates = 0
            iterator = iter(train_loader)
            for _ in range(int(training["iterations_per_epoch"])):
                try:
                    raw = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    raw = next(iterator)
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
                allowed = [stride for stride in training["strides"] if rollout * int(stride) <= batch["states"].shape[1] - 1]
                stride = int(allowed[int(torch.randint(0, len(allowed), (1,)))])
                optimizer.zero_grad(set_to_none=True)
                loss = rollout_loss(
                    problem, model, batch, stride, rollout,
                    float(training["gradient_weight"]), intervention=intervention,
                    component_scales=training.get("component_scales"),
                )
                if not torch.isfinite(loss) or float(loss.detach()) > float(training.get("maximum_loss", 1.0e6)):
                    skipped += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                optimizer.step()
                total += float(loss.detach())
                updates += 1
            scheduler.step()
            model.eval()
            validate_now = (epoch + 1) % int(training.get("validation_interval", 1)) == 0 or epoch + 1 == epochs
            if validate_now:
                large = validation_metrics(
                    problem, model, val_loader, int(training["validation_stride"]),
                    int(training["validation_steps"]), device, intervention=intervention,
                    component_scales=training.get("component_scales"),
                )
                mixed_l2 = 0.0
                for stride in training["strides"]:
                    steps = min(int(training["validation_steps"]), (val_data.states.shape[1] - 1) // int(stride))
                    mixed_l2 += validation_metrics(
                        problem, model, val_loader, int(stride), steps, device,
                        intervention=intervention,
                        component_scales=training.get("component_scales"),
                    )["median_l2"]
                mixed_l2 /= len(training["strides"])
            else:
                large = {"failures": math.inf, "samples": len(val_data), "median_l2": math.inf}
                mixed_l2 = math.inf
            payload = {
                "epoch": epoch + 1, "seed": seed, "problem": problem, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": config,
                "training_intervention": intervention,
                "validation_large": large, "validation_mixed_l2": mixed_l2,
            }
            torch.save(payload, output_dir / "checkpoint_latest.pt")
            large_key = (large["failures"], large["median_l2"])
            if large_key < best_large:
                best_large = large_key
                torch.save(payload, output_dir / "checkpoint_best_large_step.pt")
            if mixed_l2 < best_mixed:
                best_mixed = mixed_l2
                torch.save(payload, output_dir / "checkpoint_best_mixed.pt")
            row = {
                "epoch": epoch + 1, "rollout": rollout, "loss": total / max(updates, 1),
                "val_failures": large["failures"], "val_l2": large["median_l2"],
                "lr": scheduler.get_last_lr()[0], "skipped": skipped,
            }
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
