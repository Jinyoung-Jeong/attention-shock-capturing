from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .equations import InviscidBurgers1D, ShallowWater1D
from .utils import load_tensor_file
from .initial_conditions import periodic_rectangles_2d, periodic_top_hats_1d, random_direction, shallow_water_segments_1d
from .numerics import burgers_weno5_rhs_2d, integrate_to_times, shallow_water_weno5_flux


class TrajectoryDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = load_tensor_file(path, map_location="cpu")
        self.states = payload["states"]
        self.dt_base = payload["dt_base"]
        self.parameters = payload.get("parameters", {})
        self.metadata = payload["metadata"]

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {"states": self.states[index], "dt_base": self.dt_base[index]}
        for key, value in self.parameters.items():
            item[key] = value[index]
        return item


def _save_dataset(path: Path, states: list[torch.Tensor], dts: list[float], parameters: dict[str, list], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_tensors = {key: torch.as_tensor(value) for key, value in parameters.items()}
    torch.save(
        {
            "states": torch.stack(states).float(),
            "dt_base": torch.tensor(dts, dtype=torch.float32),
            "parameters": param_tensors,
            "metadata": metadata,
        },
        path,
    )


def generate_burgers_1d(config: dict, count: int, seed: int, path: Path, device: torch.device) -> None:
    nx = int(config["grid_size"])
    frames = int(config["base_frames"])
    base_cfl = float(config["base_cfl"])
    quadrature = int(config.get("lax_hopf_quadrature", 8))
    generator = torch.Generator().manual_seed(seed)
    equation = InviscidBurgers1D()
    states, dts = [], []
    for sample in range(count):
        u0 = periodic_top_hats_1d(nx, generator).to(device)
        dt = base_cfl / (nx * float(u0.abs().amax().clamp_min(1.0e-6)))
        trajectory = [equation.lax_hopf_cell_average(u0, frame * dt, quadrature).cpu() for frame in range(frames + 1)]
        states.append(torch.stack(trajectory))
        dts.append(dt)
        if (sample + 1) % max(1, count // 10) == 0:
            print(f"[{path.stem}] {sample + 1}/{count}", flush=True)
    _save_dataset(path, states, dts, {}, {"problem": "burgers_1d", "grid_size": nx, "base_cfl": base_cfl})


def generate_burgers_2d(config: dict, count: int, seed: int, path: Path, device: torch.device) -> None:
    grid = int(config["grid_size"])
    factor = int(config.get("reference_factor", 2))
    reference_grid = grid * factor
    frames = int(config["base_frames"])
    base_cfl = float(config["base_cfl"])
    reference_cfl = float(config.get("reference_cfl", 0.1))
    generator = torch.Generator().manual_seed(seed)
    states, dts, parameters = [], [], {"beta_x": [], "beta_y": []}
    for sample in range(count):
        low = periodic_rectangles_2d(grid, grid, generator)
        # Reconstruct the same IC at high resolution by conservative nearest-cell replication.
        high = low.repeat_interleave(factor, 0).repeat_interleave(factor, 1).to(device)
        bx, by = random_direction(generator)
        dt = base_cfl / (grid * float(low.abs().amax().clamp_min(1.0e-6)))
        output_times = [frame * dt for frame in range(frames + 1)]
        dx_ref = 1.0 / reference_grid
        rhs = lambda u: burgers_weno5_rhs_2d(u, bx, by, dx_ref, dx_ref)
        stable = lambda u: reference_cfl * dx_ref / float(u.abs().amax().clamp_min(1.0e-6))
        high_trajectory = integrate_to_times(high, output_times, rhs, stable)
        shape = high_trajectory.shape
        coarse = high_trajectory.reshape(shape[0], grid, factor, grid, factor).mean((2, 4)).cpu()
        states.append(coarse)
        dts.append(dt)
        parameters["beta_x"].append(bx)
        parameters["beta_y"].append(by)
        if (sample + 1) % max(1, count // 10) == 0:
            print(f"[{path.stem}] {sample + 1}/{count}", flush=True)
    _save_dataset(
        path, states, dts, parameters,
        {"problem": "burgers_2d", "grid_size": grid, "reference_grid": reference_grid, "base_cfl": base_cfl},
    )


def generate_shallow_water_1d(config: dict, count: int, seed: int, path: Path, device: torch.device) -> None:
    nx = int(config["grid_size"])
    factor = int(config.get("reference_factor", 4))
    reference_nx = nx * factor
    frames = int(config["base_frames"])
    base_cfl = float(config["base_cfl"])
    reference_cfl = float(config.get("reference_cfl", 0.1))
    gravity = float(config.get("gravity", 1.0))
    equation = ShallowWater1D(gravity=gravity)
    generator = torch.Generator().manual_seed(seed)
    states, dts = [], []
    for sample in range(count):
        low = shallow_water_segments_1d(nx, generator)
        high = low.repeat_interleave(factor, 0).to(device)
        dt = base_cfl / (nx * float(equation.max_speed(low).clamp_min(1.0e-6)))
        output_times = [frame * dt for frame in range(frames + 1)]
        dx_ref = 1.0 / reference_nx

        def rhs(u: torch.Tensor) -> torch.Tensor:
            face_flux = shallow_water_weno5_flux(u[None], equation)[0]
            return -(face_flux - torch.roll(face_flux, 1, 0)) / dx_ref

        stable = lambda u: reference_cfl * dx_ref / float(equation.max_speed(u).clamp_min(1.0e-6))
        high_trajectory = integrate_to_times(high, output_times, rhs, stable)
        coarse = high_trajectory.reshape(frames + 1, nx, factor, 2).mean(2).cpu()
        states.append(coarse)
        dts.append(dt)
        if (sample + 1) % max(1, count // 10) == 0:
            print(f"[{path.stem}] {sample + 1}/{count}", flush=True)
    _save_dataset(
        path, states, dts, {},
        {"problem": "shallow_water_1d", "grid_size": nx, "reference_grid": reference_nx,
         "base_cfl": base_cfl, "gravity": gravity},
    )


def generate_split(problem: str, config: dict, count: int, seed: int, path: Path, device: torch.device) -> None:
    if problem == "burgers_1d":
        generate_burgers_1d(config, count, seed, path, device)
    elif problem == "burgers_2d":
        generate_burgers_2d(config, count, seed, path, device)
    elif problem == "shallow_water_1d":
        generate_shallow_water_1d(config, count, seed, path, device)
    else:
        raise ValueError(problem)
