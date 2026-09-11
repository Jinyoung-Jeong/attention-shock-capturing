"""Reproduce the appendix pre-shock Burgers reconstruction check (no training)."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from transport_attention_fv.numerics import burgers_godunov_flux, burgers_weno5_flux_1d
from transport_attention_fv.utils import save_json


def initial_cell_averages(nx: int) -> torch.Tensor:
    dx = 1.0 / nx
    left = torch.arange(nx, dtype=torch.float64) * dx
    right = left + dx
    avg_sin = (torch.cos(2 * torch.pi * left) - torch.cos(2 * torch.pi * right)) / (2 * torch.pi * dx)
    return (1.0 + 0.5 * avg_sin).view(1, -1)


def paper_eno3_flux(state: torch.Tensor) -> torch.Tensor:
    """Paper convergence check: recursive undivided-difference ENO selection.

    Kept distinct from the production extension's minimum-indicator variant.
    """
    im2, im1, i0, ip1, ip2, ip3 = [torch.roll(state, s, -1) for s in (2, 1, 0, -1, -2, -3)]
    q0 = im2 / 3 - 7 * im1 / 6 + 11 * i0 / 6
    q1 = -im1 / 6 + 5 * i0 / 6 + ip1 / 3
    q2 = i0 / 3 + 5 * ip1 / 6 - ip2 / 6
    left1 = (i0 - im1).abs() < (ip1 - i0).abs()
    left2 = (i0 - 2 * im1 + im2).abs() < (ip1 - 2 * i0 + im1).abs()
    right2 = (ip1 - 2 * i0 + im1).abs() < (ip2 - 2 * ip1 + i0).abs()
    left = torch.where(left1, torch.where(left2, q0, q1), torch.where(right2, q1, q2))
    r0 = 11 * ip1 / 6 - 7 * ip2 / 6 + ip3 / 3
    r1 = i0 / 3 + 5 * ip1 / 6 - ip2 / 6
    r2 = -im1 / 6 + 5 * i0 / 6 + ip1 / 3
    grow_left = (ip1 - i0).abs() < (ip2 - ip1).abs()
    first_left = (ip1 - 2 * i0 + im1).abs() < (ip2 - 2 * ip1 + i0).abs()
    first_right = (ip2 - 2 * ip1 + i0).abs() < (ip3 - 2 * ip2 + ip1).abs()
    right = torch.where(grow_left, torch.where(first_left, r2, r1), torch.where(first_right, r1, r0))
    return burgers_godunov_flux(left, right)


def exact_cell_averages(nx: int, time: float, subcells: int = 128) -> torch.Tensor:
    """Historical 128-point midpoint quadrature of the pre-shock characteristic solution."""
    dx = 1.0 / nx
    centers = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    offsets = (torch.arange(subcells, dtype=torch.float64) + 0.5) / subcells - 0.5
    x = centers[:, None] + dx * offsets[None, :]
    foot = x - time
    for _ in range(40):
        residual = foot + time * (1 + 0.5 * torch.sin(2 * torch.pi * foot)) - x
        derivative = 1 + time * torch.pi * torch.cos(2 * torch.pi * foot)
        step = residual / derivative
        foot = foot - step
        if float(step.abs().max()) < 1e-14:
            break
    else:
        raise RuntimeError("Characteristic inversion did not converge")
    return (1 + 0.5 * torch.sin(2 * torch.pi * foot)).mean(1).view(1, -1)


@torch.no_grad()
def convergence() -> list[dict]:
    rows, previous = [], {}
    final_time = 0.15
    for nx in (32, 64, 128, 256):
        dx = 1.0 / nx
        steps = math.ceil(final_time / (0.05 * dx / 1.5))
        dt = final_time / steps
        reference = exact_cell_averages(nx, final_time)
        for name, flux_function in (("ENO-3", paper_eno3_flux), ("WENO-5", burgers_weno5_flux_1d)):
            state = initial_cell_averages(nx)

            def rhs(value):
                flux = flux_function(value)
                return -(flux - torch.roll(flux, 1, -1)) / dx

            for _ in range(steps):
                first = state + dt * rhs(state)
                second = 0.75 * state + 0.25 * (first + dt * rhs(first))
                state = (1 / 3) * state + (2 / 3) * (second + dt * rhs(second))
            if not torch.isfinite(state).all():
                raise RuntimeError(f"Non-finite {name} solution at nx={nx}")
            error = float((state - reference).abs().mean())
            order = math.log2(previous[name] / error) if name in previous else None
            previous[name] = error
            rows.append(dict(nx=nx, solver=name, integrator="SSP-RK3", dt=dt,
                             steps=steps, L1=error, observed_order=order, nonfinite=False))
            print(f"{name}: nx={nx}, L1={error:.8e}, order={order}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/classical_convergence"))
    parser.add_argument("--check", action="store_true", help="Compare with the released appendix values")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = convergence()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(args.output / "protocol.json", {
        "equation": "u_t + (u^2/2)_x = 0", "boundary": "periodic on [0,1]",
        "initial_condition": "1 + 0.5 sin(2 pi x), exact initial cell averages",
        "final_time": 0.15, "breaking_time": 1 / math.pi,
        "cfl_bound": 0.05, "reference_midpoint_subcells": 128,
        "dtype": "float64", "torch": str(torch.__version__),
        "interpretation": "Finite-grid spatial-accuracy check with small time steps; not fifth-order accuracy of SSP-RK3 in time.",
        "eno_selection": "Recursive undivided-difference selection used by the paper convergence check; distinct from the extension production minimum-indicator variant.",
    })
    if args.check:
        expected_path = Path(__file__).resolve().parents[1] / "results/appendix/classical_convergence_reported.csv"
        with expected_path.open(encoding="utf-8") as handle:
            expected = {(int(r["nx"]), r["solver"]): float(r["L1"]) for r in csv.DictReader(handle)}
        for row in rows:
            if not math.isclose(row["L1"], expected[(row["nx"], row["solver"])], rel_tol=3e-6, abs_tol=1e-13):
                raise AssertionError(f"Reported-value mismatch: {row}")
        print("Reported-value check: PASS")


if __name__ == "__main__":
    main()
