#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = []
    for path in sorted(args.runs.glob("burgers_1d/seed_*/generalization/summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No generalization results under {args.runs}")
    all_results = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output / "generalization_by_seed.csv", index=False)
    all_results.groupby(
        ["case", "grid_size", "cfl", "horizon_multiplier"], as_index=False
    ).agg(
        seeds=("seed", "count"),
        success_rate_median=("success_rate", "median"),
        success_rate_min=("success_rate", "min"),
        median_l2_successful=("median_l2_successful", "median"),
        median_l2_successful_min=("median_l2_successful", "min"),
        median_l2_successful_max=("median_l2_successful", "max"),
        mean_conservation_error=("mean_conservation_error", "median"),
    ).to_csv(args.output / "generalization_aggregate.csv", index=False)

    smooth = all_results[all_results["case"] == "smooth_sine_pre_shock"].copy()
    convergence_rows = []
    for (seed, cfl, horizon), group in smooth.groupby(
        ["seed", "cfl", "horizon_multiplier"]
    ):
        ordered = group.sort_values("grid_size")
        records = ordered.to_dict("records")
        for coarse, fine in zip(records[:-1], records[1:]):
            coarse_error = float(coarse["median_l2_successful"])
            fine_error = float(fine["median_l2_successful"])
            ratio = float(fine["grid_size"]) / float(coarse["grid_size"])
            order = (
                np.log(coarse_error / fine_error) / np.log(ratio)
                if coarse_error > 0 and fine_error > 0 and ratio > 1 else float("nan")
            )
            convergence_rows.append({
                "seed": int(seed), "cfl": float(cfl),
                "horizon_multiplier": int(horizon),
                "coarse_grid": int(coarse["grid_size"]),
                "fine_grid": int(fine["grid_size"]),
                "coarse_l2": coarse_error, "fine_l2": fine_error,
                "observed_l2_order": float(order),
            })
    if convergence_rows:
        convergence = pd.DataFrame(convergence_rows)
        convergence.to_csv(args.output / "smooth_convergence_order_by_seed.csv", index=False)
        convergence.groupby(
            ["cfl", "horizon_multiplier", "coarse_grid", "fine_grid"], as_index=False
        ).agg(
            seeds=("seed", "count"),
            observed_l2_order_median=("observed_l2_order", "median"),
            observed_l2_order_min=("observed_l2_order", "min"),
            observed_l2_order_max=("observed_l2_order", "max"),
        ).to_csv(args.output / "smooth_convergence_order_aggregate.csv", index=False)
    print(f"generalization aggregate: {args.output}")


if __name__ == "__main__":
    main()
