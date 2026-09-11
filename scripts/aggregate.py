#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = []
    for path in sorted(args.runs.glob("*/seed_*/evaluation/summary.csv")):
        frame = pd.read_csv(path)
        for column in ("mean_depth_conservation_error", "mean_discharge_conservation_error"):
            if column not in frame:
                frame[column] = float("nan")
        frame.insert(0, "seed", int(path.parts[-3].split("_")[-1]))
        frame.insert(0, "problem", path.parts[-4])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No evaluation summaries under {args.runs}")
    all_results = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output / "all_seed_results.csv", index=False)
    group = all_results.groupby(["problem", "method", "cfl"], as_index=False)
    aggregate = group.agg(
        seeds=("seed", "count"),
        success_rate_median=("success_rate", "median"),
        success_rate_min=("success_rate", "min"),
        failures_mean=("failures", "mean"), failures_max=("failures", "max"),
        median_l1_successful=("median_l1_successful", "median"),
        median_l2_successful=("median_l2_successful", "median"),
        median_l2_successful_min=("median_l2_successful", "min"),
        median_l2_successful_max=("median_l2_successful", "max"),
        median_linf_successful=("median_linf_successful", "median"),
        sequential_steps=("sequential_steps", "first"),
        flux_evaluations=("flux_evaluations", "first"),
        timing_samples=("timing_samples", "first"),
        mean_conservation_error=("mean_conservation_error", "median"),
        mean_depth_conservation_error=("mean_depth_conservation_error", "median"),
        mean_discharge_conservation_error=("mean_discharge_conservation_error", "median"),
        wall_clock_seconds=("wall_clock_seconds", "median"),
        wall_clock_seconds_min=("wall_clock_seconds", "min"),
        wall_clock_seconds_max=("wall_clock_seconds", "max"),
        median_realized_cfl=("median_realized_cfl", "median"),
        max_realized_cfl=("max_realized_cfl", "max"),
        nonpositive_depth_count=("nonpositive_depth_count", "sum"),
        median_depth_l2=("median_depth_l2", "median"),
        median_discharge_l2=("median_discharge_l2", "median"),
    )
    aggregate.to_csv(args.output / "aggregate.csv", index=False)

    ablation_frames = []
    for path in sorted(args.runs.glob("*/ablations/*/seed_*/evaluation/summary.csv")):
        frame = pd.read_csv(path)
        for column in ("mean_depth_conservation_error", "mean_discharge_conservation_error"):
            if column not in frame:
                frame[column] = float("nan")
        frame.insert(0, "seed", int(path.parts[-3].split("_")[-1]))
        frame.insert(0, "variant", path.parts[-4])
        frame.insert(0, "problem", path.parts[-6])
        ablation_frames.append(frame)
    if ablation_frames:
        ablations = pd.concat(ablation_frames, ignore_index=True)
        ablations.to_csv(args.output / "all_seed_retrained_ablations.csv", index=False)
        ablations.groupby(["problem", "variant", "method", "cfl"], as_index=False).agg(
            seeds=("seed", "count"),
            success_rate_median=("success_rate", "median"),
            success_rate_min=("success_rate", "min"),
            failures_mean=("failures", "mean"), failures_max=("failures", "max"),
            median_l1_successful=("median_l1_successful", "median"),
            median_l2_successful=("median_l2_successful", "median"),
            median_linf_successful=("median_linf_successful", "median"),
            sequential_steps=("sequential_steps", "first"),
            flux_evaluations=("flux_evaluations", "first"),
            timing_samples=("timing_samples", "first"),
            mean_conservation_error=("mean_conservation_error", "median"),
            mean_depth_conservation_error=("mean_depth_conservation_error", "median"),
            mean_discharge_conservation_error=("mean_discharge_conservation_error", "median"),
            wall_clock_seconds=("wall_clock_seconds", "median"),
            median_realized_cfl=("median_realized_cfl", "median"),
            max_realized_cfl=("max_realized_cfl", "max"),
            nonpositive_depth_count=("nonpositive_depth_count", "sum"),
            median_depth_l2=("median_depth_l2", "median"),
            median_discharge_l2=("median_discharge_l2", "median"),
        ).to_csv(args.output / "aggregate_retrained_ablations.csv", index=False)
    intervention_frames = []
    reach_rows = []
    attention_region_frames = []
    for path in sorted(args.runs.glob("*/seed_*/diagnostics/interventions.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(path.parts[-3].split("_")[-1]))
        frame.insert(0, "problem", path.parts[-4])
        intervention_frames.append(frame)
        reach_path = path.parent / "reach_regression_by_cfl.csv"
        if reach_path.exists():
            reach_frame = pd.read_csv(reach_path)
            reach_frame.insert(0, "seed", int(path.parts[-3].split("_")[-1]))
            reach_frame.insert(0, "problem", path.parts[-4])
            reach_rows.extend(reach_frame.to_dict("records"))
        region_path = path.parent / "attention_region_stats_by_cfl.csv"
        if region_path.exists():
            region_frame = pd.read_csv(region_path, dtype={"head": str})
            region_frame.insert(0, "seed", int(path.parts[-3].split("_")[-1]))
            region_frame.insert(0, "problem", path.parts[-4])
            attention_region_frames.append(region_frame)
    if intervention_frames:
        interventions = pd.concat(intervention_frames, ignore_index=True)
        interventions.to_csv(args.output / "all_seed_interventions.csv", index=False)
        interventions.groupby(["problem", "intervention"], as_index=False).agg(
            seeds=("seed", "count"),
            success_rate_median=("success_rate", "median"),
            success_rate_min=("success_rate", "min"),
            failures_mean=("failures", "mean"), failures_max=("failures", "max"),
            median_l2_successful=("median_l2_successful", "median"),
        ).to_csv(args.output / "aggregate_interventions.csv", index=False)
    if reach_rows:
        reach = pd.DataFrame(reach_rows)
        reach.to_csv(args.output / "reach_by_seed_and_cfl.csv", index=False)
        reach.groupby(["problem", "cfl"], as_index=False).agg(
            seeds=("seed", "count"),
            slope_median=("slope", "median"), slope_min=("slope", "min"), slope_max=("slope", "max"),
            pearson_r_median=("pearson_r", "median"),
            pearson_r_min=("pearson_r", "min"), pearson_r_max=("pearson_r", "max"),
            mean_abs_transport_reach=("mean_abs_transport_reach", "median"),
            mean_abs_attention_reach=("mean_abs_attention_reach", "median"),
            max_abs_attention_reach=("max_abs_attention_reach", "max"),
            fraction_abs_attention_reach_ge_2_25=("fraction_abs_attention_reach_ge_2_25", "median"),
        ).to_csv(args.output / "reach_aggregate.csv", index=False)
    if attention_region_frames:
        regions = pd.concat(attention_region_frames, ignore_index=True)
        regions.to_csv(args.output / "attention_region_stats_by_seed.csv", index=False)
        regions.groupby(["problem", "cfl", "region", "head"], as_index=False).agg(
            seeds=("seed", "count"),
            faces=("faces", "median"),
            attention_entropy=("attention_entropy", "median"),
            left_right_asymmetry=("left_right_asymmetry", "median"),
            upwind_bias=("upwind_bias", "median"),
            center_mass=("center_mass", "median"),
            far_mass=("far_mass", "median"),
            absolute_center_of_mass=("absolute_center_of_mass", "median"),
            upwind_center_of_mass=("upwind_center_of_mass", "median"),
        ).to_csv(args.output / "attention_region_stats_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
