#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROFILE_FEATURES = [
    ("smooth", "upwind_center_of_mass"),
    ("smooth", "attention_entropy"),
    ("shock", "upwind_center_of_mass"),
    ("shock", "attention_entropy"),
    ("shock", "upwind_bias"),
    ("shock", "center_mass"),
]


def _profile_matrix(frame: pd.DataFrame, cfl: float) -> tuple[list[int], np.ndarray]:
    selected = frame[(frame["cfl"] == cfl) & (frame["head"].astype(str) != "mean")].copy()
    selected["head"] = selected["head"].astype(int)
    heads = sorted(selected["head"].unique())
    matrix = []
    for head in heads:
        values = []
        for region, metric in PROFILE_FEATURES:
            row = selected[(selected["head"] == head) & (selected["region"] == region)]
            if len(row) != 1:
                raise ValueError((head, region, metric, len(row)))
            values.append(float(row.iloc[0][metric]))
        matrix.append(values)
    return heads, np.asarray(matrix, dtype=float)


def _reference_roles(reference: np.ndarray) -> dict[int, str]:
    reach = reference[:, 0]
    entropy = reference[:, 3]
    order = np.argsort(reach)
    shorter, longer = order[:2], order[2:]
    roles: dict[int, str] = {}
    for reach_name, members in (("shorter_reach", shorter), ("longer_reach", longer)):
        selective = int(members[np.argmin(entropy[members])])
        diffuse = int(members[np.argmax(entropy[members])])
        roles[selective] = f"{reach_name}_selective"
        roles[diffuse] = f"{reach_name}_diffuse"
    return roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Align and aggregate functional attention-head profiles")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cfl", default=1.6, type=float)
    args = parser.parse_args()

    region_frames, intervention_frames = [], []
    for path in sorted(args.runs.glob("burgers_1d/seed_*/diagnostics/attention_region_stats_by_cfl.csv")):
        seed = int(path.parts[-3].split("_")[-1])
        intervention_path = path.parent / "head_interventions.csv"
        if not intervention_path.exists():
            print(f"skipping seed {seed}: missing {intervention_path}")
            continue
        frame = pd.read_csv(path, dtype={"head": str})
        frame.insert(0, "seed", seed)
        region_frames.append(frame)
        intervention_frames.append(pd.read_csv(intervention_path, dtype={"head": str}))
    if not region_frames:
        raise FileNotFoundError(f"No head profiles under {args.runs}")

    profiles = pd.concat(region_frames, ignore_index=True)
    seeds = sorted(profiles["seed"].unique())
    matrices = {}
    head_lists = {}
    for seed in seeds:
        heads, matrix = _profile_matrix(profiles[profiles["seed"] == seed], args.cfl)
        head_lists[seed], matrices[seed] = heads, matrix
    stacked = np.vstack(list(matrices.values()))
    center = stacked.mean(0)
    scale = stacked.std(0)
    scale[scale < 1.0e-12] = 1.0
    normalized = {seed: (matrix - center) / scale for seed, matrix in matrices.items()}

    reference_seed = seeds[0]
    reference = normalized[reference_seed]
    roles_by_reference_index = _reference_roles(matrices[reference_seed])
    mappings = []
    for seed in seeds:
        cost = ((normalized[seed][:, None, :] - reference[None, :, :]) ** 2).sum(-1)
        source_indices, reference_indices = linear_sum_assignment(cost)
        for source_index, reference_index in zip(source_indices, reference_indices):
            mappings.append({
                "seed": seed,
                "raw_head": head_lists[seed][source_index],
                "reference_seed": reference_seed,
                "reference_head": head_lists[reference_seed][reference_index],
                "functional_role": roles_by_reference_index[int(reference_index)],
                "matching_cost": float(cost[source_index, reference_index]),
            })
    mapping = pd.DataFrame(mappings)
    args.output.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output / "head_alignment.csv", index=False)

    numeric_profiles = profiles[profiles["head"].astype(str) != "mean"].copy()
    numeric_profiles["raw_head"] = numeric_profiles["head"].astype(int)
    aligned_profiles = numeric_profiles.merge(mapping, on=["seed", "raw_head"], how="left")
    aligned_profiles.to_csv(args.output / "head_profiles_aligned_by_seed.csv", index=False)
    metrics = [
        "attention_entropy", "left_right_asymmetry", "upwind_bias", "center_mass",
        "far_mass", "absolute_center_of_mass", "upwind_center_of_mass",
    ]
    aggregations = {metric: ["median", "min", "max"] for metric in metrics}
    aligned_profiles.groupby(["cfl", "region", "functional_role"], as_index=False).agg(
        aggregations
    ).to_csv(args.output / "head_profiles_aggregate.csv", index=False)

    interventions = pd.concat(intervention_frames, ignore_index=True)
    interventions = interventions[interventions["head"] != "all"].copy()
    interventions["raw_head"] = interventions["head"].astype(int)
    aligned_interventions = interventions.merge(mapping, on=["seed", "raw_head"], how="left")
    aligned_interventions.to_csv(args.output / "head_interventions_aligned_by_seed.csv", index=False)
    aligned_interventions.groupby(
        ["cfl", "intervention", "functional_role"], as_index=False
    ).agg(
        seeds=("seed", "count"),
        success_rate_median=("success_rate", "median"),
        success_rate_min=("success_rate", "min"),
        relative_l2_change_median=("relative_l2_change_from_learned", "median"),
        relative_l2_change_min=("relative_l2_change_from_learned", "min"),
        relative_l2_change_max=("relative_l2_change_from_learned", "max"),
        relative_flux_change_median=("median_relative_first_step_flux_change", "median"),
    ).to_csv(args.output / "head_interventions_aggregate.csv", index=False)
    print(f"head analysis: {args.output}")


if __name__ == "__main__":
    main()
