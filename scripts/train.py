#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transport_attention_fv.training import train
from transport_attention_fv.utils import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=("main", "cfl_blind", "uniform"), default="main")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = resolve_path(args.config, config["output"]["directory"]) / config["problem"]
    output = root / f"seed_{args.seed}" if args.variant == "main" else root / "ablations" / args.variant / f"seed_{args.seed}"
    intervention = "learned" if args.variant == "main" else args.variant
    train(
        config, args.config.resolve(), args.seed, torch.device(args.device), output,
        intervention=intervention,
    )


if __name__ == "__main__":
    main()
