#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transport_attention_fv.evaluation import evaluate
from transport_attention_fv.utils import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--variant", choices=("main", "cfl_blind", "uniform"), default="main")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = resolve_path(args.config, config["output"]["directory"]) / config["problem"]
    run = root / f"seed_{args.seed}" if args.variant == "main" else root / "ablations" / args.variant / f"seed_{args.seed}"
    checkpoint = run / "checkpoint_best_large_step.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    evaluate(
        config, args.config.resolve(), checkpoint, torch.device(args.device), run / "evaluation",
        include_baselines=not args.attention_only,
        attention_intervention="learned" if args.variant == "main" else args.variant,
    )


if __name__ == "__main__":
    main()
