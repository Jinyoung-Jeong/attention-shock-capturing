#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transport_attention_fv.models import build_model
from transport_attention_fv.utils import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record trainable parameter counts")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.config:
        config = load_yaml(path)
        model = build_model(config["problem"], config["model"])
        rows.append({
            "config": str(path),
            "problem": config["problem"],
            "method_label": config["evaluation"]["method_label"],
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        })
    if len(rows) == 2:
        rows[1]["relative_parameter_difference_from_first"] = (
            rows[1]["trainable_parameters"] / rows[0]["trainable_parameters"] - 1.0
        )
    save_json(args.output, {"models": rows})
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
