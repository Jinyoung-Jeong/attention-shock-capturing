#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-gib", type=float, default=0.0)
    args = parser.parse_args()
    expected = (2, 5, 1)
    current = tuple(int(part.split("+")[0]) for part in torch.__version__.split(".")[:3])
    if current < expected:
        raise RuntimeError(f"PyTorch >=2.5.1 required, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 2**30
    if free_gib < args.minimum_free_gib:
        raise RuntimeError(
            f"GPU is not sufficiently free: {free_gib:.1f} GiB available, "
            f"{args.minimum_free_gib:.1f} GiB required"
        )
    probe = torch.randn(64, 64, device="cuda")
    _ = probe @ probe
    torch.cuda.synchronize()
    print(json.dumps({
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "capability": capability,
        "free_gib": round(free_gib, 2), "total_gib": round(total_bytes / 2**30, 2),
    }))


if __name__ == "__main__":
    main()
