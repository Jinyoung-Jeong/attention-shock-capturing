from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_tensor_file(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load tensor/dictionary artifacts without unrestricted pickle fallback."""
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 14):
        raise RuntimeError("The public loader requires PyTorch >= 2.14; see README.md.")
    return torch.load(path, map_location=map_location, weights_only=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def runtime_manifest() -> dict[str, Any]:
    result = {
        "python_pid": os.getpid(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return result


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    config_path = Path(config_path).resolve()
    root = next((parent for parent in config_path.parents if (parent / "pyproject.toml").exists()), config_path.parent)
    return (root / value).resolve()
