from __future__ import annotations

from pathlib import Path

import torch
import yaml

from transport_attention_fv.models import build_model
from transport_attention_fv.training import advance_model
from transport_attention_fv.utils import load_tensor_file


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ("burgers_1d", "burgers_2d", "shallow_water_1d")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _conserved_sum(problem: str, state: torch.Tensor) -> torch.Tensor:
    if problem == "burgers_2d":
        return state.sum(dim=(-2, -1))
    if problem == "shallow_water_1d":
        return state.sum(dim=-2)
    return state.sum(dim=-1)


def run_problem(problem: str) -> None:
    config = _load_yaml(ROOT / "configs" / "production" / f"{problem}.yaml")
    sample = load_tensor_file(
        ROOT / "data" / "samples" / f"{problem}.pt",
        map_location="cpu",
    )
    checkpoint = load_tensor_file(
        ROOT / "checkpoints" / "samples" / f"{problem}_seed_2027.pt",
        map_location="cpu",
    )

    model = build_model(problem, config["model"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    state = sample["states"][:, 0]
    dt = sample["dt_base"]
    batch = {key: value for key, value in sample.get("parameters", {}).items()}

    with torch.no_grad():
        prediction = advance_model(problem, model, state, dt, batch)

    if not torch.isfinite(prediction).all():
        raise RuntimeError(f"{problem}: non-finite prediction")
    before = _conserved_sum(problem, state)
    after = _conserved_sum(problem, prediction)
    conservation_error = float((after - before).abs().max())
    if conservation_error > 1.0e-4:
        raise RuntimeError(
            f"{problem}: total-sum conservation error {conservation_error:.3e}"
        )
    if problem == "shallow_water_1d" and float(prediction[..., 0].min()) <= 0.0:
        raise RuntimeError("shallow_water_1d: non-positive depth")

    target = sample["states"][:, 1]
    l2 = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    print(
        f"{problem}: OK (one-step L2={l2:.6e}, "
        f"conservation={conservation_error:.3e})"
    )


def main() -> None:
    for problem in PROBLEMS:
        run_problem(problem)
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
