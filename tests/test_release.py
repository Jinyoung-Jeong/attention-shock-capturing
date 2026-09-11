import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from transport_attention_fv.evaluation import _conserved_totals, evaluate
from transport_attention_fv.utils import load_tensor_file


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("problem", ["burgers_1d", "burgers_2d", "shallow_water_1d"])
def test_released_checkpoint_evaluates_without_training_metadata(tmp_path, problem):
    config_path = ROOT / "configs/production" / f"{problem}.yaml"
    config = yaml.safe_load(config_path.read_text())
    sample = load_tensor_file(ROOT / "data/samples" / f"{problem}.pt")
    sample["states"] = sample["states"][:1, :2]
    sample["dt_base"] = sample["dt_base"][:1]
    sample["parameters"] = {k: v[:1] for k, v in sample["parameters"].items()}
    data_dir = tmp_path / "data" / problem
    data_dir.mkdir(parents=True)
    torch.save(sample, data_dir / "test.pt")
    config["data"].update(directory=str(data_dir.parent), base_frames=1)
    config["evaluation"].update(cfl_values=[0.4], batch_size=1, timing_samples=1,
                                timing_warmup=0, timing_repeats=1)
    checkpoint = ROOT / "checkpoints/samples" / f"{problem}_seed_2027.pt"
    assert "epoch" not in load_tensor_file(checkpoint)
    output = tmp_path / "evaluation"
    evaluate(config, config_path, checkpoint, torch.device("cpu"), output)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["checkpoint_epoch"] is None
    assert metadata["checkpoint"] == checkpoint.name
    with (output / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(int(r["samples"]) == 1 for r in rows)
    if problem == "shallow_water_1d":
        assert all(float(r["mean_depth_conservation_error"]) >= 0 for r in rows)
        assert all(float(r["mean_discharge_conservation_error"]) >= 0 for r in rows)


def test_conservation_does_not_cancel_depth_against_discharge():
    before = torch.ones(1, 4, 2, dtype=torch.float64)
    after = before + torch.tensor([0.125, -0.125], dtype=torch.float64)
    assert after.sum() == before.sum()  # Old mixed-component diagnostic misses this.
    errors = (_conserved_totals("shallow_water_1d", after)
              - _conserved_totals("shallow_water_1d", before)).abs()
    torch.testing.assert_close(errors, torch.tensor([[0.5, 0.5]], dtype=torch.float64))


def test_loader_rejects_unapproved_pickle_objects(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save({"value": datetime.datetime(2026, 1, 1)}, path)
    import pickle
    with pytest.raises(pickle.UnpicklingError):
        load_tensor_file(path)


def test_loader_rejects_old_runtime_before_deserialization(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    with pytest.raises(RuntimeError, match="PyTorch >= 2.14"):
        load_tensor_file(tmp_path / "not_read.pt")


def test_nonpositive_depth_is_not_erased_by_a_later_nan():
    minimum = torch.tensor([-0.1])
    minimum = torch.fmin(minimum, torch.tensor([float("nan")]))
    assert bool(minimum <= 0)


def test_release_parts_reassemble_and_reject_corruption(tmp_path):
    pieces = [b"first segment", b"second segment"]
    names = [f"burgers_2d_train.zip.part{i}" for i in (1, 2)]
    for name, data in zip(names, pieces):
        (tmp_path / name).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in zip(names, pieces)}
    hashes["burgers_2d_train.zip"] = hashlib.sha256(b"".join(pieces)).hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
    command = [sys.executable, str(ROOT / "scripts/prepare_release_assets.py"),
               "--assets", str(tmp_path), "--join"]
    subprocess.run(command, check=True, capture_output=True)
    assert (tmp_path / "burgers_2d_train.zip").read_bytes() == b"".join(pieces)
    (tmp_path / names[0]).write_bytes(b"corrupt")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
