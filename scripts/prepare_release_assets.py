"""Verify downloaded release assets and optionally rejoin the 2D training ZIP."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--join", action="store_true")
    args = parser.parse_args()
    root = args.assets.resolve()
    expected = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
            raise ValueError("Checksum manifest must contain plain filenames")
        expected[name] = checksum
    checked = 0
    for name, checksum in expected.items():
        path = root / name
        if path.is_file():
            if digest(path) != checksum:
                raise ValueError(f"SHA-256 mismatch: {name}")
            checked += 1
            print(f"verified: {name}", flush=True)
    if not checked:
        raise FileNotFoundError("No listed assets were found")
    if args.join:
        parts = [root / f"burgers_2d_train.zip.part{i}" for i in (1, 2)]
        if any(p.name not in expected or not p.is_file() for p in parts):
            raise FileNotFoundError("Download both 2D training parts and SHA256SUMS.txt")
        target = root / "burgers_2d_train.zip"
        if target.name not in expected:
            raise ValueError("Missing checksum for the reconstructed ZIP")
        if target.exists():
            print("Reconstructed ZIP already verified; leaving it unchanged.")
            return
        temporary = target.with_suffix(".zip.joining")
        with temporary.open("xb") as output:
            for part in parts:
                with part.open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
        if digest(temporary) != expected[target.name]:
            raise ValueError("Reconstructed ZIP checksum mismatch; not installing it")
        temporary.rename(target)
        print("Reconstructed ZIP matches the original byte for byte.")


if __name__ == "__main__":
    main()
