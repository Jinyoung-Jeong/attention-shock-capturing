#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transport_attention_fv.utils import save_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suffixes = {".py", ".yaml", ".toml", ".sh", ".md"}
    files = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path.suffix in suffixes and "artifacts" not in path.parts
    )
    save_json(args.output, {
        "root": str(args.root.resolve()),
        "files": {
            path.relative_to(args.root).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        },
    })
    print(f"source manifest: {args.output}")


if __name__ == "__main__":
    main()
