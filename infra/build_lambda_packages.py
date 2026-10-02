#!/usr/bin/env python3
"""Build the two AWS Lambda deployment packages.

Uses only the standard library so it works on any machine, unlike `zip`, which
is not always installed. Packages are written next to the handlers and are
gitignored build output.
"""

from __future__ import annotations

import pathlib
import zipfile

BASE = pathlib.Path(__file__).resolve().parent / "lambda"


def build(name: str) -> pathlib.Path:
    source = BASE / name
    if not source.is_dir():
        raise SystemExit(f"handler directory not found: {source}")

    target = BASE / f"{name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            archive.write(path, path.relative_to(BASE))

    print(f"{target}  {target.stat().st_size} bytes")
    return target


def main() -> int:
    for name in ("ingest", "spike"):
        build(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
