#!/usr/bin/env python3
"""Inline the Lambda handler sources into the CloudFormation template.

The handlers under ``infra/lambda/*/handler.py`` are the tested code. Embedding
that exact text in the template's ``ZipFile`` block means a single
``aws cloudformation deploy`` produces working functions, instead of creating
placeholder stubs that then need a manual ``update-function-code`` step.

Both handlers import only the standard library plus ``boto3``, which the Lambda
runtime already provides, so no layer or dependency packaging is required.

Run this after editing a handler. ``infra/tests/test_template_sync.py`` fails if
the two ever drift.

    python infra/sync_template_code.py           # rewrite the template
    python infra/sync_template_code.py --check   # verify only, exit 1 if stale
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "infra" / "cloudformation" / "airshield-pulse.yaml"
HANDLERS = {
    "ingest": ROOT / "infra" / "lambda" / "ingest" / "handler.py",
    "spike": ROOT / "infra" / "lambda" / "spike" / "handler.py",
}

# `ZipFile: |` sits at 8 spaces, so its literal block content is indented 10.
_BLOCK = re.compile(
    r"(?P<header>        ZipFile: \|\n)(?P<body>(?: {10,}\S.*\n| {10,}\n|\n)*)",
)


def _block_code(source: str) -> str:
    """Render handler source as an indented YAML literal block."""
    lines = source.rstrip("\n").split("\n")
    return "".join(f"          {line}\n" if line else "\n" for line in lines)


def render() -> str:
    """Return the template text with both ZipFile blocks replaced."""
    template = TEMPLATE.read_text(encoding="utf-8")

    matches = list(_BLOCK.finditer(template))
    if len(matches) != len(HANDLERS):
        raise SystemExit(
            f"expected {len(HANDLERS)} ZipFile blocks, found {len(matches)}"
        )

    # The blocks appear in the same order as the handlers are declared.
    ordered = list(HANDLERS.values())
    pieces: list[str] = []
    cursor = 0
    for match, handler in zip(matches, ordered):
        pieces.append(template[cursor : match.start()])
        pieces.append(match.group("header"))
        pieces.append(_block_code(handler.read_text(encoding="utf-8")))
        cursor = match.end()
    pieces.append(template[cursor:])
    return "".join(pieces)


def main(argv: list[str]) -> int:
    rendered = render()
    current = TEMPLATE.read_text(encoding="utf-8")

    if "--check" in argv:
        if rendered != current:
            print("template ZipFile blocks are stale; run: python infra/sync_template_code.py")
            return 1
        print("template ZipFile blocks match the handler sources")
        return 0

    if rendered == current:
        print("template already in sync")
        return 0
    TEMPLATE.write_text(rendered, encoding="utf-8")
    print(f"updated {TEMPLATE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
