"""Fetch and extract the AirShield knowledge base.

One job: turn the URLs in ``manifest.json`` into plain-text files under
``sources/``, verbatim. There is no summarising, rewriting or generating here —
if a page cannot be fetched or yields no usable text, that is recorded as a
failure rather than filled in with invented content.

    python knowledge/fetch_sources.py            # fetch anything missing
    python knowledge/fetch_sources.py --refresh  # re-fetch everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
SOURCES = ROOT / "sources"
USER_AGENT = "AirShield-Pulse/1.0 (+knowledge-base; research use)"

#: Minimum characters for an extract to be considered usable. A page that
#: yields less than this is almost certainly a consent wall or an error page.
MIN_CHARS = 400


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _extract_html(raw: bytes, url: str) -> str:
    import trafilatura

    text = trafilatura.extract(
        raw,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    return text or ""


def _extract_pdf(raw: bytes) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def extract(raw: bytes, url: str) -> str:
    """Extract readable text from a response body, dispatching on content type."""
    if url.lower().endswith(".pdf") or raw[:5] == b"%PDF-":
        return _extract_pdf(raw)
    return _extract_html(raw, url)


def _normalise(text: str) -> str:
    """Collapse whitespace and drop obvious boilerplate lines."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # Skip nav/cookie boilerplate that survives extraction.
        lowered = stripped.lower()
        if lowered in {"share", "print", "subscribe", "sign up", "menu", "search"}:
            continue
        if len(stripped) <= 2 and not stripped.isdigit():
            continue
        lines.append(stripped)
    # Collapse runs of blank lines.
    out: list[str] = []
    for line in lines:
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out).strip()


def fetch_all(refresh: bool = False) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    SOURCES.mkdir(parents=True, exist_ok=True)

    failures: list[dict] = []
    written: list[dict] = []

    for document in manifest["documents"]:
        doc_id = document["id"]
        target = SOURCES / f"{doc_id}.txt"
        if target.is_file() and not refresh:
            written.append({"id": doc_id, "chars": target.stat().st_size, "cached": True})
            continue

        try:
            raw = _fetch(document["url"])
            text = _normalise(extract(raw, document["url"]))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            failures.append({"id": doc_id, "url": document["url"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # extraction errors are recorded, never hidden
            failures.append({"id": doc_id, "url": document["url"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        if len(text) < MIN_CHARS:
            failures.append(
                {
                    "id": doc_id,
                    "url": document["url"],
                    "error": f"extracted only {len(text)} chars (minimum {MIN_CHARS}); likely a consent wall or empty page",
                }
            )
            print(f"  FAIL {doc_id}: only {len(text)} chars extracted", file=sys.stderr)
            continue

        target.write_text(text, encoding="utf-8")
        written.append({"id": doc_id, "chars": len(text), "cached": False})
        print(f"  ok   {doc_id}: {len(text)} chars")

    # Provenance record for the whole run.
    record = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": USER_AGENT,
        "documents_ok": written,
        "documents_failed": failures,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    }
    (ROOT / "fetch_report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n{len(written)} document(s) ready, {len(failures)} failed")
    if failures:
        print("Failures are recorded in knowledge/fetch_report.json and are NOT")
        print("substituted with generated text. Fix the URL or remove the entry.")
    return 1 if failures and not written else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached sources")
    args = parser.parse_args(argv)
    return fetch_all(refresh=args.refresh)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
