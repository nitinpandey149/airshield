"""Build the bundled DEMO_MODE dataset from real Open-Meteo measurements.

Run this once to capture a frozen historical window into
``ml/data/demo/demo_hourly.csv``. The file is real measured data - it is
"demo" only in the sense that it is a fixed past window rather than live.

    python scripts/build_demo_dataset.py --days 45
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "core" / "src"))

import pandas as pd  # noqa: E402

from airshield_core.schema import HOURLY_COLUMNS  # noqa: E402
from airshield_core.sources.openmeteo import (  # noqa: E402
    ATTRIBUTION,
    AIR_QUALITY_URL,
    UpstreamError,
    default_history_window,
    fetch_history,
)
from airshield_core.sources.registry import LOCATIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=45, help="length of the captured window")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else REPO_ROOT / "ml" / "data" / "demo" / "demo_hourly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start, end = default_history_window(args.days)
    print(f"capturing real Open-Meteo measurements {start} -> {end}")

    frames = []
    for location in LOCATIONS:
        try:
            part = fetch_history(location.latitude, location.longitude, start, end, timeout=45)
        except UpstreamError as exc:
            print(f"  ! {location.slug}: {exc}")
            continue
        part["location_name"] = location.name
        part["location_slug"] = location.slug
        part["latitude"] = location.latitude
        part["longitude"] = location.longitude
        frames.append(part)
        print(f"  + {location.slug:<12} {len(part):>6} rows")

    if not frames:
        print("ERROR: no data captured; the demo dataset was not written.")
        return 1

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["pm2_5"]).sort_values(["location_slug", "time"])

    ordered = ["time", "location_slug", "location_name", "latitude", "longitude"] + list(HOURLY_COLUMNS)
    frame = frame[[c for c in ordered if c in frame.columns]]
    frame.to_csv(out_path, index=False)

    manifest = {
        "description": (
            "Frozen historical window of real hourly air-quality and weather "
            "measurements, captured from Open-Meteo for DEMO_MODE."
        ),
        "source": "Open-Meteo",
        "url": AIR_QUALITY_URL,
        "licence": ATTRIBUTION,
        "is_synthetic": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": str(start), "end": str(end)},
        "rows": int(len(frame)),
        "locations": sorted(frame["location_slug"].unique().tolist()),
        "columns": list(frame.columns),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {len(frame)} real rows -> {out_path}")
    print(f"wrote manifest      -> {out_path.parent / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
