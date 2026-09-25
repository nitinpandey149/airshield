"""DEMO_MODE data source backed by a bundled sample dataset.

The CSV in ``ml/data/demo/`` is a frozen, real historical window captured from
Open-Meteo (see ``ml/data/demo/README.md`` for the exact provenance). It is used
only when live upstream data is unavailable, and every response built from it is
labelled ``mode="demo"`` / ``is_synthetic=false`` with a clear notice so a user
can never mistake it for a current reading.

The file is real measured data, not generated numbers; "demo" refers to the fact
that it is a fixed historical snapshot rather than *now*.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from airshield_core.schema import Observation, Observations, SourceInfo

DEMO_NOTICE = (
    "DEMO MODE: showing a bundled historical sample captured from Open-Meteo, "
    "not current conditions. Live upstream data was unavailable."
)


class DemoSource:
    """Reads a frozen historical window from a local CSV."""

    name = "Open-Meteo (bundled sample)"

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "demo" / "demo_hourly.csv"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "demo" / "manifest.json"

    @property
    def is_available(self) -> bool:
        return self.csv_path.is_file()

    def _manifest(self) -> dict:
        if self.manifest_path.is_file():
            import json

            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def fetch_frame(self, location_name: str | None = None) -> pd.DataFrame:
        """Load the bundled frame, optionally filtered to one location."""
        if not self.is_available:
            raise FileNotFoundError(
                f"demo dataset missing at {self.csv_path}. "
                "Run `python scripts/build_demo_dataset.py` to create it."
            )
        frame = pd.read_csv(self.csv_path)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        if location_name and "location_name" in frame.columns:
            frame = frame[frame["location_name"] == location_name]
        return frame.sort_values("time").reset_index(drop=True)

    def source_info(self) -> SourceInfo:
        manifest = self._manifest()
        return SourceInfo(
            name=self.name,
            url=manifest.get("url", "https://open-meteo.com/"),
            mode="demo",
            licence=manifest.get(
                "licence", "Weather and air-quality data by Open-Meteo.com (CC BY 4.0)"
            ),
            fetched_at=datetime.now(timezone.utc),
            is_synthetic=False,
        )

    def fetch(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        *,
        window_hours: int = 72,
    ) -> Observations:
        """Return the trailing ``window_hours`` of the bundled series as observations."""
        frame = self.fetch_frame(location_name)
        if frame.empty:
            raise FileNotFoundError(
                f"demo dataset has no rows for location {location_name!r}"
            )
        frame = frame.tail(window_hours)
        rows = [
            Observation(**{k: v for k, v in record.items() if k in Observation.model_fields})
            for record in frame.to_dict("records")
        ]
        return Observations(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            source=self.source_info(),
            rows=rows,
        )

    def fetch_frame_for_prediction(
        self, location_name: str, *, window_hours: int = 72
    ) -> pd.DataFrame:
        """Bundled frame trimmed to end with a forecast-weather row.

        The bundled CSV has measured PM2.5 on every row, so the last row is used
        as the "next hour" weather row and its PM2.5 is cleared, mirroring the
        live path exactly.
        """
        frame = self.fetch_frame(location_name).tail(window_hours + 1).reset_index(drop=True)
        if len(frame) < 2:
            raise ValueError(f"not enough bundled rows for {location_name!r}")
        frame.loc[frame.index[-1], "pm2_5"] = float("nan")
        return frame
