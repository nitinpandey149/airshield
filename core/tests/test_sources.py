"""Tests for the data sources: schema conformance and demo-mode labelling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airshield_core.schema import HOURLY_COLUMNS, Observation
from airshield_core.sources.demo import DEMO_NOTICE, DemoSource
from airshield_core.sources.registry import LOCATIONS, LocationNotFoundError, get_location


def test_registry_lookup() -> None:
    assert get_location("delhi").name == "Delhi"
    with pytest.raises(LocationNotFoundError, match="unknown location"):
        get_location("atlantis")


def test_registry_locations_are_unique_and_well_formed() -> None:
    slugs = [loc.slug for loc in LOCATIONS]
    assert len(slugs) == len(set(slugs))
    for loc in LOCATIONS:
        assert -90 <= loc.latitude <= 90
        assert -180 <= loc.longitude <= 180
        assert loc.name and loc.country


def test_demo_dataset_matches_the_schema(demo_frame: pd.DataFrame) -> None:
    for column in HOURLY_COLUMNS:
        assert column in demo_frame.columns, f"demo dataset missing {column}"


def test_demo_dataset_is_real_not_synthetic(demo_frame: pd.DataFrame) -> None:
    """The bundled data must carry real measured variation, not constants."""
    pm = demo_frame["pm2_5"].dropna()
    assert len(pm) > 1000
    assert pm.std() > 1.0, "PM2.5 has no variation - suspicious for real data"
    assert pm.min() >= 0
    assert pm.max() < 1000


def test_demo_dataset_covers_multiple_locations(demo_frame: pd.DataFrame) -> None:
    locations = demo_frame["location_name"].unique()
    assert len(locations) >= 5


def test_demo_source_labels_itself_as_demo(demo_csv: Path) -> None:
    source = DemoSource(demo_csv.parents[1])
    info = source.source_info()
    assert info.mode == "demo"
    assert info.is_synthetic is False, "the bundled data is real, just historical"
    assert info.licence
    assert DEMO_NOTICE.startswith("DEMO MODE")


def test_demo_source_returns_observations(demo_csv: Path) -> None:
    source = DemoSource(demo_csv.parents[1])
    observations = source.fetch(28.6139, 77.209, "Delhi", window_hours=48)
    assert observations.location_name == "Delhi"
    assert len(observations.rows) == 48
    assert all(isinstance(row, Observation) for row in observations.rows)
    assert observations.source.mode == "demo"


def test_demo_prediction_frame_clears_the_final_pm25(demo_csv: Path) -> None:
    """The forecast rows must mimic the live path: forecast weather, no measurement."""
    source = DemoSource(demo_csv.parents[1])
    frame, base_time = source.fetch_frame_for_prediction("Delhi", window_hours=48)
    forecast_rows = frame[frame["is_forecast"]]
    assert len(forecast_rows) == 1
    assert forecast_rows["pm2_5"].isna().all()
    assert frame.loc[~frame["is_forecast"], "pm2_5"].notna().all()
    assert base_time == frame.loc[~frame["is_forecast"], "time"].max()


def test_demo_multi_horizon_frame(demo_csv: Path) -> None:
    source = DemoSource(demo_csv.parents[1])
    frame, base_time = source.fetch_frame_for_prediction("Delhi", window_hours=48, future_hours=6)
    assert frame["is_forecast"].sum() == 6
    assert frame[frame["is_forecast"]]["pm2_5"].isna().all()


def test_demo_source_reports_a_missing_file(tmp_path: Path) -> None:
    source = DemoSource(tmp_path)
    assert not source.is_available
    with pytest.raises(FileNotFoundError, match="demo dataset missing"):
        source.fetch_frame()
