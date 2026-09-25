"""Shared pytest fixtures for the AirShield core tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "ml" / "data" / "demo" / "demo_hourly.csv"


@pytest.fixture(scope="session")
def demo_csv() -> Path:
    if not DEMO_CSV.is_file():
        pytest.skip(f"bundled demo dataset not present at {DEMO_CSV}")
    return DEMO_CSV


@pytest.fixture(scope="session")
def demo_frame(demo_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(demo_csv)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, demo_csv: Path) -> Path:
    """Train a real model on the real bundled dataset, once per session.

    Several test modules need a trained artifact. Training here means the tests
    exercise genuine inference rather than a stub.
    """
    from airshield_core.train import train_model

    frame = pd.read_csv(demo_csv)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    out = tmp_path_factory.mktemp("artifact")
    train_model(frame, artifact_dir=out, num_rounds=50)
    return out


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A deterministic, clearly synthetic frame for unit-level assertions.

    Only used to test *mechanics* (lag alignment, leakage, warm-up). All
    reported model metrics come from real data, never from this fixture.
    """
    rng = np.random.default_rng(7)
    hours = 200
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(hours)]

    # Smooth daily cycle plus noise, so lags are meaningfully correlated.
    pm = 20 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.normal(0, 1.5, hours)
    pm = np.clip(pm, 1.0, None)

    return pd.DataFrame(
        {
            "time": times,
            "pm2_5": pm,
            "pm10": pm * 1.6,
            "nitrogen_dioxide": rng.uniform(5, 40, hours),
            "ozone": rng.uniform(10, 80, hours),
            "temperature_2m": 15 + 8 * np.sin(np.arange(hours) * 2 * np.pi / 24),
            "relative_humidity_2m": rng.uniform(30, 90, hours),
            "wind_speed_10m": rng.uniform(0.5, 25, hours),
            "wind_direction_10m": rng.uniform(0, 360, hours),
            "surface_pressure": rng.uniform(1000, 1025, hours),
            "precipitation": rng.choice([0.0, 0.0, 0.0, 1.2], hours),
        }
    )
