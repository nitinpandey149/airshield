"""Shared fixtures for backend tests.

Every test that needs a model trains a small one into a temp directory, so the
suite never depends on a pre-existing artifact and never mocks a prediction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "ml" / "data" / "demo" / "demo_hourly.csv"

# Make `app` importable when pytest is run from the repository root.
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))


@pytest.fixture(scope="session")
def demo_csv() -> Path:
    if not DEMO_CSV.is_file():
        pytest.skip(f"bundled demo dataset not present at {DEMO_CSV}")
    return DEMO_CSV


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, demo_csv: Path) -> Path:
    """Train a real model on the real bundled data for the test session."""
    from airshield_core.train import train_model

    frame = pd.read_csv(demo_csv)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    out = tmp_path_factory.mktemp("backend-artifact")
    train_model(frame, artifact_dir=out, num_rounds=50)
    return out


@pytest.fixture
def client(artifact_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """A TestClient wired to the trained artifact in DEMO data mode."""
    monkeypatch.setenv("AIRSHIELD_INFERENCE_BACKEND", "local")
    monkeypatch.setenv("AIRSHIELD_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("AIRSHIELD_DATA_MODE", "demo")

    from airshield_core.config import reset_settings_cache

    reset_settings_cache()

    from app import deps
    from app.main import app

    deps.reset_service()
    app.dependency_overrides.clear()

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    deps.reset_service()
    reset_settings_cache()


@pytest.fixture
def live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the process to require live upstream data."""
    monkeypatch.setenv("AIRSHIELD_DATA_MODE", "live")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()


@pytest.fixture(autouse=True)
def _no_real_aws_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test accidentally tries to reach AWS without opting in."""
    if os.environ.get("AIRSHIELD_ALLOW_AWS_TESTS") == "1":
        return
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "")
