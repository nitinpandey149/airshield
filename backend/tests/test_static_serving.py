"""Tests for the optional single-origin dashboard mount.

`AIRSHIELD_SERVE_FRONTEND` is off by default so a local dev server keeps a plain
API at "/" while the frontend runs on its own Vite server. Docker turns it on so
the image serves API and dashboard from one origin.

The mount decision happens at import time, so these tests re-import `app.main`
with the flag set either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def built_dist(tmp_path: Path) -> Path:
    """A stand-in for frontend/dist: an index.html and one asset."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>AirShield dashboard</title></head>"
        '<body><div id="root"></div>'
        '<script src="/assets/app.js"></script></body></html>'
    )
    (dist / "assets" / "app.js").write_text("console.log('airshield')")
    return dist


def _client(monkeypatch, artifact_dir: Path, *, serve_frontend: bool, dist: Path) -> TestClient:
    """Import app.main with a given AIRSHIELD_SERVE_FRONTEND value and wrap it."""
    monkeypatch.setenv("AIRSHIELD_SERVE_FRONTEND", "true" if serve_frontend else "false")
    monkeypatch.setenv("AIRSHIELD_FRONTEND_DIST", str(dist))
    monkeypatch.setenv("AIRSHIELD_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("AIRSHIELD_DATA_MODE", "demo")
    monkeypatch.setenv("AIRSHIELD_INFERENCE_BACKEND", "local")

    from airshield_core.config import reset_settings_cache

    reset_settings_cache()

    # Drop cached modules so the module-level `if settings.serve_frontend` re-runs.
    for name in [m for m in sys.modules if m == "app.main" or m.startswith("app.main.")]:
        del sys.modules[name]

    import app.main as main

    from app import deps

    deps.reset_service()
    return TestClient(main.app)


def test_api_only_by_default(monkeypatch, artifact_dir, built_dist) -> None:
    """With the flag off, "/" stays a JSON banner even if dist exists."""
    with _client(monkeypatch, artifact_dir, serve_frontend=False, dist=built_dist) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.headers["content-type"].startswith("application/json")
        assert root.json()["name"] == "AirShield API"


def test_dashboard_is_served_when_enabled(monkeypatch, artifact_dir, built_dist) -> None:
    with _client(monkeypatch, artifact_dir, serve_frontend=True, dist=built_dist) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.headers["content-type"].startswith("text/html")
        assert "AirShield dashboard" in root.text

        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "airshield" in asset.text


def test_api_routes_win_over_the_static_mount(monkeypatch, artifact_dir, built_dist) -> None:
    """Mounting "/" must not shadow the API."""
    with _client(monkeypatch, artifact_dir, serve_frontend=True, dist=built_dist) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/info").json()["name"] == "AirShield API"
        assert client.get("/api/locations").status_code == 200


def test_api_404_stays_json_and_client_routes_get_the_shell(
    monkeypatch, artifact_dir, built_dist
) -> None:
    with _client(monkeypatch, artifact_dir, serve_frontend=True, dist=built_dist) as client:
        missing = client.get("/api/does-not-exist")
        assert missing.status_code == 404
        assert missing.headers["content-type"].startswith("application/json")

        page = client.get("/some/client/route")
        assert page.status_code == 200
        assert "AirShield dashboard" in page.text


def test_enabled_without_a_build_falls_back_to_the_banner(
    monkeypatch, artifact_dir, tmp_path
) -> None:
    """Flag on but no build present must degrade to the API banner, not 500."""
    with _client(
        monkeypatch, artifact_dir, serve_frontend=True, dist=tmp_path / "missing"
    ) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["name"] == "AirShield API"


def test_banner_never_leaks_credentials(monkeypatch, artifact_dir, built_dist) -> None:
    for flag in (False, True):
        with _client(monkeypatch, artifact_dir, serve_frontend=flag, dist=built_dist) as client:
            body = client.get("/api/info").json()
            assert body["name"] == "AirShield API"
            assert "secret" not in json.dumps(body).lower()
