"""End-to-end API tests against a real trained model and real data."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_root_reports_configuration(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "AirShield API"
    assert body["ml_service"] == "Amazon SageMaker AI (XGBoost)"
    assert "forecast" in body["endpoints"]


def test_health_reports_ok_with_a_model(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["model_version"].startswith("xgboost-pm25-1h-")
    assert body["data_mode"] == "demo"


def test_locations_are_listed(client) -> None:
    response = client.get("/api/locations")
    assert response.status_code == 200
    locations = response.json()
    assert len(locations) >= 5
    berlin = next(loc for loc in locations if loc["slug"] == "berlin")
    assert berlin["label"] == "Berlin, Germany"


def test_forecast_returns_a_real_prediction(client) -> None:
    response = client.get("/api/forecast/berlin")
    assert response.status_code == 200
    body = response.json()

    prediction = body["forecast"]["predicted_pm25"]
    assert isinstance(prediction, float)
    assert prediction >= 0
    assert prediction < 1000, "implausible magnitude suggests a broken model"
    assert body["forecast"]["horizon_hours"] == 1
    assert body["forecast"]["backend"] == "local"


def test_forecast_reports_real_training_metrics(client) -> None:
    """The metrics in the response must be the measured ones, not placeholders."""
    body = client.get("/api/forecast/berlin").json()
    metrics = body["forecast"]["model_metrics"]
    assert metrics, "training metrics should be surfaced"
    assert metrics["rmse"] > 0
    assert metrics["n"] > 0
    assert -1 <= metrics["r2"] <= 1


def test_prediction_targets_the_next_hour(client) -> None:
    body = client.get("/api/forecast/berlin").json()
    base = datetime.fromisoformat(body["forecast"]["base_time"].replace("Z", "+00:00"))
    target = datetime.fromisoformat(body["forecast"]["target_time"].replace("Z", "+00:00"))
    assert (target - base).total_seconds() == 3600


def test_forecast_includes_alert_and_aqi(client) -> None:
    body = client.get("/api/forecast/berlin").json()
    alert, aqi = body["alert"], body["aqi"]

    assert alert["severity"] in {
        "good", "moderate", "elevated", "high", "very_high", "hazardous"
    }
    assert alert["headline"] and alert["advice"] and alert["sensitive_group_advice"]
    assert 0 <= aqi["aqi"] <= 500
    # The alert AQI must agree with the AQI block - one source of truth.
    assert alert["aqi"] == aqi["aqi"]
    assert alert["category"] == aqi["category"]


def test_aqi_matches_the_prediction(client) -> None:
    """Recompute AQI from the returned prediction and compare."""
    from airshield_core.aqi import aqi_from_pm25

    body = client.get("/api/forecast/berlin").json()
    expected = aqi_from_pm25(body["forecast"]["predicted_pm25"])
    assert body["aqi"]["aqi"] == expected.aqi
    assert body["aqi"]["category"] == expected.category


def test_history_ends_with_the_prediction(client) -> None:
    body = client.get("/api/forecast/berlin").json()
    history = body["history"]
    assert len(history) > 10
    assert history[-1]["predicted"] is True
    assert history[-1]["pm2_5"] == pytest.approx(body["forecast"]["predicted_pm25"])
    assert all(point["predicted"] is False for point in history[:-1])


def test_demo_mode_is_clearly_labelled(client) -> None:
    """A demo response must never look like a live reading."""
    body = client.get("/api/forecast/berlin").json()
    assert body["source"]["mode"] == "demo"
    assert body["notice"] and "DEMO MODE" in body["notice"]
    assert body["source"]["is_synthetic"] is False
    assert body["source"]["licence"]


def test_all_locations_produce_a_forecast(client) -> None:
    for location in client.get("/api/locations").json():
        response = client.get(f"/api/forecast/{location['slug']}")
        assert response.status_code == 200, f"{location['slug']} failed"
        assert response.json()["forecast"]["predicted_pm25"] >= 0


def test_unknown_location_returns_404(client) -> None:
    response = client.get("/api/forecast/atlantis")
    assert response.status_code == 404
    assert "unknown location" in response.json()["detail"]


def test_model_card_exposes_provenance(client) -> None:
    body = client.get("/api/model").json()
    assert body["available"] is True
    assert body["train_rows"] > 0
    assert body["test_rows"] > 0
    assert body["metrics"]["rmse"] > 0
    assert body["baseline_metrics"]["rmse"] > 0
    assert body["feature_count"] > 0
    assert body["train_window"]["start"]
    assert body["top_features"], "feature importances should be reported"


def test_model_card_metrics_come_from_metadata(client, artifact_dir) -> None:
    """Cross-check the API against the artifact on disk."""
    import json

    on_disk = json.loads((artifact_dir / "metadata.json").read_text())
    body = client.get("/api/model").json()
    assert body["metrics"]["rmse"] == on_disk["metrics"]["rmse"]
    assert body["trained_at"] == on_disk["trained_at"]


def test_no_secrets_are_exposed(client) -> None:
    """Responses must never leak credentials."""
    import json

    for path in ("/", "/api/health", "/api/model", "/api/forecast/berlin"):
        payload = json.dumps(client.get(path).json()).lower()
        for needle in ("secret", "access_key", "password", "github_pat", "token"):
            assert needle not in payload, f"{needle} leaked via {path}"
