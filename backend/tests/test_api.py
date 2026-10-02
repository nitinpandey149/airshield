"""End-to-end API tests against a real trained model and real data."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_root_reports_configuration(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "AirShield Pulse API"
    assert body["ml_service"] == "Amazon SageMaker AI (XGBoost)"
    assert "forecast" in body["endpoints"]
    assert "plan" in body["endpoints"]
    assert "routes" in body["endpoints"]


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
    delhi = next(loc for loc in locations if loc["slug"] == "delhi")
    assert delhi["label"] == "Delhi, India"


def test_forecast_returns_a_real_prediction(client) -> None:
    response = client.get("/api/forecast/delhi")
    assert response.status_code == 200
    body = response.json()

    prediction = body["forecast"]["predicted_pm25"]
    assert isinstance(prediction, float)
    assert prediction >= 0
    assert prediction < 1000, "implausible magnitude suggests a broken model"
    assert body["forecast"]["horizon_hours"] == 1
    assert body["forecast"]["backend"] == "local"


def test_forecast_reports_only_trained_horizons(client) -> None:
    """Horizons are reported per model; none is claimed without an artifact."""
    body = client.get("/api/forecast/delhi").json()
    horizons = [f["horizon_hours"] for f in body["forecasts"]]
    assert horizons, "at least the 1h horizon should be present"
    assert horizons == sorted(horizons)
    # Every reported horizon has its own model version and AQI.
    for entry in body["forecasts"]:
        assert entry["model_version"]
        assert 0 <= entry["aqi"] <= 500
    # Anything missing is named explicitly, with a reason.
    for missing in body["unavailable_horizons"]:
        assert "h" in missing


def test_forecast_reports_real_training_metrics(client) -> None:
    """The metrics in the response must be the measured ones, not placeholders."""
    body = client.get("/api/forecast/delhi").json()
    metrics = body["forecast"]["model_metrics"]
    assert metrics, "training metrics should be surfaced"
    assert metrics["rmse"] > 0
    assert metrics["n"] > 0
    assert -1 <= metrics["r2"] <= 1


def test_prediction_targets_the_next_hour(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    base = datetime.fromisoformat(body["forecast"]["base_time"].replace("Z", "+00:00"))
    target = datetime.fromisoformat(body["forecast"]["target_time"].replace("Z", "+00:00"))
    assert (target - base).total_seconds() == 3600


def test_each_horizon_targets_its_own_hour(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    for entry in body["forecasts"]:
        base = datetime.fromisoformat(entry["base_time"].replace("Z", "+00:00"))
        target = datetime.fromisoformat(entry["target_time"].replace("Z", "+00:00"))
        assert (target - base).total_seconds() == 3600 * entry["horizon_hours"]


def test_forecast_includes_current_conditions(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    current = body["current"]
    assert current["pm2_5"] >= 0
    assert 0 <= current["aqi"] <= 500
    assert current["time"]
    # Current AQI must be derived from the current PM2.5 - one source of truth.
    from airshield_core.aqi import aqi_from_pm25

    expected = aqi_from_pm25(current["pm2_5"])
    assert current["aqi"] == expected.aqi
    assert current["category"] == expected.category
    assert current["weather"] is not None


def test_forecast_includes_alert_and_aqi(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    alert, aqi = body["alert"], body["aqi"]

    assert alert["severity"] in {
        "good", "moderate", "elevated", "high", "very_high", "hazardous"
    }
    assert alert["headline"] and alert["advice"] and alert["sensitive_group_advice"]
    assert 0 <= aqi["aqi"] <= 500
    # The alert reports the India CPCB National AQI *PM2.5 sub-index* of the
    # prediction. The `aqi` block is the US EPA AQI for the same value. They are
    # different standards, so they legitimately differ; what must hold is that
    # each agrees with its own standard.
    from airshield_core.aqi import aqi_from_pm25, national_sub_index

    predicted = body["forecast"]["predicted_pm25"]
    assert aqi["aqi"] == aqi_from_pm25(predicted).aqi
    assert aqi["category"] == aqi_from_pm25(predicted).category
    assert alert["aqi"] == national_sub_index("pm2_5", predicted).sub_index
    assert alert["category"] == national_sub_index("pm2_5", predicted).category
    assert aqi["standard"] == "US EPA AQI"


def test_forecast_includes_the_india_national_aqi(client) -> None:
    """The India CPCB National AQI must be present and internally consistent."""
    body = client.get("/api/forecast/delhi").json()
    national = body["national_aqi"]
    assert national is not None
    assert national["standard"] == "India CPCB National AQI"
    assert 0 <= national["aqi"] <= 500
    # The reported value must be the worst sub-index, recomputed here.
    sub_indices = {s["pollutant"]: s["sub_index"] for s in national["sub_indices"]}
    assert sub_indices, "at least one pollutant sub-index is required"
    assert national["aqi"] == max(sub_indices.values())
    assert national["dominant_pollutant"] in sub_indices
    # The source lacks CO/SO2/NH3, so the response must say the value is partial.
    assert national["is_partial"] is True
    assert set(national["missing_pollutants"]) == {"CO", "SO2", "NH3"}
    assert national["health_guidance"]


def test_aqi_matches_the_prediction(client) -> None:
    """Recompute AQI from the returned prediction and compare."""
    from airshield_core.aqi import aqi_from_pm25

    body = client.get("/api/forecast/delhi").json()
    expected = aqi_from_pm25(body["forecast"]["predicted_pm25"])
    assert body["aqi"]["aqi"] == expected.aqi
    assert body["aqi"]["category"] == expected.category


def test_history_ends_with_the_prediction(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    history = body["history"]
    assert len(history) > 10
    predicted = [p for p in history if p["predicted"]]
    measured = [p for p in history if not p["predicted"]]
    assert predicted and measured
    # The first predicted point is the 1-hour forecast, matching `forecast`.
    assert predicted[0]["pm2_5"] == pytest.approx(body["forecast"]["predicted_pm25"])
    # Measured points must all precede predicted points.
    assert history.index(measured[-1]) < history.index(predicted[0])


def test_demo_mode_is_clearly_labelled(client) -> None:
    """A demo response must never look like a live reading."""
    body = client.get("/api/forecast/delhi").json()
    assert body["source"]["mode"] == "demo"
    assert body["notice"] and "DEMO MODE" in body["notice"]
    assert body["source"]["is_synthetic"] is False
    assert body["source"]["licence"]


def test_spike_block_is_present_and_honest(client) -> None:
    body = client.get("/api/forecast/delhi").json()
    spike = body["spike"]
    assert spike is not None
    assert 0.0 <= spike["confidence"] <= 1.0
    assert spike["confidence_basis"]
    # Associated signals, when present, must carry the non-causal disclaimer.
    if spike["associated_signals"]:
        assert "not proven causes" in spike["signal_disclaimer"]


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

    for path in ("/", "/api/health", "/api/model", "/api/forecast/delhi"):
        payload = json.dumps(client.get(path).json()).lower()
        for needle in ("secret", "access_key", "password", "github_pat", "token"):
            assert needle not in payload, f"{needle} leaked via {path}"
