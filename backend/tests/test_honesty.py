"""Tests for the honesty guarantees.

These are the most important tests in the repository: they check that AirShield
fails loudly instead of inventing data, predictions or AWS responses.
"""

from __future__ import annotations

import pytest

from airshield_core.predict import ModelNotFoundError
from airshield_core.sources.demo import DEMO_NOTICE
from airshield_core.sources.openmeteo import UpstreamError


def test_live_mode_reports_failure_instead_of_inventing_data(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With data_mode=live and a dead upstream, the API must error, not fake it."""
    from airshield_core.config import get_settings
    from app import deps

    monkeypatch.setenv("AIRSHIELD_DATA_MODE", "live")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    deps.reset_service()

    def boom(*args, **kwargs):
        raise UpstreamError("simulated upstream outage")

    monkeypatch.setattr(
        "app.services.forecast_service.fetch_frame_with_fallback", boom
    )

    response = client.get("/api/forecast/berlin")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "live data unavailable" in detail
    assert "simulated upstream outage" in detail

    deps.reset_service()
    reset_settings_cache()


def test_auto_mode_falls_back_to_demo_and_says_so(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In auto mode a dead upstream yields a labelled demo response, with a notice."""
    monkeypatch.setenv("AIRSHIELD_DATA_MODE", "auto")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()

    def boom(*args, **kwargs):
        raise UpstreamError("simulated upstream outage")

    monkeypatch.setattr(
        "app.services.forecast_service.fetch_frame_with_fallback", boom
    )

    response = client.get("/api/forecast/berlin")
    assert response.status_code == 200
    body = response.json()
    assert body["source"]["mode"] == "demo"
    assert body["notice"] == DEMO_NOTICE
    # Still a genuine model prediction, just on historical inputs.
    assert body["forecast"]["predicted_pm25"] >= 0

    deps.reset_service()
    reset_settings_cache()


def test_sagemaker_requires_an_endpoint_name() -> None:
    from app.services.sagemaker_client import SageMakerClient, SageMakerError

    with pytest.raises(SageMakerError, match="SAGEMAKER_ENDPOINT_NAME is not set"):
        SageMakerClient("", "us-east-1")


def test_aws_backend_without_endpoint_fails_loudly(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing the AWS backend without configuring an endpoint must error."""
    monkeypatch.setenv("AIRSHIELD_INFERENCE_BACKEND", "aws")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()

    health = client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert health["model_available"] is False
    assert "SAGEMAKER_ENDPOINT_NAME" in health["detail"]

    deps.reset_service()
    reset_settings_cache()


def test_missing_model_makes_health_degraded(
    client, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("AIRSHIELD_ARTIFACT_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AIRSHIELD_INFERENCE_BACKEND", "local")
    from airshield_core.config import reset_settings_cache

    reset_settings_cache()
    from app import deps

    deps.reset_service()

    health = client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert health["model_available"] is False
    assert "make train" in health["detail"]

    forecast = client.get("/api/forecast/berlin")
    assert forecast.status_code == 500
    assert "make train" in forecast.json()["detail"]

    deps.reset_service()
    reset_settings_cache()


def test_build_predictor_raises_for_missing_artifact(tmp_path) -> None:
    from app.services.sagemaker_client import build_predictor
    from airshield_core.config import Settings

    settings = Settings(
        AIRSHIELD_INFERENCE_BACKEND="local",
        AIRSHIELD_ARTIFACT_DIR=str(tmp_path / "nothing"),
    )
    with pytest.raises(ModelNotFoundError):
        build_predictor(settings)


def test_sagemaker_response_parsing_is_defensive(monkeypatch) -> None:
    """A malformed SageMaker response must raise, never return a made-up number."""
    from app.services.sagemaker_client import SageMakerClient, SageMakerError

    client = SageMakerClient("endpoint", "us-east-1")

    class FakeBody:
        def __init__(self, payload: str):
            self._payload = payload

        def read(self) -> bytes:
            return self._payload.encode()

    class FakeRuntime:
        def __init__(self, payload: str):
            self._payload = payload

        def invoke_endpoint(self, **kwargs):
            return {"Body": FakeBody(self._payload)}

    import pandas as pd

    features = pd.DataFrame({"a": [1.0], "b": [2.0]})

    # Plain scalar, named field, and list forms are all accepted.
    for payload, expected in (
        ('{"prediction": 12.5}', 12.5),
        ('{"predicted_pm25": 7.25}', 7.25),
        ('[3.5]', 3.5),
        ('9.0', 9.0),
    ):
        client._client = FakeRuntime(payload)
        assert client.predict(features) == expected

    # Unknown shape must fail rather than guess.
    client._client = FakeRuntime('{"unexpected": 1}')
    with pytest.raises(SageMakerError, match="missing a prediction field"):
        client.predict(features)

    client._client = FakeRuntime("[]")
    with pytest.raises(SageMakerError, match="empty prediction list"):
        client.predict(features)

    client._client = FakeRuntime("not json at all")
    with pytest.raises(SageMakerError, match="non-JSON body"):
        client.predict(features)


def test_invocation_errors_are_surfaced(monkeypatch) -> None:
    from app.services.sagemaker_client import SageMakerClient, SageMakerError

    client = SageMakerClient("endpoint", "us-east-1")

    class FailingRuntime:
        def invoke_endpoint(self, **kwargs):
            raise RuntimeError("ValidationException: endpoint not found")

    client._client = FailingRuntime()

    import pandas as pd

    with pytest.raises(SageMakerError, match="endpoint not found"):
        client.predict(pd.DataFrame({"a": [1.0]}))
