"""Tests for the AWS status, horizon and model-card endpoints."""

from __future__ import annotations


def test_aws_status_reports_only_configured_components(client) -> None:
    response = client.get("/api/aws/status")
    assert response.status_code == 200
    body = response.json()

    # Local inference in the test environment: SageMaker must not be claimed.
    assert body["sagemaker_active"] is False
    assert "sagemaker" not in body["configured_components"]
    assert body["inference_backend"] == "local"

    names = {c["key"] for c in body["components"]}
    assert {"sagemaker", "s3", "dynamodb", "lambda", "eventbridge", "sns", "cloudwatch"} <= names

    # CloudWatch is always available (logs go to stdout).
    cloudwatch = next(c for c in body["components"] if c["key"] == "cloudwatch")
    assert cloudwatch["configured"] is True

    # Unconfigured services explain what would enable them.
    s3 = next(c for c in body["components"] if c["key"] == "s3")
    assert s3["configured"] is False
    assert "AIRSHIELD_S3_BUCKET" in s3["detail"]

    # Credentials are never guessed.
    assert body["credentials"]["verified"] is False
    assert "not probed" in body["credentials"]["detail"]


def test_aws_status_does_not_claim_deployment(client) -> None:
    body = client.get("/api/aws/status").json()
    assert "does not claim" in body["note"]
    assert "SageMaker" in body["note"]


def test_aws_status_reflects_configured_sagemaker(client, monkeypatch) -> None:
    """When SageMaker is configured, the status says so - and only then."""
    from airshield_core.config import reset_settings_cache
    from app import deps

    monkeypatch.setenv("AIRSHIELD_INFERENCE_BACKEND", "aws")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "airshield-pm25-test")
    reset_settings_cache()
    deps.reset_service()

    body = client.get("/api/aws/status").json()
    assert body["sagemaker_active"] is True
    assert "sagemaker" in body["configured_components"]

    deps.reset_service()
    reset_settings_cache()


def test_aws_architecture_lists_the_pipeline(client) -> None:
    body = client.get("/api/aws/architecture").json()
    assert body["ml_service"] == "Amazon SageMaker AI (the only AWS AI/ML service used)"
    assert any("SageMaker" in step for step in body["pipeline"])
    assert any("SNS" in step for step in body["notification_pipeline"])
    assert body["infrastructure_as_code"]


def test_horizons_endpoint_reports_trained_models(client) -> None:
    body = client.get("/api/horizons").json()
    assert body["available"], "the test fixture trains all horizons"
    assert set(body["available"]) == {1, 3, 6}
    for card in body["horizons"]:
        assert card["model_version"]
        assert card["metrics"], "real held-out metrics must be present"
        assert card["baseline_metrics"]
        assert card["moving_average_metrics"]
    assert "separately trained" in body["note"]


def test_model_card_includes_baselines(client) -> None:
    body = client.get("/api/model").json()
    assert body["available"] is True
    assert body["inference_backend"] == "local"
    assert body["metrics"]["rmse"] > 0
    assert "moving_average_metrics" in body
    assert body["hyperparameters"]
    assert body["library_versions"]
    assert body["top_features"]
    for feature in body["top_features"]:
        assert 0.0 <= feature["gain_share"] <= 1.0
