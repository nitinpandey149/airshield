"""Amazon SageMaker AI inference client.

This is the only AWS AI/ML service AirShield uses. The backend sends the exact
feature row to a real-time SageMaker endpoint and returns whatever the endpoint
returns. If the endpoint is unreachable the error is raised to the caller - a
prediction is never invented as a fallback.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from airshield_core.predict import Forecast, ModelNotFoundError

logger = logging.getLogger(__name__)


class SageMakerError(RuntimeError):
    """Raised when SageMaker cannot serve a prediction."""


class SageMakerClient:
    """Invokes a SageMaker real-time endpoint hosting the AirShield model."""

    def __init__(self, endpoint_name: str, region: str):
        if not endpoint_name:
            raise SageMakerError(
                "SAGEMAKER_ENDPOINT_NAME is not set. Deploy the model with "
                "`python ml/sagemaker/deploy.py` or set AIRSHIELD_INFERENCE_BACKEND=local."
            )
        self.endpoint_name = endpoint_name
        self.region = region
        self._client = None

    @property
    def client(self):
        """Lazily create the boto3 client so import never requires credentials."""
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - boto3 is a dependency
                raise SageMakerError("boto3 is required for SageMaker inference") from exc
            self._client = boto3.client("sagemaker-runtime", region_name=self.region)
        return self._client

    def predict(self, features: pd.DataFrame) -> float:
        """Send one feature row and return the predicted PM2.5 value."""
        payload = {
            "feature_names": list(features.columns),
            "rows": features.to_numpy(dtype=float).tolist(),
        }
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            body = response["Body"].read().decode("utf-8")
        except Exception as exc:  # botocore raises many distinct types
            raise SageMakerError(
                f"SageMaker invocation failed for endpoint {self.endpoint_name!r}: {exc}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SageMakerError(f"SageMaker returned non-JSON body: {body[:200]!r}") from exc

        if isinstance(parsed, dict):
            for key in ("prediction", "predicted_pm25", "predictions", "pm2_5"):
                if key in parsed:
                    value = parsed[key]
                    break
            else:
                raise SageMakerError(f"SageMaker response missing a prediction field: {parsed}")
        else:
            value = parsed

        if isinstance(value, list):
            if not value:
                raise SageMakerError("SageMaker returned an empty prediction list")
            value = value[0]

        return float(value)

    def forecast(self, features: pd.DataFrame, base_time: datetime) -> Forecast:
        """Produce a :class:`Forecast` from a SageMaker endpoint response."""
        from datetime import timedelta

        value = self.predict(features)
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=timezone.utc)
        return Forecast(
            predicted_pm25=value,
            base_time=base_time,
            target_time=base_time + timedelta(hours=1),
            model_version=f"sagemaker:{self.endpoint_name}",
            backend="aws",
            horizon_hours=1,
            metadata={"endpoint_name": self.endpoint_name, "region": self.region},
        )


def build_predictor(settings):
    """Return the predictor selected by ``AIRSHIELD_INFERENCE_BACKEND``.

    Raises :class:`ModelNotFoundError` for a missing local artifact and
    :class:`SageMakerError` for a misconfigured endpoint, so the caller can
    report the real problem instead of degrading silently.
    """
    from airshield_core.predict import LocalPredictor

    if settings.inference_backend == "aws":
        client = SageMakerClient(settings.sagemaker_endpoint_name, settings.aws_region)
        logger.info("inference backend: SageMaker endpoint %s", settings.sagemaker_endpoint_name)
        return client

    predictor = LocalPredictor(settings.artifact_path)
    if not predictor.is_available:
        raise ModelNotFoundError(
            f"no local model artifact in {settings.artifact_path}. Run `make train`."
        )
    predictor.load()
    logger.info("inference backend: local artifact at %s", settings.artifact_path)
    return predictor
