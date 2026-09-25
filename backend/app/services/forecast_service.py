"""Forecast orchestration: data acquisition -> features -> prediction -> alert.

Data-mode policy, applied explicitly rather than implicitly:

* ``live``  - require real upstream data; surface any failure to the client.
* ``demo``  - always use the bundled historical sample.
* ``auto``  - try live, and on failure fall back to demo while setting a visible
  ``notice`` and ``source.mode == "demo"`` on the response.

A prediction is never fabricated: if no model can run, the request fails with an
explanatory error instead of returning an invented number.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from airshield_core.aqi import aqi_from_pm25, exposure_alert
from airshield_core.config import Settings
from airshield_core.features import latest_feature_row
from airshield_core.schema import SourceInfo
from airshield_core.sources.demo import DEMO_NOTICE, DemoSource
from airshield_core.sources.openmeteo import UpstreamError, fetch_frame_with_fallback
from airshield_core.sources.registry import Location

logger = logging.getLogger(__name__)

#: Number of measured hours returned for charting.
HISTORY_HOURS = 48


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced from real inputs."""


class ForecastService:
    """Builds a complete forecast response for a location."""

    def __init__(self, settings: Settings, predictor=None):
        self.settings = settings
        self._predictor = predictor
        self._demo = DemoSource(settings.data_path)

    # ------------------------------------------------------------ predictor
    @property
    def predictor(self):
        if self._predictor is None:
            from app.services.sagemaker_client import build_predictor

            self._predictor = build_predictor(self.settings)
        return self._predictor

    def set_predictor(self, predictor) -> None:
        self._predictor = predictor

    # ----------------------------------------------------------------- data
    def _acquire(self, location: Location) -> tuple[pd.DataFrame, SourceInfo, str | None]:
        """Return ``(frame, source_info, notice)`` honouring the configured mode."""
        mode = self.settings.data_mode

        if mode == "demo":
            frame = self._demo.fetch_frame_for_prediction(location.name)
            return frame, self._demo.source_info(), DEMO_NOTICE

        if mode == "live":
            try:
                frame, info = fetch_frame_with_fallback(
                    location.latitude,
                    location.longitude,
                    timeout=self.settings.http_timeout,
                    retries=self.settings.http_retries,
                )
            except UpstreamError as exc:
                raise ForecastError(
                    f"live data unavailable for {location.label}: {exc}. "
                    "Set AIRSHIELD_DATA_MODE=auto to allow the bundled demo fallback."
                ) from exc
            return frame, info, None

        # auto: prefer live, fall back loudly
        try:
            frame, info = fetch_frame_with_fallback(
                location.latitude,
                location.longitude,
                timeout=self.settings.http_timeout,
                retries=self.settings.http_retries,
            )
            return frame, info, None
        except UpstreamError as exc:
            logger.warning("live data unavailable (%s); falling back to DEMO data", exc)
            frame = self._demo.fetch_frame_for_prediction(location.name)
            return frame, self._demo.source_info(), DEMO_NOTICE

    # --------------------------------------------------------------- output
    def build_forecast(self, location: Location) -> dict:
        """Produce the full response payload for one location."""
        frame, source, notice = self._acquire(location)

        try:
            features, base_time = latest_feature_row(frame)
        except ValueError as exc:
            raise ForecastError(f"could not build features for {location.label}: {exc}") from exc

        forecast = self.predictor.forecast(features, base_time.to_pydatetime())

        prediction = forecast.predicted_pm25
        result = aqi_from_pm25(prediction)
        alert = exposure_alert(prediction, horizon_hours=forecast.horizon_hours)

        measured = frame[frame["pm2_5"].notna()].tail(HISTORY_HOURS)
        history = [
            {"time": row.time, "pm2_5": float(row.pm2_5), "predicted": False}
            for row in measured.itertuples()
        ]
        history.append(
            {"time": forecast.target_time, "pm2_5": float(prediction), "predicted": True}
        )

        metrics: dict[str, float] = {}
        metadata = getattr(self.predictor, "metadata", None)
        if metadata is not None:
            metrics = dict(metadata.metrics)

        return {
            "location": {
                "slug": location.slug,
                "name": location.name,
                "country": location.country,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "timezone": location.timezone,
                "label": location.label,
            },
            "generated_at": datetime.now(timezone.utc),
            "source": source,
            "notice": notice,
            "forecast": {
                "predicted_pm25": prediction,
                "base_time": forecast.base_time,
                "target_time": forecast.target_time,
                "horizon_hours": forecast.horizon_hours,
                "model_version": forecast.model_version,
                "backend": forecast.backend,
                "model_metrics": metrics,
            },
            "aqi": {
                "aqi": result.aqi,
                "category": result.category,
                "band": result.band,
                "who_ratio": result.who_ratio,
            },
            "alert": alert,
            "history": history,
        }

    # ------------------------------------------------------------ model card
    def model_info(self) -> dict:
        """Return an honest model card, or an explanation of why there isn't one."""
        metadata = getattr(self.predictor, "metadata", None)
        if metadata is None:
            return {
                "available": True,
                "inference_backend": self.settings.inference_backend,
                "note": (
                    "Connected to a SageMaker endpoint; training metrics are held "
                    "by the training job rather than this process. "
                    "See ml/artifacts/metadata.json for the local artifact."
                ),
            }

        importance = metadata.feature_importance
        if not importance:
            top_features: list[dict] = []
        else:
            top_features = [
                {"feature": k, "gain_share": v} for k, v in list(importance.items())[:15]
            ]

        return {
            "available": True,
            "inference_backend": self.settings.inference_backend,
            "model_version": getattr(self.predictor, "model_version", None),
            "trained_at": metadata.trained_at,
            "feature_count": len(metadata.feature_columns),
            "train_rows": metadata.train_rows,
            "test_rows": metadata.test_rows,
            "metrics": metadata.metrics,
            "baseline_metrics": metadata.baseline_metrics,
            "train_window": metadata.train_window,
            "locations": metadata.locations,
            "data_source": metadata.data_source,
            "hyperparameters": metadata.hyperparameters,
            "library_versions": metadata.library_versions,
            "top_features": top_features,
        }
