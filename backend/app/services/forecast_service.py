"""Forecast orchestration: data acquisition -> features -> prediction -> alert.

Data-mode policy, applied explicitly rather than implicitly:

* ``live``  - require real upstream data; surface any failure to the client.
* ``demo``  - always use the bundled historical sample.
* ``auto``  - try live, and on failure fall back to demo while setting a visible
  ``notice`` and ``source.mode == "demo"`` on the response.

A prediction is never fabricated: if no model can run, the request fails with an
explanatory error instead of returning an invented number.

Multi-horizon: each horizon has its own trained booster. Only horizons for which
a model artifact actually exists are served, so a 1-hour model is never
presented as a 6-hour forecast.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from airshield_core.aqi import aqi_from_pm25, exposure_alert
from airshield_core.config import Settings
from airshield_core.events import detect_spike
from airshield_core.features import HORIZONS, latest_feature_row
from airshield_core.schema import SourceInfo
from airshield_core.sources.demo import DEMO_NOTICE, DemoSource
from airshield_core.sources.openmeteo import UpstreamError, fetch_frame_with_fallback
from airshield_core.sources.registry import Location
from airshield_core.spike_calibration import CalibrationTable

logger = logging.getLogger(__name__)

#: Number of measured hours returned for charting.
HISTORY_HOURS = 48
#: Number of forecast hours the acquisition requests (enough for the longest
#: horizon plus the exposure planner's forward view).
FORECAST_HOURS = 12


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced from real inputs."""


class ForecastService:
    """Builds a complete forecast response for a location."""

    def __init__(self, settings: Settings, predictor=None):
        self.settings = settings
        self._predictor = predictor
        self._demo = DemoSource(settings.data_path)
        self._calibration: CalibrationTable | None = None

    # ------------------------------------------------------------ predictor
    @property
    def predictor(self):
        if self._predictor is None:
            from app.services.sagemaker_client import build_predictor

            self._predictor = build_predictor(self.settings)
        return self._predictor

    def set_predictor(self, predictor) -> None:
        self._predictor = predictor

    # -------------------------------------------------------- horizon models
    def available_horizons(self) -> list[int]:
        """Horizons for which a usable predictor exists.

        A SageMaker client serves a single endpoint whose horizon is whatever it
        was trained for; a local artifact directory may hold several horizons.
        """
        if self.settings.inference_backend == "aws":
            return [1]
        from airshield_core.predict import LocalPredictor

        return [
            h
            for h in HORIZONS
            if LocalPredictor(self.settings.artifact_path, horizon=h).is_available
        ]

    def horizon_predictor(self, horizon: int):
        """Return a predictor for a horizon, or raise a clear error."""
        if self.settings.inference_backend == "aws":
            if horizon != 1:
                raise ForecastError(
                    "the SageMaker endpoint serves the 1-hour model only; "
                    f"a {horizon}-hour forecast needs a local artifact for that horizon"
                )
            return self.predictor

        from airshield_core.predict import LocalPredictor, ModelNotFoundError

        predictor = LocalPredictor(self.settings.artifact_path, horizon=horizon)
        if not predictor.is_available:
            raise ModelNotFoundError(
                f"no trained {horizon}-hour model in {self.settings.artifact_path}. "
                f"Train it with `make train` (horizons {HORIZONS})."
            )
        predictor.load()
        return predictor

    # ------------------------------------------------------------- calibration
    @property
    def calibration(self) -> CalibrationTable:
        if self._calibration is None:
            self._calibration = CalibrationTable.load(
                self.settings.artifact_path / "spike_calibration.json"
            )
        return self._calibration

    # ----------------------------------------------------------------- data
    def acquire(
        self, location: Location, *, future_hours: int = FORECAST_HOURS
    ) -> tuple[pd.DataFrame, SourceInfo, str | None, pd.Timestamp]:
        """Return ``(frame, source_info, notice, base_time)`` for the data mode."""
        mode = self.settings.data_mode

        if mode == "demo":
            frame, base_time = self._demo.fetch_frame_for_prediction(
                location.name, future_hours=future_hours
            )
            return frame, self._demo.source_info(), DEMO_NOTICE, base_time

        if mode == "live":
            try:
                frame, info, base_time = fetch_frame_with_fallback(
                    location.latitude,
                    location.longitude,
                    timeout=self.settings.http_timeout,
                    retries=self.settings.http_retries,
                    future_hours=future_hours,
                )
            except UpstreamError as exc:
                raise ForecastError(
                    f"live data unavailable for {location.label}: {exc}. "
                    "Set AIRSHIELD_DATA_MODE=auto to allow the bundled demo fallback."
                ) from exc
            return frame, info, None, base_time

        # auto: prefer live, fall back loudly
        try:
            frame, info, base_time = fetch_frame_with_fallback(
                location.latitude,
                location.longitude,
                timeout=self.settings.http_timeout,
                retries=self.settings.http_retries,
                future_hours=future_hours,
            )
            return frame, info, None, base_time
        except UpstreamError as exc:
            logger.warning("live data unavailable (%s); falling back to DEMO data", exc)
            frame, base_time = self._demo.fetch_frame_for_prediction(
                location.name, future_hours=future_hours
            )
            return frame, self._demo.source_info(), DEMO_NOTICE, base_time

    def _acquire(self, location: Location):
        """Backwards-compatible alias returning ``(frame, source, notice)``."""
        frame, source, notice, _ = self.acquire(location, future_hours=1)
        return frame, source, notice

    # --------------------------------------------------------------- current
    @staticmethod
    def current_conditions(frame: pd.DataFrame) -> dict:
        """The latest measured hour: PM2.5, other pollutants and weather."""
        measured = frame[frame["pm2_5"].notna()]
        if measured.empty:
            raise ForecastError("no measured PM2.5 rows available for current conditions")
        row = measured.iloc[-1]
        result = aqi_from_pm25(float(row["pm2_5"]))

        def value(name):
            if name not in frame.columns:
                return None
            raw = row.get(name)
            return None if raw is None or raw != raw else float(raw)

        return {
            "time": row["time"],
            "pm2_5": float(row["pm2_5"]),
            "pm10": value("pm10"),
            "nitrogen_dioxide": value("nitrogen_dioxide"),
            "ozone": value("ozone"),
            "temperature_2m": value("temperature_2m"),
            "relative_humidity_2m": value("relative_humidity_2m"),
            "wind_speed_10m": value("wind_speed_10m"),
            "wind_direction_10m": value("wind_direction_10m"),
            "surface_pressure": value("surface_pressure"),
            "precipitation": value("precipitation"),
            "aqi": result.aqi,
            "category": result.category,
            "band": result.band,
            "who_ratio": result.who_ratio,
        }

    # --------------------------------------------------------------- forecast
    def predict_horizons(
        self, frame: pd.DataFrame, base_time: pd.Timestamp
    ) -> tuple[list[dict], list[str]]:
        """Predict every available horizon from one feature frame.

        Returns ``(forecasts, unavailable)`` where ``unavailable`` names the
        horizons that were requested but have no trained model, so the client can
        see exactly what is missing instead of assuming coverage.
        """
        forecasts: list[dict] = []
        unavailable: list[str] = []

        for horizon in HORIZONS:
            if len(frame) < horizon + 1:
                unavailable.append(f"{horizon}h (insufficient forecast weather rows)")
                continue
            try:
                predictor = self.horizon_predictor(horizon)
                features, _ = latest_feature_row(frame, horizon=horizon, base_time=base_time)
                forecast = predictor.forecast(features, base_time.to_pydatetime())
            except Exception as exc:
                unavailable.append(f"{horizon}h ({exc})")
                continue

            result = aqi_from_pm25(forecast.predicted_pm25)
            forecasts.append(
                {
                    "horizon_hours": horizon,
                    "predicted_pm25": forecast.predicted_pm25,
                    "base_time": forecast.base_time,
                    "target_time": forecast.target_time,
                    "model_version": forecast.model_version,
                    "backend": forecast.backend,
                    "aqi": result.aqi,
                    "category": result.category,
                    "band": result.band,
                    "who_ratio": result.who_ratio,
                    "model_metrics": dict(
                        getattr(getattr(predictor, "metadata", None), "metrics", {}) or {}
                    ),
                }
            )
        return forecasts, unavailable

    def forecast_timeline(
        self, frame: pd.DataFrame, base_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Predicted PM2.5 for each future hour, for the timeline and planner.

        Each future hour is produced by the nearest available horizon model whose
        target is at or before that hour: hours 1-2 use the 1h model, hours 3-4
        the 3h model, hours 5+ the 6h model. Because only a few horizons are
        trained, a model's prediction is held forward for the hours between its
        target and the next horizon's target. Every value therefore comes from a
        real booster; ``horizon_hours`` records which one, and ``held_hours``
        records how far the value was carried. Nothing is interpolated or invented.
        """
        available = sorted(self.available_horizons())
        if not available:
            return pd.DataFrame(columns=["time", "pm2_5", "horizon_hours", "step_hours", "held_hours"])

        # One real prediction per trained horizon, at its true target time.
        predictions: dict[int, float] = {}
        for horizon in available:
            if len(frame) < horizon + 1:
                continue
            try:
                predictor = self.horizon_predictor(horizon)
                features, _ = latest_feature_row(
                    frame, horizon=horizon, base_time=base_time
                )
                predictions[horizon] = float(predictor.predict(features))
            except Exception:
                continue
        if not predictions:
            return pd.DataFrame(columns=["time", "pm2_5", "horizon_hours", "step_hours", "held_hours"])

        rows: list[dict] = []
        for step in range(1, FORECAST_HOURS + 1):
            # Largest trained horizon at or before this hour; fall back to the
            # smallest horizon for steps before the first target.
            candidates = [h for h in predictions if h <= step]
            horizon = max(candidates) if candidates else min(predictions)
            rows.append(
                {
                    "time": base_time.to_pydatetime() + timedelta(hours=step),
                    "pm2_5": predictions[horizon],
                    "horizon_hours": horizon,
                    "step_hours": step,
                    "held_hours": step - horizon,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ event/spike
    def spike(self, timeline: pd.DataFrame, current_pm25: float) -> dict:
        """Detect a pollution spike in the predicted timeline."""
        if timeline.empty:
            return {
                "spike_detected": False,
                "kind": "none",
                "severity": "none",
                "expected_time": None,
                "expected_change_percent": 0.0,
                "peak_pm25": None,
                "baseline_pm25": round(current_pm25, 2),
                "horizon_hours": 0,
                "confidence": 0.0,
                "confidence_basis": "no_forecast",
                "associated_signals": [],
                "message": "No forecast timeline is available, so no spike can be assessed.",
            }
        event = detect_spike(
            timeline,
            baseline_value=current_pm25,
            horizon_hours=int(timeline["step_hours"].max()),
            calibration=self.calibration,
        )
        return event.to_dict()

    # --------------------------------------------------------------- output
    def build_forecast(self, location: Location) -> dict:
        """Produce the full response payload for one location."""
        frame, source, notice, base_time = self.acquire(
            location, future_hours=FORECAST_HOURS
        )
        try:
            current = self.current_conditions(frame)
        except ForecastError as exc:
            raise ForecastError(
                f"could not read current conditions for {location.label}: {exc}"
            ) from exc

        forecasts, unavailable = self.predict_horizons(frame, base_time)
        if not forecasts:
            raise ForecastError(
                "no forecast model could run for this location: "
                + "; ".join(unavailable or ["no trained artifacts found"])
            )

        primary = forecasts[0]
        alert = exposure_alert(
            primary["predicted_pm25"], horizon_hours=primary["horizon_hours"]
        )

        timeline = self.forecast_timeline(frame, base_time)
        spike = self.spike(timeline, current["pm2_5"])

        measured = frame[frame["pm2_5"].notna()].tail(HISTORY_HOURS)
        history = [
            {"time": row.time, "pm2_5": float(row.pm2_5), "predicted": False}
            for row in measured.itertuples()
        ]
        for row in timeline.itertuples():
            history.append({"time": row.time, "pm2_5": float(row.pm2_5), "predicted": True})

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
            "current": {
                "time": current["time"],
                "pm2_5": current["pm2_5"],
                "pm10": current["pm10"],
                "nitrogen_dioxide": current["nitrogen_dioxide"],
                "ozone": current["ozone"],
                "aqi": current["aqi"],
                "category": current["category"],
                "band": current["band"],
                "who_ratio": current["who_ratio"],
                "weather": {
                    "temperature_2m": current["temperature_2m"],
                    "relative_humidity_2m": current["relative_humidity_2m"],
                    "wind_speed_10m": current["wind_speed_10m"],
                    "wind_direction_10m": current["wind_direction_10m"],
                    "surface_pressure": current["surface_pressure"],
                    "precipitation": current["precipitation"],
                },
            },
            "forecast": {
                "predicted_pm25": primary["predicted_pm25"],
                "base_time": primary["base_time"],
                "target_time": primary["target_time"],
                "horizon_hours": primary["horizon_hours"],
                "model_version": primary["model_version"],
                "backend": primary["backend"],
                "model_metrics": primary["model_metrics"],
            },
            "forecasts": forecasts,
            "unavailable_horizons": unavailable,
            "aqi": {
                "aqi": primary["aqi"],
                "category": primary["category"],
                "band": primary["band"],
                "who_ratio": primary["who_ratio"],
            },
            "alert": alert,
            "spike": spike,
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
            "horizon_hours": metadata.horizon_hours,
            "feature_count": len(metadata.feature_columns),
            "train_rows": metadata.train_rows,
            "test_rows": metadata.test_rows,
            "metrics": metadata.metrics,
            "baseline_metrics": metadata.baseline_metrics,
            "moving_average_metrics": metadata.moving_average_metrics,
            "train_window": metadata.train_window,
            "locations": metadata.locations,
            "data_source": metadata.data_source,
            "hyperparameters": metadata.hyperparameters,
            "library_versions": metadata.library_versions,
            "top_features": top_features,
        }

    def horizons_info(self) -> list[dict]:
        """Model card for each horizon that has an artifact on disk."""
        cards: list[dict] = []
        for horizon in self.available_horizons():
            try:
                predictor = self.horizon_predictor(horizon)
            except Exception:
                continue
            metadata = getattr(predictor, "metadata", None)
            cards.append(
                {
                    "horizon_hours": horizon,
                    "model_version": getattr(predictor, "model_version", None),
                    "trained_at": getattr(metadata, "trained_at", None),
                    "metrics": getattr(metadata, "metrics", {}) or {},
                    "baseline_metrics": getattr(metadata, "baseline_metrics", {}) or {},
                    "moving_average_metrics": getattr(
                        metadata, "moving_average_metrics", {}
                    )
                    or {},
                    "train_rows": getattr(metadata, "train_rows", None),
                    "test_rows": getattr(metadata, "test_rows", None),
                }
            )
        return cards
