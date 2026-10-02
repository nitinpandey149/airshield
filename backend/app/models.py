"""API response models.

Every response carries its provenance. The frontend renders ``source.mode`` and
``notice`` verbatim, which is what prevents demo data from ever being presented
as a live reading.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from airshield_core.schema import SourceInfo


class LocationOut(BaseModel):
    slug: str
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    label: str


class ForecastOut(BaseModel):
    predicted_pm25: float = Field(description="Predicted PM2.5 for the target hour, ug/m3")
    base_time: datetime = Field(description="Hour the prediction starts from (UTC)")
    target_time: datetime = Field(description="Hour being predicted (UTC)")
    horizon_hours: int = 1
    model_version: str
    backend: Literal["local", "aws"]
    model_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Real held-out metrics from training; empty if unknown",
    )


class HorizonForecastOut(BaseModel):
    """One forecast horizon with its own model and AQI."""

    horizon_hours: int
    predicted_pm25: float
    base_time: datetime
    target_time: datetime
    model_version: str
    backend: Literal["local", "aws"]
    aqi: int
    category: str
    band: str
    who_ratio: float
    model_metrics: dict[str, float] = Field(default_factory=dict)


class WeatherOut(BaseModel):
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    surface_pressure: float | None = None
    precipitation: float | None = None


class CurrentOut(BaseModel):
    """The latest measured hour."""

    time: datetime
    pm2_5: float
    pm10: float | None = None
    nitrogen_dioxide: float | None = None
    ozone: float | None = None
    aqi: int
    category: str
    band: str
    who_ratio: float
    weather: WeatherOut


class SpikeOut(BaseModel):
    spike_detected: bool
    kind: str
    severity: str
    expected_time: datetime | None = None
    expected_change_percent: float
    peak_pm25: float | None = None
    baseline_pm25: float
    horizon_hours: int
    confidence: float
    confidence_basis: str
    associated_signals: list[dict] = Field(default_factory=list)
    signal_disclaimer: str | None = None
    message: str = ""


class AqiOut(BaseModel):
    aqi: int
    category: str
    band: str
    who_ratio: float


class AlertOut(BaseModel):
    severity: str
    headline: str
    advice: str
    sensitive_group_advice: str
    category: str
    aqi: int
    horizon: str


class HistoryPoint(BaseModel):
    time: datetime
    pm2_5: float | None = None
    predicted: bool = False


class ForecastResponse(BaseModel):
    location: LocationOut
    generated_at: datetime
    source: SourceInfo
    notice: str | None = Field(
        default=None,
        description="Set when the response is not live, e.g. DEMO MODE fallback",
    )
    current: CurrentOut
    forecast: ForecastOut
    forecasts: list[HorizonForecastOut] = Field(
        default_factory=list,
        description="One entry per horizon with a trained model",
    )
    unavailable_horizons: list[str] = Field(
        default_factory=list,
        description="Horizons requested but without a trained model, with the reason",
    )
    aqi: AqiOut
    alert: AlertOut
    spike: SpikeOut | None = None
    history: list[HistoryPoint] = Field(
        default_factory=list,
        description="Recent measured PM2.5 followed by the predicted points",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    inference_backend: str
    data_mode: str
    model_available: bool
    model_version: str | None = None
    sagemaker_endpoint: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Honest model card surfaced by the API."""

    available: bool
    inference_backend: str
    model_version: str | None = None
    trained_at: str | None = None
    horizon_hours: int | None = None
    feature_count: int | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    moving_average_metrics: dict[str, float] = Field(default_factory=dict)
    train_window: dict[str, str] = Field(default_factory=dict)
    locations: list[str] = Field(default_factory=list)
    data_source: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    top_features: list[dict] = Field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------- planning
class ActivityOption(BaseModel):
    activity: str
    label: str
    intensity: float


class ExposureEstimateOut(BaseModel):
    score: float
    level: str
    level_label: str
    mean_pm25: float
    peak_pm25: float
    duration_minutes: float
    activity: str
    activity_label: str
    activity_intensity: float
    location_factor: float
    micro_scale: float
    basis: str


class WindowOut(BaseModel):
    start: datetime
    end: datetime
    exposure: ExposureEstimateOut
    relative_reduction_percent: float | None = None


class TimelinePoint(BaseModel):
    time: datetime
    pm2_5: float
    horizon_hours: int


class ExposurePlanResponse(BaseModel):
    location: str
    location_slug: str
    activity: str
    activity_label: str
    duration_minutes: int
    basis: str
    generated_at: datetime
    base_time: datetime
    source: SourceInfo
    notice: str | None = None
    best_window: WindowOut
    alternative_window: WindowOut | None = None
    highest_exposure_window: WindowOut | None = None
    relative_reduction_percent: float | None = None
    reason: str
    note: str
    exposure_note: str
    candidates: list[WindowOut] = Field(default_factory=list)
    timeline: list[TimelinePoint] = Field(default_factory=list)


# ------------------------------------------------------------------ routes
class RouteOut(BaseModel):
    route_id: str
    label: str
    distance_km: float
    duration_minutes: float
    average_pm25: float
    peak_pm25: float
    exposure_score: float
    exposure_level: str
    activity: str
    activity_label: str
    relative_reduction_percent: float | None = None
    provider: str
    geometry: list[list[float]] = Field(default_factory=list)


class RouteComparisonResponse(BaseModel):
    mode: str
    activity: str
    provider: str | None = None
    provider_profile: str | None = None
    generated_at: datetime
    routes: list[RouteOut]
    recommended_route_id: str
    relative_reduction_percent: float | None = None
    recommendation: str
    note: str


# --------------------------------------------------------------------- aws
class AwsComponentOut(BaseModel):
    key: str
    name: str
    purpose: str
    configured: bool
    detail: str = ""


class AwsStatusResponse(BaseModel):
    region: str
    inference_backend: str
    sagemaker_active: bool
    components: list[AwsComponentOut]
    configured_components: list[str]
    credentials: dict
    note: str
