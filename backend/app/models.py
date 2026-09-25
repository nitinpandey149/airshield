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
    forecast: ForecastOut
    aqi: AqiOut
    alert: AlertOut
    history: list[HistoryPoint] = Field(
        default_factory=list,
        description="Recent measured PM2.5 followed by the predicted point",
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
    feature_count: int | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    train_window: dict[str, str] = Field(default_factory=dict)
    locations: list[str] = Field(default_factory=list)
    data_source: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    top_features: list[dict] = Field(default_factory=list)
    note: str | None = None
