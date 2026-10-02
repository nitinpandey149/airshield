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


class SubIndexOut(BaseModel):
    """One pollutant's contribution to the India CPCB National AQI."""

    pollutant: str
    concentration: float
    sub_index: int
    category: str
    band: str


class NationalAqiOut(BaseModel):
    """India CPCB National AQI: the worst sub-index across available pollutants."""

    aqi: int
    category: str
    dominant_pollutant: str
    band: str
    who_ratio: float
    sub_indices: list[SubIndexOut] = Field(default_factory=list)
    health_guidance: str
    missing_pollutants: list[str] = Field(
        default_factory=list,
        description="Pollutants the National AQI covers that our source does not provide",
    )
    is_partial: bool = Field(
        description="True when the value could be understated due to missing pollutants"
    )
    standard: str = "India CPCB National AQI"


class AqiOut(BaseModel):
    aqi: int
    category: str
    band: str
    who_ratio: float
    standard: str = "US EPA AQI"


class AlertOut(BaseModel):
    severity: str
    headline: str
    advice: str
    sensitive_group_advice: str
    category: str
    aqi: int
    horizon: str
    basis: str = "India CPCB National AQI PM2.5 sub-index"
    health_guidance: str = ""


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
    national_aqi: NationalAqiOut | None = Field(
        default=None,
        description="India CPCB National AQI across all pollutants the source provides",
    )
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


# --------------------------------------------------------------- assistant
class AssistantContextIn(BaseModel):
    """AirShield's own outputs, supplied by the client for the assistant.

    These are the same payloads ``/api/forecast`` and ``/api/plan`` return. The
    assistant explains them; it never recomputes or invents them.
    """

    location: str | None = None
    forecast: dict | None = Field(
        default=None, description="A /api/forecast payload (observed + predicted blocks)"
    )
    exposure: dict | None = Field(
        default=None, description="A /api/plan payload (recommendation block)"
    )
    recommendation: dict | None = Field(
        default=None, description="Alias for `exposure`, accepted for convenience"
    )


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    location_slug: str | None = Field(
        default=None, description="Location to load live AirShield context for"
    )
    activity: str | None = Field(default=None, description="Activity for exposure context")
    duration_minutes: int | None = Field(default=None, ge=1, le=600)
    context: AssistantContextIn | None = Field(
        default=None,
        description="Client-supplied AirShield payloads; when absent, the server fetches them",
    )


class AssistantSourceOut(BaseModel):
    """A citation. Only documents actually retrieved are ever returned."""

    doc_id: str
    title: str
    source: str
    url: str
    category: str = ""
    document_type: str = ""
    publication_date: str | None = None
    licence: str = ""


class AssistantRetrievedOut(BaseModel):
    """Auditable record of one retrieved passage."""

    doc_id: str
    title: str
    source: str
    url: str
    score: float
    ordinal: int


class AssistantChatResponse(BaseModel):
    answer: str
    sources: list[AssistantSourceOut] = Field(default_factory=list)
    retrieved_chunks: int = 0
    #: True only when the answer is grounded in retrieved knowledge.
    grounded: bool = False
    #: True when the assistant declined for lack of verified information.
    insufficient_knowledge: bool = False
    #: "llm" when a model generated the prose, "extractive" when no model is
    #: configured and the answer quotes the retrieved passages, "refusal" when
    #: nothing relevant was found.
    mode: str = "extractive"
    llm_available: bool = False
    llm_model: str = ""
    #: The exact AirShield values supplied to the model, for auditability.
    context_used: dict = Field(default_factory=dict)
    retrieved: list[AssistantRetrievedOut] = Field(default_factory=list)
    notice: str | None = None
    context_error: str | None = None


class SuggestedQuestion(BaseModel):
    id: str
    label: str
    question: str
    needs_context: bool = False


class AssistantStatusResponse(BaseModel):
    enabled: bool
    ready: bool
    index: dict | None = None
    embedder_semantic: bool | None = None
    llm_available: bool
    llm_model: str | None = None
    llm_detail: str | None = None
    suggested_questions: list[SuggestedQuestion] = Field(default_factory=list)
    notice: str | None = None
    detail: str | None = None
