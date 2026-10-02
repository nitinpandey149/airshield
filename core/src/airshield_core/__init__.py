"""AirShield core - shared domain logic for data, features, models and alerts.

This package is imported by the FastAPI backend, the training pipeline and the
SageMaker entry points so that feature engineering and AQI maths are defined
exactly once.
"""

from airshield_core.aqi import (
    PM25_BREAKPOINTS,
    AqiResult,
    aqi_from_pm25,
    aqi_category,
    exposure_alert,
)
from airshield_core.events import SpikeEvent, detect_spike
from airshield_core.exposure import (
    ACTIVITY_INTENSITY,
    ACTIVITY_LABELS,
    ExposureEstimate,
    exposure_level,
    exposure_over_series,
    relative_reduction_percent,
    segment_exposure,
)
from airshield_core.features import (
    FEATURE_COLUMNS,
    HORIZONS,
    build_features,
    build_training_frame,
    latest_feature_row,
    target_column,
)
from airshield_core.predict import Forecast, LocalPredictor
from airshield_core.route_exposure import (
    RouteCandidate,
    RouteExposure,
    RouteSegment,
    rank_routes,
    score_route,
)
from airshield_core.schema import (
    HOURLY_COLUMNS,
    Observation,
    Observations,
    SourceInfo,
)
from airshield_core.spike_calibration import CalibrationTable, build_calibration
from airshield_core.windows import CandidateWindow, WindowPlan, optimize_window

__all__ = [
    "PM25_BREAKPOINTS",
    "AqiResult",
    "aqi_from_pm25",
    "aqi_category",
    "exposure_alert",
    "SpikeEvent",
    "detect_spike",
    "ACTIVITY_INTENSITY",
    "ACTIVITY_LABELS",
    "ExposureEstimate",
    "exposure_level",
    "exposure_over_series",
    "relative_reduction_percent",
    "segment_exposure",
    "FEATURE_COLUMNS",
    "HORIZONS",
    "build_features",
    "build_training_frame",
    "latest_feature_row",
    "target_column",
    "Forecast",
    "LocalPredictor",
    "RouteCandidate",
    "RouteExposure",
    "RouteSegment",
    "rank_routes",
    "score_route",
    "HOURLY_COLUMNS",
    "Observation",
    "Observations",
    "SourceInfo",
    "CalibrationTable",
    "build_calibration",
    "CandidateWindow",
    "WindowPlan",
    "optimize_window",
]

__version__ = "0.1.0"
