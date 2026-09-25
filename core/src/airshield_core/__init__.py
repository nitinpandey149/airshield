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
from airshield_core.features import FEATURE_COLUMNS, build_features, build_training_frame
from airshield_core.predict import Forecast, LocalPredictor
from airshield_core.schema import (
    HOURLY_COLUMNS,
    Observation,
    Observations,
    SourceInfo,
)

__all__ = [
    "PM25_BREAKPOINTS",
    "AqiResult",
    "aqi_from_pm25",
    "aqi_category",
    "exposure_alert",
    "FEATURE_COLUMNS",
    "build_features",
    "build_training_frame",
    "Forecast",
    "LocalPredictor",
    "HOURLY_COLUMNS",
    "Observation",
    "Observations",
    "SourceInfo",
]

__version__ = "0.1.0"
