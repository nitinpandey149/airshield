"""FastAPI dependency wiring.

The forecast service is created once per process and cached, because loading an
XGBoost booster on every request would be wasteful.
"""

from __future__ import annotations

from functools import lru_cache

from airshield_core.config import get_settings
from app.services.forecast_service import ForecastService

_service: ForecastService | None = None


@lru_cache(maxsize=1)
def _cached_service() -> ForecastService:
    return ForecastService(get_settings())


def get_service() -> ForecastService:
    """Return the process-wide forecast service."""
    return _cached_service()


def reset_service() -> None:
    """Drop the cached service - used by tests that swap configuration."""
    global _service
    _service = None
    _cached_service.cache_clear()
