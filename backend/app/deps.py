"""FastAPI dependency wiring.

The forecast service is created once per process and cached, because loading an
XGBoost booster on every request would be wasteful.
"""

from __future__ import annotations

from functools import lru_cache

from airshield_core.config import get_settings
from app.services.assistant_service import AssistantService
from app.services.aws_status import AwsStatusService
from app.services.forecast_service import ForecastService
from app.services.planning_service import PlanningService
from app.services.route_service import RouteService

_service: ForecastService | None = None


@lru_cache(maxsize=1)
def _cached_service() -> ForecastService:
    return ForecastService(get_settings())


def get_service() -> ForecastService:
    """Return the process-wide forecast service."""
    return _cached_service()


@lru_cache(maxsize=1)
def _cached_planning() -> PlanningService:
    return PlanningService(_cached_service())


def get_planning_service() -> PlanningService:
    """Return the process-wide exposure-planning service."""
    return _cached_planning()


@lru_cache(maxsize=1)
def _cached_routes() -> RouteService:
    return RouteService(_cached_service())


def get_route_service() -> RouteService:
    """Return the process-wide route-comparison service."""
    return _cached_routes()


@lru_cache(maxsize=1)
def _cached_aws_status() -> AwsStatusService:
    return AwsStatusService(get_settings())


def get_aws_status_service() -> AwsStatusService:
    """Return the process-wide AWS status service."""
    return _cached_aws_status()


@lru_cache(maxsize=1)
def _cached_assistant() -> AssistantService:
    return AssistantService(get_settings(), _cached_service(), _cached_planning())


def get_assistant_service() -> AssistantService:
    """Return the process-wide Ask AirShield service."""
    return _cached_assistant()


def reset_service() -> None:
    """Drop the cached services - used by tests that swap configuration."""
    global _service
    _service = None
    _cached_service.cache_clear()
    _cached_planning.cache_clear()
    _cached_routes.cache_clear()
    _cached_aws_status.cache_clear()
    _cached_assistant.cache_clear()
