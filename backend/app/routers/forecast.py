"""Forecast endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from airshield_core.sources.demo import DEMO_NOTICE
from airshield_core.sources.openmeteo import UpstreamError
from airshield_core.sources.registry import (
    LOCATIONS,
    LocationNotFoundError,
    get_location,
)
from app.deps import get_service
from app.models import ForecastResponse, LocationOut
from app.services.forecast_service import ForecastError, ForecastService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["forecast"])


@router.get("/locations", response_model=list[LocationOut])
def list_locations() -> list[LocationOut]:
    """List the locations AirShield can forecast for."""
    return [
        LocationOut(
            slug=loc.slug,
            name=loc.name,
            country=loc.country,
            latitude=loc.latitude,
            longitude=loc.longitude,
            timezone=loc.timezone,
            label=loc.label,
        )
        for loc in LOCATIONS
    ]


@router.get("/forecast/{location_slug}", response_model=ForecastResponse)
def forecast(
    location_slug: str,
    service: ForecastService = Depends(get_service),
) -> ForecastResponse:
    """Predict next-hour PM2.5 and derive an exposure alert.

    Errors are reported as they are: a missing model or unreachable data source
    produces a 4xx/5xx explanation, never a placeholder prediction.
    """
    try:
        location = get_location(location_slug)
    except LocationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        payload = service.build_forecast(location)
    except ForecastError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UpstreamError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # model/runtime failures must be visible
        logger.exception("forecast failed for %s", location_slug)
        raise HTTPException(status_code=500, detail=f"forecast failed: {exc}") from exc

    return ForecastResponse(**payload)


@router.get("/demo-notice")
def demo_notice() -> dict:
    """Return the exact wording used when DEMO_MODE data is served."""
    return {"notice": DEMO_NOTICE}
