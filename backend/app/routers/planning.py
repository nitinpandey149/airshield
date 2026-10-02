"""Exposure planner endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from airshield_core.exposure import UnknownActivityError
from airshield_core.sources.registry import LocationNotFoundError, get_location
from app.deps import get_planning_service
from app.models import ActivityOption, ExposurePlanResponse
from app.services.forecast_service import ForecastError
from app.services.planning_service import PlanningError, PlanningService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["planning"])


@router.get("/activities", response_model=list[ActivityOption])
def list_activities(
    service: PlanningService = Depends(get_planning_service),
) -> list[ActivityOption]:
    """Activity options and their exposure intensity multipliers.

    The multipliers are engineering approximations used for relative comparison,
    not medical measurements.
    """
    return [ActivityOption(**option) for option in service.activity_options]


@router.get("/plan/{location_slug}", response_model=ExposurePlanResponse)
def plan_exposure(
    location_slug: str,
    activity: str = Query("walking", description="walking, running, cycling, outdoor_work, ..."),
    duration_minutes: int = Query(45, ge=1, le=600),
    start_time: str | None = Query(None, description="ISO-8601 lower bound for the window"),
    end_time: str | None = Query(None, description="ISO-8601 upper bound for the window"),
    step_minutes: int = Query(30, ge=5, le=180),
    service: PlanningService = Depends(get_planning_service),
) -> ExposurePlanResponse:
    """Recommend the lowest-exposure activity window for a location.

    Every candidate window is scored over the whole activity duration using
    predicted PM2.5, then ranked. The response always describes *lower predicted
    exposure*, never safety.
    """
    try:
        location = get_location(location_slug)
    except LocationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        payload = service.plan(
            location,
            activity=activity,
            duration_minutes=duration_minutes,
            start_time=start_time,
            end_time=end_time,
            step_minutes=step_minutes,
        )
    except UnknownActivityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ForecastError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # model/runtime failures must be visible
        logger.exception("planning failed for %s", location_slug)
        raise HTTPException(status_code=500, detail=f"planning failed: {exc}") from exc

    return ExposurePlanResponse(**payload)
