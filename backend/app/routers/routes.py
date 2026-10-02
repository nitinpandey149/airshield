"""Route comparison endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from airshield_core.exposure import UnknownActivityError
from app.deps import get_route_service
from app.models import RouteComparisonResponse
from app.services.route_service import RouteService, RouteServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["routes"])


@router.get("/routes/compare", response_model=RouteComparisonResponse)
def compare_routes(
    origin_lat: float = Query(..., ge=-90, le=90),
    origin_lon: float = Query(..., ge=-180, le=180),
    dest_lat: float = Query(..., ge=-90, le=90),
    dest_lon: float = Query(..., ge=-180, le=180),
    mode: str = Query("walking", description="walking, cycling or running"),
    activity: str | None = Query(None, description="activity used for exposure; defaults to mode"),
    max_routes: int = Query(3, ge=1, le=5),
    service: RouteService = Depends(get_route_service),
) -> RouteComparisonResponse:
    """Compare real route alternatives by predicted exposure.

    Route geometry comes from a real routing engine (Valhalla, with OSRM as a
    fallback). Predicted PM2.5 is sampled from Open-Meteo at each segment
    midpoint. If no routing engine is reachable the endpoint returns 503 rather
    than inventing a route.
    """
    if mode not in {"walking", "cycling", "running"}:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported travel mode {mode!r}. Use walking, cycling or running.",
        )

    try:
        payload = service.compare(
            origin=(origin_lat, origin_lon),
            destination=(dest_lat, dest_lon),
            mode=mode,
            activity=activity,
            max_routes=max_routes,
        )
    except UnknownActivityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RouteServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("route comparison failed")
        raise HTTPException(status_code=500, detail=f"route comparison failed: {exc}") from exc

    return RouteComparisonResponse(**payload)
