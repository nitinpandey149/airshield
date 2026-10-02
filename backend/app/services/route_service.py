"""Route comparison service: rank alternative routes by predicted exposure.

Pipeline per route:

1. Ask a real routing engine for alternatives (Valhalla, then OSRM).
2. Split each route into segments.
3. Fetch real predicted PM2.5 at every segment midpoint, in one batched
   Open-Meteo request (the API accepts multiple coordinates).
4. Score and rank routes with the deterministic route-exposure engine.

If the routing engine is unreachable the endpoint reports the feature as
unavailable. If the pollution sampling fails, the request fails. Route
pollution is never fabricated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

from airshield_core.route_exposure import (
    RouteCandidate,
    RouteExposureError,
    RouteSegment,
    rank_routes,
)
from airshield_core.routing import RoutingError, route_alternatives
from airshield_core.sources.openmeteo import AIR_QUALITY_URL, UpstreamError
from app.services.forecast_service import ForecastService

logger = logging.getLogger(__name__)

#: Cap on sample points so a long route does not fan out into a huge request.
MAX_SAMPLE_POINTS = 40


class RouteServiceError(RuntimeError):
    """Raised when routes cannot be scored."""


def _fetch_predictions_at(
    points: list[tuple[float, float]], *, timeout: float, retries: int
) -> list[float]:
    """Fetch the current predicted PM2.5 at each ``(lat, lon)`` sample point.

    Open-Meteo accepts comma-separated coordinates and returns one entry per
    location. The value returned is the air-quality model's forecast for the
    current hour at each coordinate - a real upstream value, not an interpolation.
    """
    if not points:
        raise RouteServiceError("no sample points supplied")

    latitudes = ",".join(f"{lat:.4f}" for lat, _ in points)
    longitudes = ",".join(f"{lon:.4f}" for _, lon in points)
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "pm2_5",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(AIR_QUALITY_URL, params=params)
                response.raise_for_status()
                payload = response.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
    else:
        raise UpstreamError(
            f"could not sample pollution along the route after {retries + 1} attempts: {last_error}"
        )

    entries = payload if isinstance(payload, list) else [payload]
    if len(entries) != len(points):
        raise UpstreamError(
            f"expected {len(points)} pollution samples, received {len(entries)}"
        )

    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    values: list[float] = []
    for entry in entries:
        hourly = entry.get("hourly") or {}
        times = hourly.get("time") or []
        series = hourly.get("pm2_5") or []
        value = None
        # Prefer the current hour; fall back to the last non-null value.
        for stamp, candidate in zip(times, series):
            if candidate is None:
                continue
            if pd.Timestamp(stamp).tz_localize("UTC") <= now_hour:
                value = float(candidate)
        if value is None:
            for candidate in reversed(series):
                if candidate is not None:
                    value = float(candidate)
                    break
        if value is None:
            raise UpstreamError("Open-Meteo returned no usable PM2.5 for a route sample point")
        values.append(value)
    return values


class RouteService:
    """Scores and ranks route alternatives by predicted exposure."""

    def __init__(self, forecast_service: ForecastService):
        self.forecasts = forecast_service

    def _segments_with_pollution(
        self, path, *, timeout: float, retries: int
    ) -> RouteCandidate:
        specs = list(path.segments)
        if not specs:
            raise RouteServiceError(f"route {path.route_id!r} has no segments")

        # Thin the sample points if the route is very long.
        if len(specs) > MAX_SAMPLE_POINTS:
            stride = len(specs) // MAX_SAMPLE_POINTS + 1
            specs = specs[::stride]

        points = [(s.latitude, s.longitude) for s in specs]
        try:
            values = _fetch_predictions_at(points, timeout=timeout, retries=retries)
        except UpstreamError as exc:
            raise RouteServiceError(str(exc)) from exc

        segments = tuple(
            RouteSegment(
                latitude=spec.latitude,
                longitude=spec.longitude,
                distance_km=spec.distance_km,
                duration_minutes=spec.duration_minutes,
                pm25=value,
            )
            for spec, value in zip(specs, values)
        )
        return RouteCandidate(
            route_id=path.route_id,
            label=path.label,
            distance_km=path.distance_km,
            duration_minutes=path.duration_minutes,
            segments=segments,
            geometry=path.geometry,
            provider=f"{path.provider}:{path.provider_profile}",
        )

    def compare(
        self,
        *,
        origin: tuple[float, float],
        destination: tuple[float, float],
        mode: str,
        activity: str | None = None,
        max_routes: int = 3,
    ) -> dict:
        """Compare real route alternatives by predicted exposure."""
        settings = self.forecasts.settings
        try:
            paths = route_alternatives(
                origin,
                destination,
                mode,
                timeout=settings.http_timeout,
                retries=settings.http_retries,
                max_routes=max_routes,
            )
        except RoutingError as exc:
            raise RouteServiceError(str(exc)) from exc

        activity_key = activity or mode
        candidates = [
            self._segments_with_pollution(
                path,
                timeout=settings.http_timeout,
                retries=settings.http_retries,
            )
            for path in paths
        ]

        try:
            ranked = rank_routes(candidates, activity_key, basis="predicted")
        except RouteExposureError as exc:
            raise RouteServiceError(str(exc)) from exc

        routes = [item.to_dict() for item in ranked]
        best = ranked[0]
        return {
            "mode": mode,
            "activity": activity_key,
            "provider": paths[0].provider if paths else None,
            "provider_profile": paths[0].provider_profile if paths else None,
            "generated_at": datetime.now(timezone.utc),
            "routes": routes,
            "recommended_route_id": best.route.route_id,
            "relative_reduction_percent": best.relative_reduction_percent,
            "recommendation": (
                f"{best.route.label} has lower predicted exposure than the other "
                f"route(s) considered"
                + (
                    f" (approximately {best.relative_reduction_percent:.0f}% lower "
                    "predicted exposure than the highest-exposure route)."
                    if best.relative_reduction_percent
                    else "."
                )
            ),
            "note": (
                "Route exposure is an engineering estimate that sums predicted "
                "ambient PM2.5 along the route. It is not a measurement of what a "
                "person inhales, and lower predicted exposure is not a guarantee "
                "of safety. Route geometry comes from OpenStreetMap via the named "
                "routing provider; travel duration uses an assumed activity speed."
            ),
        }
