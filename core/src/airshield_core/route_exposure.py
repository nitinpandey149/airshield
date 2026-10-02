"""Route exposure: aggregate predicted pollution along alternative routes.

A route is split into segments. Each segment carries a predicted PM2.5 (from a
real forecast sampled at the segment's coordinates), a distance and a duration.
Exposure is the sum over segments:

    route exposure = sum(segment pm25 x segment duration x activity intensity)

Routes are ranked by that total. The result is always described as *lower
predicted exposure*, never as a safe route: the concentration is an ambient
forecast at sample points, not a measurement of what a person inhales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from airshield_core.exposure import (
    ACTIVITY_LABELS,
    exposure_over_series,
    normalize_activity,
    relative_reduction_percent,
)


class RouteExposureError(ValueError):
    """Raised when a route cannot be scored."""


@dataclass(frozen=True)
class RouteSegment:
    """One leg of a route with its predicted ambient PM2.5."""

    latitude: float
    longitude: float
    distance_km: float
    duration_minutes: float
    pm25: float

    def to_dict(self) -> dict:
        return {
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "distance_km": round(self.distance_km, 3),
            "duration_minutes": round(self.duration_minutes, 2),
            "pm25": round(self.pm25, 2),
        }


@dataclass(frozen=True)
class RouteCandidate:
    """A route alternative returned by the routing engine."""

    route_id: str
    label: str
    distance_km: float
    duration_minutes: float
    segments: tuple[RouteSegment, ...]
    geometry: tuple[tuple[float, float], ...] = ()
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "label": self.label,
            "distance_km": round(self.distance_km, 3),
            "duration_minutes": round(self.duration_minutes, 1),
            "provider": self.provider,
            "geometry": [[round(lat, 5), round(lon, 5)] for lat, lon in self.geometry],
            "segments": [s.to_dict() for s in self.segments],
        }


@dataclass
class RouteExposure:
    """Predicted exposure for one route."""

    route: RouteCandidate
    activity: str
    exposure_score: float
    exposure_level: str
    average_pm25: float
    peak_pm25: float
    relative_reduction_percent: float | None = None

    def to_dict(self) -> dict:
        return {
            "route_id": self.route.route_id,
            "label": self.route.label,
            "distance_km": round(self.route.distance_km, 3),
            "duration_minutes": round(self.route.duration_minutes, 1),
            "average_pm25": round(self.average_pm25, 2),
            "peak_pm25": round(self.peak_pm25, 2),
            "exposure_score": round(self.exposure_score, 1),
            "exposure_level": self.exposure_level,
            "activity": self.activity,
            "activity_label": ACTIVITY_LABELS.get(self.activity, self.activity),
            "relative_reduction_percent": self.relative_reduction_percent,
            "provider": self.route.provider,
            "geometry": [
                [round(lat, 5), round(lon, 5)] for lat, lon in self.route.geometry
            ],
        }


def score_route(
    route: RouteCandidate,
    activity: str = "walking",
    *,
    micro_scale: float = 1.0,
    basis: str = "predicted",
) -> RouteExposure:
    """Compute predicted exposure for one route."""
    if not route.segments:
        raise RouteExposureError(f"route {route.route_id!r} has no segments")
    if route.duration_minutes <= 0:
        raise RouteExposureError(f"route {route.route_id!r} has no duration")

    activity_key = normalize_activity(activity)
    estimate = exposure_over_series(
        [(seg.pm25, seg.duration_minutes) for seg in route.segments],
        activity_key,
        micro_scale=micro_scale,
        basis=basis,
    )
    return RouteExposure(
        route=route,
        activity=activity_key,
        exposure_score=estimate.score,
        exposure_level=estimate.level,
        average_pm25=estimate.mean_pm25,
        peak_pm25=estimate.peak_pm25,
    )


def rank_routes(
    routes: list[RouteCandidate],
    activity: str = "walking",
    *,
    micro_scale: float = 1.0,
    basis: str = "predicted",
) -> list[RouteExposure]:
    """Score and rank routes from lowest to highest predicted exposure.

    ``relative_reduction_percent`` on each route is measured against the
    *highest*-exposure route in the set, so the best route carries the largest
    reduction. The lowest-exposure route has ``relative_reduction_percent=None``
    because it is the baseline for the comparison.
    """
    if not routes:
        raise RouteExposureError("at least one route is required")

    scored = [score_route(r, activity, micro_scale=micro_scale, basis=basis) for r in routes]
    worst = max(s.exposure_score for s in scored)

    ranked: list[RouteExposure] = []
    for item in sorted(scored, key=lambda s: s.exposure_score):
        if item.exposure_score >= worst:
            item.relative_reduction_percent = None
        else:
            item.relative_reduction_percent = relative_reduction_percent(
                worst, item.exposure_score
            )
        ranked.append(item)
    return ranked
