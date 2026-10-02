"""Route geometry and segmentation from real routing engines.

Two public OpenStreetMap routing engines are supported, tried in order:

1. **Valhalla** (``valhalla1.openstreetmap.de``) - honours per-mode costing
   (``pedestrian``, ``bicycle``, ``auto``) and returns alternatives.
2. **OSRM** (``router.project-osrm.org``) - fallback. Its public demo server
   currently serves *car* routing regardless of the profile in the URL, so the
   profile it returns is reported as ``car`` rather than silently presented as a
   walking route.

Nothing here is simulated: if no engine answers, :class:`RoutingError` is raised
and the caller must report the feature as unavailable rather than invent a route.

Segmentation splits a route into equal-distance slices so that predicted PM2.5
can be sampled at each slice's midpoint. This is what makes route exposure a
sum over the route rather than a single number for the whole trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import httpx

VALHALLA_URL = "https://valhalla1.openstreetmap.de/route"
OSRM_URL = "https://router.project-osrm.org/route/v1"

#: AirShield travel mode -> Valhalla costing model.
MODE_TO_COSTING: dict[str, str] = {
    "walking": "pedestrian",
    "running": "pedestrian",
    "cycling": "bicycle",
}

#: AirShield travel mode -> OSRM profile path segment.
MODE_TO_OSRM_PROFILE: dict[str, str] = {
    "walking": "foot",
    "running": "foot",
    "cycling": "bike",
}

#: Assumed travel speeds (km/h) used to turn a distance into an activity
#: duration. These are stated assumptions, not measurements: the routing
#: engine's own duration reflects its profile, not the user's activity.
ACTIVITY_SPEED_KMH: dict[str, float] = {
    "walking": 5.0,
    "running": 9.5,
    "cycling": 15.0,
    "outdoor_work": 4.0,
    "child_outdoor_activity": 4.5,
}

#: Number of segments a route is split into for pollution sampling.
DEFAULT_SEGMENTS = 8


class RoutingError(RuntimeError):
    """Raised when no routing engine can return a route."""


@dataclass(frozen=True)
class RouteSegmentSpec:
    """One route slice, before environmental data is attached."""

    latitude: float
    longitude: float
    distance_km: float
    duration_minutes: float
    fraction: float


@dataclass
class RoutePath:
    """A route returned by a routing engine."""

    route_id: str
    label: str
    distance_km: float
    duration_minutes: float
    geometry: tuple[tuple[float, float], ...]
    provider: str
    provider_profile: str
    provider_duration_minutes: float
    segments: tuple[RouteSegmentSpec, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "label": self.label,
            "distance_km": round(self.distance_km, 3),
            "duration_minutes": round(self.duration_minutes, 1),
            "provider": self.provider,
            "provider_profile": self.provider_profile,
            "provider_duration_minutes": round(self.provider_duration_minutes, 1),
            "geometry": [[round(lat, 5), round(lon, 5)] for lat, lon in self.geometry],
            "segments": [
                {
                    "latitude": round(s.latitude, 5),
                    "longitude": round(s.longitude, 5),
                    "distance_km": round(s.distance_km, 3),
                    "duration_minutes": round(s.duration_minutes, 2),
                }
                for s in self.segments
            ],
        }


def decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
    """Decode a Google/Valhalla encoded polyline into ``(lat, lon)`` pairs."""
    coordinates: list[tuple[float, float]] = []
    index = 0
    latitude = 0
    longitude = 0
    factor = 10**precision
    length = len(encoded)

    while index < length:
        for is_latitude in (True, False):
            shift = 0
            result = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if is_latitude:
                latitude += delta
            else:
                longitude += delta
        coordinates.append((latitude / factor, longitude / factor))
    return coordinates


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two ``(lat, lon)`` points."""
    radius = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def _cumulative_distances(geometry: list[tuple[float, float]]) -> list[float]:
    cumulative = [0.0]
    for previous, current in zip(geometry, geometry[1:]):
        cumulative.append(cumulative[-1] + _haversine_km(previous, current))
    return cumulative


def segment_geometry(
    geometry: list[tuple[float, float]],
    distance_km: float,
    duration_minutes: float,
    *,
    segments: int = DEFAULT_SEGMENTS,
) -> tuple[RouteSegmentSpec, ...]:
    """Split a route into equal-distance slices with midpoint coordinates.

    Distances are measured along the geometry (haversine); the engine's total
    distance is used to scale them so the slices sum to the reported route.
    Duration is apportioned by distance.
    """
    if len(geometry) < 2:
        raise RoutingError("geometry needs at least two points to be segmented")
    if segments < 1:
        raise ValueError("segments must be >= 1")

    cumulative = _cumulative_distances(geometry)
    total = cumulative[-1]
    if total <= 0:
        raise RoutingError("geometry has zero length")

    # Use the engine's distance when available; otherwise fall back to the sum
    # of the geometry. Scale keeps slices summing to the reported distance.
    reported = distance_km if distance_km > 0 else total
    scale = reported / total

    specs: list[RouteSegmentSpec] = []
    for i in range(segments):
        start_target = total * i / segments
        end_target = total * (i + 1) / segments
        mid_target = (start_target + end_target) / 2

        # Slice lengths are exact by construction; scaling by the engine's
        # reported distance keeps the slices summing to that distance.
        segment_km = (end_target - start_target) * scale
        midpoint = _interpolate(geometry, cumulative, mid_target)
        specs.append(
            RouteSegmentSpec(
                latitude=midpoint[0],
                longitude=midpoint[1],
                distance_km=segment_km,
                duration_minutes=(segment_km / reported) * duration_minutes
                if reported > 0
                else 0.0,
                fraction=(i + 0.5) / segments,
            )
        )
    return tuple(specs)


def _locate(cumulative: list[float], target: float) -> int:
    """Index of the geometry vertex at or before ``target`` distance."""
    lo, hi = 0, len(cumulative) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cumulative[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _interpolate(
    geometry: list[tuple[float, float]], cumulative: list[float], target: float
) -> tuple[float, float]:
    """Point at ``target`` distance along the polyline."""
    index = _locate(cumulative, target)
    index = min(index, len(geometry) - 2)
    span = cumulative[index + 1] - cumulative[index]
    if span <= 0:
        return geometry[index]
    ratio = (target - cumulative[index]) / span
    lat = geometry[index][0] + ratio * (geometry[index + 1][0] - geometry[index][0])
    lon = geometry[index][1] + ratio * (geometry[index + 1][1] - geometry[index][1])
    return (lat, lon)


def _activity_duration(distance_km: float, mode: str, provider_minutes: float) -> float:
    """Activity duration from distance at the stated assumed speed.

    The routing engine's duration reflects its own profile, not the user's
    activity (a run over a pedestrian network is not walked). We therefore
    derive the duration from the distance and a documented assumed speed, and
    report the engine's own duration alongside for transparency.
    """
    speed = ACTIVITY_SPEED_KMH.get(mode)
    if speed and distance_km > 0:
        return distance_km / speed * 60.0
    return provider_minutes


def _detour_midpoint(
    origin: tuple[float, float], destination: tuple[float, float], offset: float
) -> tuple[float, float]:
    """A point ``offset`` km to one side of the origin->destination midpoint.

    Used to request genuine alternative routes: routing through a via-point makes
    the engine return a real, distinct path rather than an invented one.
    """
    mid_lat = (origin[0] + destination[0]) / 2
    mid_lon = (origin[1] + destination[1]) / 2
    dlat = destination[0] - origin[0]
    dlon = destination[1] - origin[1]
    span = math.hypot(dlat, dlon)
    if span == 0:
        return (mid_lat, mid_lon)
    # Perpendicular in degree space, normalised then scaled to km.
    perp_lat = -dlon / span
    perp_lon = dlat / span
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(0.2, math.cos(math.radians(mid_lat)))
    return (
        mid_lat + (perp_lat * offset) / km_per_deg_lat,
        mid_lon + (perp_lon * offset) / km_per_deg_lon,
    )


def route_alternatives(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    *,
    timeout: float = 15.0,
    retries: int = 2,
    max_routes: int = 3,
    segments: int = DEFAULT_SEGMENTS,
) -> list[RoutePath]:
    """Return real route alternatives between two ``(lat, lon)`` points.

    Engines do not always return as many alternatives as requested, so when too
    few come back we ask for genuine detours through via-points offset to either
    side of the direct line. Every route still comes from the routing engine.

    Raises :class:`RoutingError` if no engine can serve the request. The caller
    must then report the route feature as unavailable - never invent geometry.
    """
    if mode not in MODE_TO_COSTING:
        raise RoutingError(f"unsupported travel mode {mode!r}")

    errors: list[str] = []
    for provider in (_route_valhalla, _route_osrm):
        try:
            routes = provider(
                origin,
                destination,
                mode,
                timeout=timeout,
                retries=retries,
                max_routes=max_routes,
                segments=segments,
            )
        except RoutingError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:  # network/parse failures must not crash the API
            errors.append(f"{provider.__name__}: {type(exc).__name__}: {exc}")
            continue

        if not routes:
            errors.append(f"{provider.__name__}: no routes returned")
            continue

        if len(routes) < max_routes and provider is _route_valhalla:
            routes = _add_detours(
                routes,
                origin,
                destination,
                mode,
                target=max_routes,
                timeout=timeout,
                retries=retries,
                segments=segments,
            )
        return routes[:max_routes]

    raise RoutingError(
        "no routing engine returned a route for this origin/destination. "
        + "; ".join(errors)
    )


def _route_signature(geometry: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    """A point at the middle of the geometry, used to tell routes apart.

    The start of a route is shared between alternatives (same first street), so
    the midpoint is a far better discriminator than the leading vertices.
    """
    if not geometry:
        return (0.0, 0.0)
    mid = geometry[len(geometry) // 2]
    return (round(mid[0], 4), round(mid[1], 4))


def _add_detours(
    routes: list[RoutePath],
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    *,
    target: int,
    timeout: float,
    retries: int,
    segments: int,
) -> list[RoutePath]:
    """Top up a short route list with real via-point detours."""
    direct_km = max(0.5, _haversine_km(origin, destination))
    offsets = [direct_km * 0.15, -direct_km * 0.15, direct_km * 0.30, -direct_km * 0.30]

    seen = {_route_signature(route.geometry) for route in routes}
    index = len(routes)

    for offset in offsets:
        if len(routes) >= target:
            break
        via = _detour_midpoint(origin, destination, offset)
        try:
            detour = _route_valhalla_waypoints(
                origin, via, destination, mode, timeout=timeout, retries=retries
            )
        except Exception:  # a failed detour just means fewer alternatives
            continue
        if detour is None:
            continue
        signature = _route_signature(detour.geometry)
        if signature in seen:
            continue
        seen.add(signature)
        route_id = f"route_{chr(ord('a') + index)}"
        label = f"Route {chr(ord('A') + index)}"
        index += 1
        routes.append(
            RoutePath(
                route_id=route_id,
                label=label,
                distance_km=detour.distance_km,
                duration_minutes=detour.duration_minutes,
                geometry=detour.geometry,
                provider="valhalla",
                provider_profile=detour.provider_profile,
                provider_duration_minutes=detour.provider_duration_minutes,
                segments=segment_geometry(
                    list(detour.geometry),
                    detour.distance_km,
                    detour.duration_minutes,
                    segments=segments,
                ),
            )
        )
    return routes


def _route_valhalla_waypoints(
    origin: tuple[float, float],
    via: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    *,
    timeout: float,
    retries: int,
) -> RoutePath | None:
    """Route through a via-point, returning one real path (no alternatives)."""
    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": via[0], "lon": via[1], "type": "through"},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": MODE_TO_COSTING[mode],
        "units": "kilometers",
    }
    payload = _post_json(VALHALLA_URL, body, timeout, retries)
    trip = payload.get("trip") or {}
    if trip.get("status") not in (0, None):
        return None
    legs = trip.get("legs") or []
    if not legs:
        return None
    summary = trip.get("summary") or {}
    geometry: list[tuple[float, float]] = []
    for leg in legs:
        decoded = decode_polyline(leg.get("shape", ""))
        geometry.extend(decoded if not geometry else decoded[1:])
    distance_km = float(summary.get("length") or 0.0)
    provider_minutes = float(summary.get("time") or 0.0) / 60.0
    if distance_km <= 0 or len(geometry) < 2:
        return None
    return RoutePath(
        route_id="via",
        label="Via",
        distance_km=distance_km,
        duration_minutes=_activity_duration(distance_km, mode, provider_minutes),
        geometry=tuple(geometry),
        provider="valhalla",
        provider_profile=MODE_TO_COSTING[mode],
        provider_duration_minutes=provider_minutes,
    )


def _route_valhalla(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    *,
    timeout: float,
    retries: int,
    max_routes: int,
    segments: int,
) -> list[RoutePath]:
    body = {
        "locations": [
            {"lat": origin[0], "lon": origin[1]},
            {"lat": destination[0], "lon": destination[1]},
        ],
        "costing": MODE_TO_COSTING[mode],
        "alternates": max(0, max_routes - 1),
        "units": "kilometers",
    }
    payload = _post_json(VALHALLA_URL, body, timeout, retries)
    trip = payload.get("trip") or {}
    if trip.get("status") not in (0, None):
        raise RoutingError(
            f"Valhalla returned status {trip.get('status')}: {trip.get('status_message')}"
        )

    legs = trip.get("legs") or []
    if not legs:
        raise RoutingError("Valhalla returned no legs")

    paths: list[RoutePath] = []
    for index, entry in enumerate([trip, *(payload.get("alternates") or [])]):
        entry_legs = entry.get("legs") or []
        if not entry_legs:
            continue
        summary = entry.get("summary") or {}
        geometry: list[tuple[float, float]] = []
        for leg in entry_legs:
            decoded = decode_polyline(leg.get("shape", ""))
            geometry.extend(decoded if not geometry else decoded[1:])
        distance_km = float(summary.get("length") or 0.0)
        provider_minutes = float(summary.get("time") or 0.0) / 60.0
        if distance_km <= 0 or len(geometry) < 2:
            continue
        duration = _activity_duration(distance_km, mode, provider_minutes)
        paths.append(
            RoutePath(
                route_id=f"route_{chr(ord('a') + index)}",
                label=f"Route {chr(ord('A') + index)}",
                distance_km=distance_km,
                duration_minutes=duration,
                geometry=tuple(geometry),
                provider="valhalla",
                provider_profile=MODE_TO_COSTING[mode],
                provider_duration_minutes=provider_minutes,
                segments=segment_geometry(
                    geometry, distance_km, duration, segments=segments
                ),
            )
        )
    return paths[:max_routes]


def _route_osrm(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    *,
    timeout: float,
    retries: int,
    max_routes: int,
    segments: int,
) -> list[RoutePath]:
    profile = MODE_TO_OSRM_PROFILE[mode]
    url = (
        f"{OSRM_URL}/{profile}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    )
    payload = _get_json(
        url,
        {
            "alternatives": "true",
            "overview": "simplified",
            "geometries": "geojson",
        },
        timeout,
        retries,
    )
    if payload.get("code") != "Ok":
        raise RoutingError(f"OSRM returned code {payload.get('code')}")

    paths: list[RoutePath] = []
    for index, route in enumerate(payload.get("routes") or []):
        coordinates = (route.get("geometry") or {}).get("coordinates") or []
        geometry = [(lat, lon) for lon, lat in coordinates]
        distance_km = float(route.get("distance") or 0.0) / 1000.0
        provider_minutes = float(route.get("duration") or 0.0) / 60.0
        if distance_km <= 0 or len(geometry) < 2:
            continue
        duration = _activity_duration(distance_km, mode, provider_minutes)
        paths.append(
            RoutePath(
                route_id=f"route_{chr(ord('a') + index)}",
                label=f"Route {chr(ord('A') + index)}",
                distance_km=distance_km,
                duration_minutes=duration,
                geometry=tuple(geometry),
                provider="osrm",
                # The public OSRM demo server ignores the profile and serves car
                # routing; report that honestly rather than as a walking route.
                provider_profile="car",
                provider_duration_minutes=provider_minutes,
                segments=segment_geometry(
                    geometry, distance_km, duration, segments=segments
                ),
            )
        )
    return paths[:max_routes]


def _get_json(url: str, params: dict, timeout: float, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
    raise RoutingError(f"GET {url} failed after {retries + 1} attempts: {last_error}")


def _post_json(url: str, body: dict, timeout: float, retries: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=body)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
    raise RoutingError(f"POST {url} failed after {retries + 1} attempts: {last_error}")
