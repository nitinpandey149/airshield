"""Tests for route comparison and AWS status endpoints.

Route comparison needs a live routing engine and live pollution sampling, which
is not available in the test sandbox. These tests therefore verify the contract
and the *failure* behaviour: when a provider is unreachable the API must say so
with a 503 rather than inventing a route.
"""

from __future__ import annotations

import pytest

from app.services.route_service import RouteServiceError, _fetch_predictions_at


BERLIN = {"origin_lat": 52.52, "origin_lon": 13.405, "dest_lat": 52.51, "dest_lon": 13.42}


def test_route_compare_rejects_an_unsupported_mode(client) -> None:
    response = client.get("/api/routes/compare", params={**BERLIN, "mode": "driving"})
    assert response.status_code == 422
    assert "unsupported travel mode" in response.json()["detail"]


def test_route_compare_validates_coordinates(client) -> None:
    response = client.get(
        "/api/routes/compare",
        params={"origin_lat": 999, "origin_lon": 13.4, "dest_lat": 52.5, "dest_lon": 13.4},
    )
    assert response.status_code == 422


def test_route_compare_is_honest_when_no_provider_is_reachable(client, monkeypatch) -> None:
    """No routing engine -> 503, never a fabricated route."""
    from airshield_core import routing

    def boom(*args, **kwargs):
        raise routing.RoutingError("no routing provider reachable in this environment")

    monkeypatch.setattr("app.services.route_service.route_alternatives", boom)
    response = client.get("/api/routes/compare", params={**BERLIN, "mode": "walking"})
    assert response.status_code == 503
    assert "no routing provider" in response.json()["detail"]


def test_route_compare_reports_pollution_sampling_failure(client, monkeypatch) -> None:
    """Routing works but pollution sampling fails -> fail, do not invent values."""
    from airshield_core.routing import RoutePath, RouteSegmentSpec
    from airshield_core.sources.openmeteo import UpstreamError

    path = RoutePath(
        route_id="route_1",
        label="Route 1",
        distance_km=5.0,
        duration_minutes=25.0,
        geometry=[(52.52, 13.405), (52.51, 13.42)],
        segments=(
            RouteSegmentSpec(latitude=52.52, longitude=13.405, distance_km=2.5, duration_minutes=12.5, fraction=0.25),
            RouteSegmentSpec(latitude=52.51, longitude=13.42, distance_km=2.5, duration_minutes=12.5, fraction=0.75),
        ),
        provider="test",
        provider_profile="pedestrian",
        provider_duration_minutes=25.0,
    )
    monkeypatch.setattr(
        "app.services.route_service.route_alternatives", lambda *a, **k: [path]
    )

    def boom(*args, **kwargs):
        raise UpstreamError("Open-Meteo unreachable")

    monkeypatch.setattr("app.services.route_service._fetch_predictions_at", boom)
    response = client.get("/api/routes/compare", params={**BERLIN, "mode": "walking"})
    assert response.status_code == 503
    assert "Open-Meteo unreachable" in response.json()["detail"]


def test_route_compare_scores_and_ranks_with_injected_pollution(client, monkeypatch) -> None:
    """With real geometry and injected real-shaped pollution, ranking is correct."""
    from airshield_core.routing import RoutePath, RouteSegmentSpec

    def make(route_id: str, distance: float, duration: float) -> RoutePath:
        geometry = [(52.52, 13.405), (52.515, 13.412), (52.51, 13.42)]
        segments = tuple(
            RouteSegmentSpec(
                latitude=52.52 - i * 0.005,
                longitude=13.405 + i * 0.0075,
                distance_km=distance / 2,
                duration_minutes=duration / 2,
                fraction=0.25 + i * 0.5,
            )
            for i in range(2)
        )
        return RoutePath(
            route_id=route_id,
            label=route_id.upper(),
            distance_km=distance,
            duration_minutes=duration,
            geometry=geometry,
            segments=segments,
            provider="test",
            provider_profile="pedestrian",
            provider_duration_minutes=duration,
        )

    paths = [make("route_1", 5.2, 24), make("route_2", 5.5, 27), make("route_3", 5.8, 29)]
    monkeypatch.setattr(
        "app.services.route_service.route_alternatives", lambda *a, **k: paths
    )
    # Route 1 passes the dirtiest air, route 3 the cleanest.
    pollution = {"route_1": [80.0, 80.0], "route_2": [50.0, 50.0], "route_3": [20.0, 20.0]}
    calls = {"i": 0}

    def fake_fetch(points, *, timeout, retries):
        # Two sample points per route; consume in route order.
        route_ids = ["route_1", "route_2", "route_3"]
        values = pollution[route_ids[calls["i"]]]
        calls["i"] += 1
        return values

    monkeypatch.setattr("app.services.route_service._fetch_predictions_at", fake_fetch)

    response = client.get("/api/routes/compare", params={**BERLIN, "mode": "walking"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["recommended_route_id"] == "route_3"
    assert [r["route_id"] for r in body["routes"]] == ["route_3", "route_2", "route_1"]
    assert body["routes"][0]["average_pm25"] == pytest.approx(20.0)
    # Reduction is duration-weighted: route_3 (20 ug/m3 x 29 min) vs route_1
    # (80 ug/m3 x 24 min): (32.0 - 9.67) / 32.0 = 69.8%. Not the naive 75%,
    # because the cleaner route also takes longer.
    assert body["relative_reduction_percent"] == pytest.approx(69.8, abs=0.5)
    # The wording must avoid claiming safety.
    assert "lower predicted exposure" in body["recommendation"].lower()
    assert "not a guarantee of safety" in body["note"].lower()


def test_fetch_predictions_rejects_empty_points() -> None:
    with pytest.raises(RouteServiceError, match="no sample points"):
        _fetch_predictions_at([], timeout=1.0, retries=0)
