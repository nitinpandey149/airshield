"""Tests for the safe-window optimizer and route exposure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from airshield_core.route_exposure import (
    RouteCandidate,
    RouteExposureError,
    RouteSegment,
    rank_routes,
    score_route,
)
from airshield_core.windows import WindowOptimizationError, optimize_window

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _horizon(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [BASE + timedelta(hours=i) for i in range(len(values))],
            "pm2_5": values,
        }
    )


def test_optimizer_considers_the_whole_duration() -> None:
    """A window that starts clean but ends dirty must not win on its minimum."""
    # Hours: 0..5. A 90-minute activity starting at hour 0 covers [10, 10, 80]
    # (bad end); starting at hour 3 covers [5, 5, 5] - the best.
    horizon = _horizon([10.0, 10.0, 80.0, 5.0, 5.0, 5.0])
    plan = optimize_window(
        horizon,
        activity="walking",
        duration_minutes=90,
        start_time=BASE,
        end_time=BASE + timedelta(hours=6),
        step_minutes=60,
    )
    assert plan.best.start == BASE + timedelta(hours=3)
    # The best window's peak is clean; the hour-2 window looks for a clean hour
    # but its dirty start dominates once the whole duration is integrated.
    assert plan.best.peak_pm25 == pytest.approx(5.0)
    worst = plan.worst
    assert worst is not None and worst.start == BASE + timedelta(hours=2)
    assert worst.peak_pm25 == pytest.approx(80.0)


def test_optimizer_ranks_by_exposure_not_single_hour() -> None:
    horizon = _horizon([50.0, 50.0, 5.0, 5.0, 50.0, 50.0])
    plan = optimize_window(
        horizon, activity="walking", duration_minutes=60, step_minutes=30
    )
    # The single cleanest hour is index 2 or 3, but a 60-min window must cover
    # two hours; the best two-hour window is [5, 5].
    assert plan.best.start == BASE + timedelta(hours=2)
    assert plan.best.estimate.mean_pm25 == pytest.approx(5.0)
    assert plan.worst is not None
    assert plan.worst.estimate.score > plan.best.estimate.score


def test_optimizer_reports_relative_reduction() -> None:
    horizon = _horizon([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    plan = optimize_window(
        horizon, activity="running", duration_minutes=60, step_minutes=60
    )
    payload = plan.to_dict()
    assert payload["relative_reduction_percent"] > 0
    assert "lower predicted exposure" in payload["reason"].lower()
    assert "not a guarantee of safety" in payload["note"]


def test_optimizer_uses_activity_intensity() -> None:
    horizon = _horizon([20.0] * 6)
    walk = optimize_window(horizon, activity="walking", duration_minutes=30)
    run = optimize_window(horizon, activity="running", duration_minutes=30)
    assert run.best.estimate.score == pytest.approx(walk.best.estimate.score * 2)


def test_optimizer_rejects_impossible_window() -> None:
    horizon = _horizon([10.0, 10.0])
    with pytest.raises(WindowOptimizationError):
        optimize_window(horizon, activity="walking", duration_minutes=600)


def test_optimizer_rejects_bad_arguments() -> None:
    horizon = _horizon([10.0] * 4)
    with pytest.raises(WindowOptimizationError):
        optimize_window(horizon, activity="walking", duration_minutes=0)
    with pytest.raises(WindowOptimizationError):
        optimize_window(horizon, activity="walking", duration_minutes=30, step_minutes=0)
    with pytest.raises(WindowOptimizationError, match="empty"):
        optimize_window(_horizon([]), activity="walking", duration_minutes=30)


def _route(route_id: str, pm25: float, duration: float = 30.0, distance: float = 5.0):
    segments = tuple(
        RouteSegment(
            latitude=52.5 + i * 0.001,
            longitude=13.4,
            distance_km=distance / 4,
            duration_minutes=duration / 4,
            pm25=pm25,
        )
        for i in range(4)
    )
    return RouteCandidate(
        route_id=route_id,
        label=route_id.upper(),
        distance_km=distance,
        duration_minutes=duration,
        segments=segments,
        provider="test",
    )


def test_route_exposure_sums_segments() -> None:
    route = _route("a", 40.0, duration=60.0)
    scored = score_route(route, "walking")
    # 40 ug/m3 * 1h * 1.0 = 40
    assert scored.exposure_score == pytest.approx(40.0)
    assert scored.average_pm25 == pytest.approx(40.0)


def test_rank_routes_orders_by_exposure() -> None:
    routes = [_route("a", 80.0), _route("b", 40.0), _route("c", 20.0)]
    ranked = rank_routes(routes, "walking")
    assert [r.route.route_id for r in ranked] == ["c", "b", "a"]
    # Best route carries the largest reduction; the worst is the baseline.
    assert ranked[0].relative_reduction_percent == pytest.approx(75.0)
    assert ranked[-1].relative_reduction_percent is None


def test_route_exposure_uses_activity_intensity() -> None:
    route = _route("a", 30.0)
    walk = score_route(route, "walking")
    run = score_route(route, "running")
    assert run.exposure_score == pytest.approx(walk.exposure_score * 2)


def test_route_rejects_empty_segments() -> None:
    route = RouteCandidate(
        route_id="x", label="X", distance_km=1.0, duration_minutes=10.0, segments=()
    )
    with pytest.raises(RouteExposureError, match="no segments"):
        score_route(route, "walking")


def test_rank_routes_requires_routes() -> None:
    with pytest.raises(RouteExposureError, match="at least one route"):
        rank_routes([], "walking")


def test_route_to_dict_is_serialisable() -> None:
    ranked = rank_routes([_route("a", 50.0), _route("b", 20.0)], "cycling")
    for item in ranked:
        payload = item.to_dict()
        assert payload["route_id"]
        assert payload["average_pm25"] >= 0
        assert payload["activity"] == "cycling"
