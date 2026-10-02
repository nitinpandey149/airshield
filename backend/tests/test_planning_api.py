"""Tests for the exposure planner API contract."""

from __future__ import annotations

import pytest


def test_activities_are_listed_with_intensity(client) -> None:
    response = client.get("/api/activities")
    assert response.status_code == 200
    activities = {a["activity"]: a for a in response.json()}
    assert activities["walking"]["intensity"] == 1.0
    assert activities["running"]["intensity"] == 2.0
    assert activities["cycling"]["intensity"] == 1.5
    assert activities["outdoor_work"]["intensity"] == 1.4


def test_plan_returns_a_best_window(client) -> None:
    response = client.get("/api/plan/delhi", params={"activity": "running", "duration_minutes": 45})
    assert response.status_code == 200, response.text
    body = response.json()

    best = body["best_window"]
    start, end = best["start"], best["end"]
    assert start < end
    assert body["activity"] == "running"
    assert body["duration_minutes"] == 45
    assert body["best_window"]["exposure"]["score"] >= 0
    # The reason must use non-medical language.
    assert "lower predicted exposure" in body["reason"].lower()
    assert "not a guarantee of safety" in body["note"].lower()


def test_plan_scores_the_whole_duration(client) -> None:
    """A 90-minute plan must integrate more than one hour of forecast."""
    body = client.get(
        "/api/plan/delhi", params={"activity": "walking", "duration_minutes": 90}
    ).json()
    best = body["best_window"]
    assert best["exposure"]["duration_minutes"] == pytest.approx(90, abs=1)
    # Candidate windows each carry a full estimate.
    for candidate in body["candidates"]:
        assert candidate["exposure"]["duration_minutes"] > 0
        assert candidate["exposure"]["level"] in {"low", "moderate", "high", "very_high"}


def test_plan_reports_relative_reduction_and_alternative(client) -> None:
    body = client.get(
        "/api/plan/delhi", params={"activity": "running", "duration_minutes": 45}
    ).json()
    if body["relative_reduction_percent"] is not None:
        assert body["relative_reduction_percent"] >= 0
    if body["alternative_window"] is not None:
        assert (
            body["alternative_window"]["exposure"]["score"]
            >= body["best_window"]["exposure"]["score"]
        )


def test_plan_activity_intensity_changes_the_score(client) -> None:
    walk = client.get(
        "/api/plan/delhi", params={"activity": "walking", "duration_minutes": 30}
    ).json()
    run = client.get(
        "/api/plan/delhi", params={"activity": "running", "duration_minutes": 30}
    ).json()
    # Same windows and same pollution, but running doubles the exposure score.
    assert run["best_window"]["exposure"]["score"] == pytest.approx(
        walk["best_window"]["exposure"]["score"] * 2
    )


def test_plan_timeline_is_measured_from_real_models(client) -> None:
    body = client.get(
        "/api/plan/delhi", params={"activity": "walking", "duration_minutes": 30}
    ).json()
    timeline = body["timeline"]
    assert timeline
    for point in timeline:
        assert point["pm2_5"] >= 0
        assert point["horizon_hours"] in {1, 3, 6}


def test_plan_rejects_an_unknown_activity(client) -> None:
    response = client.get("/api/plan/delhi", params={"activity": "skydiving"})
    assert response.status_code == 422
    assert "unknown activity" in response.json()["detail"]


def test_plan_rejects_an_impossible_duration(client) -> None:
    response = client.get(
        "/api/plan/delhi", params={"activity": "walking", "duration_minutes": 1000}
    )
    # FastAPI validates the query bound before the service is reached.
    assert response.status_code == 422


def test_plan_rejects_an_unknown_location(client) -> None:
    response = client.get("/api/plan/atlantis", params={"activity": "walking"})
    assert response.status_code == 404


def test_plan_provenance_is_attached(client) -> None:
    body = client.get(
        "/api/plan/delhi", params={"activity": "walking", "duration_minutes": 30}
    ).json()
    assert body["source"]["mode"] in {"live", "demo"}
    assert body["source"]["name"]
    assert body["exposure_note"]
