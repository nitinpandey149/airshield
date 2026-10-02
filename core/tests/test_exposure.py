"""Tests for the exposure engine."""

from __future__ import annotations

import pytest

from airshield_core.exposure import (
    ACTIVITY_INTENSITY,
    UnknownActivityError,
    activity_intensity,
    exposure_level,
    exposure_over_series,
    normalize_activity,
    relative_reduction_percent,
    segment_exposure,
)


def test_activity_factors_are_the_documented_values() -> None:
    assert ACTIVITY_INTENSITY["walking"] == 1.0
    assert ACTIVITY_INTENSITY["cycling"] == 1.5
    assert ACTIVITY_INTENSITY["running"] == 2.0
    assert ACTIVITY_INTENSITY["outdoor_work"] == 1.4


def test_activity_aliases_normalise() -> None:
    assert normalize_activity("Run") == "running"
    assert normalize_activity("outdoor work") == "outdoor_work"
    assert normalize_activity("Bike") == "cycling"
    assert activity_intensity("running") == 2.0


def test_unknown_activity_is_rejected() -> None:
    with pytest.raises(UnknownActivityError, match="unknown activity"):
        normalize_activity("teleporting")


def test_exposure_scales_with_duration() -> None:
    short = segment_exposure(20.0, 30, "walking")
    long = segment_exposure(20.0, 60, "walking")
    assert long.score == pytest.approx(short.score * 2)


def test_exposure_scales_with_intensity() -> None:
    walk = segment_exposure(20.0, 45, "walking")
    run = segment_exposure(20.0, 45, "running")
    assert run.score == pytest.approx(walk.score * 2.0)


def test_exposure_matches_the_formula() -> None:
    """exposure = pm25 x hours x intensity x location_factor."""
    estimate = segment_exposure(30.0, 30, "cycling")
    assert estimate.score == pytest.approx(30.0 * 0.5 * 1.5 * 1.0)
    assert estimate.mean_pm25 == pytest.approx(30.0)
    assert estimate.peak_pm25 == pytest.approx(30.0)


def test_micro_scale_multiplies_exposure() -> None:
    base = segment_exposure(30.0, 30, "walking")
    canyon = segment_exposure(30.0, 30, "walking", micro_scale=1.5)
    assert canyon.score == pytest.approx(base.score * 1.5)


def test_negative_duration_and_nan_are_rejected() -> None:
    with pytest.raises(ValueError):
        segment_exposure(10.0, -5, "walking")
    with pytest.raises(ValueError):
        segment_exposure(float("nan"), 10, "walking")
    with pytest.raises(ValueError):
        segment_exposure(-1.0, 10, "walking")


def test_exposure_level_bands() -> None:
    assert exposure_level(0.0) == "low"
    assert exposure_level(25.0) == "moderate"
    assert exposure_level(50.0) == "high"
    assert exposure_level(500.0) == "very_high"


def test_series_aggregation_integrates_concentration_time() -> None:
    """A window whose second half is dirty must not win on its minimum."""
    clean_then_dirty = exposure_over_series([(10.0, 30), (80.0, 30)], "walking")
    steady = exposure_over_series([(45.0, 30), (45.0, 30)], "walking")
    # Same mean (45) but the varying window has the same integral too; both
    # average to 45, so scores match. The point is that the peak is captured.
    assert clean_then_dirty.mean_pm25 == pytest.approx(45.0)
    assert steady.mean_pm25 == pytest.approx(45.0)
    assert clean_then_dirty.peak_pm25 == pytest.approx(80.0)


def test_series_aggregation_weights_by_duration() -> None:
    mostly_clean = exposure_over_series([(10.0, 50), (80.0, 10)], "walking")
    mostly_dirty = exposure_over_series([(80.0, 50), (10.0, 10)], "walking")
    assert mostly_clean.score < mostly_dirty.score


def test_series_requires_samples() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        exposure_over_series([], "walking")


def test_relative_reduction_formula() -> None:
    # (82 - 47) / 82 * 100 = 42.68 -> 42.7
    assert relative_reduction_percent(82.0, 47.0) == pytest.approx(42.7)


def test_relative_reduction_never_negative() -> None:
    assert relative_reduction_percent(10.0, 20.0) == 0.0
    assert relative_reduction_percent(0.0, 5.0) == 0.0
