"""Tests for pollution event / spike detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airshield_core.events import EventDetectionError, detect_spike
from airshield_core.spike_calibration import (
    CalibrationTable,
    build_calibration,
)


def _series(values: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=len(values), freq="h", tz="UTC"),
            "pm2_5": values,
            "wind_speed_10m": np.linspace(10, 2, len(values)),
            "surface_pressure": np.linspace(1005, 1020, len(values)),
            "nitrogen_dioxide": np.linspace(10, 40, len(values)),
        }
    )


def test_flat_series_detects_nothing() -> None:
    event = detect_spike(_series([20.0] * 6))
    assert event.detected is False
    assert event.kind == "none"
    assert event.severity == "none"
    assert event.confidence == 0.0


def test_small_rise_below_absolute_floor_is_not_a_spike() -> None:
    event = detect_spike(_series([2.0, 2.5, 3.0, 3.5]))
    assert event.detected is False


def test_sharp_rise_is_a_spike() -> None:
    event = detect_spike(_series([20.0, 24.0, 30.0, 34.0]), baseline_value=20.0)
    assert event.detected is True
    assert event.kind == "spike"
    assert event.expected_change_percent == pytest.approx(70.0)
    assert event.peak_pm25 == pytest.approx(34.0)
    assert event.expected_time is not None


def test_sustained_elevation_is_flagged() -> None:
    # A ~36% rise held for several hours is flagged and classified.
    values = [20.0, 26.0, 26.5, 27.0, 26.8, 27.2]
    event = detect_spike(_series(values), baseline_value=20.0)
    assert event.detected is True
    assert event.kind in {"sustained", "sudden", "spike"}
    assert event.severity in {"minor", "moderate", "severe"}


def test_associated_signals_are_reported_without_causal_claims() -> None:
    event = detect_spike(_series([20.0, 24.0, 30.0, 36.0]), baseline_value=20.0)
    assert event.detected
    labels = {s["label"] for s in event.associated_signals}
    # Wind falls and NO2 rises in the fixture, in the expected directions.
    assert "reduced wind" in labels
    assert "increased NO2" in labels
    payload = event.to_dict()
    assert "not proven causes" in payload["signal_disclaimer"]


def test_confidence_is_bounded_and_has_a_basis() -> None:
    event = detect_spike(_series([20.0, 30.0, 45.0]), baseline_value=20.0)
    assert 0.0 <= event.confidence <= 1.0
    assert event.confidence_basis == "rule_based_estimate"


def test_calibrated_confidence_uses_measured_frequency() -> None:
    table = CalibrationTable(
        horizons={"1": {"<0.10": 0.02, "<0.25": 0.10, "<0.50": 0.55, "<1.00": 0.9, ">=1.00": 0.99}},
        counts={"1": {"<0.50": 500}},
        source="test",
    )
    assert table.available
    event = detect_spike(
        _series([20.0, 24.0, 28.0]),
        baseline_value=20.0,
        horizon_hours=1,
        calibration=table,
    )
    # change = 0.4 -> bucket "<0.50" -> calibrated 0.55
    assert event.confidence == pytest.approx(0.55)
    assert event.confidence_basis == "empirical_calibration"


def test_calibration_round_trip(tmp_path) -> None:
    table = CalibrationTable(horizons={"1": {"<0.50": 0.4}}, source="unit")
    path = table.save(tmp_path / "cal.json")
    loaded = CalibrationTable.load(path)
    assert loaded.available
    assert loaded.probability(0.4, 1) == pytest.approx(0.4)


def test_calibration_missing_file_is_empty(tmp_path) -> None:
    table = CalibrationTable.load(tmp_path / "absent.json")
    assert not table.available
    assert table.basis == "rule_based_estimate"


def test_build_calibration_from_real_frame(demo_frame: pd.DataFrame) -> None:
    """Calibration must be derived from real data and be a valid probability."""
    table = build_calibration(demo_frame, horizons=(1, 3), source="unit-test")
    assert table.available
    for horizon, buckets in table.horizons.items():
        for bucket, rate in buckets.items():
            assert 0.0 <= rate <= 1.0, f"{horizon}/{bucket} not a probability"
        assert table.counts[horizon]


def test_detector_rejects_bad_input() -> None:
    with pytest.raises(EventDetectionError, match="empty"):
        detect_spike(pd.DataFrame({"time": [], "pm2_5": []}))
    with pytest.raises(EventDetectionError, match="at least two"):
        detect_spike(_series([20.0]))
    bad = _series([20.0, float("nan")])
    with pytest.raises(EventDetectionError, match="NaN"):
        detect_spike(bad)
