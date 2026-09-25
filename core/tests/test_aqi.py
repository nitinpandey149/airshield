"""Tests for the EPA AQI conversion and the exposure alert."""

from __future__ import annotations

import pytest

from airshield_core.aqi import (
    PM25_BREAKPOINTS,
    aqi_category,
    aqi_from_pm25,
    exposure_alert,
)


@pytest.mark.parametrize(
    ("pm25", "expected_aqi", "expected_category"),
    [
        (0.0, 0, "Good"),
        (9.0, 38, "Good"),
        (12.0, 50, "Good"),          # top of the Good band
        (12.1, 51, "Moderate"),      # bottom of Moderate
        (35.4, 100, "Moderate"),     # top of Moderate
        (35.5, 101, "Unhealthy for Sensitive Groups"),
        (55.4, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 151, "Unhealthy"),
        (150.4, 200, "Unhealthy"),
        (150.5, 201, "Very Unhealthy"),
        (250.4, 300, "Very Unhealthy"),
        (250.5, 301, "Hazardous"),
        (500.4, 500, "Hazardous"),
    ],
)
def test_aqi_breakpoints(pm25: float, expected_aqi: int, expected_category: str) -> None:
    result = aqi_from_pm25(pm25)
    assert result.aqi == expected_aqi
    assert result.category == expected_category


def test_aqi_interpolation_is_monotonic() -> None:
    """AQI must never decrease as concentration rises."""
    values = [aqi_from_pm25(p).aqi for p in range(0, 300)]
    assert values == sorted(values)


def test_aqi_above_table_clamps_to_500() -> None:
    result = aqi_from_pm25(900.0)
    assert result.aqi == 500
    assert result.category == "Hazardous"
    assert result.band == "Above AQI table"


def test_aqi_rejects_nan_and_negative() -> None:
    with pytest.raises(ValueError):
        aqi_from_pm25(float("nan"))
    with pytest.raises(ValueError):
        aqi_from_pm25(-1.0)


def test_who_ratio_uses_the_2021_guideline() -> None:
    result = aqi_from_pm25(15.0)
    assert result.who_ratio == pytest.approx(1.0)


def test_every_breakpoint_band_is_covered() -> None:
    """Sanity check the table itself: contiguous bands, ascending AQI."""
    previous_high = -1.0
    for c_low, c_high, aqi_low, aqi_high, _ in PM25_BREAKPOINTS:
        assert c_low > previous_high, "concentration bands must be contiguous"
        assert aqi_high > aqi_low
        previous_high = c_high


@pytest.mark.parametrize(
    ("pm25", "severity"),
    [
        (5.0, "good"),
        (20.0, "moderate"),
        (45.0, "elevated"),
        (80.0, "high"),
        (200.0, "very_high"),
        (400.0, "hazardous"),
    ],
)
def test_exposure_alert_severity_mapping(pm25: float, severity: str) -> None:
    alert = exposure_alert(pm25)
    assert alert["severity"] == severity
    assert alert["headline"]
    assert alert["advice"]
    assert alert["sensitive_group_advice"]


def test_alert_horizon_wording() -> None:
    assert exposure_alert(5.0, 1)["horizon"] == "next 1 hour"
    assert exposure_alert(5.0, 3)["horizon"] == "next 3 hours"


def test_alert_is_stricter_as_pollution_rises() -> None:
    """Advice must escalate, and never tell a user to exercise in dirty air."""
    good = exposure_alert(5.0)
    hazardous = exposure_alert(400.0)
    assert good["severity"] != hazardous["severity"]
    assert "Remain indoors" in hazardous["advice"]
    assert "run" in good["advice"]


def test_aqi_category_helper_matches_lookup() -> None:
    assert aqi_category(0) == "Good"
    assert aqi_category(75) == "Moderate"
    assert aqi_category(500) == "Hazardous"
