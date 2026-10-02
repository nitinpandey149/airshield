"""Tests for the EPA AQI conversion and the exposure alert."""

from __future__ import annotations

import pytest

from airshield_core.aqi import (
    NATIONAL_AQI_BREAKPOINTS,
    NATIONAL_AQI_MISSING_POLLUTANTS,
    PM25_BREAKPOINTS,
    aqi_category,
    aqi_from_pm25,
    exposure_alert,
    national_aqi_from_pollutants,
    national_sub_index,
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
        (5.0, "good"),        # National AQI: Good
        (45.0, "moderate"),   # Satisfactory
        (75.0, "elevated"),   # Moderate
        (105.0, "high"),      # Poor
        (200.0, "very_high"), # Very Poor
        (300.0, "hazardous"), # Severe
    ],
)
def test_exposure_alert_severity_mapping(pm25: float, severity: str) -> None:
    alert = exposure_alert(pm25)
    assert alert["severity"] == severity
    assert alert["headline"]
    assert alert["advice"]
    assert alert["sensitive_group_advice"]
    # The alert is driven by the Indian National AQI PM2.5 sub-index, and says so.
    assert alert["aqi"] == national_sub_index("pm2_5", pm25).sub_index
    assert alert["category"] == national_sub_index("pm2_5", pm25).category
    assert alert["basis"] == "India CPCB National AQI PM2.5 sub-index"


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


# ------------------------------------------------------- India CPCB National


@pytest.mark.parametrize(
    ("pm25", "expected_aqi", "expected_category"),
    [
        (0.0, 0, "Good"),
        (30.0, 50, "Good"),            # top of Good
        (45.0, 76, "Satisfactory"),    # mid Satisfactory (75.5 rounds up)
        (60.0, 100, "Satisfactory"),   # top of Satisfactory
        (75.0, 150, "Moderate"),
        (90.0, 200, "Moderate"),       # top of Moderate
        (105.0, 250, "Poor"),
        (120.0, 300, "Poor"),          # top of Poor
        (185.0, 350, "Very Poor"),
        (250.0, 400, "Very Poor"),     # top of Very Poor
        (375.0, 450, "Severe"),
        (500.0, 500, "Severe"),
    ],
)
def test_national_pm25_sub_index_breakpoints(
    pm25: float, expected_aqi: int, expected_category: str
) -> None:
    """India CPCB PM2.5 sub-index values and categories, at real boundaries."""
    result = national_sub_index("pm2_5", pm25)
    assert result.sub_index == expected_aqi
    assert result.category == expected_category


def test_national_aqi_is_the_worst_sub_index() -> None:
    """The National AQI is the maximum across pollutants, not the PM2.5 value."""
    result = national_aqi_from_pollutants(
        pm2_5=25.0,   # sub-index 42 (Good)
        pm10=260.0,   # sub-index ~213 (Poor) - dominant
        nitrogen_dioxide=50.0,
        ozone=30.0,
    )
    assert result.dominant_pollutant == "pm10"
    assert result.aqi == max(s.sub_index for s in result.sub_indices)
    assert result.category == "Poor"


def test_national_aqi_names_the_missing_pollutants() -> None:
    """CO, SO2 and NH3 are not in our source; the result must say so."""
    result = national_aqi_from_pollutants(pm2_5=80.0)
    assert result.is_partial is True
    assert set(result.missing_pollutants) == set(NATIONAL_AQI_MISSING_POLLUTANTS)
    assert result.health_guidance


def test_national_aqi_requires_at_least_one_pollutant() -> None:
    with pytest.raises(ValueError, match="at least one pollutant"):
        national_aqi_from_pollutants()


def test_national_aqi_ignores_nan_but_keeps_the_rest() -> None:
    result = national_aqi_from_pollutants(pm2_5=float("nan"), pm10=120.0)
    assert result.dominant_pollutant == "pm10"
    assert {s.pollutant for s in result.sub_indices} == {"pm10"}


def test_national_sub_index_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        national_sub_index("pm2_5", float("nan"))
    with pytest.raises(ValueError):
        national_sub_index("pm2_5", -1.0)
    with pytest.raises(ValueError, match="no National AQI breakpoints"):
        national_sub_index("methane", 10.0)


def test_national_bands_are_contiguous_and_ascending() -> None:
    """Sanity-check the CPCB table itself."""
    for pollutant, bands in NATIONAL_AQI_BREAKPOINTS.items():
        previous_high = None
        previous_aqi = -1
        for c_low, c_high, aqi_low, aqi_high, _ in bands:
            assert c_high > c_low, f"{pollutant}: empty band"
            assert aqi_high > aqi_low, f"{pollutant}: AQI band not ascending"
            assert aqi_low > previous_aqi, f"{pollutant}: AQI bands not contiguous"
            if previous_high is not None:
                assert c_low == previous_high, f"{pollutant}: concentration gap"
            previous_high = c_high
            previous_aqi = aqi_high


def test_india_and_epa_scales_disagree_by_design() -> None:
    """The two standards must not be conflated: the same PM2.5 differs."""
    # 40 ug/m3 is 'Satisfactory' in India but 'Unhealthy for Sensitive Groups' in the US.
    assert national_sub_index("pm2_5", 40.0).category == "Satisfactory"
    assert aqi_from_pm25(40.0).category == "Unhealthy for Sensitive Groups"


def test_national_who_ratio_uses_pm25() -> None:
    result = national_aqi_from_pollutants(pm2_5=30.0)
    assert result.who_ratio == pytest.approx(2.0)
