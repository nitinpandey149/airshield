"""US EPA AQI maths for PM2.5 plus the plain-language exposure alert.

The breakpoints and the piecewise-linear interpolation below follow the US EPA
``Technical Assistance Document for the Reporting of Daily Air Quality``
(EPA-454/B-18-007). PM2.5 concentrations are 24-hour averages in ug/m3.
"""

from __future__ import annotations

from dataclasses import dataclass

#: (concentration_low, concentration_high, aqi_low, aqi_high, category)
#: Concentration values are truncated to one decimal place per EPA guidance.
PM25_BREAKPOINTS: tuple[tuple[float, float, int, int, str], ...] = (
    (0.0, 12.0, 0, 50, "Good"),
    (12.1, 35.4, 51, 100, "Moderate"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, 151, 200, "Unhealthy"),
    (150.5, 250.4, 201, 300, "Very Unhealthy"),
    (250.5, 500.4, 301, 500, "Hazardous"),
)

#: Concentration ceiling of the AQI table; anything above reports 500.
MAX_CONCENTRATION = 500.4

#: WHO 2021 global air quality guideline for PM2.5, 24-hour mean (ug/m3).
WHO_24H_GUIDELINE = 15.0
#: WHO 2021 interim target 4 for PM2.5, 24-hour mean (ug/m3).
WHO_24H_INTERIM_TARGET_4 = 25.0


@dataclass(frozen=True)
class AqiResult:
    """AQI value, its category, and the concentration band it fell into."""

    aqi: int
    category: str
    pm25: float
    band: str
    who_ratio: float
    """Predicted PM2.5 divided by the WHO 24-hour guideline (1.0 == at guideline)."""


def aqi_category(aqi: int) -> str:
    """Map an AQI value to its EPA category name."""
    for _, _, aqi_low, aqi_high, category in PM25_BREAKPOINTS:
        if aqi_low <= aqi <= aqi_high:
            return category
    return "Hazardous" if aqi > 500 else "Good"


def aqi_from_pm25(pm25: float) -> AqiResult:
    """Convert a PM2.5 concentration (ug/m3) to a US EPA AQI value.

    Uses the standard piecewise-linear interpolation between breakpoints.
    Concentrations are truncated (not rounded) to one decimal, matching EPA's
    reporting convention, and values above the table clamp to AQI 500.
    """
    if pm25 != pm25:  # NaN guard - never let a bad number become a fake reading
        raise ValueError("pm25 must be a real number, got NaN")
    if pm25 < 0:
        raise ValueError(f"pm25 must be non-negative, got {pm25}")

    concentration = round(pm25, 1)
    if concentration >= MAX_CONCENTRATION:
        return AqiResult(
            aqi=500,
            category="Hazardous",
            pm25=pm25,
            band="Above AQI table",
            who_ratio=pm25 / WHO_24H_GUIDELINE,
        )

    for c_low, c_high, i_low, i_high, category in PM25_BREAKPOINTS:
        if c_low <= concentration <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (concentration - c_low) + i_low
            # EPA rounds the interpolated AQI to the nearest integer.
            rounded = int(round(aqi))
            return AqiResult(
                aqi=rounded,
                category=category,
                pm25=pm25,
                band=f"{c_low:g}-{c_high:g} ug/m3",
                who_ratio=pm25 / WHO_24H_GUIDELINE,
            )

    # Unreachable for finite inputs, but keeps the contract explicit.
    raise ValueError(f"pm25={pm25} did not match any AQI breakpoint")


def exposure_alert(pm25: float, horizon_hours: int = 1) -> dict:
    """Build a plain-language exposure alert for a predicted concentration.

    Returns a dict with a severity level, a short headline, concrete advice,
    and the sensitive-group guidance that applies at that severity.
    """
    result = aqi_from_pm25(pm25)
    category = result.category
    horizon = f"next {horizon_hours} hour" if horizon_hours == 1 else f"next {horizon_hours} hours"

    if category == "Good":
        severity, headline = "good", "Air looks clean"
        advice = (
            "PM2.5 is expected to stay low. Normal outdoor activity is fine - "
            "a good window for a run or a walk."
        )
        sensitive = "No extra precautions needed."
    elif category == "Moderate":
        severity, headline = "moderate", "Air is acceptable"
        advice = (
            "Air quality is acceptable for most people. If you are unusually "
            "sensitive to particle pollution, consider a shorter outdoor session."
        )
        sensitive = "Unusually sensitive people may want to limit long or intense outdoor exertion."
    elif category == "Unhealthy for Sensitive Groups":
        severity, headline = "elevated", "Sensitive groups should take care"
        advice = (
            "Reduce prolonged or heavy outdoor exertion. Move workouts indoors or "
            "shift them to a cleaner hour."
        )
        sensitive = (
            "People with asthma or heart conditions, older adults, and children "
            "should limit outdoor exertion."
        )
    elif category == "Unhealthy":
        severity, headline = "high", "Unhealthy air expected"
        advice = (
            "Avoid prolonged outdoor exertion. Keep windows closed and run "
            "filtration indoors if you have it."
        )
        sensitive = "Sensitive groups should remain indoors and avoid outdoor exertion entirely."
    elif category == "Very Unhealthy":
        severity, headline = "very_high", "Very unhealthy air expected"
        advice = (
            "Stay indoors with windows closed. Use an air purifier or a well-fitted "
            "mask (N95/FFP2) if you must go outside."
        )
        sensitive = "Everyone should avoid outdoor exertion; sensitive groups should stay indoors."
    else:
        severity, headline = "hazardous", "Hazardous air expected"
        advice = (
            "Remain indoors and keep indoor air filtered. Avoid all outdoor "
            "activity and wear respiratory protection if evacuation is necessary."
        )
        sensitive = "Emergency conditions for everyone, including healthy adults."

    return {
        "severity": severity,
        "headline": headline,
        "advice": advice,
        "sensitive_group_advice": sensitive,
        "category": category,
        "aqi": result.aqi,
        "horizon": horizon,
    }
