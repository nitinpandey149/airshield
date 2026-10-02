"""Air quality index maths, in two standards, plus the exposure alert.

Two indexes are implemented because they answer different questions and must
never be confused with one another:

* **India CPCB National AQI** - the index Indian users see. Six categories
  (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) with breakpoints from
  the Central Pollution Control Board. It is a *multi-pollutant* index: the
  reported value is the worst sub-index across pollutants. AirShield computes
  sub-indices for the pollutants its upstream source provides (PM2.5, PM10, NO2,
  O3); CO, SO2 and NH3 are not available, so the value can be *lower* than the
  official National AQI when one of those dominates. This is stated in the
  response rather than hidden.
* **US EPA AQI** - kept because the EPA breakpoints are a well-known reference
  and the ``who_ratio`` is derived independently of either index.

Both share the same piecewise-linear interpolation, and both live here so the
``aqi`` and ``alert`` response blocks can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------- US EPA

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


# ------------------------------------------------------- India CPCB National

#: India CPCB National AQI bands, in order. ``c_low`` equals the previous band's
#: ``c_high`` so the bands are contiguous; a concentration is assigned to the
#: first band whose ``c_high`` it does not exceed.
#: (concentration_low, concentration_high, aqi_low, aqi_high, category)
NATIONAL_AQI_BREAKPOINTS: dict[str, tuple[tuple[float, float, int, int, str], ...]] = {
    # 24-hour average, ug/m3
    "pm2_5": (
        (0.0, 30.0, 0, 50, "Good"),
        (30.0, 60.0, 51, 100, "Satisfactory"),
        (60.0, 90.0, 101, 200, "Moderate"),
        (90.0, 120.0, 201, 300, "Poor"),
        (120.0, 250.0, 301, 400, "Very Poor"),
        (250.0, 500.0, 401, 500, "Severe"),
    ),
    # 24-hour average, ug/m3
    "pm10": (
        (0.0, 50.0, 0, 50, "Good"),
        (50.0, 100.0, 51, 100, "Satisfactory"),
        (100.0, 250.0, 101, 200, "Moderate"),
        (250.0, 350.0, 201, 300, "Poor"),
        (350.0, 430.0, 301, 400, "Very Poor"),
        (430.0, 600.0, 401, 500, "Severe"),
    ),
    # 24-hour average, ug/m3
    "nitrogen_dioxide": (
        (0.0, 40.0, 0, 50, "Good"),
        (40.0, 80.0, 51, 100, "Satisfactory"),
        (80.0, 180.0, 101, 200, "Moderate"),
        (180.0, 280.0, 201, 300, "Poor"),
        (280.0, 400.0, 301, 400, "Very Poor"),
        (400.0, 800.0, 401, 500, "Severe"),
    ),
    # 8-hour average, ug/m3
    "ozone": (
        (0.0, 50.0, 0, 50, "Good"),
        (50.0, 100.0, 51, 100, "Satisfactory"),
        (100.0, 168.0, 101, 200, "Moderate"),
        (168.0, 208.0, 201, 300, "Poor"),
        (208.0, 748.0, 301, 400, "Very Poor"),
        (748.0, 1000.0, 401, 500, "Severe"),
    ),
}

#: Pollutants the National AQI formally covers but our upstream source does not
#: provide. Named explicitly so the response can say what is missing.
NATIONAL_AQI_MISSING_POLLUTANTS: tuple[str, ...] = ("CO", "SO2", "NH3")

#: Plain-language health guidance per National AQI category, from CPCB.
NATIONAL_AQI_HEALTH_GUIDANCE: dict[str, str] = {
    "Good": "Minimal impact. Air quality is considered satisfactory for outdoor activity.",
    "Satisfactory": (
        "Minor breathing discomfort to sensitive people. Sensitive groups may "
        "notice mild symptoms during prolonged outdoor exertion."
    ),
    "Moderate": (
        "Breathing discomfort to people with asthma, and to people with heart "
        "or lung disease. Sensitive groups should reduce prolonged outdoor exertion."
    ),
    "Poor": (
        "Breathing discomfort to most people on prolonged exposure, and to "
        "people with heart disease on short exposure. Reduce outdoor exertion."
    ),
    "Very Poor": (
        "Respiratory illness on prolonged exposure. Everyone should avoid "
        "outdoor exertion; sensitive groups should remain indoors."
    ),
    "Severe": (
        "Affects healthy people and seriously impacts those with existing "
        "disease. Remain indoors and keep indoor air filtered."
    ),
}


@dataclass(frozen=True)
class SubIndex:
    """One pollutant's contribution to the National AQI."""

    pollutant: str
    concentration: float
    sub_index: int
    category: str
    band: str
    """Concentration band that produced the sub-index, e.g. ``30-60 ug/m3``."""


@dataclass(frozen=True)
class NationalAqiResult:
    """The India CPCB National AQI: the worst sub-index across pollutants."""

    aqi: int
    category: str
    dominant_pollutant: str
    """The pollutant whose sub-index set the reported value."""
    band: str
    """Concentration band of the dominant pollutant."""
    sub_indices: list[SubIndex] = field(default_factory=list)
    missing_pollutants: tuple[str, ...] = NATIONAL_AQI_MISSING_POLLUTANTS
    who_ratio: float = 0.0
    """PM2.5 divided by the WHO 24-hour guideline, when PM2.5 is known."""

    @property
    def health_guidance(self) -> str:
        return NATIONAL_AQI_HEALTH_GUIDANCE[self.category]

    @property
    def is_partial(self) -> bool:
        """True when the National AQI could be understated because of missing inputs."""
        return bool(self.missing_pollutants)


@dataclass(frozen=True)
class AqiResult:
    """AQI value, its category, and the concentration band it fell into."""

    aqi: int
    category: str
    pm25: float
    band: str
    who_ratio: float
    """Predicted PM2.5 divided by the WHO 24-hour guideline (1.0 == at guideline)."""


def _interpolate(bands, concentration: float) -> tuple[int, str, float, float]:
    """Piecewise-linear sub-index lookup over contiguous bands.

    Returns ``(aqi, category, band_low, band_high)``. A concentration is placed
    in the first band whose upper bound it does not exceed, which keeps the
    shared boundary of two adjacent bands in the lower band.
    """
    for c_low, c_high, i_low, i_high, category in bands:
        if concentration <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (concentration - c_low) + i_low
            return int(round(aqi)), category, c_low, c_high
    # Above the table: clamp to the top of the scale.
    _, c_high, i_low, i_high, category = bands[-1]
    return i_high, category, bands[-1][0], c_high


def national_sub_index(pollutant: str, concentration: float) -> SubIndex:
    """Compute one pollutant's National AQI sub-index."""
    if pollutant not in NATIONAL_AQI_BREAKPOINTS:
        raise ValueError(f"no National AQI breakpoints for {pollutant!r}")
    if concentration != concentration:
        raise ValueError("concentration must be a real number, got NaN")
    if concentration < 0:
        raise ValueError(f"concentration must be non-negative, got {concentration}")
    aqi, category, c_low, c_high = _interpolate(
        NATIONAL_AQI_BREAKPOINTS[pollutant], concentration
    )
    return SubIndex(
        pollutant=pollutant,
        concentration=concentration,
        sub_index=min(aqi, 500),
        category=category,
        band=f"{c_low:g}-{c_high:g} ug/m3",
    )


def national_aqi_from_pollutants(
    *,
    pm2_5: float | None = None,
    pm10: float | None = None,
    nitrogen_dioxide: float | None = None,
    ozone: float | None = None,
) -> NationalAqiResult:
    """Compute the India CPCB National AQI from whatever pollutants are known.

    The National AQI is the maximum sub-index across pollutants. Only the
    pollutants supplied are considered, so the result records which are missing
    (see :attr:`NationalAqiResult.missing_pollutants`) rather than implying full
    coverage. At least one pollutant is required.
    """
    supplied = {
        "pm2_5": pm2_5,
        "pm10": pm10,
        "nitrogen_dioxide": nitrogen_dioxide,
        "ozone": ozone,
    }
    sub_indices = [
        national_sub_index(name, float(value))
        for name, value in supplied.items()
        if value is not None and value == value
    ]
    if not sub_indices:
        raise ValueError("at least one pollutant concentration is required")

    worst = max(sub_indices, key=lambda s: s.sub_index)
    who_ratio = pm2_5 / WHO_24H_GUIDELINE if pm2_5 is not None and pm2_5 == pm2_5 else 0.0
    return NationalAqiResult(
        aqi=worst.sub_index,
        category=worst.category,
        dominant_pollutant=worst.pollutant,
        band=worst.band,
        sub_indices=sub_indices,
        who_ratio=who_ratio,
    )


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
    """Build a plain-language exposure alert for a *predicted* PM2.5 value.

    The alert is driven by the PM2.5 sub-index of the India CPCB National AQI,
    because PM2.5 is the only pollutant AirShield forecasts. It is therefore a
    PM2.5 exposure alert and not the full multi-pollutant National AQI; the
    ``basis`` field says so, and ``severity`` is a presentation token that maps
    the six Indian categories onto the dashboard's colour scale.
    """
    sub_index = national_sub_index("pm2_5", pm25)
    category = sub_index.category
    horizon = f"next {horizon_hours} hour" if horizon_hours == 1 else f"next {horizon_hours} hours"

    if category == "Good":
        severity, headline = "good", "Air looks clean"
        advice = (
            "PM2.5 is expected to stay low. Normal outdoor activity is fine - "
            "a good window for a run or a walk."
        )
        sensitive = "Minimal impact. No extra precautions needed."
    elif category == "Satisfactory":
        severity, headline = "moderate", "Air is acceptable"
        advice = (
            "Air quality is acceptable for most people. If you are unusually "
            "sensitive to particle pollution, consider a shorter outdoor session."
        )
        sensitive = (
            "Minor breathing discomfort to sensitive people. Those with asthma "
            "may want to limit long or intense outdoor exertion."
        )
    elif category == "Moderate":
        severity, headline = "elevated", "Sensitive groups should take care"
        advice = (
            "Reduce prolonged or heavy outdoor exertion. Move workouts indoors or "
            "shift them to a cleaner hour."
        )
        sensitive = (
            "Breathing discomfort to people with asthma, and to people with heart "
            "or lung disease. Sensitive groups should limit outdoor exertion."
        )
    elif category == "Poor":
        severity, headline = "high", "Poor air expected"
        advice = (
            "Avoid prolonged outdoor exertion. Keep windows closed and run "
            "filtration indoors if you have it."
        )
        sensitive = (
            "Breathing discomfort on prolonged exposure for most people. People "
            "with heart disease should limit even short outdoor exposure."
        )
    elif category == "Very Poor":
        severity, headline = "very_high", "Very poor air expected"
        advice = (
            "Stay indoors with windows closed. Use an air purifier or a well-fitted "
            "N95 mask if you must go outside."
        )
        sensitive = (
            "Respiratory illness on prolonged exposure. Everyone should avoid "
            "outdoor exertion; sensitive groups should stay indoors."
        )
    else:  # Severe
        severity, headline = "hazardous", "Severe air expected"
        advice = (
            "Remain indoors and keep indoor air filtered. Avoid all outdoor "
            "activity and wear a well-fitted N95 mask if you must go out."
        )
        sensitive = (
            "Affects healthy people and seriously impacts those with existing "
            "disease. Remain indoors."
        )

    return {
        "severity": severity,
        "headline": headline,
        "advice": advice,
        "sensitive_group_advice": sensitive,
        "category": category,
        "aqi": sub_index.sub_index,
        "horizon": horizon,
        "basis": "India CPCB National AQI PM2.5 sub-index",
        "health_guidance": NATIONAL_AQI_HEALTH_GUIDANCE[category],
    }
