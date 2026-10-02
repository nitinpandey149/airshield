"""Personal exposure engine.

This module turns a PM2.5 concentration and an activity into an *exposure
estimate*. It is deliberately simple and deterministic, and it is not a health
model:

    exposure = concentration x duration x activity_intensity x location_factor

**These multipliers are engineering approximations, not medical measurements.**
They exist so that two options (a 45-minute run at 06:00 versus 11:00, or two
routes of different length) can be compared on a like-for-like basis. They are
not derived from inhalation-rate studies and must not be presented as such.

The engine is intentionally ML-compatible: :func:`segment_exposure` accepts any
concentration series, so an ML model can supply the concentrations while the
aggregation stays deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Activity intensity multipliers, relative to a relaxed walk (1.0).
#: Engineering approximations for *relative* comparison only - see module docstring.
ACTIVITY_INTENSITY: dict[str, float] = {
    "walking": 1.0,
    "running": 2.0,
    "cycling": 1.5,
    "outdoor_work": 1.4,
    "child_outdoor_activity": 1.2,
    "custom": 1.0,
}

#: Human-readable labels for the API and UI.
ACTIVITY_LABELS: dict[str, str] = {
    "walking": "Walking",
    "running": "Running",
    "cycling": "Cycling",
    "outdoor_work": "Outdoor work",
    "child_outdoor_activity": "Child outdoor activity",
    "custom": "Custom",
}

#: Aliases accepted from clients, mapped onto canonical activity names.
_ACTIVITY_ALIASES: dict[str, str] = {
    "walk": "walking",
    "run": "running",
    "bike": "cycling",
    "bicycling": "cycling",
    "work": "outdoor_work",
    "outdoor-work": "outdoor_work",
    "child": "child_outdoor_activity",
    "children": "child_outdoor_activity",
    "kids": "child_outdoor_activity",
}

#: Exposure-score band edges (inclusive upper bound) mapped to a level name.
#: Score units are ug/m3 * hour * intensity; the edges are engineering choices
#: chosen so a clean 45-minute walk is "low" and a polluted hour is "high".
EXPOSURE_BANDS: tuple[tuple[float, str], ...] = (
    (20.0, "low"),
    (45.0, "moderate"),
    (90.0, "high"),
    (float("inf"), "very_high"),
)

EXPOSURE_LEVEL_LABELS: dict[str, str] = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "very_high": "Very high",
}


class UnknownActivityError(ValueError):
    """Raised when an activity name is not recognised."""


@dataclass(frozen=True)
class ExposureEstimate:
    """A single exposure estimate for one continuous activity."""

    score: float
    level: str
    mean_pm25: float
    duration_minutes: float
    activity: str
    activity_intensity: float
    location_factor: float
    micro_scale: float
    peak_pm25: float
    basis: str

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "level_label": EXPOSURE_LEVEL_LABELS[self.level],
            "mean_pm25": round(self.mean_pm25, 2),
            "peak_pm25": round(self.peak_pm25, 2),
            "duration_minutes": round(self.duration_minutes, 1),
            "activity": self.activity,
            "activity_label": ACTIVITY_LABELS.get(self.activity, self.activity),
            "activity_intensity": self.activity_intensity,
            "location_factor": round(self.location_factor, 3),
            "micro_scale": self.micro_scale,
            "basis": self.basis,
        }


def normalize_activity(name: str) -> str:
    """Map a client-supplied activity name onto a canonical activity key."""
    key = (name or "").strip().lower().replace(" ", "_")
    key = _ACTIVITY_ALIASES.get(key, key)
    if key not in ACTIVITY_INTENSITY:
        known = ", ".join(sorted(ACTIVITY_INTENSITY))
        raise UnknownActivityError(f"unknown activity {name!r}. Known: {known}")
    return key


def activity_intensity(name: str) -> float:
    """Return the intensity multiplier for an activity."""
    return ACTIVITY_INTENSITY[normalize_activity(name)]


def location_factor(pm25: float, micro_scale: float = 1.0) -> float:
    """Location/micro-environment factor.

    ``micro_scale`` lets a caller express a locally more (or less) polluted
    micro-environment (a street canyon, a park) relative to the ambient
    station reading. It defaults to ``1.0`` - i.e. no adjustment - so the
    engine never invents a local multiplier that was not supplied.
    """
    if micro_scale <= 0:
        raise ValueError("micro_scale must be positive")
    return float(micro_scale)


def exposure_level(score: float) -> str:
    """Map an exposure score onto a level name."""
    for upper, level in EXPOSURE_BANDS:
        if score < upper:
            return level
    return "very_high"  # pragma: no cover - the last band is infinite


def relative_reduction_percent(baseline: float, improved: float) -> float:
    """Percentage reduction of ``improved`` relative to ``baseline``.

    ``(baseline - improved) / baseline * 100``. Returns ``0.0`` when the
    baseline is non-positive, and never returns a negative number: a choice
    that is *worse* than the baseline shows no reduction.
    """
    if baseline <= 0:
        return 0.0
    reduction = (baseline - improved) / baseline * 100.0
    return round(max(0.0, reduction), 1)


def segment_exposure(
    pm25: float,
    duration_minutes: float,
    activity: str = "walking",
    *,
    micro_scale: float = 1.0,
    basis: str = "measured",
) -> ExposureEstimate:
    """Exposure for one constant-concentration segment."""
    if duration_minutes < 0:
        raise ValueError("duration_minutes must be non-negative")
    if pm25 != pm25:  # NaN
        raise ValueError("pm25 must be a real number, got NaN")
    if pm25 < 0:
        raise ValueError(f"pm25 must be non-negative, got {pm25}")

    activity_key = normalize_activity(activity)
    intensity = ACTIVITY_INTENSITY[activity_key]
    loc = location_factor(pm25, micro_scale)
    hours = duration_minutes / 60.0
    score = pm25 * hours * intensity * loc
    return ExposureEstimate(
        score=score,
        level=exposure_level(score),
        mean_pm25=pm25,
        duration_minutes=duration_minutes,
        activity=activity_key,
        activity_intensity=intensity,
        location_factor=loc,
        micro_scale=micro_scale,
        peak_pm25=pm25,
        basis=basis,
    )


def exposure_over_series(
    samples: list[tuple[float, float]],
    activity: str = "walking",
    *,
    micro_scale: float = 1.0,
    basis: str = "measured",
) -> ExposureEstimate:
    """Aggregate exposure over a series of ``(pm25, minutes)`` slices.

    Each slice contributes ``pm25 * minutes`` to the concentration-time integral.
    Slices are expected to tile the activity duration without overlap.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    activity_key = normalize_activity(activity)
    intensity = ACTIVITY_INTENSITY[activity_key]
    loc = location_factor(samples[0][0], micro_scale)

    total_minutes = 0.0
    concentration_minutes = 0.0
    peak = 0.0
    for pm25, minutes in samples:
        if minutes < 0:
            raise ValueError("slice minutes must be non-negative")
        if pm25 != pm25:
            raise ValueError("pm25 must be a real number, got NaN")
        if pm25 < 0:
            raise ValueError(f"pm25 must be non-negative, got {pm25}")
        total_minutes += minutes
        concentration_minutes += pm25 * minutes
        peak = max(peak, pm25)

    if total_minutes <= 0:
        raise ValueError("total duration must be positive")

    mean_pm25 = concentration_minutes / total_minutes
    hours = total_minutes / 60.0
    score = concentration_minutes / 60.0 * intensity * loc
    return ExposureEstimate(
        score=score,
        level=exposure_level(score),
        mean_pm25=mean_pm25,
        duration_minutes=total_minutes,
        activity=activity_key,
        activity_intensity=intensity,
        location_factor=loc,
        micro_scale=micro_scale,
        peak_pm25=peak,
        basis=basis,
    )
