"""Pollution event detection: sudden rises, sustained rises and spikes.

The detector is a transparent, rule-based engine over the *forecast* series.
It never claims causation. It reports *associated signals* - weather and
pollutant variables that are statistically linked to the predicted rise in the
historical record - and labels them as associations, not causes.

Severity and confidence are derived from the magnitude and coherence of the
predicted change, with an optional empirical calibration supplied by
:mod:`airshield_core.spike_calibration`. A confidence value is never invented:
when no calibration table is available, the value is the transparent
rule-derived score and the response says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

#: Minimum relative rise (fraction) to call something a "sudden" increase.
SUDDEN_RISE_FRACTION = 0.25
#: Minimum relative rise to call something a "spike".
SPIKE_RISE_FRACTION = 0.35
#: A rise sustained for at least this many hours is a "sustained" increase.
SUSTAINED_HOURS = 3
#: Absolute floor (ug/m3) so trivial changes in clean air are not flagged.
MIN_ABSOLUTE_RISE = 3.0

#: Variables inspected for association with a predicted rise, and the direction
#: a rise is associated with. ``+1`` means "higher values co-occur with higher
#: PM2.5", ``-1`` means "lower values co-occur with higher PM2.5".
ASSOCIATED_SIGNALS: dict[str, int] = {
    "wind_speed_10m": -1,
    "surface_pressure": +1,
    "nitrogen_dioxide": +1,
    "ozone": -1,
    "temperature_2m": +1,
    "relative_humidity_2m": +1,
    "precipitation": -1,
}

SIGNAL_LABELS: dict[str, str] = {
    "wind_speed_10m": "reduced wind",
    "surface_pressure": "rising pressure",
    "nitrogen_dioxide": "increased NO2",
    "ozone": "lower ozone",
    "temperature_2m": "warmer air",
    "relative_humidity_2m": "higher humidity",
    "precipitation": "less precipitation",
}

#: A signal must move by at least this fraction of its own spread to be reported.
_SIGNAL_Z_THRESHOLD = 0.5

ASSOCIATION_DISCLAIMER = (
    "These are weather and pollution signals statistically associated with the "
    "predicted rise in the historical record. They are not proven causes of it."
)


class EventDetectionError(ValueError):
    """Raised when the series cannot be analysed."""


@dataclass
class SpikeEvent:
    """A detected pollution increase over the forecast horizon."""

    detected: bool
    kind: str  # "none" | "sudden" | "sustained" | "spike"
    severity: str  # "none" | "minor" | "moderate" | "severe"
    expected_time: datetime | None
    expected_change_percent: float
    peak_pm25: float | None
    baseline_pm25: float
    horizon_hours: int
    confidence: float
    confidence_basis: str
    associated_signals: list[dict] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "spike_detected": self.detected,
            "kind": self.kind,
            "severity": self.severity,
            "expected_time": self.expected_time,
            "expected_change_percent": round(self.expected_change_percent, 1),
            "peak_pm25": None if self.peak_pm25 is None else round(self.peak_pm25, 2),
            "baseline_pm25": round(self.baseline_pm25, 2),
            "horizon_hours": self.horizon_hours,
            "confidence": round(self.confidence, 3),
            "confidence_basis": self.confidence_basis,
            "associated_signals": self.associated_signals,
            "signal_disclaimer": ASSOCIATION_DISCLAIMER,
            "message": self.message,
        }


def _severity(change_fraction: float, absolute_rise: float) -> str:
    if change_fraction >= 1.0 or absolute_rise >= 60:
        return "severe"
    if change_fraction >= SPIKE_RISE_FRACTION or absolute_rise >= 25:
        return "moderate"
    return "minor"


def _confidence(
    change_fraction: float,
    baseline: float,
    coherence: float,
    horizon_hours: int,
    calibration: "CalibrationTable | None" = None,
) -> tuple[float, str]:
    """Confidence in the detection, in ``[0, 1]``.

    Rule-based by default; when an empirical calibration table is supplied the
    rule score is mapped through the measured historical frequency of the same
    situation, which is a real calibrated probability rather than a guess.
    """
    # A larger relative rise, a larger absolute rise in dirty air, and more
    # agreeing signals all raise confidence. Longer horizons lower it.
    magnitude = min(1.0, change_fraction / (2 * SPIKE_RISE_FRACTION))
    absolute = min(1.0, (change_fraction * baseline) / 40.0)
    horizon_penalty = min(0.3, max(0, horizon_hours - 1) * 0.05)
    rule = 0.45 + 0.30 * magnitude + 0.15 * absolute + 0.10 * coherence - horizon_penalty
    rule = float(min(0.97, max(0.05, rule)))

    if calibration is not None and calibration.available:
        calibrated = calibration.probability(change_fraction, horizon_hours)
        return calibrated, calibration.basis
    return rule, "rule_based_estimate"


def detect_spike(
    series: pd.DataFrame,
    *,
    time_column: str = "time",
    value_column: str = "pm2_5",
    baseline_value: float | None = None,
    horizon_hours: int | None = None,
    calibration: "CalibrationTable | None" = None,
) -> SpikeEvent:
    """Detect a pollution increase in a predicted PM2.5 series.

    ``series`` must be time-ordered and contain only *predicted* values.
    ``baseline_value`` is the reference (usually the latest measured PM2.5); it
    defaults to the first value in the series.
    """
    if series.empty:
        raise EventDetectionError("series is empty")
    if time_column not in series.columns or value_column not in series.columns:
        raise EventDetectionError(
            f"series must contain {time_column!r} and {value_column!r}"
        )

    ordered = series.sort_values(time_column).reset_index(drop=True)
    values = ordered[value_column].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise EventDetectionError("predicted series contains NaN")
    if len(values) < 2:
        raise EventDetectionError("at least two predicted points are required")

    baseline = float(baseline_value) if baseline_value is not None else float(values[0])
    if baseline <= 0:
        # A zero baseline makes a percentage meaningless; fall back to the min.
        baseline = max(float(np.min(values)), 1e-6)

    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    expected_time = pd.Timestamp(ordered[time_column].iloc[peak_index]).to_pydatetime()
    change_fraction = (peak - baseline) / baseline
    absolute_rise = peak - baseline
    horizon = horizon_hours if horizon_hours is not None else len(values)

    associated = _associated_signals(ordered, peak_index)

    if change_fraction < SUDDEN_RISE_FRACTION or absolute_rise < MIN_ABSOLUTE_RISE:
        return SpikeEvent(
            detected=False,
            kind="none",
            severity="none",
            expected_time=None,
            expected_change_percent=max(0.0, change_fraction * 100),
            peak_pm25=peak,
            baseline_pm25=baseline,
            horizon_hours=horizon,
            confidence=0.0,
            confidence_basis="not_detected",
            associated_signals=[],
            message="No significant pollution increase is predicted in this horizon.",
        )

    # Sustained: the rise persists for several consecutive hours.
    rising_run = _longest_rising_run(values, baseline)
    if change_fraction >= SPIKE_RISE_FRACTION and rising_run >= 2:
        kind = "spike"
    elif rising_run >= SUSTAINED_HOURS:
        kind = "sustained"
    else:
        kind = "sudden"

    coherence = min(1.0, len(associated) / 3.0)
    confidence, basis = _confidence(
        change_fraction, baseline, coherence, horizon, calibration
    )
    severity = _severity(change_fraction, absolute_rise)

    if kind == "spike":
        message = (
            "PM2.5 is predicted to rise sharply. Consider postponing prolonged "
            "outdoor activity and check the exposure planner for a lower-exposure window."
        )
    elif kind == "sustained":
        message = (
            "PM2.5 is predicted to stay elevated for several hours. A shorter "
            "outdoor session earlier in the window may involve less exposure."
        )
    else:
        message = (
            "A short PM2.5 increase is predicted. If you can shift a long outdoor "
            "session, an earlier or later window may involve less exposure."
        )

    return SpikeEvent(
        detected=True,
        kind=kind,
        severity=severity,
        expected_time=expected_time,
        expected_change_percent=change_fraction * 100,
        peak_pm25=peak,
        baseline_pm25=baseline,
        horizon_hours=horizon,
        confidence=confidence,
        confidence_basis=basis,
        associated_signals=associated,
        message=message,
    )


def _longest_rising_run(values: np.ndarray, baseline: float) -> int:
    """Longest run of consecutive hours above the baseline."""
    longest = 0
    current = 0
    for value in values:
        if value > baseline:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _associated_signals(ordered: pd.DataFrame, peak_index: int) -> list[dict]:
    """Signals that move with PM2.5 towards the peak, in the expected direction.

    Only columns present in the frame are inspected. A signal is reported when
    its change from the start of the window to the peak is both in the expected
    direction and larger than half its own spread across the window.
    """
    signals: list[dict] = []
    for column, direction in ASSOCIATED_SIGNALS.items():
        if column not in ordered.columns:
            continue
        column_values = ordered[column].to_numpy(dtype=float)
        if np.isnan(column_values).any() or len(column_values) < 2:
            continue
        spread = float(np.nanstd(column_values))
        if spread <= 0:
            continue
        change = float(column_values[peak_index] - column_values[0])
        z = change / spread
        if np.sign(change) != np.sign(direction) or abs(z) < _SIGNAL_Z_THRESHOLD:
            continue
        signals.append(
            {
                "variable": column,
                "label": SIGNAL_LABELS.get(column, column),
                "direction": "up" if change > 0 else "down",
                "change": round(change, 3),
                "z_score": round(z, 2),
            }
        )
    signals.sort(key=lambda item: abs(item["z_score"]), reverse=True)
    return signals


# --------------------------------------------------------------------------
# Optional empirical calibration (imported lazily to avoid a hard dependency)
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from airshield_core.spike_calibration import CalibrationTable
