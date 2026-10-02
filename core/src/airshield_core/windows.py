"""Safe-time optimizer: choose the lowest-exposure window for an activity.

The optimizer scores *whole candidate windows*, not single hours. A 45-minute
run is integrated over its full duration, so a window whose first hour is clean
but whose second half is dirty is scored correctly rather than winning on its
minimum value.

The word "safe" is avoided throughout. Every recommendation is phrased as
*lower predicted exposure* relative to the other windows considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from airshield_core.exposure import (
    ACTIVITY_LABELS,
    ExposureEstimate,
    exposure_over_series,
    normalize_activity,
    relative_reduction_percent,
)


class WindowOptimizationError(ValueError):
    """Raised when no candidate window fits the available forecast."""


@dataclass
class CandidateWindow:
    """One candidate activity window and its predicted exposure."""

    start: datetime
    end: datetime
    estimate: ExposureEstimate
    peak_pm25: float

    def to_dict(self, baseline_score: float | None = None) -> dict:
        payload = {
            "start": self.start,
            "end": self.end,
            "exposure": self.estimate.to_dict(),
        }
        if baseline_score is not None:
            payload["relative_reduction_percent"] = relative_reduction_percent(
                baseline_score, self.estimate.score
            )
        return payload


@dataclass
class WindowPlan:
    """The optimizer's recommendation."""

    activity: str
    duration_minutes: int
    best: CandidateWindow
    alternative: CandidateWindow | None
    worst: CandidateWindow | None
    candidates: list[CandidateWindow] = field(default_factory=list)
    basis: str = "predicted"

    def to_dict(self) -> dict:
        worst_score = self.worst.estimate.score if self.worst else None
        reduction = (
            relative_reduction_percent(worst_score, self.best.estimate.score)
            if worst_score is not None
            else None
        )
        return {
            "activity": self.activity,
            "activity_label": ACTIVITY_LABELS.get(self.activity, self.activity),
            "duration_minutes": self.duration_minutes,
            "basis": self.basis,
            "best_window": self.best.to_dict(worst_score),
            "alternative_window": (
                self.alternative.to_dict(worst_score) if self.alternative else None
            ),
            "highest_exposure_window": self.worst.to_dict() if self.worst else None,
            "relative_reduction_percent": reduction,
            "reason": self._reason(reduction),
            "note": (
                "Windows are ranked by predicted exposure over the whole activity "
                "duration. Lower predicted exposure is not a guarantee of safety."
            ),
            "candidates": [c.to_dict(worst_score) for c in self.candidates],
        }

    def _reason(self, reduction: float | None) -> str:
        label = ACTIVITY_LABELS.get(self.activity, self.activity).lower()
        if reduction is None:
            return (
                f"The {self.duration_minutes}-minute {label} window with the lowest "
                "predicted exposure in the requested period."
            )
        return (
            f"Approximately {reduction:.0f}% lower predicted exposure than the "
            f"highest-exposure {self.duration_minutes}-minute window in the requested period."
        )


def _window_samples(
    ordered: pd.DataFrame,
    start: datetime,
    end: datetime,
    *,
    time_column: str,
    value_column: str,
) -> list[tuple[float, float]] | None:
    """Integrate hourly predictions over ``[start, end)``.

    Returns ``None`` when the window is not fully covered by the series, so the
    caller can skip it rather than extrapolate a value that was never predicted.
    """
    if start >= end:
        return None

    index = pd.DatetimeIndex(ordered[time_column])
    values = ordered[value_column].to_numpy(dtype=float)

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Each prediction is valid for the hour beginning at its timestamp.
    first = index[0].to_pydatetime()
    last_valid_end = (index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
    if start < first or end > last_valid_end:
        return None

    samples: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        next_hour = (pd.Timestamp(cursor).floor("h") + pd.Timedelta(hours=1)).to_pydatetime()
        slice_end = min(next_hour, end)
        position = int(index.searchsorted(pd.Timestamp(cursor), side="right")) - 1
        position = max(0, min(position, len(values) - 1))
        minutes = (slice_end - cursor).total_seconds() / 60.0
        samples.append((float(values[position]), minutes))
        cursor = slice_end
    return samples


def optimize_window(
    horizon: pd.DataFrame,
    *,
    activity: str,
    duration_minutes: int,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    step_minutes: int = 30,
    time_column: str = "time",
    value_column: str = "pm2_5",
    micro_scale: float = 1.0,
    basis: str = "predicted",
) -> WindowPlan:
    """Rank candidate activity windows by predicted exposure.

    ``horizon`` must be a time-ordered frame of *predicted* PM2.5 covering the
    requested period. Windows that the forecast does not fully cover are
    skipped; if none fit, :class:`WindowOptimizationError` explains why.
    """
    if horizon.empty:
        raise WindowOptimizationError("forecast horizon is empty")
    if duration_minutes <= 0:
        raise WindowOptimizationError("duration_minutes must be positive")
    if step_minutes <= 0:
        raise WindowOptimizationError("step_minutes must be positive")

    activity_key = normalize_activity(activity)
    ordered = horizon.sort_values(time_column).reset_index(drop=True)

    series_start = pd.Timestamp(ordered[time_column].iloc[0]).to_pydatetime()
    series_end = (
        pd.Timestamp(ordered[time_column].iloc[-1]) + pd.Timedelta(hours=1)
    ).to_pydatetime()

    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time is not None and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    window_start = start_time or series_start
    window_end = end_time or series_end
    # Never propose a window the forecast cannot cover.
    window_start = max(window_start, series_start)
    window_end = min(window_end, series_end)

    if window_end - window_start < timedelta(minutes=duration_minutes):
        raise WindowOptimizationError(
            "the requested period is shorter than the activity duration, or the "
            "forecast does not cover it. Extend the time range or shorten the activity."
        )

    candidates: list[CandidateWindow] = []
    cursor = window_start
    duration_delta = timedelta(minutes=duration_minutes)
    step = timedelta(minutes=step_minutes)
    while cursor + duration_delta <= window_end:
        samples = _window_samples(
            ordered,
            cursor,
            cursor + duration_delta,
            time_column=time_column,
            value_column=value_column,
        )
        if samples is not None:
            estimate = exposure_over_series(
                samples, activity_key, micro_scale=micro_scale, basis=basis
            )
            candidates.append(
                CandidateWindow(
                    start=cursor,
                    end=cursor + duration_delta,
                    estimate=estimate,
                    peak_pm25=max(value for value, _ in samples),
                )
            )
        cursor += step

    if not candidates:
        raise WindowOptimizationError(
            "no candidate window could be scored from the available forecast"
        )

    # De-duplicate identical windows produced by overlapping steps.
    unique: dict[tuple[datetime, datetime], CandidateWindow] = {}
    for candidate in candidates:
        unique.setdefault((candidate.start, candidate.end), candidate)
    ranked = sorted(unique.values(), key=lambda c: c.estimate.score)

    return WindowPlan(
        activity=activity_key,
        duration_minutes=duration_minutes,
        best=ranked[0],
        alternative=ranked[1] if len(ranked) > 1 else None,
        worst=ranked[-1] if len(ranked) > 1 else None,
        candidates=ranked,
        basis=basis,
    )
