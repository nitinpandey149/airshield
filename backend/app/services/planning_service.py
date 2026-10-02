"""Exposure planning service: rank activity windows by predicted exposure.

The service builds the predicted PM2.5 timeline for a location (via the forecast
service, so the same models and the same provenance are used), then hands it to
the deterministic window optimizer. Nothing is invented here: if the timeline is
too short to cover the requested duration, the request fails with a clear reason.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from airshield_core.exposure import ACTIVITY_LABELS, ACTIVITY_INTENSITY
from airshield_core.sources.registry import Location
from airshield_core.windows import WindowOptimizationError, optimize_window
from app.services.forecast_service import ForecastError, ForecastService

logger = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    """Raised when an exposure plan cannot be produced from real forecasts."""


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanningError(f"invalid ISO-8601 time {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class PlanningService:
    """Plans the lowest-exposure activity window for a location."""

    def __init__(self, forecast_service: ForecastService):
        self.forecasts = forecast_service

    @property
    def activity_options(self) -> list[dict]:
        return [
            {
                "activity": key,
                "label": ACTIVITY_LABELS[key],
                "intensity": ACTIVITY_INTENSITY[key],
            }
            for key in ACTIVITY_INTENSITY
        ]

    def plan(
        self,
        location: Location,
        *,
        activity: str,
        duration_minutes: int,
        start_time: str | None = None,
        end_time: str | None = None,
        step_minutes: int = 30,
    ) -> dict:
        """Return the recommended window plus alternatives and provenance."""
        if duration_minutes <= 0:
            raise PlanningError("duration_minutes must be positive")

        frame, source, notice, base_time = self.forecasts.acquire(location)
        timeline = self.forecasts.forecast_timeline(frame, base_time)
        if timeline.empty:
            raise PlanningError(
                "no forecast timeline is available, so windows cannot be scored. "
                "Train a model with `make train`."
            )

        try:
            plan = optimize_window(
                timeline,
                activity=activity,
                duration_minutes=duration_minutes,
                start_time=_parse_time(start_time),
                end_time=_parse_time(end_time),
                step_minutes=step_minutes,
                basis="predicted",
            )
        except WindowOptimizationError as exc:
            raise PlanningError(str(exc)) from exc

        payload = plan.to_dict()
        payload.update(
            {
                "location": location.label,
                "location_slug": location.slug,
                "generated_at": datetime.now(timezone.utc),
                "source": source.model_dump(),
                "notice": notice,
                "base_time": base_time.to_pydatetime(),
                "timeline": [
                    {
                        "time": row.time,
                        "pm2_5": float(row.pm2_5),
                        "horizon_hours": int(row.horizon_hours),
                    }
                    for row in timeline.itertuples()
                ],
                "exposure_note": (
                    "Exposure scores are comparative engineering estimates "
                    "(concentration x duration x activity intensity), not medical "
                    "measurements. Lower predicted exposure is not a guarantee of safety."
                ),
            }
        )
        return payload
