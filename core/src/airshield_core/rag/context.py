"""The bridge between AirShield's numerical engines and the language model.

This module is the honesty boundary of the assistant. Everything the LLM is
allowed to say about the *current situation* is assembled here, from values that
AirShield already computed. The LLM receives them as labelled facts; it does not
compute, estimate or interpolate any of them.

Four categories are kept deliberately distinct, because merging them is how an
assistant ends up misleading a user:

* **observed**    - measured environmental data (from Open-Meteo)
* **predicted**   - XGBoost model output, with its horizon and model version
* **recommendation** - deterministic exposure-engine output
* **knowledge**   - retrieved from the curated corpus, handled in ``answering``

:class:`ForecastContext` is built from the existing API payloads rather than a
parallel schema, so the numbers the assistant explains are the same numbers the
dashboard shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _num(value: Any) -> float | None:
    """Coerce to float, preserving None. Never invents a value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(container: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or a pydantic model.

    The service layer returns plain dicts, but nested blocks such as
    ``SourceInfo`` are pydantic objects. Accepting both keeps this bridge usable
    from either the API payloads or the internal service calls.
    """
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _fmt(value: float | None, unit: str = "", places: int = 1) -> str:
    if value is None:
        return "not available"
    return f"{value:.{places}f}{unit}"


@dataclass
class ForecastContext:
    """A snapshot of AirShield's own outputs, ready to be explained.

    All fields are optional: a user may ask a general knowledge question with no
    location selected, or ask about exposure with no forecast loaded. Missing
    values are reported as missing; they are never filled in.
    """

    location: str | None = None
    location_slug: str | None = None

    # ---- observed ----------------------------------------------------------
    observed_pm25: float | None = None
    observed_pm10: float | None = None
    observed_no2: float | None = None
    observed_o3: float | None = None
    observed_time: str | None = None
    observed_aqi: int | None = None
    observed_category: str | None = None

    # ---- predicted ---------------------------------------------------------
    predicted_pm25: float | None = None
    forecast_horizon_hours: int | None = None
    predicted_target_time: str | None = None
    model_version: str | None = None
    inference_backend: str | None = None
    predicted_aqi: int | None = None
    predicted_category: str | None = None
    spike_detected: bool | None = None
    spike_severity: str | None = None
    spike_expected_time: str | None = None
    spike_change_percent: float | None = None
    spike_confidence: float | None = None
    spike_confidence_basis: str | None = None

    # ---- recommendation ----------------------------------------------------
    activity: str | None = None
    activity_label: str | None = None
    duration_minutes: int | None = None
    recommended_window: str | None = None
    recommended_window_exposure: float | None = None
    recommended_window_level: str | None = None
    relative_reduction_percent: float | None = None
    alternative_window: str | None = None
    highest_exposure_window: str | None = None
    exposure_basis: str | None = None

    # ---- provenance --------------------------------------------------------
    data_mode: str | None = None
    data_source: str | None = None
    notice: str | None = None

    # ---- weather (observed) ------------------------------------------------
    wind_speed_10m: float | None = None
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    surface_pressure: float | None = None
    precipitation: float | None = None

    #: Extra values supplied by the caller that do not fit the fields above.
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------- factories
    @classmethod
    def from_payloads(
        cls,
        *,
        forecast: dict | None = None,
        plan: dict | None = None,
        routes: dict | None = None,
        location: str | None = None,
        location_slug: str | None = None,
    ) -> "ForecastContext":
        """Build a context from existing API response payloads.

        This reads the same dicts the API already returns, so there is no second
        source of truth for the numbers.
        """
        ctx = cls(location=location, location_slug=location_slug)

        if forecast:
            loc = _field(forecast, "location", {})
            ctx.location = ctx.location or _field(loc, "label") or _field(loc, "name")
            ctx.location_slug = ctx.location_slug or _field(loc, "slug")

            source = _field(forecast, "source", {})
            ctx.data_mode = _field(source, "mode")
            ctx.data_source = _field(source, "name")
            ctx.notice = _field(forecast, "notice")

            current = _field(forecast, "current", {})
            ctx.observed_pm25 = _num(_field(current, "pm2_5"))
            ctx.observed_pm10 = _num(_field(current, "pm10"))
            ctx.observed_no2 = _num(_field(current, "nitrogen_dioxide"))
            ctx.observed_o3 = _num(_field(current, "ozone"))
            ctx.observed_time = _field(current, "time")
            ctx.observed_aqi = _field(current, "aqi")
            ctx.observed_category = _field(current, "category")

            weather = _field(current, "weather", {})
            ctx.wind_speed_10m = _num(_field(weather, "wind_speed_10m"))
            ctx.temperature_2m = _num(_field(weather, "temperature_2m"))
            ctx.relative_humidity_2m = _num(_field(weather, "relative_humidity_2m"))
            ctx.surface_pressure = _num(_field(weather, "surface_pressure"))
            ctx.precipitation = _num(_field(weather, "precipitation"))

            primary = _field(forecast, "forecast", {})
            ctx.predicted_pm25 = _num(_field(primary, "predicted_pm25"))
            ctx.forecast_horizon_hours = _field(primary, "horizon_hours")
            ctx.predicted_target_time = _field(primary, "target_time")
            ctx.model_version = _field(primary, "model_version")
            ctx.inference_backend = _field(primary, "backend")

            aqi_block = _field(forecast, "aqi", {})
            ctx.predicted_aqi = _field(aqi_block, "aqi")
            ctx.predicted_category = _field(aqi_block, "category")

            spike = _field(forecast, "spike", {})
            if spike:
                ctx.spike_detected = bool(_field(spike, "spike_detected"))
                ctx.spike_severity = _field(spike, "severity")
                ctx.spike_expected_time = _field(spike, "expected_time")
                ctx.spike_change_percent = _num(_field(spike, "expected_change_percent"))
                ctx.spike_confidence = _num(_field(spike, "confidence"))
                ctx.spike_confidence_basis = _field(spike, "confidence_basis")

        if plan:
            ctx.location = ctx.location or _field(plan, "location")
            ctx.location_slug = ctx.location_slug or _field(plan, "location_slug")
            ctx.activity = _field(plan, "activity")
            ctx.activity_label = _field(plan, "activity_label")
            ctx.duration_minutes = _field(plan, "duration_minutes")
            ctx.exposure_basis = _field(plan, "basis")
            ctx.relative_reduction_percent = _num(_field(plan, "relative_reduction_percent"))

            best = _field(plan, "best_window", {}) or {}
            ctx.recommended_window = _window_label(best)
            estimate = _field(best, "exposure", {}) or {}
            ctx.recommended_window_exposure = _num(_field(estimate, "score"))
            ctx.recommended_window_level = _field(estimate, "level")

            alt = _field(plan, "alternative_window", {}) or {}
            ctx.alternative_window = _window_label(alt) if alt else None
            worst = _field(plan, "highest_exposure_window", {}) or {}
            ctx.highest_exposure_window = _window_label(worst) if worst else None

            if ctx.notice is None:
                ctx.notice = _field(plan, "notice")
                source = _field(plan, "source", {})
                ctx.data_mode = ctx.data_mode or _field(source, "mode")
                ctx.data_source = ctx.data_source or _field(source, "name")

        if routes:
            comparison = _field(routes, "recommendation")
            if comparison:
                ctx.extras["route_recommendation"] = comparison
            best_route = _field(routes, "recommended_route_id")
            if best_route:
                ctx.extras["recommended_route_id"] = best_route
            options = _field(routes, "routes", []) or []
            if options:
                ctx.extras["route_count"] = len(options)
                for option in options:
                    if _field(option, "route_id") == best_route:
                        ctx.extras["recommended_route_distance_km"] = _num(
                            _field(option, "distance_km")
                        )
                        ctx.extras["recommended_route_exposure"] = _num(
                            _field(option, "exposure_score")
                        )
                        ctx.extras["recommended_route_reduction_percent"] = _num(
                            _field(option, "relative_reduction_percent")
                        )
                        break

        return ctx

    # -------------------------------------------------------------- rendering
    def has_any_facts(self) -> bool:
        return any(
            value is not None
            for value in (
                self.observed_pm25,
                self.predicted_pm25,
                self.recommended_window,
                self.observed_aqi,
            )
        )

    def fact_block(self) -> str:
        """Render the context as labelled facts for the language model.

        The headings are part of the contract: the model is instructed to keep
        OBSERVED, PREDICTED and RECOMMENDATION separate, and to never state a
        number that is absent here.
        """
        lines: list[str] = []
        where = self.location or "the selected location"

        if self.observed_pm25 is not None or self.observed_aqi is not None:
            lines.append("OBSERVED (measured by the upstream network, not modelled):")
            lines.append(f"- Location: {where}")
            if self.observed_time:
                lines.append(f"- Measured at: {self.observed_time}")
            if self.observed_pm25 is not None:
                lines.append(f"- PM2.5: {_fmt(self.observed_pm25, ' ug/m3')}")
            if self.observed_pm10 is not None:
                lines.append(f"- PM10: {_fmt(self.observed_pm10, ' ug/m3')}")
            if self.observed_no2 is not None:
                lines.append(f"- NO2: {_fmt(self.observed_no2, ' ug/m3')}")
            if self.observed_o3 is not None:
                lines.append(f"- O3: {_fmt(self.observed_o3, ' ug/m3')}")
            if self.observed_aqi is not None:
                lines.append(
                    f"- AQI: {self.observed_aqi} ({self.observed_category or 'unclassified'})"
                )
            weather_bits = [
                f"wind {_fmt(self.wind_speed_10m, ' km/h')}" if self.wind_speed_10m is not None else None,
                f"temperature {_fmt(self.temperature_2m, ' degC')}" if self.temperature_2m is not None else None,
                f"humidity {_fmt(self.relative_humidity_2m, ' %')}" if self.relative_humidity_2m is not None else None,
                f"pressure {_fmt(self.surface_pressure, ' hPa')}" if self.surface_pressure is not None else None,
                f"precipitation {_fmt(self.precipitation, ' mm')}" if self.precipitation is not None else None,
            ]
            weather = [bit for bit in weather_bits if bit]
            if weather:
                lines.append(f"- Weather: {', '.join(weather)}")

        if self.predicted_pm25 is not None:
            if lines:
                lines.append("")
            lines.append("PREDICTED (AirShield XGBoost model output, not a measurement):")
            horizon = self.forecast_horizon_hours
            horizon_text = f"{horizon}-hour" if horizon is not None else "near-term"
            lines.append(f"- Predicted PM2.5 ({horizon_text} horizon): {_fmt(self.predicted_pm25, ' ug/m3')}")
            if self.predicted_target_time:
                lines.append(f"- Target time: {self.predicted_target_time}")
            if self.predicted_aqi is not None:
                lines.append(
                    f"- Predicted AQI: {self.predicted_aqi} ({self.predicted_category or 'unclassified'})"
                )
            if self.model_version:
                lines.append(f"- Model version: {self.model_version}")
            if self.inference_backend:
                lines.append(f"- Served by: {self.inference_backend}")

        if self.spike_detected is not None:
            if lines:
                lines.append("")
            lines.append("SPIKE ASSESSMENT (from the forecast timeline):")
            if self.spike_detected:
                lines.append(
                    f"- A {self.spike_severity or 'possible'} pollution increase is expected"
                )
                if self.spike_expected_time:
                    lines.append(f"- Expected around: {self.spike_expected_time}")
                if self.spike_change_percent is not None:
                    lines.append(f"- Expected change: +{_fmt(self.spike_change_percent, '%')}")
                if self.spike_confidence is not None:
                    basis = self.spike_confidence_basis or "unspecified"
                    lines.append(f"- Confidence: {_fmt(self.spike_confidence, '', 2)} (basis: {basis})")
            else:
                lines.append("- No significant spike is predicted in the current horizon")

        if self.recommended_window:
            if lines:
                lines.append("")
            lines.append("RECOMMENDATION (deterministic exposure engine output):")
            activity = self.activity_label or self.activity or "activity"
            if self.duration_minutes:
                lines.append(f"- Activity: {activity}, {self.duration_minutes} minutes")
            else:
                lines.append(f"- Activity: {activity}")
            lines.append(f"- Lowest predicted-exposure window: {self.recommended_window}")
            if self.recommended_window_exposure is not None:
                lines.append(
                    f"- Exposure score in that window: "
                    f"{_fmt(self.recommended_window_exposure, '', 1)}"
                    f" ({self.recommended_window_level or 'unclassified'})"
                )
            if self.relative_reduction_percent is not None:
                lines.append(
                    f"- Relative reduction vs the highest-exposure window: "
                    f"{_fmt(self.relative_reduction_percent, '%')}"
                )
            if self.alternative_window:
                lines.append(f"- Alternative window: {self.alternative_window}")
            if self.highest_exposure_window:
                lines.append(f"- Highest-exposure window to avoid: {self.highest_exposure_window}")
            if self.exposure_basis:
                lines.append(f"- Basis: {self.exposure_basis}")

        if self.extras.get("route_recommendation"):
            if lines:
                lines.append("")
            lines.append("ROUTE COMPARISON (deterministic route exposure output):")
            lines.append(f"- {self.extras['route_recommendation']}")
            if self.extras.get("recommended_route_distance_km") is not None:
                lines.append(
                    f"- Recommended route distance: "
                    f"{_fmt(self.extras['recommended_route_distance_km'], ' km', 2)}"
                )
            if self.extras.get("recommended_route_exposure") is not None:
                lines.append(
                    f"- Recommended route exposure score: "
                    f"{_fmt(self.extras['recommended_route_exposure'], '', 1)}"
                )
            if self.extras.get("recommended_route_reduction_percent") is not None:
                lines.append(
                    f"- Relative reduction: "
                    f"{_fmt(self.extras['recommended_route_reduction_percent'], '%')}"
                )

        if self.data_mode:
            if lines:
                lines.append("")
            lines.append(f"PROVENANCE: data mode is '{self.data_mode}'"
                         + (f", source {self.data_source}" if self.data_source else ""))
            if self.notice:
                lines.append(f"NOTICE: {self.notice}")

        return "\n".join(lines) if lines else ""

    def as_dict(self) -> dict:
        """Structured form, used by tests to assert the LLM got real API values."""
        return {
            "location": self.location,
            "observed_pm25": self.observed_pm25,
            "observed_aqi": self.observed_aqi,
            "predicted_pm25": self.predicted_pm25,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "activity": self.activity,
            "duration_minutes": self.duration_minutes,
            "recommended_window": self.recommended_window,
            "recommended_window_exposure": self.recommended_window_exposure,
            "relative_reduction_percent": self.relative_reduction_percent,
            "data_mode": self.data_mode,
            "extras": dict(self.extras),
        }


def _window_label(window: Any) -> str | None:
    """Format a window as 'HH:MM-HH:MM' from its ISO timestamps."""
    start = _field(window, "start")
    end = _field(window, "end")
    if not start or not end:
        return None
    return f"{_clock(start)}-{_clock(end)}"


def _clock(value: Any) -> str:
    """Extract HH:MM from a timestamp.

    Accepts a datetime or an ISO-8601 string, with or without a ``T`` separator
    and with or without a UTC offset.
    """
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    text = str(value)
    for separator in ("T", " "):
        if separator in text:
            text = text.split(separator, 1)[1]
            break
    return text[:5]
