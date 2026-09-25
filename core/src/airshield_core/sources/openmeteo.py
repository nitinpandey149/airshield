"""Open-Meteo data client.

Open-Meteo is a public, open-data weather and air-quality API that needs no API
key. Two endpoints are used:

* air quality  - ``pm2_5``, ``pm10``, ``NO2``, ``O3``
* weather      - temperature, humidity, wind, pressure, precipitation

``/v1/air-quality`` serves recent observations plus a short forecast and also
accepts ``start_date``/``end_date`` for historical ranges, which is how the
training set is built. Weather history comes from the ERA5 archive endpoint.

Attribution: Weather and air-quality data by Open-Meteo.com (CC BY 4.0).
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from airshield_core.schema import Observation, Observations, SourceInfo

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ATTRIBUTION = "Weather and air-quality data by Open-Meteo.com (CC BY 4.0)"

AIR_QUALITY_HOURLY = "pm2_5,pm10,nitrogen_dioxide,ozone"
WEATHER_HOURLY = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,surface_pressure,precipitation"
)


class UpstreamError(RuntimeError):
    """Raised when Open-Meteo cannot be reached or returns unusable data."""


def _get(url: str, params: dict, timeout: float, retries: int) -> dict:
    """GET JSON with a small retry budget and honest error reporting."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
            if "hourly" not in payload:
                raise UpstreamError(
                    f"unexpected response from {url}: missing 'hourly' key "
                    f"({str(payload)[:200]})"
                )
            return payload
        except (httpx.HTTPError, ValueError, UpstreamError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    raise UpstreamError(f"request to {url} failed after {retries + 1} attempts: {last_error}")


def _to_frame(payload: dict, columns: list[str]) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for column in columns:
        values = hourly.get(column)
        frame[column] = pd.to_numeric(values, errors="coerce") if values is not None else float("nan")
    return frame


def _merge(air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = air.merge(weather, on="time", how="inner")
    return merged.sort_values("time").reset_index(drop=True)


def fetch_history(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    *,
    timeout: float = 30.0,
    retries: int = 2,
) -> pd.DataFrame:
    """Fetch historical hourly air-quality and weather for a date range."""
    air = _to_frame(
        _get(
            AIR_QUALITY_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": AIR_QUALITY_HOURLY,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": "UTC",
            },
            timeout,
            retries,
        ),
        AIR_QUALITY_HOURLY.split(","),
    )
    weather = _to_frame(
        _get(
            WEATHER_ARCHIVE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": WEATHER_HOURLY,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": "UTC",
            },
            timeout,
            retries,
        ),
        WEATHER_HOURLY.split(","),
    )
    return _merge(air, weather)


class OpenMeteoSource:
    """Live source: recent measured history plus forecast weather for the next hour."""

    name = "Open-Meteo"
    url = AIR_QUALITY_URL

    def __init__(self, *, timeout: float = 15.0, retries: int = 2):
        self.timeout = timeout
        self.retries = retries

    def fetch(self, latitude: float, longitude: float, location_name: str) -> Observations:
        """Return recent observations including one hour of forecast weather.

        The final row carries forecast weather with ``pm2_5`` set to ``NaN``;
        it exists solely to supply the ``wx_next_*`` features. A naive NaN would
        break the pydantic schema, so that row is stripped here and re-appended
        by :meth:`fetch_frame`.
        """
        frame = self.fetch_frame(latitude, longitude)
        rows = [
            Observation(**record)
            for record in frame.dropna(subset=["pm2_5"]).to_dict("records")
        ]
        return Observations(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            source=SourceInfo(
                name=self.name,
                url=self.url,
                mode="live",
                licence=ATTRIBUTION,
                fetched_at=datetime.now(timezone.utc),
                is_synthetic=False,
            ),
            rows=rows,
        )

    def fetch_frame(self, latitude: float, longitude: float) -> pd.DataFrame:
        """Recent hourly frame, including the forecast hour as the last row."""
        air_payload = _get(
            AIR_QUALITY_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": AIR_QUALITY_HOURLY,
                "past_days": 7,
                "forecast_days": 2,
                "timezone": "UTC",
            },
            self.timeout,
            self.retries,
        )
        weather_payload = _get(
            WEATHER_FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": WEATHER_HOURLY,
                "past_days": 7,
                "forecast_days": 2,
                "timezone": "UTC",
            },
            self.timeout,
            self.retries,
        )
        air = _to_frame(air_payload, AIR_QUALITY_HOURLY.split(","))
        weather = _to_frame(weather_payload, WEATHER_HOURLY.split(","))
        frame = _merge(air, weather)

        # The air-quality endpoint returns forecast PM2.5 as well as measured
        # values, so "not NaN" is not the same as "observed". Anchor on the
        # current hour instead: keep everything up to now, plus the single
        # forecast hour that follows it (which supplies the wx_next_* features).
        now_hour = pd.Timestamp.now(tz="UTC").floor("h")
        observed = frame[frame["time"] <= now_hour]
        if observed.empty:
            raise UpstreamError(
                "Open-Meteo returned no data at or before the current hour "
                f"({now_hour.isoformat()})"
            )
        if observed["pm2_5"].notna().sum() < 2:
            raise UpstreamError(
                "Open-Meteo returned fewer than two measured pm2_5 values up to now"
            )

        cutoff = observed.index[-1]
        return frame.iloc[: cutoff + 2].reset_index(drop=True)


def fetch_frame_with_fallback(
    latitude: float,
    longitude: float,
    *,
    timeout: float = 15.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, SourceInfo]:
    """Fetch live data, reporting provenance; raises :class:`UpstreamError` on failure.

    Callers decide how to fall back so that the demo path is always explicit.
    """
    source = OpenMeteoSource(timeout=timeout, retries=retries)
    frame = source.fetch_frame(latitude, longitude)
    info = SourceInfo(
        name=source.name,
        url=source.url,
        mode="live",
        licence=ATTRIBUTION,
        fetched_at=datetime.now(timezone.utc),
        is_synthetic=False,
    )
    return frame, info


def default_history_window(days: int) -> tuple[date, date]:
    """Return a ``(start, end)`` window ending yesterday.

    Yesterday is the cutoff because archive endpoints lag real time; asking for
    today would return partially-filled or empty hours.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)
    return start, end
