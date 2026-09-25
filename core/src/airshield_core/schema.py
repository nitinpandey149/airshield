"""Canonical data schema shared by every AirShield component.

A single ``Observation`` row represents one hour at one location. Keeping the
column set in one place is what allows the bundled demo dataset, the Open-Meteo
client and the SageMaker training job to agree on features.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

#: Numeric hourly columns, in canonical order. ``pm2_5`` is the prediction
#: target; the rest are model inputs plus context shown in the UI.
HOURLY_COLUMNS: tuple[str, ...] = (
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
)

#: Columns that may legitimately be missing and are therefore nullable.
NULLABLE_COLUMNS: frozenset[str] = frozenset(
    {"pm10", "nitrogen_dioxide", "ozone", "precipitation"}
)

DataMode = Literal["live", "demo"]
"""``live`` = real upstream measurements, ``demo`` = bundled sample dataset."""


class Observation(BaseModel):
    """One hour of air-quality and weather measurements at a location."""

    time: datetime
    pm2_5: float = Field(description="Fine particulate matter, ug/m3")
    pm10: float | None = Field(default=None, description="Coarse PM, ug/m3")
    nitrogen_dioxide: float | None = Field(default=None, description="NO2, ug/m3")
    ozone: float | None = Field(default=None, description="O3, ug/m3")
    temperature_2m: float = Field(description="Air temperature at 2m, degC")
    relative_humidity_2m: float = Field(description="Relative humidity, %")
    wind_speed_10m: float = Field(description="Wind speed at 10m, km/h")
    wind_direction_10m: float = Field(description="Wind direction at 10m, degrees")
    surface_pressure: float = Field(description="Surface pressure, hPa")
    precipitation: float | None = Field(default=None, description="Precipitation, mm")


class SourceInfo(BaseModel):
    """Provenance for a dataset so the UI can never silently mislead a user."""

    name: str = Field(description="Human-readable source name")
    url: str = Field(description="Upstream URL the data came from")
    mode: DataMode = Field(description="Whether these are live or demo readings")
    licence: str = Field(description="Licence/attribution string for the data")
    fetched_at: datetime | None = Field(
        default=None, description="When the data was retrieved (UTC)"
    )
    is_synthetic: bool = Field(
        default=False,
        description="True only for the bundled sample dataset, never for live data",
    )


class Observations(BaseModel):
    """A location-tagged, time-ordered batch of observations."""

    latitude: float
    longitude: float
    location_name: str
    source: SourceInfo
    rows: list[Observation]

    def to_records(self) -> list[dict]:
        """Flatten to plain dicts for pandas/feature engineering."""
        return [r.model_dump() for r in self.rows]
