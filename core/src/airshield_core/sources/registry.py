"""Location registry and the source-selection helper used by the backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from airshield_core.schema import SourceInfo


@dataclass(frozen=True)
class Location:
    """A place AirShield can forecast for."""

    slug: str
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str = "UTC"

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}"


#: Built-in locations. Each was chosen because it has a full Open-Meteo history
#: (required for training) and a distinct pollution profile.
LOCATIONS: tuple[Location, ...] = (
    Location("berlin", "Berlin", "Germany", 52.52, 13.405, "Europe/Berlin"),
    Location("new-york", "New York", "United States", 40.7128, -74.006, "America/New_York"),
    Location("delhi", "Delhi", "India", 28.6139, 77.209, "Asia/Kolkata"),
    Location("los-angeles", "Los Angeles", "United States", 34.0522, -118.2437, "America/Los_Angeles"),
    Location("sao-paulo", "Sao Paulo", "Brazil", -23.5505, -46.6333, "America/Sao_Paulo"),
    Location("beijing", "Beijing", "China", 39.9042, 116.4074, "Asia/Shanghai"),
)

LOCATIONS_BY_SLUG: dict[str, Location] = {loc.slug: loc for loc in LOCATIONS}

DEFAULT_LOCATION = LOCATIONS_BY_SLUG["berlin"]


class LocationNotFoundError(KeyError):
    """Raised when a requested location slug is unknown."""


def get_location(slug: str) -> Location:
    try:
        return LOCATIONS_BY_SLUG[slug]
    except KeyError as exc:
        known = ", ".join(sorted(LOCATIONS_BY_SLUG))
        raise LocationNotFoundError(
            f"unknown location {slug!r}. Available: {known}"
        ) from exc


def get_source(data_dir: Path | str):
    """Return the bundled demo source. Live access is handled by the backend.

    The demo source is constructed here so both the CLI trainer and the API use
    the same dataset path resolution.
    """
    from airshield_core.sources.demo import DemoSource

    return DemoSource(Path(data_dir))
