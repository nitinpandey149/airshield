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


#: Built-in locations: Indian cities, where PM2.5 exposure avoidance matters most.
#: Each was chosen because it has a full Open-Meteo history (required for
#: training) and a distinct pollution profile across India's climate zones.
#: Air quality is measured by Open-Meteo's CAMS-based model, not by CPCB
#: monitors; the AQI reported is India's CPCB National AQI.
LOCATIONS: tuple[Location, ...] = (
    Location("delhi", "Delhi", "India", 28.6139, 77.209, "Asia/Kolkata"),
    Location("mumbai", "Mumbai", "India", 19.076, 72.8777, "Asia/Kolkata"),
    Location("bengaluru", "Bengaluru", "India", 12.9716, 77.5946, "Asia/Kolkata"),
    Location("chennai", "Chennai", "India", 13.0827, 80.2707, "Asia/Kolkata"),
    Location("kolkata", "Kolkata", "India", 22.5726, 88.3639, "Asia/Kolkata"),
    Location("hyderabad", "Hyderabad", "India", 17.385, 78.4867, "Asia/Kolkata"),
)

LOCATIONS_BY_SLUG: dict[str, Location] = {loc.slug: loc for loc in LOCATIONS}

DEFAULT_LOCATION = LOCATIONS_BY_SLUG["delhi"]


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
