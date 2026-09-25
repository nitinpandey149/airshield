"""Data acquisition: real upstream measurements and the bundled demo dataset."""

from airshield_core.sources.demo import DemoSource
from airshield_core.sources.openmeteo import OpenMeteoSource, fetch_history
from airshield_core.sources.registry import LOCATIONS, Location, get_source

__all__ = [
    "DemoSource",
    "OpenMeteoSource",
    "fetch_history",
    "LOCATIONS",
    "Location",
    "get_source",
]
