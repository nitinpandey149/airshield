"""Scheduled ingestion Lambda for AirShield Pulse.

Runs on an EventBridge schedule. Fetches real hourly air-quality and weather
observations from Open-Meteo for the configured location, stores them as a JSON
sample in S3 (training-data archive) and records the latest reading in DynamoDB
(serving state).

Nothing is synthesised: if Open-Meteo cannot be reached, the invocation fails
with a clear error so CloudWatch records it and the sample is simply not written.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_HOURLY = "pm2_5,pm10,nitrogen_dioxide,ozone"
WEATHER_HOURLY = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,surface_pressure,precipitation"
)


def _get(url: str, params: dict, timeout: float = 20.0) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "AirShield-Pulse/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch(latitude: float, longitude: float) -> dict:
    common = {"latitude": latitude, "longitude": longitude, "past_days": 2, "forecast_days": 1, "timezone": "UTC"}
    air = _get(AIR_QUALITY_URL, {**common, "hourly": AIR_HOURLY})
    weather = _get(WEATHER_URL, {**common, "hourly": WEATHER_HOURLY})

    times = (air.get("hourly") or {}).get("time") or []
    if not times:
        raise RuntimeError("Open-Meteo returned no hourly air-quality times")

    air_hourly = air.get("hourly") or {}
    weather_hourly = weather.get("hourly") or {}
    rows = []
    for index, stamp in enumerate(times):
        row = {"time": stamp}
        for key in AIR_HOURLY.split(","):
            series = air_hourly.get(key) or []
            row[key] = series[index] if index < len(series) else None
        for key in WEATHER_HOURLY.split(","):
            series = weather_hourly.get(key) or []
            row[key] = series[index] if index < len(series) else None
        rows.append(row)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "Open-Meteo",
            "url": AIR_QUALITY_URL,
            "licence": "CC BY 4.0 (https://open-meteo.com/en/license)",
        },
        "rows": rows,
    }


def handler(event, context):  # noqa: ANN001 - AWS Lambda signature
    import boto3

    bucket = os.environ["AIRSHIELD_BUCKET"]
    table = os.environ["AIRSHIELD_TABLE"]
    slug = os.environ.get("AIRSHIELD_LOCATION_SLUG", "berlin")
    latitude = float(os.environ.get("AIRSHIELD_LATITUDE", "52.52"))
    longitude = float(os.environ.get("AIRSHIELD_LONGITUDE", "13.405"))

    payload = _fetch(latitude, longitude)
    now = datetime.now(timezone.utc)
    key = f"samples/{slug}/{now:%Y/%m/%d}/{now:%H%M%S}.json"

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )

    # Latest measured row (PM2.5 not null) for the serving state.
    measured = [row for row in payload["rows"] if row.get("pm2_5") is not None]
    if not measured:
        raise RuntimeError("no measured PM2.5 in the fetched sample")
    latest = measured[-1]

    dynamodb = boto3.resource("dynamodb")
    dynamodb.Table(table).put_item(
        Item={
            "pk": f"LOCATION#{slug}",
            "sk": f"OBS#{latest['time']}",
            "location_slug": slug,
            "time": latest["time"],
            "pm2_5": str(latest["pm2_5"]),
            "latitude": str(latitude),
            "longitude": str(longitude),
            "s3_key": key,
            "ingested_at": now.isoformat(),
            # Samples expire after 90 days unless promoted to a training set.
            "expires_at": int(now.timestamp()) + 90 * 86400,
        }
    )

    return {"stored": key, "latest_time": latest["time"], "pm2_5": latest["pm2_5"]}
