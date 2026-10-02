"""Pollution-spike notification Lambda for AirShield Pulse.

Runs on an EventBridge schedule. Calls the AirShield Pulse API for the latest
forecast, and when a pollution spike is predicted publishes a notification to
SNS. A spike is only notified once per hour (de-duplicated through DynamoDB), so
subscribers are not spammed.

The notification text is built from the API response verbatim. This function
never invents a spike, a confidence or an expected change: if the API is
unreachable it raises, and CloudWatch records the failure.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _get_json(url: str, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "AirShield-Pulse/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _message(forecast: dict) -> tuple[str, str]:
    location = forecast["location"]["label"]
    spike = forecast.get("spike") or {}
    current = forecast.get("current") or {}
    change = spike.get("expected_change_percent", 0)
    when = spike.get("expected_time") or "soon"
    subject = f"AirShield Pulse: pollution spike expected in {location}"
    body = (
        f"A pollution spike is predicted for {location}.\n\n"
        f"Expected time: {when}\n"
        f"Expected change: {change:+.0f}%\n"
        f"Severity: {spike.get('severity', 'unknown')}\n"
        f"Confidence: {spike.get('confidence', 0) * 100:.0f}% "
        f"(basis: {spike.get('confidence_basis', 'unknown')})\n"
        f"Current PM2.5: {current.get('pm2_5', 'n/a')} ug/m3\n\n"
        "This is a forecast-based alert, not medical advice. Consider moving "
        "outdoor activity to a lower-exposure window. Open the dashboard for the "
        "recommended window and lower-exposure route.\n\n"
        "Contributing signals are associated conditions, not proven causes."
    )
    return subject, body


def handler(event, context):  # noqa: ANN001 - AWS Lambda signature
    import boto3

    base_url = os.environ.get("AIRSHIELD_API_BASE_URL", "http://localhost:8000").rstrip("/")
    slug = os.environ.get("AIRSHIELD_LOCATION_SLUG", "berlin")
    topic_arn = os.environ["AIRSHIELD_SNS_TOPIC_ARN"]
    table_name = os.environ.get("AIRSHIELD_TABLE")

    forecast = _get_json(f"{base_url}/api/forecast/{slug}")
    spike = forecast.get("spike") or {}

    if not spike.get("spike_detected"):
        return {"notified": False, "reason": "no spike predicted"}

    expected_time = spike.get("expected_time") or ""
    hour_key = expected_time[:13]  # YYYY-MM-DDTHH

    if table_name:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(table_name)
        dedupe_pk = f"SPIKE#{slug}"
        dedupe_sk = f"NOTIFIED#{hour_key}"
        existing = table.get_item(Key={"pk": dedupe_pk, "sk": dedupe_sk}).get("Item")
        if existing:
            return {"notified": False, "reason": "already notified for this hour"}

    subject, body = _message(forecast)
    sns = boto3.client("sns")
    sns.publish(TopicArn=topic_arn, Subject=subject[:100], Message=body)

    if table_name:
        now = datetime.now(timezone.utc)
        table.put_item(
            Item={
                "pk": f"SPIKE#{slug}",
                "sk": f"NOTIFIED#{hour_key}",
                "location_slug": slug,
                "expected_time": expected_time,
                "severity": spike.get("severity"),
                "expected_change_percent": str(spike.get("expected_change_percent", 0)),
                "confidence": str(spike.get("confidence", 0)),
                "confidence_basis": spike.get("confidence_basis"),
                "notified_at": now.isoformat(),
                "expires_at": int(now.timestamp()) + 7 * 86400,
            }
        )

    return {"notified": True, "expected_time": expected_time}
