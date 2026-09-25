"""Verify a SageMaker endpoint with a real invocation.

    python ml/sagemaker/smoke_test.py --endpoint-name airshield-pm25-endpoint

Sends one real feature row built from real current measurements and prints the
prediction the endpoint returns. Nothing is simulated: if the endpoint is not
reachable the script exits non-zero and says why.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "src"))

from airshield_core.aqi import aqi_from_pm25, exposure_alert  # noqa: E402
from airshield_core.features import latest_feature_row  # noqa: E402
from airshield_core.sources.openmeteo import UpstreamError, fetch_frame_with_fallback  # noqa: E402
from airshield_core.sources.registry import DEFAULT_LOCATION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-name", default=os.environ.get("SAGEMAKER_ENDPOINT_NAME"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--location", default=DEFAULT_LOCATION.slug)
    args = parser.parse_args()

    if not args.endpoint_name:
        print("ERROR: --endpoint-name (or SAGEMAKER_ENDPOINT_NAME) is required.")
        return 1

    from airshield_core.sources.registry import get_location

    location = get_location(args.location)

    try:
        frame, info = fetch_frame_with_fallback(
            location.latitude, location.longitude, timeout=30, retries=2
        )
    except UpstreamError as exc:
        print(f"ERROR: could not fetch real measurements: {exc}")
        return 1

    features, base_time = latest_feature_row(frame)
    print(f"location   : {location.label}")
    print(f"data source: {info.name} ({info.mode})")
    print(f"base time  : {base_time}")

    try:
        from app.services.sagemaker_client import SageMakerClient, SageMakerError
    except ImportError:
        # Allow running this script without the backend package installed.
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        from app.services.sagemaker_client import SageMakerClient, SageMakerError

    client = SageMakerClient(args.endpoint_name, args.region)
    try:
        prediction = client.predict(features)
    except SageMakerError as exc:
        print(f"ERROR: {exc}")
        return 1

    result = aqi_from_pm25(prediction)
    alert = exposure_alert(prediction)
    print(f"prediction : {prediction:.2f} ug/m3 PM2.5 for the next hour")
    print(f"AQI        : {result.aqi} ({result.category})")
    print(f"alert      : {alert['headline']} [{alert['severity']}]")
    print(json.dumps({"prediction": round(prediction, 3), "aqi": result.aqi}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
