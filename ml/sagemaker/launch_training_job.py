"""Launch a real SageMaker training job for the AirShield PM2.5 model.

This is the only AWS AI/ML service AirShield uses. The job runs
:mod:`airshield_core.train` inside the managed XGBoost container and writes a
``model.tar.gz`` artifact to S3.

Requirements: AWS credentials with SageMaker + S3 permissions, and the
``AIRSHIELD_S3_BUCKET`` environment variable set to a bucket you own.

    export AIRSHIELD_S3_BUCKET=my-airshield-bucket
    python ml/sagemaker/launch_training_job.py --role-arn arn:aws:iam::...:role/SageMakerRole

Everything this script prints comes from the actual AWS API response.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "src"))

DEFAULT_INSTANCE = "ml.m5.xlarge"
TRAINING_JOB_PREFIX = "airshield-pm25"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    parser.add_argument("--bucket", default=os.environ.get("AIRSHIELD_S3_BUCKET"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--days", type=int, default=int(os.environ.get("AIRSHIELD_TRAIN_DAYS", "90")))
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--wait", action="store_true", help="block until the job finishes")
    args = parser.parse_args()

    if not args.role_arn:
        print("ERROR: --role-arn (or SAGEMAKER_ROLE_ARN) is required.")
        return 1
    if not args.bucket:
        print("ERROR: --bucket (or AIRSHIELD_S3_BUCKET) is required.")
        return 1

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is required. Install with `pip install boto3`.")
        return 1

    import sagemaker
    from sagemaker.inputs import TrainingInput
    from sagemaker.xgboost import XGBoost

    session = sagemaker.Session(boto3.Session(region_name=args.region))
    job_name = f"{TRAINING_JOB_PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"region      : {args.region}")
    print(f"bucket      : {args.bucket}")
    print(f"job name    : {job_name}")
    print(f"data window : {args.days} days of real Open-Meteo measurements")

    # Fetch the real training data locally, then stage it to S3. Fetching here
    # (rather than in the container) keeps the container offline and simple.
    from airshield_core.sources.openmeteo import (
        UpstreamError,
        default_history_window,
        fetch_history,
    )
    from airshield_core.sources.registry import LOCATIONS
    import pandas as pd

    start, end = default_history_window(args.days)
    frames = []
    for location in LOCATIONS:
        try:
            part = fetch_history(location.latitude, location.longitude, start, end, timeout=60)
        except UpstreamError as exc:
            print(f"  ! {location.slug}: {exc}")
            continue
        part["location_name"] = location.name
        frames.append(part)
        print(f"  + {location.slug:<12} {len(part):>6} rows")

    if not frames:
        print("ERROR: no training data could be fetched.")
        return 1

    frame = pd.concat(frames, ignore_index=True)
    local_csv = REPO_ROOT / "ml" / "data" / "training_data.csv"
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(local_csv, index=False)
    print(f"\nstaged {len(frame)} real rows -> {local_csv}")

    train_s3 = session.upload_data(
        str(local_csv), bucket=args.bucket, key_prefix=f"{TRAINING_JOB_PREFIX}/input"
    )
    print(f"uploaded to {train_s3}")

    source_dir = str(Path(__file__).resolve().parent)
    estimator = XGBoost(
        entry_point="train_entry.py",
        source_dir=source_dir,
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance,
        framework_version="1.7-1",
        py_version="py3",
        output_path=f"s3://{args.bucket}/{TRAINING_JOB_PREFIX}/output",
        base_job_name=TRAINING_JOB_PREFIX,
        hyperparameters={"num-rounds": args.rounds, "days": args.days},
        disable_profiler=True,
    )

    print("\nstarting SageMaker training job (this is a real AWS resource)...")
    estimator.fit({"training": TrainingInput(train_s3, content_type="text/csv")}, wait=args.wait)

    print(f"\njob name    : {estimator.latest_training_job.name}")
    print(f"model artifact: {estimator.model_data}")
    print("\nDeploy it with:")
    print(f"  python ml/sagemaker/deploy.py --model-data {estimator.model_data} --role-arn {args.role_arn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
