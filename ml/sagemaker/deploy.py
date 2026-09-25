"""Deploy a trained AirShield model to a real SageMaker endpoint.

    python ml/sagemaker/deploy.py \
        --model-data s3://bucket/airshield-pm25/output/<job>/output/model.tar.gz \
        --role-arn arn:aws:iam::123456789012:role/SageMakerRole

Prints the real endpoint name on success. Point the backend at it with
``SAGEMAKER_ENDPOINT_NAME`` and ``AIRSHIELD_INFERENCE_BACKEND=aws``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "src"))

DEFAULT_ENDPOINT = "airshield-pm25-endpoint"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", default=os.environ.get("AIRSHIELD_MODEL_DATA"))
    parser.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--endpoint-name", default=os.environ.get("SAGEMAKER_ENDPOINT_NAME") or DEFAULT_ENDPOINT)
    parser.add_argument("--instance", default="ml.m5.large")
    args = parser.parse_args()

    if not args.model_data:
        print("ERROR: --model-data (or AIRSHIELD_MODEL_DATA) is required.")
        return 1
    if not args.role_arn:
        print("ERROR: --role-arn (or SAGEMAKER_ROLE_ARN) is required.")
        return 1

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is required. Install with `pip install boto3`.")
        return 1

    import sagemaker
    from sagemaker.xgboost import XGBoostModel

    session = sagemaker.Session(boto3.Session(region_name=args.region))
    source_dir = str(Path(__file__).resolve().parent)

    model = XGBoostModel(
        model_data=args.model_data,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir=source_dir,
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=session,
    )

    print(f"deploying to endpoint {args.endpoint_name!r} on {args.instance} ...")
    model.deploy(
        initial_instance_count=1,
        instance_type=args.instance,
        endpoint_name=args.endpoint_name,
    )

    print(f"\nendpoint is live: {args.endpoint_name}")
    print("Configure the backend with:")
    print("  AIRSHIELD_INFERENCE_BACKEND=aws")
    print(f"  SAGEMAKER_ENDPOINT_NAME={args.endpoint_name}")
    print(f"  AWS_REGION={args.region}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
