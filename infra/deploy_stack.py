#!/usr/bin/env python3
"""Deploy the AirShield Pulse CloudFormation stack with boto3.

This exists so the repository can be deployed from the project virtualenv,
without requiring the AWS CLI to be installed. It performs a real deployment
and prints only values that AWS returned.

It refuses to pretend: with no credentials it exits non-zero with the real
reason, rather than reporting a success that did not happen.

    python infra/deploy_stack.py --env dev
    python infra/deploy_stack.py --env dev --email you@example.com \
        --api-base-url https://your-api.example.com
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import boto3
import botocore.exceptions

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "infra" / "cloudformation" / "airshield-pulse.yaml"

# CloudFormation terminal states we wait through.
_IN_PROGRESS = {
    "CREATE_IN_PROGRESS",
    "UPDATE_IN_PROGRESS",
    "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_ROLLBACK_IN_PROGRESS",
    "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS",
    "ROLLBACK_IN_PROGRESS",
}
_FAILED = {
    "CREATE_FAILED",
    "ROLLBACK_COMPLETE",
    "UPDATE_ROLLBACK_COMPLETE",
    "ROLLBACK_FAILED",
    "UPDATE_ROLLBACK_FAILED",
    "DELETE_FAILED",
}


def _identity(region: str) -> dict:
    """Return the caller identity, or exit with the real AWS error."""
    try:
        return boto3.client("sts", region_name=region).get_caller_identity()
    except botocore.exceptions.NoCredentialsError:
        print(
            "ERROR: no AWS credentials found. Configure the standard boto3 chain "
            "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an "
            "instance role) and try again.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except botocore.exceptions.ClientError as exc:
        print(f"ERROR: AWS rejected the credentials: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _wait(cfn, stack_name: str, timeout: float) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            status = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
        except botocore.exceptions.ClientError as exc:
            if "does not exist" in str(exc):
                return "DOES_NOT_EXIST"
            raise
        if status != last:
            print(f"  stack status: {status}")
            last = status
        if status not in _IN_PROGRESS:
            return status
        time.sleep(10)
    print(f"ERROR: timed out after {timeout:.0f}s waiting for {stack_name}", file=sys.stderr)
    raise SystemExit(1)


def _failure_reason(cfn, stack_name: str) -> str:
    events = cfn.describe_stack_events(StackName=stack_name)["StackEvents"]
    for event in events:
        if event.get("ResourceStatus", "").endswith("_FAILED"):
            return (
                f"{event.get('LogicalResourceId')}: "
                f"{event.get('ResourceStatusReason', 'no reason given')}"
            )
    return "no failing resource event found"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Deploy the AirShield Pulse stack.")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--region", default=None, help="defaults to AWS_REGION or us-east-1")
    parser.add_argument("--email", default="", help="optional SNS spike-notification email")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--location", default="berlin")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    region = args.region or _default_region()
    template_body = TEMPLATE.read_text(encoding="utf-8")
    stack_name = f"airshield-pulse-{args.env}"

    identity = _identity(region)
    print(f"Account : {identity['Account']}")
    print(f"Region  : {region}")
    print(f"Stack   : {stack_name}")

    cfn = boto3.client("cloudformation", region_name=region)

    # Real server-side validation, before creating anything.
    cfn.validate_template(TemplateBody=template_body)
    print("template validated by CloudFormation")
    if args.validate_only:
        return 0

    parameters = [
        {"ParameterKey": "Environment", "ParameterValue": args.env},
        {"ParameterKey": "LocationSlug", "ParameterValue": args.location},
        {"ParameterKey": "ApiBaseUrl", "ParameterValue": args.api_base_url},
        {"ParameterKey": "NotificationEmail", "ParameterValue": args.email},
    ]

    exists = _exists(cfn, stack_name)
    action = "update" if exists else "create"
    print(f"{action} in progress ...")
    try:
        if exists:
            cfn.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
        else:
            cfn.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
    except botocore.exceptions.ClientError as exc:
        message = str(exc)
        if "No updates are to be performed" in message:
            print("stack is already up to date")
        else:
            print(f"ERROR: {message}", file=sys.stderr)
            return 1

    status = _wait(cfn, stack_name, args.timeout)
    if status in _FAILED:
        print(f"ERROR: deployment failed ({status})", file=sys.stderr)
        print(f"  reason: {_failure_reason(cfn, stack_name)}", file=sys.stderr)
        return 1

    outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0].get("Outputs", [])
    print(f"\ndeployed ({status}). Outputs:")
    for output in outputs:
        print(f"  {output['OutputKey']:22s} {output['OutputValue']}")
    print(
        "\nSet these in your .env to point the API at the deployed state store:\n"
        f"  AIRSHIELD_S3_BUCKET=<DataBucketName>\n"
        f"  AIRSHIELD_DYNAMODB_TABLE=<StateTableName>\n"
        f"  AIRSHIELD_SNS_TOPIC_ARN=<SpikeTopicArn>"
    )
    return 0


def _default_region() -> str:
    import os

    session = boto3.session.Session()
    return os.environ.get("AWS_REGION") or session.region_name or "us-east-1"


def _exists(cfn, stack_name: str) -> bool:
    try:
        cfn.describe_stacks(StackName=stack_name)
        return True
    except botocore.exceptions.ClientError as exc:
        if "does not exist" in str(exc):
            return False
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
