"""AWS architecture status.

AirShield Pulse uses AWS for a small number of clearly-scoped jobs. This module
reports what is *actually configured* in the running process, so the UI and the
docs can never claim a service is deployed when it is not.

It never invents an AWS response. ``configured`` reflects environment settings;
``verified`` is only ever set by a real API call (see ``probe_credentials``),
which is opt-in because it requires network access and credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from airshield_core.config import Settings


@dataclass
class AwsComponent:
    """One AWS service in the architecture and whether it is wired up."""

    key: str
    name: str
    purpose: str
    configured: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "purpose": self.purpose,
            "configured": self.configured,
            "detail": self.detail,
        }


class AwsStatusService:
    """Describes the AWS architecture and its real configuration state."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def components(self) -> list[AwsComponent]:
        sagemaker_ready = bool(self.settings.sagemaker_endpoint_name) and (
            self.settings.inference_backend == "aws"
        )
        bucket = os.environ.get("AIRSHIELD_S3_BUCKET", "").strip()
        table = os.environ.get("AIRSHIELD_DYNAMODB_TABLE", "").strip()
        topic = os.environ.get("AIRSHIELD_SNS_TOPIC_ARN", "").strip()
        schedule = os.environ.get("AIRSHIELD_EVENTBRIDGE_SCHEDULE", "").strip()
        function = os.environ.get("AIRSHIELD_LAMBDA_FUNCTION", "").strip()

        return [
            AwsComponent(
                key="sagemaker",
                name="Amazon SageMaker AI",
                purpose="Primary production ML path for PM2.5 forecasting.",
                configured=sagemaker_ready,
                detail=(
                    f"endpoint {self.settings.sagemaker_endpoint_name} "
                    f"({self.settings.aws_region})"
                    if sagemaker_ready
                    else "not active: set AIRSHIELD_INFERENCE_BACKEND=aws and "
                    "SAGEMAKER_ENDPOINT_NAME, then deploy with ml/sagemaker/deploy.py"
                ),
            ),
            AwsComponent(
                key="s3",
                name="Amazon S3",
                purpose="Training datasets and versioned model artifacts.",
                configured=bool(bucket),
                detail=bucket or "set AIRSHIELD_S3_BUCKET to enable",
            ),
            AwsComponent(
                key="dynamodb",
                name="Amazon DynamoDB",
                purpose="Persistent location preferences, saved plans and spike events.",
                configured=bool(table),
                detail=table or "set AIRSHIELD_DYNAMODB_TABLE to enable",
            ),
            AwsComponent(
                key="lambda",
                name="AWS Lambda",
                purpose="Scheduled ingestion of Open-Meteo data into S3/DynamoDB.",
                configured=bool(function),
                detail=function or "set AIRSHIELD_LAMBDA_FUNCTION to enable",
            ),
            AwsComponent(
                key="eventbridge",
                name="Amazon EventBridge",
                purpose="Schedule for the spike-detection and notification pipeline.",
                configured=bool(schedule),
                detail=schedule or "set AIRSHIELD_EVENTBRIDGE_SCHEDULE to enable",
            ),
            AwsComponent(
                key="sns",
                name="Amazon SNS",
                purpose="Delivery of pollution-spike notifications.",
                configured=bool(topic),
                detail=topic or "set AIRSHIELD_SNS_TOPIC_ARN to enable",
            ),
            AwsComponent(
                key="cloudwatch",
                name="Amazon CloudWatch",
                purpose="Logs, errors and model-inference metrics.",
                configured=True,
                detail="always on: uvicorn writes structured logs to stdout",
            ),
        ]

    def probe_credentials(self) -> dict:
        """Make a real STS call to verify credentials. Never guesses.

        Returns the account and ARN the credentials resolve to, or an explicit
        failure. This is opt-in because it needs network access and credentials.
        """
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - boto3 is a dependency
            return {"verified": False, "detail": f"boto3 unavailable: {exc}"}

        try:
            client = boto3.client("sts", region_name=self.settings.aws_region)
            identity = client.get_caller_identity()
        except Exception as exc:
            return {
                "verified": False,
                "detail": f"AWS credentials could not be verified: {exc}",
            }
        return {
            "verified": True,
            "account": identity.get("Account"),
            "arn": identity.get("Arn"),
            "region": self.settings.aws_region,
        }

    def status(self, *, verify: bool = False) -> dict:
        components = self.components()
        configured = [c.key for c in components if c.configured]
        return {
            "region": self.settings.aws_region,
            "inference_backend": self.settings.inference_backend,
            "sagemaker_active": self.settings.inference_backend == "aws"
            and bool(self.settings.sagemaker_endpoint_name),
            "components": [c.to_dict() for c in components],
            "configured_components": configured,
            "credentials": (
                self.probe_credentials()
                if verify
                else {
                    "verified": False,
                    "detail": "not probed; call /api/aws/status?verify=true to run a real STS check",
                }
            ),
            "note": (
                "AirShield Pulse does not claim any AWS service is deployed unless it "
                "is configured here. Local inference remains fully functional without "
                "AWS. AWS AI/ML usage is limited to Amazon SageMaker AI."
            ),
        }
