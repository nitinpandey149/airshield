"""AWS architecture status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from airshield_core.config import get_settings
from app.deps import get_aws_status_service
from app.models import AwsStatusResponse
from app.services.aws_status import AwsStatusService

router = APIRouter(prefix="/api", tags=["aws"])


@router.get("/aws/status", response_model=AwsStatusResponse)
def aws_status(
    verify: bool = Query(
        False,
        description="Run a real STS credential check (requires network + credentials)",
    ),
    service: AwsStatusService = Depends(get_aws_status_service),
) -> AwsStatusResponse:
    """Report which AWS components are actually configured.

    This endpoint never claims a service is deployed unless it is configured in
    the running process. Local inference works fully without AWS.
    """
    return AwsStatusResponse(**service.status(verify=verify))


@router.get("/aws/architecture")
def aws_architecture() -> dict:
    """The AWS architecture as designed, with each service's purpose."""
    settings = get_settings()
    return {
        "pipeline": [
            "Open-Meteo (air quality + weather)",
            "AWS Lambda (scheduled ingestion)",
            "Amazon S3 (training data) / Amazon DynamoDB (state)",
            "Amazon SageMaker AI (training + inference endpoint)",
            "FastAPI (this service)",
            "React frontend",
        ],
        "notification_pipeline": [
            "Amazon EventBridge (schedule)",
            "AWS Lambda (spike check)",
            "Amazon SNS (user notification)",
        ],
        "observability": "Amazon CloudWatch (logs, errors, inference metrics)",
        "region": settings.aws_region,
        "ml_service": "Amazon SageMaker AI (the only AWS AI/ML service used)",
        "infrastructure_as_code": "infra/cloudformation/airshield-pulse.yaml",
        "note": (
            "Every service listed has a specific job. No service is included to "
            "enlarge the diagram. See /api/aws/status for what is configured."
        ),
    }
