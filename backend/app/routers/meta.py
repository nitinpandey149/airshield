"""Health and model-card endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from airshield_core import __version__
from app.deps import get_service
from app.models import HealthResponse, ModelInfoResponse
from app.services.forecast_service import ForecastService

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(service: ForecastService = Depends(get_service)) -> HealthResponse:
    """Report whether a usable predictor is wired up."""
    settings = service.settings
    try:
        predictor = service.predictor
        metadata = getattr(predictor, "metadata", None)
        version = getattr(predictor, "model_version", None)
        return HealthResponse(
            status="ok",
            version=__version__,
            inference_backend=settings.inference_backend,
            data_mode=settings.data_mode,
            model_available=True,
            model_version=version,
            sagemaker_endpoint=(
                settings.sagemaker_endpoint_name
                if settings.inference_backend == "aws"
                else None
            ),
        )
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            version=__version__,
            inference_backend=settings.inference_backend,
            data_mode=settings.data_mode,
            model_available=False,
            detail=str(exc),
        )


@router.get("/model", response_model=ModelInfoResponse)
def model_info(service: ForecastService = Depends(get_service)) -> ModelInfoResponse:
    """Return the model card: real metrics, provenance and hyperparameters."""
    try:
        return ModelInfoResponse(**service.model_info())
    except Exception as exc:
        return ModelInfoResponse(
            available=False,
            inference_backend=service.settings.inference_backend,
            note=f"model information unavailable: {exc}",
        )
