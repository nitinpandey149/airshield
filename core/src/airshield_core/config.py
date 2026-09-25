"""Configuration loading.

All settings come from environment variables (optionally via a local ``.env``
file). Secrets are never hardcoded; AWS credentials are resolved by the
standard boto3 credential chain.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_file() -> Path | None:
    candidate = REPO_ROOT / ".env"
    return candidate if candidate.is_file() else None


class Settings(BaseSettings):
    """Runtime configuration for AirShield."""

    model_config = SettingsConfigDict(
        env_file=_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- inference ---------------------------------------------------------
    inference_backend: Literal["local", "aws"] = Field(
        default="local", alias="AIRSHIELD_INFERENCE_BACKEND"
    )
    data_mode: Literal["live", "demo", "auto"] = Field(
        default="auto", alias="AIRSHIELD_DATA_MODE"
    )
    artifact_dir: str = Field(default="ml/artifacts", alias="AIRSHIELD_ARTIFACT_DIR")

    # ---- upstream http -----------------------------------------------------
    http_timeout: float = Field(default=15.0, alias="AIRSHIELD_HTTP_TIMEOUT")
    http_retries: int = Field(default=2, alias="AIRSHIELD_HTTP_RETRIES")

    # ---- backend -----------------------------------------------------------
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="AIRSHIELD_CORS_ORIGINS",
    )
    # Serve the built dashboard from this API (single-origin deployment).
    # Off by default: in development the frontend runs on its own Vite server.
    serve_frontend: bool = Field(default=False, alias="AIRSHIELD_SERVE_FRONTEND")

    # ---- sagemaker ---------------------------------------------------------
    sagemaker_endpoint_name: str = Field(default="", alias="SAGEMAKER_ENDPOINT_NAME")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # ---- training ----------------------------------------------------------
    train_days: int = Field(default=60, alias="AIRSHIELD_TRAIN_DAYS")
    data_dir: str = Field(default="ml/data", alias="AIRSHIELD_DATA_DIR")

    # ------------------------------------------------------------------ paths
    @property
    def artifact_path(self) -> Path:
        p = Path(self.artifact_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings (env is read once per process)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache - used by tests that patch the environment."""
    get_settings.cache_clear()
    os.environ.pop("_AIRSHIELD_SETTINGS_SENTINEL", None)
