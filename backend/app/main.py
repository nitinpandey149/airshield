"""AirShield FastAPI application.

Run locally with:
    uvicorn app.main:app --reload --port 8000   (from the backend/ directory)
or via ``make dev`` from the repository root.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from airshield_core import __version__
from airshield_core.config import REPO_ROOT, get_settings
from app.routers import forecast as forecast_router
from app.routers import meta as meta_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="AirShield API",
    description=(
        "Near-term PM2.5 forecasting and air-exposure alerts. "
        "Primary ML: XGBoost on Amazon SageMaker AI."
    ),
    version=__version__,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta_router.router)
app.include_router(forecast_router.router)


def service_banner() -> dict:
    """Service banner with the active configuration, so a client can self-check."""
    return {
        "name": "AirShield API",
        "version": __version__,
        "tagline": "Know the air before you step outside.",
        "inference_backend": settings.inference_backend,
        "data_mode": settings.data_mode,
        "endpoints": {
            "health": "/api/health",
            "model": "/api/model",
            "locations": "/api/locations",
            "forecast": "/api/forecast/{location_slug}",
            "docs": "/docs",
        },
        "ml_service": "Amazon SageMaker AI (XGBoost)",
        "data_source": "Open-Meteo (CC BY 4.0)",
    }


# The banner is reachable at /api/info always, and also at / when the API is
# not serving the dashboard.
app.get("/api/info", tags=["meta"])(service_banner)
root = service_banner


def _not_found(exc: Exception) -> JSONResponse:
    detail = getattr(exc, "detail", "Not Found")
    status = getattr(exc, "status_code", 404)
    return JSONResponse({"detail": detail}, status_code=status)


# Serving the built dashboard from the API is opt-in, and Docker enables it.
# Keeping it off by default means a local dev server keeps a plain API at "/"
# (the frontend runs separately on :5173), while the image is a single origin.
if settings.serve_frontend:
    FRONTEND_DIST = Path(
        os.environ.get("AIRSHIELD_FRONTEND_DIST") or (REPO_ROOT / "frontend" / "dist")
    )

    if FRONTEND_DIST.is_dir():
        # Mounted last so every /api/* route registered above wins.
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="dashboard")

        @app.exception_handler(404)
        async def spa_fallback(request: Request, exc: Exception):  # pragma: no cover
            """Serve index.html for client-side routes; keep API 404s as JSON."""
            if request.url.path.startswith("/api"):
                return _not_found(exc)
            index = FRONTEND_DIST / "index.html"
            return FileResponse(index) if index.is_file() else _not_found(exc)

else:
    app.get("/", tags=["meta"])(service_banner)
