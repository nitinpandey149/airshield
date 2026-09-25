# syntax=docker/dockerfile:1

# --- Stage 1: build the React frontend ---------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# --- Stage 2: runtime for the FastAPI backend --------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the backend dependencies first so this layer caches well.
COPY backend/requirements.txt ./backend/requirements.txt
COPY core/pyproject.toml ./core/pyproject.toml
COPY core/src ./core/src
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY core/ ./core/
COPY ml/data ./ml/data
COPY scripts/ ./scripts/

# Serve the built frontend as static files from the same origin.
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

ENV AIRSHIELD_ARTIFACT_DIR=/app/ml/artifacts \
    AIRSHIELD_DATA_MODE=auto \
    AIRSHIELD_INFERENCE_BACKEND=local \
    AIRSHIELD_SERVE_FRONTEND=true \
    PYTHONPATH=/app/backend:/app/core/src

# Train from the bundled dataset so the image is self-contained and works with
# no network at runtime. Trained artifacts are gitignored, so they cannot be
# copied in from the build context.
RUN python -m airshield_core.train --out /app/ml/artifacts --rounds 400 \
        --from-csv /app/ml/data/demo/demo_hourly.csv \
    && test -f /app/ml/artifacts/model.ubj

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
