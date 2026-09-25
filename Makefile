# AirShield developer commands.
#
# The Python toolchain lives in ./.venv. Create it with `make setup`.

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REPO_ROOT := $(shell pwd)

# Backend tests need the backend package on the path.
export PYTHONPATH := $(REPO_ROOT)/backend:$(REPO_ROOT)/core/src

.DEFAULT_GOAL := help
.PHONY: help setup install train train-offline build-demo-data api frontend-dev \
        frontend-build test test-core test-backend test-frontend clean \
        docker-build docker-run sagemaker-train sagemaker-deploy sagemaker-smoke check

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- setup
$(VENV):
	python3 -m venv $(VENV)

setup: $(VENV) ## Create the virtualenv and install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ./core
	$(PIP) install -r backend/requirements.txt
	$(PIP) install -r ml/requirements.txt
	cd frontend && npm install

install: setup ## Alias for setup

# ------------------------------------------------------------------ data/ML
build-demo-data: ## Rebuild the bundled demo dataset from real Open-Meteo data
	$(PY) scripts/build_demo_dataset.py

train: ## Train the XGBoost PM2.5 model and write ml/artifacts
	$(PY) -m airshield_core.train --out ml/artifacts --rounds 400

train-offline: ## Retrain from the bundled dataset (no network required)
	$(PY) -m airshield_core.train --out ml/artifacts --rounds 400 \
		--from-csv ml/data/demo/demo_hourly.csv

# ---------------------------------------------------------------- run locally
api: ## Run the FastAPI backend with autoreload on :8000
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

frontend-dev: ## Run the Vite dev server on :5173
	cd frontend && npm run dev

frontend-build: ## Type-check and build the frontend
	cd frontend && npm run build

# --------------------------------------------------------------------- tests
test: test-core test-backend ## Run all Python tests

test-core: ## Run the core library tests
	cd core && ../$(PY) -m pytest tests -q

test-backend: ## Run the API tests
	cd backend && ../$(PY) -m pytest tests -q

test-frontend: ## Type-check the frontend
	cd frontend && npx tsc -b --noEmit

check: test test-frontend ## Run tests and type checks

# -------------------------------------------------------------------- docker
docker-build: ## Build the combined frontend+backend image
	docker build -t airshield:local .

docker-run: ## Run the image on :8000
	docker run --rm -p 8000:8000 --env-file .env.docker airshield:local

# ----------------------------------------------------------------- sagemaker
sagemaker-train: ## Launch a real SageMaker training job (needs AWS creds)
	$(PY) ml/sagemaker/launch_training_job.py --wait

sagemaker-deploy: ## Deploy the trained artifact to a SageMaker endpoint
	$(PY) ml/sagemaker/deploy.py

sagemaker-smoke: ## Invoke the deployed SageMaker endpoint once
	$(PY) ml/sagemaker/smoke_test.py

# --------------------------------------------------------------------- clean
clean: ## Remove build output and caches
	rm -rf frontend/dist frontend/node_modules/.vite
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -rf core/src/*.egg-info core/*.egg-info .pytest_cache
