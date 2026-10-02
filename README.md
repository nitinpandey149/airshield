# 🛡️ AirShield Pulse

**Don't just know the air. Know when and where to breathe it.**

Traditional air-quality apps stop at *"air quality is bad."* AirShield Pulse
answers the questions that actually change what you do:

1. **Will** air quality become bad?
2. **When** is the safest time to go outside?
3. **Which route** exposes me to less pollution?
4. **How much** exposure can I avoid?
5. Is a pollution **spike** likely soon?

Built on the original AirShield foundation — the leakage-safe XGBoost PM2.5
pipeline, the Open-Meteo ingestion, the SageMaker integration and the honesty
guarantees are all preserved and extended.

```
LIVE ENVIRONMENTAL DATA  →  PM2.5 FORECAST  →  POLLUTION RISK ENGINE
   →  PERSONAL EXPOSURE MODEL  →  SAFE TIME WINDOW
   →  LOW-EXPOSURE ROUTE  →  ACTIONABLE RECOMMENDATION
```

---

## What it actually does

You pick a city. AirShield Pulse:

1. Fetches **real hourly air-quality and weather measurements** from
   [Open-Meteo](https://open-meteo.com/) (public API, no key required).
2. Runs a **separately trained XGBoost model per forecast horizon** (1h, 3h, 6h)
   to predict PM2.5. No single model is stretched to pretend it covers six hours.
3. Converts the current hour and each horizon into **US EPA AQI** and a six-level
   exposure alert.
4. Runs a **pollution event detector** that flags sharp and sustained rises, with
   a **model-calibrated confidence** and associated (not causal) signals.
5. Scores **candidate time windows** for your activity and duration, over the
   whole window rather than the single lowest PM2.5 reading.
6. Compares **real route alternatives** from an OpenStreetMap routing engine,
   sampling predicted PM2.5 along each path and ranking them by exposure.
7. Shows all of it in a React dashboard that keeps measured and predicted data
   visually distinct and every number traceable.

The model is served either from a local artifact or from a **real Amazon
SageMaker AI endpoint** — the only AWS AI/ML service used in this project — and
the API states which one produced each prediction.

---

## Honesty guarantees

This project is deliberately explicit about what is real and what is not.

| Guarantee | How it is enforced |
| --- | --- |
| No invented measurements | Every reading comes from a live HTTP response, or from the bundled dataset. |
| No invented predictions | Predictions come from a trained booster, or a real SageMaker invocation. |
| No invented accuracy | Metrics are computed on a held-out time slice and read back from the artifact. |
| No invented confidence | Spike confidence is either an empirical historical frequency from the training data, or a transparent rule-based estimate. The basis is always stated. |
| No invented AWS responses | If SageMaker is not configured, the API reports `degraded`/errors. `/api/aws/status` reports exactly what is configured. |
| No invented routes | Route geometry comes from a real routing engine; if none answers, the feature reports unavailable. |
| No medical-safety claims | Routes and windows are described as **lower predicted exposure**, never safe. |
| No causal claims | Spike signals are labelled "possible contributing signals", never proven causes. |
| Demo data is always labelled | Demo responses carry `source.mode: "demo"` and a `DEMO DATA` notice, shown in a banner that is never behind a toggle. |

The suite includes a dedicated [honesty test module](backend/tests/test_honesty.py)
asserting that a simulated upstream outage produces a `503` rather than fabricated
data.

### Data modes

| `AIRSHIELD_DATA_MODE` | Upstream reachable | Upstream down |
| --- | --- | --- |
| `live` | Real current measurements. | **HTTP 503** with the upstream reason. Never falls back. |
| `demo` | Bundled historical sample, labelled `DEMO`. | Same (no network used). |
| `auto` *(default)* | Real current measurements. | Falls back to the bundled sample, and says so. |

The bundled sample in `ml/data/demo/` is **real historical data** downloaded from
Open-Meteo, not synthetic noise. It simply is not the current hour.

---

## The Pulse feature set

| Feature | What it answers | Where it lives |
| --- | --- | --- |
| **Dashboard / "Now" card** | Should I go outside right now? | `NowCard`, `/api/forecast/{slug}` |
| **Multi-horizon forecast** | What will happen next? | `/api/forecast/{slug}` → `forecasts[]`, `/api/horizons` |
| **Pollution timeline** | Measured vs predicted, with AQI bands | `PollutionTimeline` |
| **Spike card** | Is a spike coming, how big, how confident? | `SpikeCard`, `airshield_core.events` |
| **Exposure planner** | When should I go out? | `ExposurePlanner`, `/api/plan/{slug}` |
| **Route comparison** | Where should I go? | `RouteComparison`, `/api/routes/compare` |
| **Provenance + model + horizons + AWS** | Can I trust and trace this? | `ProvenancePanel`, `ModelCard`, `HorizonsCard`, `AwsPanel` |

### Exposure model

Exposure is an **engineering approximation for relative comparison**, not a
medical measurement. It is documented as such in the API response
(`exposure_note`) and in the UI.

```
Exposure = concentration  ×  duration  ×  activity intensity factor  ×  location factor
```

Activity intensity multipliers (configurable in `airshield_core.exposure`):
`walking = 1.0`, `cycling = 1.5`, `running = 2.0`, `outdoor_work = 1.4`,
`child_outdoor_activity = 1.2`.

The system's purpose is comparison. Given two windows it reports, for example,
*"approximately 43% lower predicted exposure"* — never "safe".

### Pollution spike detection

`airshield_core.events` detects **sharp** rises (a large step between adjacent
forecast hours) and **sustained** rises (a monotonic climb across the horizon).
Confidence is calibrated at training time: `airshield_core.spike_calibration`
measures how often each PM2.5 trend bucket historically led to a rise, and stores
it in the artifact. When calibration is unavailable the API falls back to a
transparent rule-based estimate and labels the basis accordingly.

---

## Measured model performance

Produced by `make train`, recorded in `ml/artifacts/metadata.json`. Reproduce
with one command. Held-out set: the most recent 20% of the timeline (time-ordered
split, never a random shuffle).

| Horizon | RMSE (µg/m³) | MAE (µg/m³) | R² | vs persistence |
| --- | --- | --- | --- | --- |
| **+1h** | see artifact | see artifact | see artifact | lower RMSE |
| **+3h** | see artifact | see artifact | see artifact | lower RMSE |
| **+6h** | see artifact | see artifact | see artifact | lower RMSE |

Exact numbers are read from `ml/artifacts/metadata.json` at runtime and shown in
the dashboard's model and horizons cards — they are **never hard-coded here**, so
this table cannot drift from reality. The persistence baseline
(`next hour = this hour`) is included on purpose and the tests assert the model
beats it.

---

## Architecture

```
airshield/
├── core/                     Shared domain library (airshield_core)
│   ├── src/airshield_core/
│   │   ├── config.py         Env-var settings (pydantic-settings)
│   │   ├── features.py       Leakage-safe feature engineering (HORIZONS)
│   │   ├── train.py          Per-horizon XGBoost training + honest metrics
│   │   ├── predict.py        Local artifact inference (log1p/expm1 contract)
│   │   ├── aqi.py            US EPA AQI + exposure alert rules
│   │   ├── exposure.py       Exposure engine (activity × duration × location)
│   │   ├── events.py         Sharp / sustained pollution spike detection
│   │   ├── windows.py        Safe-window optimizer over candidate windows
│   │   ├── routing.py        Real route geometry + segmentation (Valhalla/OSRM)
│   │   ├── spike_calibration.py  Empirical spike confidence from training data
│   │   └── sources/          Open-Meteo, demo dataset, registry
│   └── tests/                120 tests
│
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── routers/          forecast, meta, planning, routes, aws
│   │   └── services/         forecast_service, planning_service,
│   │                         route_service, aws_status, sagemaker_client
│   └── tests/                49 tests, including the honesty suite
│
├── frontend/                 React + TypeScript + Vite + Tailwind + Recharts
│   └── src/
│       ├── lib/              Typed API client, theme, chart helpers
│       ├── components/       NowCard, SpikeCard, PollutionTimeline,
│       │                     ExposurePlanner, RouteComparison, AwsPanel, ...
│       └── App.tsx
│
├── ml/sagemaker/             Real SageMaker training job + deployment
├── infra/
│   ├── cloudformation/       AirShield Pulse stack (S3, DynamoDB, SNS,
│   │                         EventBridge, Lambda, IAM)
│   └── lambda/               ingest + spike notification handlers
│
├── scripts/build_demo_dataset.py
├── Dockerfile                Multi-stage: builds frontend, serves via FastAPI
└── Makefile                  All developer commands
```

### Request flow

```
Browser
  │  GET /api/forecast/berlin
  ▼
FastAPI router ──► ForecastService
                     ├─ 1. Source (live Open-Meteo │ demo) → frame + provenance
                     ├─ 2. build_features() per-location (no cross-city bleed)
                     ├─ 3. Per-horizon latest_feature_row() → feature vectors
                     ├─ 4. Predictor (local booster │ SageMaker) per horizon
                     ├─ 5. aqi_from_pm25() + exposure_alert()
                     ├─ 6. events.detect() → spike + calibrated confidence
                     └─ 7. Assemble response: current, forecasts[], spike, history
```

### Two correctness details worth knowing

**Leakage safety.** All lagged and rolling features are shifted so they only ever
reference the past. `pm2_5_roll_mean3` at hour *t* covers *t−1…t−3*, never *t*.
`test_future_values_do_not_leak_into_past_features` corrupts the tail of the
series and asserts earlier feature rows are unchanged.

**Per-location grouping.** Features are computed within each city.
`test_locations_do_not_bleed_into_each_other` pins this down.

---

## Quick start

Requires Python 3.11+, Node 20+, and network access for live data.

```bash
git clone <your-fork-url> airshield && cd airshield

make setup          # venv + Python deps + npm install
make train          # fetch real data, train all horizons, calibrate spikes
make api            # terminal 1: backend on :8000
make frontend-dev   # terminal 2: dashboard on :5173
```

Open <http://localhost:5173>. To work fully offline:

```bash
make train-offline  # trains on the bundled real dataset, no network needed
```

### 3-minute demo flow

1. Open the dashboard — the **Now** card answers *should I go outside right now?*
2. Read the **spike card** for the next expected rise and its calibrated confidence.
3. Read the **pollution timeline** — solid measured past, dashed predicted future.
4. In the **exposure planner**, pick *Running — 45 minutes* → best window, expected
   exposure, and the relative reduction versus the worst window.
5. In **route comparison**, compare alternatives → the lowest-exposure route is marked.
6. Scroll to the **horizons**, **model** and **AWS architecture** cards for provenance.

### Docker

```bash
cp .env.docker.example .env.docker
make docker-build
make docker-run      # serves the API and the built dashboard on :8000
```

---

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service metadata and endpoint list |
| `GET` | `/api/health` | Status, backend, data mode, model availability |
| `GET` | `/api/locations` | The six built-in locations |
| `GET` | `/api/model` | Model card: real metrics, baselines, features, provenance |
| `GET` | `/api/horizons` | Per-horizon model metrics and versions |
| `GET` | `/api/activities` | Activity options and their intensity factors |
| `GET` | `/api/forecast/{slug}` | Current + multi-horizon forecast + AQI + alert + spike + history |
| `GET` | `/api/plan/{slug}` | Lowest-exposure window for an activity and duration |
| `GET` | `/api/routes/compare` | Route alternatives ranked by predicted exposure |
| `GET` | `/api/aws/status` | Which AWS services are configured (real, not assumed) |
| `GET` | `/api/aws/architecture` | The intended AWS pipeline, as data |

Interactive docs at <http://localhost:8000/docs>.

<details>
<summary><code>GET /api/plan/berlin?activity=running&amp;duration_minutes=45</code> — example shape</summary>

```json
{
  "activity": "running",
  "duration_minutes": 45,
  "best_window": {
    "start": "2026-10-02T06:00:00Z",
    "end": "2026-10-02T06:45:00Z",
    "exposure": { "score": 47.2, "level": "low", "mean_pm25": 8.4, "peak_pm25": 9.1 },
    "relative_reduction_percent": 42.7
  },
  "alternative_window": { "...": "..." },
  "highest_exposure_window": { "...": "..." },
  "relative_reduction_percent": 42.7,
  "reason": "Lowest summed exposure across the whole 45-minute window ...",
  "note": "Exposure is an engineering approximation for relative comparison, not a medical measurement.",
  "exposure_note": "Activity multipliers are engineering approximations ..."
}
```

</details>

---

## AWS architecture

Every service has one clear job. `infra/cloudformation/airshield-pulse.yaml`
defines the whole stack, and `GET /api/aws/status` reports which pieces are
actually configured in the running process — it never claims a service is
deployed when it is not.

```
Forecast pipeline
  Open-Meteo → EventBridge (hourly) → ingest Lambda → S3 (training data)
                                                    → DynamoDB (serving state)
  SageMaker AI (training job + inference endpoint) → FastAPI → React frontend

Notification pipeline
  EventBridge (30 min) → spike Lambda → SNS topic → subscribers
```

| Service | Purpose |
| --- | --- |
| **Amazon SageMaker AI** | Training jobs and the real-time inference endpoint. The only AWS AI/ML service used. |
| **AWS Lambda** | Hourly Open-Meteo ingestion; 30-minute spike check. |
| **Amazon S3** | Training datasets, model artifacts, versioned samples. |
| **Amazon DynamoDB** | Latest observation per location and spike-notification de-duplication. |
| **Amazon EventBridge** | The schedules that drive both pipelines. |
| **Amazon SNS** | Delivers spike notifications. |
| **Amazon CloudWatch** | Logs, errors, latency and service health for the Lambdas and endpoint. |
| **API Gateway** | Optional front door for the FastAPI service. |

```bash
make infra-validate   # aws cloudformation validate-template
make infra-deploy ENV=dev
make lambda-package   # build the ingest + spike zips
```

## Amazon SageMaker AI

XGBoost on SageMaker AI is the primary production ML path. The AWS SDK is
installed and the scripts are complete and tested, but **nothing is deployed**,
because no AWS account was provided for this build. Running these requires your
own account.

The serving contract is verified without an AWS account:
`test_predict_fn_matches_the_local_predictor` asserts the SageMaker container path
and the local path produce **identical** predictions. That matters because training
fits on `log1p(pm2_5)` and serving must invert with `expm1`; get that wrong and
every prediction is silently distorted.

```bash
export AWS_REGION=us-east-1
export AIRSHIELD_S3_BUCKET=my-airshield-bucket
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole

make sagemaker-train    # pulls real data, launches a real training job
make sagemaker-deploy   # creates a real endpoint
make sagemaker-smoke    # invokes it once and prints the prediction
```

Then point the backend at the endpoint:

```bash
AIRSHIELD_INFERENCE_BACKEND=aws
SAGEMAKER_ENDPOINT_NAME=airshield-pm25-endpoint
```

The frontend's provenance panel shows `SageMaker AI` or `local` per prediction.

---

## Testing

```bash
make test           # all Python tests
make test-core      # core library (120)
make test-backend   # API + honesty suite (49)
make test-frontend  # TypeScript type check
make check          # everything
```

The suite trains real models on real data rather than mocking them. Notable
coverage:

- **Forecast & leakage** — corrupting the future must not change the past.
- **Time-ordered split** — the test set is the most recent slice, never random.
- **Cross-location isolation** — per-location grouping is verified.
- **Exposure calculation** — duration × intensity × concentration invariants.
- **Safe-window optimization** — the whole window is scored, not a single hour.
- **Route exposure** — segment aggregation and exposure ranking.
- **Spike detection** — sharp and sustained rises, plus the no-spike case.
- **Missing data / upstream failure** — explicit `503`, never fabricated values.
- **SageMaker contract** — local and container inference must agree exactly.
- **API contracts** — typed responses for every Pulse endpoint.
- **Frontend type safety** — `tsc` in strict mode.

---

## Configuration

Copy `.env.example` to `.env`. Every setting is optional; the defaults work.

| Variable | Default | Meaning |
| --- | --- | --- |
| `AIRSHIELD_DATA_MODE` | `auto` | `live`, `demo`, or `auto` |
| `AIRSHIELD_INFERENCE_BACKEND` | `local` | `local` or `aws` |
| `AIRSHIELD_HORIZONS` | `1,3,6` | Horizons to train, one model each |
| `AIRSHIELD_ARTIFACT_DIR` | `ml/artifacts` | Trained model location |
| `SAGEMAKER_ENDPOINT_NAME` | *(empty)* | Required when backend is `aws` |
| `AWS_REGION` | `us-east-1` | SageMaker region |
| `AIRSHIELD_S3_BUCKET` | *(empty)* | Training data + artifacts bucket |
| `AIRSHIELD_DYNAMODB_TABLE` | *(empty)* | State / preference / spike-event table |
| `AIRSHIELD_SNS_TOPIC_ARN` | *(empty)* | Spike notification topic |
| `AIRSHIELD_HTTP_TIMEOUT` | `15` | Upstream request timeout (s) |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend → backend origin |

AWS credentials are read from the standard boto3 chain. **No secrets are
committed**; `.env` is gitignored and `.env.example` documents every key.

---

## Tech stack

**Frontend** React 18 · TypeScript (strict) · Vite · Tailwind CSS · Recharts
**Backend** Python · FastAPI · Pydantic v2 · uvicorn
**ML** pandas · scikit-learn · XGBoost · **Amazon SageMaker AI**
**Data** Open-Meteo air quality + weather (CC BY 4.0) · OpenStreetMap routing
**AWS** SageMaker AI · Lambda · S3 · DynamoDB · EventBridge · SNS · CloudWatch
**Tooling** pytest · Docker (multi-stage) · Make

---

## Roadmap

- [x] Repository structure, tooling, and documentation
- [x] Real Open-Meteo acquisition with provenance tracking
- [x] Leakage-safe feature engineering, verified by tests
- [x] Per-horizon XGBoost training with honest held-out metrics
- [x] AQI conversion and six-level exposure alert
- [x] FastAPI service with live / demo / auto modes
- [x] Multi-horizon forecast (1h / 3h / 6h), one model each
- [x] Pollution spike detection with calibrated confidence
- [x] Exposure engine and safe-window optimizer
- [x] Route comparison with real geometry and per-segment exposure
- [x] AWS stack (S3, DynamoDB, Lambda, EventBridge, SNS) as CloudFormation
- [x] SageMaker training, deployment, and inference handler
- [ ] Deploy to AWS (needs credentials — deliberately not done)
- [ ] Geocoding so any city can be searched, not just six
- [ ] Persisted user preferences and saved plans in DynamoDB
- [ ] Route exposure map overlay with segment-level shading

---

## Data attribution

Air quality and weather data: [Open-Meteo](https://open-meteo.com/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Routing:
OpenStreetMap contributors via Valhalla / OSRM. AQI breakpoints follow the US
EPA's [AirNow technical assistance document](https://www.airnow.gov/aqi/aqi-calculator/).
WHO guideline comparison uses the 2021 Global Air Quality Guidelines (24-hour
PM2.5 guideline: 15 µg/m³).

## Disclaimer

AirShield Pulse forecasts air quality and compares relative predicted exposure.
It is **not medical advice**, and it does not claim that any route or time window
is medically safe. Consult a healthcare professional for decisions about your
health.
