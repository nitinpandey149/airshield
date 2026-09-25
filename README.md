# 🛡️ AirShield

**Know the air before you step outside.**

AirShield predicts the PM2.5 concentration for **the next hour** and turns that
number into a plain-language air-exposure alert — so you know whether it is a good
time for a run, when to keep a child with asthma indoors, or when to mask up.

Built for the **AIR / Environmental sustainability** hackathon track.

---

## What it actually does

You pick a city. AirShield:

1. Fetches **real hourly air-quality and weather measurements** for that location
   from [Open-Meteo](https://open-meteo.com/) (public API, no key required).
2. Builds feature rows from that history (lagged PM2.5, rolling means, pollutant
   ratios, meteorology, and the next hour's forecast weather).
3. Runs an **XGBoost regression** model to predict PM2.5 one hour ahead.
4. Converts the prediction into a **US EPA AQI** value and a six-level exposure
   alert with separate advice for sensitive groups.
5. Shows all of it in a React dashboard, with a chart that visibly separates
   measured history from the predicted point.

The model is served either from a local artifact or from a **real
Amazon SageMaker AI endpoint** — the only AWS AI/ML service used in this project.

---

## Honesty guarantees

This project is deliberately explicit about what is real and what is not:

| Guarantee | How it is enforced |
| --- | --- |
| No invented measurements | Every reading comes from a live HTTP response, or from the bundled dataset. |
| No invented predictions | Predictions come from the trained booster, or a real SageMaker invocation. |
| No invented accuracy | Metrics are computed on a held-out time slice and stored in the artifact; the API serves them from there. |
| No invented AWS responses | If a SageMaker endpoint is not configured, the API reports `degraded` and errors. It never guesses. |
| Demo data is always labelled | Responses in demo mode carry a `DEMO MODE` notice and `source.mode: "demo"`; the UI shows a banner. |

The test suite includes a dedicated [honesty test module](backend/tests/test_honesty.py)
that asserts these behaviours, including that a simulated upstream outage produces
a `503` rather than fabricated data.

### Data modes

| `AIRSHIELD_DATA_MODE` | Behaviour when the upstream API is reachable | Behaviour when it is not |
| --- | --- | --- |
| `live` | Real current measurements. | **HTTP 503**, with the upstream reason. Never falls back. |
| `demo` | Bundled historical sample, labelled `DEMO`. | Same (no network used). |
| `auto` *(default)* | Real current measurements. | Falls back to the bundled sample, and says so in the response. |

The bundled sample in `ml/data/demo/` is **real historical data** downloaded from
Open-Meteo, not synthetic noise. It simply is not the current hour. The manifest
records exactly which stations and window it covers.

---

## Measured model performance

These are the numbers produced by `make train-offline`, recorded in
`ml/artifacts/metadata.json`. They are **not** placeholders — reproduce them
yourself with one command.

Held-out set: the most recent 20% of the timeline (time-ordered split, never a
random shuffle, so no future information leaks into training).

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias |
| --- | --- | --- | --- | --- |
| **XGBoost (AirShield)** | **4.337** | **2.173** | **0.980** | −0.261 |
| Persistence baseline (`next hour = this hour`) | 5.601 | 2.896 | 0.966 | 0.000 |

**22.6% lower RMSE than the persistence baseline.**

The persistence baseline is included on purpose: for a one-hour-ahead air-quality
forecast, "assume nothing changes" is a genuinely strong baseline. A model that
cannot beat it is not worth shipping, so the test suite asserts that it does.

Training data: 12,954 usable rows (10,363 train / 2,591 test) drawn from the
bundled 13,104-row sample covering Berlin, New York, Delhi, Los Angeles,
São Paulo and Beijing.

Top predictive features by gain share: `pm10` (44.8%), `pm2_5` (31.8%),
`pm2_5_lag1` (18.5%), `pm2_5_roll_mean3` (1.6%).

---

## Architecture

```
airshield/
├── core/                     Shared domain library (airshield_core)
│   ├── src/airshield_core/
│   │   ├── config.py         Env-var settings (pydantic-settings)
│   │   ├── schema.py         Canonical column definitions + Observation model
│   │   ├── sources/          Data acquisition
│   │   │   ├── openmeteo.py  Real Open-Meteo air-quality + weather client
│   │   │   ├── demo.py       Bundled historical dataset reader
│   │   │   └── registry.py   The six built-in locations
│   │   ├── features.py       Leakage-safe feature engineering
│   │   ├── train.py          XGBoost training + honest metric reporting
│   │   ├── predict.py        Local artifact inference (log1p/expm1 contract)
│   │   └── aqi.py            US EPA AQI conversion + exposure alert rules
│   └── tests/                69 tests
│
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── main.py           App factory, CORS, static frontend mount
│   │   ├── deps.py           Predictor/service lifecycle
│   │   ├── routers/          forecast.py, meta.py
│   │   └── services/
│   │       ├── forecast_service.py   Orchestrates fetch -> features -> predict -> alert
│   │       └── sagemaker_client.py   Real SageMaker runtime invocation
│   └── tests/                29 tests, including the honesty suite
│
├── frontend/                 React + TypeScript + Vite + Tailwind + Recharts
│   └── src/
│       ├── lib/              Typed API client, theme, chart helpers
│       ├── components/       AlertCard, ForecastChart, AqiScale, ModelCard, ...
│       └── App.tsx
│
├── ml/
│   ├── sagemaker/            Real SageMaker training job + deployment
│   │   ├── launch_training_job.py   Launches a real training job
│   │   ├── train_entry.py           Runs inside the training container
│   │   ├── inference.py             Runs inside the endpoint container
│   │   ├── deploy.py                Creates a real endpoint
│   │   └── smoke_test.py            Invokes the endpoint once
│   ├── data/demo/            Bundled real historical dataset + manifest
│   └── artifacts/            Trained model output (gitignored)
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
                     │
                     ├─ 1. Source (live Open-Meteo │ demo dataset)
                     │       returns hourly frame + provenance metadata
                     ├─ 2. build_features()  (per-location lags: no cross-city bleed)
                     ├─ 3. latest_feature_row() -> feature vector + base_time
                     ├─ 4. Predictor  (local booster │ SageMaker endpoint)
                     ├─ 5. aqi_from_pm25() + exposure_alert()
                     └─ 6. Assemble response with source, notice, metrics, history
```

### Two correctness details worth knowing

**Leakage safety.** All lagged and rolling features are shifted so they only ever
reference the past. `pm2_5_roll_mean3` at hour *t* covers *t−1…t−3*, never *t*.
The test
[`test_future_values_do_not_leak_into_past_features`](core/tests/test_features.py)
corrupts the tail of the series and asserts that earlier feature rows are byte-for-byte
unchanged.

**Per-location grouping.** Features are computed within each city. Without
grouping, hour *t* in Berlin would pick up hour *t−1* from whichever city happened
to sort first — cross-city contamination that inflates validation scores.
[`test_locations_do_not_bleed_into_each_other`](core/tests/test_features.py)
pins this down.

---

## Quick start

Requires Python 3.11+, Node 20+, and network access for live data.

```bash
git clone <your-fork-url> airshield && cd airshield

make setup          # venv + Python deps + npm install
make train          # fetch real data and train the model (~2 min)
make api            # terminal 1: backend on :8000
make frontend-dev   # terminal 2: dashboard on :5173
```

Open <http://localhost:5173>.

To work fully offline, replace `make train` with:

```bash
make train-offline  # trains on the bundled real dataset, no network needed
```

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
| `GET` | `/api/model` | Model card: real metrics, baseline, features, provenance |
| `GET` | `/api/forecast/{slug}` | Prediction + AQI + alert + history + provenance |

Interactive docs at <http://localhost:8000/docs>.

<details>
<summary><code>GET /api/forecast/delhi</code> — example response</summary>

```json
{
  "location": { "slug": "delhi", "label": "Delhi, India", "...": "..." },
  "generated_at": "2026-09-25T08:20:18Z",
  "source": {
    "name": "Open-Meteo",
    "url": "https://open-meteo.com/",
    "mode": "live",
    "licence": "CC BY 4.0",
    "is_synthetic": false
  },
  "notice": null,
  "forecast": {
    "predicted_pm25": 34.75,
    "base_time": "2026-09-25T08:00:00Z",
    "target_time": "2026-09-25T09:00:00Z",
    "horizon_hours": 1,
    "model_version": "xgboost-pm25-1h-20260925T081725",
    "backend": "local",
    "model_metrics": { "rmse": 4.337, "mae": 2.173, "r2": 0.98, "n": 2591 }
  },
  "aqi": { "aqi": 99, "category": "Moderate", "band": "Moderate", "who_ratio": 2.32 },
  "alert": {
    "severity": "moderate",
    "headline": "Moderate air — OK for most people",
    "advice": "...",
    "sensitive_group_advice": "...",
    "horizon": "next 1 hour"
  },
  "history": [
    { "time": "2026-09-25T07:00:00Z", "pm2_5": 38.1, "predicted": false },
    { "time": "2026-09-25T08:00:00Z", "pm2_5": 34.75, "predicted": true }
  ]
}
```

</details>

---

## Amazon SageMaker AI

XGBoost regression on SageMaker AI is the primary ML path. The AWS SDK is
installed and the deployment scripts are complete and syntax-checked, but
**nothing is deployed**, because no AWS credentials or S3 bucket were provided
for this build. Running these commands requires your own AWS account.

The serving contract is covered by tests that run against the real trained
booster, so the container logic is verified without an AWS account:

```bash
cd core && ../.venv/bin/python -m pytest tests/test_sagemaker_handler.py -v
```

These assert, among other things, that
`test_predict_fn_matches_the_local_predictor` — the SageMaker container path and
the local path produce **identical** predictions. That is the check that matters,
because training fits on `log1p(pm2_5)` and serving must invert with `expm1`; get
that wrong and every prediction is silently distorted.

To run the real pipeline:

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

Each script prints only values returned by the AWS API. If a call fails, the
script exits non-zero with the real error.

---

## Testing

```bash
make test           # all Python tests (98)
make test-core      # core library (69)
make test-backend   # API, honesty, static-serving suite (29)
make test-frontend  # TypeScript type check
make check          # everything
```

The suite trains real models on real data rather than mocking them. Notable
coverage:

- **AQI boundaries** — every EPA band transition at its exact breakpoint.
- **Feature leakage** — corrupting the future must not change the past.
- **Cross-city isolation** — per-location grouping is verified explicitly.
- **Baseline comparison** — asserts the model beats persistence.
- **Honesty** — upstream outage yields a 503, never invented data.
- **SageMaker contract** — local and container inference must agree exactly.
- **Secret leakage** — no API response may contain a credential.

---

## Configuration

Copy `.env.example` to `.env`. Every setting is optional; the defaults work.

| Variable | Default | Meaning |
| --- | --- | --- |
| `AIRSHIELD_DATA_MODE` | `auto` | `live`, `demo`, or `auto` |
| `AIRSHIELD_INFERENCE_BACKEND` | `local` | `local` or `aws` |
| `AIRSHIELD_ARTIFACT_DIR` | `ml/artifacts` | Trained model location |
| `SAGEMAKER_ENDPOINT_NAME` | *(empty)* | Required when backend is `aws` |
| `AWS_REGION` | `us-east-1` | SageMaker region |
| `AIRSHIELD_HTTP_TIMEOUT` | `15` | Upstream request timeout (s) |
| `AIRSHIELD_CORS_ORIGINS` | `localhost:5173` | Comma-separated allowed origins |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend → backend origin |

AWS credentials are read from the standard boto3 chain. **No secrets are
committed**; `.env` is gitignored and `.env.example` documents every key.

---

## Tech stack

**Frontend** React 18 · TypeScript (strict) · Vite · Tailwind CSS · Recharts
**Backend** Python · FastAPI · Pydantic v2 · uvicorn
**ML** pandas · scikit-learn · XGBoost · **Amazon SageMaker AI**
**Data** Open-Meteo air quality + weather (CC BY 4.0)
**Tooling** pytest · Docker (multi-stage) · Make

---

## Roadmap

- [x] Repository structure, tooling, and documentation
- [x] Real data acquisition from Open-Meteo with provenance tracking
- [x] Bundled real historical dataset + manifest
- [x] Leakage-safe feature engineering, verified by tests
- [x] XGBoost training with honest held-out metrics and persistence baseline
- [x] AQI conversion and six-level exposure alert
- [x] FastAPI service with live / demo / auto modes
- [x] React dashboard with chart, AQI scale, model card, provenance panel
- [x] SageMaker training, deployment, and inference handler
- [x] Docker image and developer Makefile
- [ ] Deploy to AWS (needs credentials — deliberately not done)
- [ ] Multi-hour horizon (3h / 6h) and a separate hourly model per horizon
- [ ] 7-day rolling forecast with confidence intervals
- [ ] Geocoding so any city can be searched, not just six
- [ ] Personal exposure profiles (asthma, outdoor work, exercise windows)

---

## Data attribution

Air quality and weather data: [Open-Meteo](https://open-meteo.com/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). AQI breakpoints follow
the US EPA's [AirNow technical assistance document](https://www.airnow.gov/aqi/aqi-calculator/).
WHO guideline comparison uses the 2021 Global Air Quality Guidelines
(24-hour PM2.5 guideline: 15 µg/m³).

## Disclaimer

AirShield forecasts air quality; it is not medical advice. Consult a healthcare
professional for decisions about your health.
