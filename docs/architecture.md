# AirShield — architecture notes

## Design principles

1. **Nothing is fabricated.** Every number rendered in the UI traces to a real
   upstream HTTP response, the bundled real dataset, or the trained booster.
   Where something is not available, the system reports a failure rather than
   substituting a plausible value.
2. **Provenance travels with the data.** The `SourceInfo` object attached to every
   forecast response states the upstream name, URL, licence, fetch time, and
   whether the data is live or a historical sample.
3. **One source of truth for derived values.** AQI is computed once in
   `airshield_core.aqi` and reused by both the `aqi` and `alert` response blocks,
   so they can never disagree.
4. **The baseline is always reported.** A 1-hour forecast is easy to fake: repeat
   the current value. Persistence metrics are therefore computed and published
   alongside the model's, and a test asserts the model beats them.

## Layers

```
core/src/airshield_core      Domain logic. No FastAPI, no HTTP server, no UI.
  schema.py                  Canonical column names + Observation value object
  sources/                   Data acquisition (Open-Meteo live, bundled demo)
  features.py                Feature engineering; leakage-safe by construction
  train.py                   Training + metric/baseline reporting + artifact I/O
  predict.py                 Artifact loading and inference
  aqi.py                     AQI conversion and alert rules
  config.py                  Environment-backed settings

backend/app                  HTTP layer. Thin: validation, orchestration, errors.
  deps.py                    Predictor/service lifecycle (built once, reused)
  services/forecast_service  Orchestrates the core pipeline
  services/sagemaker_client  Amazon SageMaker runtime invocation
  routers/                   forecast + meta endpoints

frontend/src                 Presentation. Typed against the backend schema.
ml/sagemaker                 Real SageMaker training, deployment, serving
```

The dependency direction is strictly one-way: `frontend -> backend -> core`.
`core` imports nothing from `backend`.

## The prediction pipeline

`ForecastService.forecast(slug)`:

1. **Fetch** — `fetch_frame_with_fallback()` returns an hourly DataFrame plus
   `SourceInfo`. In `live` mode a failure raises `UpstreamError`; in `auto` it
   falls back to the bundled dataset and records that in `SourceInfo.mode`.
2. **Feature build** — `build_features()` adds lags, rolling means, deltas and
   ratios, **grouped by location**. `latest_feature_row()` then extracts the row
   at hour *t* (the most recent observed hour) whose target is *t+1*.
3. **Predict** — a `Predictor` (local booster or SageMaker client) returns PM2.5
   for *t+1*.
4. **Interpret** — `aqi_from_pm25()` produces the AQI and category;
   `exposure_alert()` produces severity, headline, general advice and
   sensitive-group advice.
5. **Assemble** — the response carries location, source, notice, forecast, aqi,
   alert and the history series (with the predicted point appended and flagged).

## Feature engineering and leakage

The `base` pollution features (`pm2_5`, `pm10`, …) are lagged or differenced;
none is used raw as a contemporaneous predictor of a *later* target. Formally, no
feature at row *t* reads a value from row *t* or later.

`pm2_5_roll_mean3` uses `shift(1).rolling(3)`, so at *t* it covers *t−3…t−1*.
The test `test_future_values_do_not_leak_into_past_features` multiplies the tail of
the series by 100 and asserts that every feature row before the corruption is
identical. This is what makes the reported R² trustworthy.

`wx_next_*` features read row *t+1* for weather only. That is legitimate because
weather forecasts are available at prediction time, and the live fetch explicitly
requests forecast weather. PM2.5 is never read from *t+1*.

### Per-location grouping

`build_features(frame, group_column="location_name")` applies every shift and
roll **within each location**. Without this, the previous hour of a different city
would leak into the current city's lags whenever multiple locations were trained
together — inflating validation scores.

## Split strategy

`train_model` splits at a quantile of the sorted timeline (80/20), not randomly.
A random split would place future rows in the training set and near-duplicate
adjacent hours across the boundary, producing a flattering but meaningless R².
The split boundary is recorded in the artifact metadata and asserted by tests.

`test_split_is_time_ordered` allows equality at the boundary, because with
several locations trained together a single wall-clock hour occupies several rows
and the split can fall inside it.

## The log1p / expm1 contract

Training fits on `log1p(pm2_5)` to stabilise the variance of a right-skewed
target. Serving must invert with `expm1`. This pair is the easiest thing to break
silently: predictions would become wrong but still finite and plausible.

Two independent guards:

- `LocalPredictor` and `ml/sagemaker/inference.py` both apply `expm1`.
- `test_predict_fn_matches_the_local_predictor` asserts the SageMaker container
  path and the local path return the same value to within `1e-6`.

## SageMaker integration

Only Amazon SageMaker AI is used as an AWS AI/ML service.

- `launch_training_job.py` fetches real data locally, stages a CSV to S3, and
  starts a real training job. It prints only values the AWS API returned.
- `train_entry.py` runs in the managed XGBoost container, calls the same
  `train_model()`, and writes `model.tar.gz` to `/opt/ml/model/`.
- `inference.py` implements the four SageMaker handler functions; it is exercised
  by tests against a real booster, so the container path is verified without an
  AWS account.
- `deploy.py` creates a real endpoint; `smoke_test.py` invokes it once.

`AIRSHIELD_INFERENCE_BACKEND=aws` with an empty `SAGEMAKER_ENDPOINT_NAME` makes
`/api/health` report `degraded` with the missing variable named, rather than
silently serving local predictions under an AWS label.

## Failure behaviour

| Situation | Behaviour |
| --- | --- |
| Live fetch fails, `data_mode=live` | `503` with the upstream reason |
| Live fetch fails, `data_mode=auto` | `200` on bundled data, `notice` set, `source.mode="demo"` |
| Model artifact missing | `/api/health` `degraded`; forecast `500` naming `make train` |
| SageMaker endpoint not configured | `degraded` naming `SAGEMAKER_ENDPOINT_NAME` |
| SageMaker returns an unknown shape | `SageMakerError`, never a guessed value |
| Unknown location slug | `404` listing the valid slugs |

`backend/tests/test_honesty.py` asserts each row of this table.

## Configuration

All settings flow through `airshield_core.config.Settings` (pydantic-settings,
`AIRSHIELD_` prefix plus explicit aliases). `get_settings()` is `lru_cache`d;
`reset_settings_cache()` lets tests patch the environment. Secrets are never
hard-coded: AWS credentials come from the standard boto3 chain, and `.env` is
gitignored while `.env.example` documents every key.

## Frontend notes

- `src/lib/api.ts` is the only place that talks to the backend, and it surfaces
  the backend's own error `detail` instead of a generic message.
- `ForecastChart` draws measured and predicted series with distinct styling
  (solid vs dashed) so the boundary between observation and forecast is visually
  unambiguous, and labels the predicted point.
- `ProvenancePanel` is always visible and never collapsed, so a demo response
  cannot be mistaken for a live one.
- `ModelCard` renders the real held-out metrics next to the persistence baseline,
  expanding to training window, feature importances and library versions.
