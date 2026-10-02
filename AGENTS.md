# AGENTS.md — working notes for automated agents and contributors

AirShield Pulse predicts PM2.5 (1h / 3h / 6h), then turns it into exposure
avoidance: when to go outside and which route exposes you less. Read this before
changing anything.

## Non-negotiables

**Never fabricate data, predictions, metrics or AWS responses.**
This is the defining constraint of the project.

- No hard-coded or interpolated "sample" values that get served as real output.
- No invented model accuracy. Metrics are computed on a held-out split and read
  back from `ml/artifacts/metadata.json`.
- No invented spike confidence. It comes from `spike_calibration.json`
  (`empirical_calibration`) or the transparent rule fallback (`rule_based_estimate`).
  The `confidence_basis` field must always name which.
- No invented AWS responses. If there is no credentials or endpoint, report a
  failure. `/api/aws/status` reports only what is really configured.
- Demo data must always be labelled. If you add a code path that serves bundled
  data, set `SourceInfo.mode = "demo"` and populate the response `notice` field.

`backend/tests/test_honesty.py` enforces this. Keep it passing.

## Language rules (product-facing copy)

These are product requirements, not style preferences.

- Never call a route or window **safe**. Use "lower predicted exposure".
- Never present spike signals as causes. They are "possible contributing signals".
- Exposure multipliers are **engineering approximations for relative comparison**,
  not medical measurements. Every exposure response repeats this (`exposure_note`).

## Layout and dependency direction

```
core/src/airshield_core   domain logic; imports nothing from backend or frontend
backend/app               HTTP layer; imports core
frontend/src              UI; talks to backend only via src/lib/api.ts
ml/sagemaker              real SageMaker train / deploy / serve
infra/cloudformation      the AWS stack (S3, DynamoDB, SNS, Lambda, EventBridge)
infra/lambda              ingest + spike-notification handlers (tested in infra/tests)
```

`core` must stay free of FastAPI and HTTP-server concerns so it remains testable
and reusable inside the SageMaker training container.

## The Pulse modules (core)

- `exposure.py` — exposure engine + `compare_exposures()` relative reduction.
- `events.py` — sharp / sustained spike detection; consumes `spike_calibration`.
- `windows.py` — safe-window optimizer; scores the **whole** candidate window.
- `routing.py` — real Valhalla/OSRM geometry, segmentation, via-point detours.
- `route_exposure.py` — per-segment aggregation and route ranking.
- `spike_calibration.py` — empirical spike frequency learned during training.

Backend services: `forecast_service`, `planning_service`, `route_service`,
`aws_status`. Routers: `forecast`, `meta`, `planning`, `routes`, `aws`.

## Commands

```bash
make setup            # venv + python deps + npm install
make train            # fetch real data, train all horizons, calibrate spikes
make train-offline    # train from bundled real dataset (no network)
make api              # backend on :8000 (reload)
make frontend-dev     # dashboard on :5173
make test             # core + backend + infra pytest
make test-frontend    # tsc type check
make check            # everything
make build-demo-data  # refresh ml/data/demo from Open-Meteo
make docker-build && make docker-run
make infra-validate   # real CloudFormation validate-template (needs AWS creds)
make infra-deploy     # deploy the stack via boto3 (needs AWS creds)
make lambda-package   # standalone handler zips
make template-sync    # re-embed handlers into the template after editing them
```

Always run `make check` before finishing a change.

## Correctness rules that are easy to break

1. **Leakage.** Every lag/rolling feature must read only the past. Rolling means
   use `shift(1).rolling(n)` so hour *t* never sees itself. `wx_next_*` may read
   *t+1* for **weather only** — never PM2.5.
2. **Per-location grouping.** Call `build_features(..., group_column="location_name")`
   when a frame holds more than one city. Otherwise lags bleed across cities.
3. **log1p/expm1.** Training fits `log1p(pm2_5)`; both `LocalPredictor` and
   `ml/sagemaker/inference.py` must invert with `expm1`. If you change one, change
   the other and keep `test_predict_fn_matches_the_local_predictor` passing.
4. **Time-ordered splits.** Never use a random split for the train/test boundary.
5. **Feature order.** The booster is trained on an ordered feature list stored in
   metadata. Serving validates order and raises on mismatch.
6. **AQI single source of truth.** Compute it in `airshield_core.aqi` only; the
   `aqi` and `alert` response blocks must agree.
7. **One model per horizon.** Never stretch the 1h model to serve 3h/6h. Each
   horizon has its own artifact; a missing one appears in `unavailable_horizons`.
8. **Exposure is duration-weighted.** Route and window comparisons weight by time,
   so a cleaner-but-slower option can still win. Do not "fix" that to a naive
   per-hour average.
9. **Routes are real.** `routing.py` only ever returns engine geometry. If no
   engine answers, raise `RoutingError` — never synthesise a path.

## Testing conventions

- Tests train real models on real data. Do not mock predictions.
- Prefer asserting a real invariant (leakage, feature order, baseline comparison)
  over asserting a specific number, which will drift as data changes.
- If a numeric assertion is unavoidable, allow a tolerance and say why.
- `synthetic_frame` in `core/tests/conftest.py` exists only to test *mechanics*.
  Never report metrics derived from it.

## Frontend conventions

- `src/lib/api.ts` is the only backend caller; surface the backend's error
  `detail` to users.
- Keep measured vs predicted visually distinct (solid vs dashed).
- Provenance and demo banners stay visible, never behind a toggle.
- `tsc` runs in strict mode with `noUnusedLocals`; keep it clean.

## Environment and secrets

- All settings go through `airshield_core.config.Settings`, `AIRSHIELD_` prefix
  (plus explicit aliases such as `SAGEMAKER_ENDPOINT_NAME`, `AWS_REGION`).
- Add new keys to `.env.example` with a comment; never commit a real `.env`.
- AWS credentials come from the standard boto3 chain. Do not hard-code them.
- AWS AI/ML usage is limited to **Amazon SageMaker AI**. Do not introduce
  Bedrock, Rekognition, Comprehend or other AWS AI/ML services.

## AWS status

Nothing is deployed. The SageMaker scripts are complete and their container logic
is tested, but no AWS account was provided. Do not describe the project as
deployed, and do not claim SageMaker accuracy that was not measured — the
reported metrics come from local training on the same data.

## Docker

The image trains the model during build from the bundled dataset, so it works
with no network at runtime. Trained artifacts are gitignored and therefore cannot
be copied from the build context — do not try. `AIRSHIELD_SERVE_FRONTEND=true`
mounts the built dashboard at `/`; `/api/*` routes always take precedence.
