# AGENTS.md — working notes for automated agents and contributors

AirShield predicts next-hour PM2.5 and turns it into an air-exposure alert.
Read this before changing anything.

## Non-negotiables

**Never fabricate data, predictions, metrics or AWS responses.**
This is the defining constraint of the project.

- No hard-coded or interpolated "sample" values that get served as real output.
- No invented model accuracy. Metrics are computed on a held-out split and read
  back from `ml/artifacts/metadata.json`.
- No invented AWS responses. If there is no credentials or endpoint, report a
  failure.
- Demo data must always be labelled. If you add a code path that serves bundled
  data, set `SourceInfo.mode = "demo"` and populate the response `notice` field.

`backend/tests/test_honesty.py` enforces this. Keep it passing.

## Layout and dependency direction

```
core/src/airshield_core   domain logic; imports nothing from backend or frontend
backend/app               HTTP layer; imports core
frontend/src              UI; talks to backend only via src/lib/api.ts
ml/sagemaker              real SageMaker train / deploy / serve
```

`core` must stay free of FastAPI and HTTP-server concerns so it remains testable
and reusable inside the SageMaker training container.

## Commands

```bash
make setup            # venv + python deps + npm install
make train            # fetch real data, train, write ml/artifacts
make train-offline    # train from bundled real dataset (no network)
make api              # backend on :8000 (reload)
make frontend-dev     # dashboard on :5173
make test             # core + backend pytest
make test-frontend    # tsc type check
make check            # everything
make build-demo-data  # refresh ml/data/demo from Open-Meteo
make docker-build && make docker-run
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
