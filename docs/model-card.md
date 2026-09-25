# AirShield — model card

## Overview

| | |
| --- | --- |
| Task | Regression: predict hourly mean PM2.5 concentration one hour ahead |
| Target | `pm2_5` at hour *t+1*, in µg/m³ |
| Algorithm | XGBoost gradient-boosted trees |
| Primary platform | Amazon SageMaker AI (the only AWS AI/ML service used) |
| Local fallback | The same booster, persisted and run in-process |
| Horizons | 1 hour |

## Intended use

Give an individual a short-horizon, plain-language signal about outdoor air
quality so they can time a run, a commute, or outdoor play, and decide whether a
sensitive person should limit exposure.

**Not intended for:** clinical decisions, regulatory reporting, long-range
planning, or any use where an error of a few µg/m³ carries legal or medical
consequence. AirShield forecasts; it does not diagnose or prescribe.

## Training data

Real hourly measurements from the Open-Meteo air-quality and weather APIs
(CC BY 4.0), covering six cities:

| City | Slug |
| --- | --- |
| Berlin, Germany | `berlin` |
| New York, United States | `new-york` |
| Delhi, India | `delhi` |
| Los Angeles, United States | `los-angeles` |
| São Paulo, Brazil | `sao-paulo` |
| Beijing, China | `beijing` |

The bundled sample (`ml/data/demo/demo_hourly.csv`) holds 13,104 real rows over a
frozen window; `manifest.json` records the source, window, licence and station
coordinates. It is real data, not synthetic — "demo" means historical rather than
current. `make train` instead pulls fresh history for the days configured in
`AIRSHIELD_TRAIN_DAYS`.

## Features

37 features, all computed within each location to prevent cross-city leakage:

| Group | Examples |
| --- | --- |
| Pollution lags | `pm2_5_lag1`, `pm2_5_lag3`, `pm2_5_lag24`, same for `pm10` |
| Rolling means (past only) | `pm2_5_roll_mean3`, `_24`, `_168` |
| Short-term change | `pm2_5_delta1`, `pm2_5_delta3`, ratios |
| Pollutant context | `pm10`, `nitrogen_dioxide`, `ozone` and their ratios |
| Meteorology | temperature, humidity, wind speed/direction, pressure, precipitation |
| Next-hour weather | `wx_next_*` (forecast weather only, never future PM2.5) |
| Calendar | hour-of-day, day-of-week, cyclical encodings |

## Evaluation

Time-ordered 80/20 split: the most recent 20% of the timeline is held out. No
random shuffling, so no future information reaches training.

Measured on the held-out slice (2,591 rows):

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias |
| --- | --- | --- | --- | --- |
| XGBoost (AirShield) | 4.337 | 2.173 | 0.980 | −0.261 |
| Persistence (`next hour = this hour`) | 5.601 | 2.896 | 0.966 | 0.000 |

RMSE improvement over persistence: **22.6%**.

Reproduce with `make train-offline`. The numbers above were produced by that
command; they are stored in `ml/artifacts/metadata.json` and served from there by
`GET /api/model`.

### Why the persistence baseline matters

Because PM2.5 is strongly autocorrelated, repeating the current value is a
genuinely competitive one-hour forecast. Reporting only R² would be misleading; a
model can reach R² 0.96 by predicting almost nothing. The baseline is therefore
computed and published beside the model's metrics, and
`test_model_beats_persistence_baseline` fails if the model stops beating it.

### Top features by gain share

| Feature | Share |
| --- | --- |
| `pm10` | 44.8% |
| `pm2_5` | 31.8% |
| `pm2_5_lag1` | 18.5% |
| `pm2_5_roll_mean3` | 1.6% |
| `pm2_5_delta1` | 0.5% |

Co-located particulate matter dominates, as expected physically: PM2.5 and PM10
share sources and transport.

## Limitations

1. **Six cities only.** No geocoding yet; a city outside the registry has no
   forecast.
2. **One-hour horizon.** Accuracy will degrade for longer horizons; the model is
   not validated beyond *t+1*.
3. **Sparse ground truth.** Open-Meteo aggregates monitoring stations. Coverage
   is uneven, and dense urban microclimates (street canyons, near-road) are not
   resolved.
4. **Weather-forecast dependence.** `wx_next_*` features inherit the error of the
   upstream weather forecast.
5. **Station-level aggregate vs personal exposure.** The prediction describes
   ambient outdoor concentration at the location's coordinates, not what an
   individual inhales.
6. **Distribution shift.** Trained on the recorded window; sensor changes or
   unusual events (wildfires, dust storms) fall outside it and need retraining.
7. **No uncertainty estimate.** A single point value is served, without
   prediction intervals.

## Ethical considerations

- **Sensitive groups are addressed separately.** Alerts carry distinct guidance
  for people with asthma, heart conditions, older adults, children and pregnant
  people, rather than one blanket message.
- **No medical claims.** Copy is phrased as advisory and avoids implying
  certainty about health outcomes.
- **Provenance is always visible.** The UI states the source, licence and fetch
  time, so a user can judge recency.
- **Demo data never masquerades as live.** Demo responses carry a notice, and the
  UI shows a banner.
- **AQI is standard.** Breakpoints follow the US EPA AirNow table; WHO comparison
  uses the 2021 guideline (24-hour PM2.5: 15 µg/m³).

## Reproducing

```bash
make train-offline   # trains from the bundled real dataset, no network
make train           # fetches fresh real history and retrains
cat ml/artifacts/metadata.json   # metrics, window, hyperparameters, versions
```
