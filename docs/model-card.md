# AirShield — model card

## Overview

| | |
| --- | --- |
| Task | Regression: predict hourly mean PM2.5 1, 3 and 6 hours ahead |
| Target | `pm2_5` at hour *t+horizon*, in µg/m³ (one booster per horizon) |
| Algorithm | XGBoost gradient-boosted trees |
| Primary platform | Amazon SageMaker AI (the only AWS AI/ML service used) |
| Local fallback | The same booster, persisted and run in-process |
| Horizons | 1, 3 and 6 hours — one separate booster per horizon |

## Intended use

Give an individual a short-horizon, plain-language signal about outdoor air
quality so they can time a run, a commute, or outdoor play, and decide whether a
sensitive person should limit exposure.

**Not intended for:** clinical decisions, regulatory reporting, long-range
planning, or any use where an error of a few µg/m³ carries legal or medical
consequence. AirShield forecasts; it does not diagnose or prescribe.

## Training data

Hourly PM2.5 and weather fields from the Open-Meteo air-quality and weather APIs
(CC BY 4.0), covering six Indian cities:

| City | Slug |
| --- | --- |
| Delhi | `delhi` |
| Mumbai | `mumbai` |
| Bengaluru | `bengaluru` |
| Chennai | `chennai` |
| Kolkata | `kolkata` |
| Hyderabad | `hyderabad` |

These were chosen because India carries a large share of the global PM2.5
exposure burden and each city has a complete Open-Meteo history, which training
requires. The air-quality series is Open-Meteo's **CAMS-based model output**
(roughly 11 km resolution), not a CPCB monitoring station: it is modelled
estimates of ambient concentration at the city coordinates, and the AQI derived
from it is India's CPCB National AQI. Do not read it as a station measurement.

The bundled sample (`ml/data/demo/demo_hourly.csv`) holds 13,104 rows over a
frozen window (2026-07-03 → 2026-10-01); `manifest.json` records the source,
window, licence and coordinates. It is real historical data, not synthetic —
"demo" here means historical rather than current, and `is_synthetic` is false.
`make train` instead pulls fresh history for the days configured in
`AIRSHIELD_TRAIN_DAYS`.

## Features

37 features, all computed within each location to prevent cross-city leakage:

| Group | Examples |
| --- | --- |
| Pollution lags | `pm2_5_lag1`, `_2`, `_3`, `_6`, `_12`, `_24` |
| Rolling means (past only) | `pm2_5_roll_mean3`, `_6`, `_24` |
| Short-term change | `pm2_5_delta1`, `pm2_5_delta3`, ratios, `pm2_5_trend` |
| Pollutant context | `pm10`, `nitrogen_dioxide`, `ozone` and their ratios |
| Meteorology | temperature, humidity, wind speed/direction, pressure, precipitation |
| Next-hour weather | `wx_next_*` (forecast weather only, never future PM2.5) |
| Calendar | hour-of-day, day-of-week, cyclical encodings |

## Evaluation

Time-ordered 80/20 split: the most recent 20% of the timeline is held out. No
random shuffling, so no future information reaches training.

Measured on the held-out slice. Each horizon is a separate booster, trained and
evaluated on its own; the test rows differ slightly because each horizon loses a
few more rows to the warm-up and target shift.

**+1 hour** (2,591 test rows):

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias |
| --- | --- | --- | --- | --- |
| XGBoost (AirShield) | 3.6706 | 2.0724 | 0.9740 | −0.4093 |
| Persistence (`next hour = this hour`) | 4.5063 | 2.5514 | 0.9608 | −0.0684 |

RMSE improvement over persistence: **18.5%**.

**+3 hours** (2,589 test rows):

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias |
| --- | --- | --- | --- | --- |
| XGBoost (AirShield) | 8.2301 | 5.1846 | 0.8692 | −1.6225 |
| Persistence | 10.0588 | 6.2219 | 0.8047 | −0.2206 |

RMSE improvement over persistence: **18.2%**.

**+6 hours** (2,585 test rows):

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias |
| --- | --- | --- | --- | --- |
| XGBoost (AirShield) | 11.9491 | 7.7495 | 0.7245 | −2.7559 |
| Persistence | 15.2423 | 9.9292 | 0.5517 | −0.4865 |

RMSE improvement over persistence: **21.6%**.

Reproduce with `make train-offline`. The numbers above were produced by that
command; they are stored in `ml/artifacts/metadata*.json` and served from there by
`GET /api/model`. A negative bias means the model slightly under-predicts on
average; it grows with horizon, as expected when the forecast leans on
climatology.

### Why the persistence baseline matters

Because PM2.5 is strongly autocorrelated, repeating the current value is a
genuinely competitive one-hour forecast. Reporting only R² would be misleading; a
model can reach R² 0.96 by predicting almost nothing. The baseline is therefore
computed and published beside the model's metrics, and
`test_model_beats_persistence_baseline` fails if the model stops beating it.

### Top features by gain share

| Feature | Share |
| --- | --- |
| `pm2_5_lag1` | 38.0% |
| `pm2_5` | 27.6% |
| `pm2_5_roll_mean3` | 25.3% |
| `pm10` | 6.0% |
| `pm2_5_delta1` | 0.5% |

Past PM2.5 — the immediately previous hour, the current hour and the recent
3-hour mean — dominates, which is what autocorrelation predicts. `pm10` adds a
co-located particulate signal; it shares sources and transport with PM2.5.

## Limitations

1. **Six cities only.** No geocoding yet; a city outside the registry has no
   forecast.
2. **Accuracy degrades with horizon.** The 6-hour booster is markedly weaker than
   the 1-hour one (R² 0.725 vs 0.974) and is not validated beyond *t+6*.
3. **Modelled grid, not monitors.** Open-Meteo's air-quality series is CAMS model
   output at roughly 11 km resolution, not a CPCB station reading. Dense urban
   microclimates (street canyons, near-road) are not resolved, and the value may
   differ from the nearest regulatory monitor.
4. **Weather-forecast dependence.** `wx_next_*` features inherit the error of the
   upstream weather forecast.
5. **Ambient grid cell vs personal exposure.** The prediction describes modelled
   ambient outdoor concentration at the location's coordinates, not what an
   individual inhales.
6. **Distribution shift.** Trained on the recorded window; upstream model or
   version changes, and unusual events (wildfires, dust storms, festival
   fireworks), fall outside it and need retraining.
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
- **AQI is standard.** The headline index is India's **CPCB National AQI** (six
  categories), computed as the worst available pollutant sub-index. The US EPA
  AQI is retained alongside it as a named reference, never substituted silently.
  WHO comparison uses the 2021 guideline (24-hour PM2.5: 15 µg/m³). Because the
  upstream source does not provide CO, SO₂ or NH₃, the National AQI can be lower
  than the official value when one of those dominates — the API says so.

## Reproducing

```bash
make train-offline   # trains from the bundled real dataset, no network
make train           # fetches fresh real history and retrains
cat ml/artifacts/metadata.json   # metrics, window, hyperparameters, versions
```
