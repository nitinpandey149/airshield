"""Feature engineering for 1-hour-ahead PM2.5 forecasting.

Three rules govern this module, and all three matter for honest results:

1. **No leakage.** Every lag and rolling statistic is computed only from
   observations at or before time ``t``. The target is PM2.5 at ``t + 1``.
2. **One implementation.** Training and serving call the same functions, so a
   model can never be served features it was not trained on.
3. **No cross-location bleed.** When several locations are trained together,
   every lag and rolling window is computed *within* a location. Without this,
   a lag would silently read a different city's row at the same timestamp.

Weather at ``t + 1`` *is* used, and that is legitimate rather than leakage:
weather forecasts for the next hour are published ahead of time, so this is
information genuinely available at prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Lags (in hours) of PM2.5 history used as predictors.
PM_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
#: Rolling windows (in hours) summarising recent PM2.5.
PM_ROLLING_WINDOWS: tuple[int, ...] = (3, 6, 24)

#: Weather variables carried forward from the forecast for hour ``t + 1``.
WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
)

#: Pollutant context available at time ``t``.
POLLUTANT_COLUMNS: tuple[str, ...] = ("pm10", "nitrogen_dioxide", "ozone")

#: Column naming the location, used to keep windows within a single series.
LOCATION_COLUMN = "location_name"

TARGET_COLUMN = "pm2_5_t_plus_1"

#: Longest lag, i.e. the warm-up period in hours before a row is usable.
WARMUP_HOURS: int = max(max(PM_LAGS), max(PM_ROLLING_WINDOWS))


def _feature_names() -> list[str]:
    names: list[str] = ["pm2_5"]
    names += [f"pm2_5_lag{h}" for h in PM_LAGS]
    names += [f"pm2_5_roll_mean{w}" for w in PM_ROLLING_WINDOWS]
    names += ["pm2_5_roll_std6", "pm2_5_delta1", "pm2_5_delta3", "pm2_5_trend"]
    names += [f"pm2_5_lag{h}_ratio" for h in (1, 3)]
    names += ["pm10", "pm10_ratio", "no2", "o3"]
    names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    names += ["wind_dir_sin", "wind_dir_cos", "ventilation_index"]
    names += [f"wx_next_{c}" for c in WEATHER_COLUMNS]
    names += [f"wx_delta_{c}" for c in ("temperature_2m", "wind_speed_10m", "surface_pressure")]
    return names


#: Exact, ordered feature vector the model expects.
FEATURE_COLUMNS: tuple[str, ...] = tuple(_feature_names())

_REQUIRED_INPUT = (
    "time",
    "pm2_5",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)


def _add_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and the ``t + 1`` target to one time-sorted, single series.

    Must not be called on a frame mixing locations.
    """
    df = df.sort_values("time").reset_index(drop=True)

    for col in POLLUTANT_COLUMNS + ("precipitation",):
        if col not in df.columns:
            df[col] = np.nan

    pm = df["pm2_5"]

    # --- PM2.5 history (strictly past) -------------------------------------
    for lag in PM_LAGS:
        df[f"pm2_5_lag{lag}"] = pm.shift(lag)

    for window in PM_ROLLING_WINDOWS:
        # shift(1) keeps the window strictly in the past.
        df[f"pm2_5_roll_mean{window}"] = pm.shift(1).rolling(window).mean()

    df["pm2_5_roll_std6"] = pm.shift(1).rolling(6).std()
    df["pm2_5_delta1"] = pm - pm.shift(1)
    df["pm2_5_delta3"] = pm - pm.shift(3)
    # Short-horizon trend: 3h mean versus 24h mean.
    df["pm2_5_trend"] = df["pm2_5_roll_mean3"] - df["pm2_5_roll_mean24"]

    # Ratios express "how unusual is now relative to a few hours ago".
    for lag in (1, 3):
        lag_col = df[f"pm2_5_lag{lag}"]
        df[f"pm2_5_lag{lag}_ratio"] = np.where(lag_col > 0, pm / lag_col, np.nan)

    # --- pollutant context -------------------------------------------------
    df["no2"] = df["nitrogen_dioxide"]
    df["o3"] = df["ozone"]
    df["pm10_ratio"] = np.where(df["pm10"] > 0, pm / df["pm10"], np.nan)

    # --- calendar ----------------------------------------------------------
    hours = df["time"].dt.hour
    dow = df["time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)

    # --- wind as a vector, plus a crude ventilation index ------------------
    radians = np.deg2rad(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(radians)
    df["wind_dir_cos"] = np.cos(radians)
    # Low wind under high pressure tends to trap particulates near the surface.
    df["ventilation_index"] = df["wind_speed_10m"] / (df["surface_pressure"] / 1000.0)

    # --- next-hour weather (from the forecast) -----------------------------
    for col in WEATHER_COLUMNS:
        df[f"wx_next_{col}"] = df[col].shift(-1)
    for col in ("temperature_2m", "wind_speed_10m", "surface_pressure"):
        df[f"wx_delta_{col}"] = df[f"wx_next_{col}"] - df[col]

    # --- target ------------------------------------------------------------
    df[TARGET_COLUMN] = pm.shift(-1)
    return df


def build_features(frame: pd.DataFrame, group_column: str | None = LOCATION_COLUMN) -> pd.DataFrame:
    """Add every model feature to an hourly observation frame.

    When ``group_column`` is present in the frame, features are computed
    independently per location so that lags never cross series. Pass
    ``group_column=None`` for a single-location frame.
    """
    missing = set(_REQUIRED_INPUT) - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if group_column and group_column in df.columns:
        if df[group_column].nunique() == 0:  # pragma: no cover - defensive
            raise ValueError(f"{group_column} is present but has no values")
        parts = [
            _add_series_features(group)
            for _, group in df.groupby(group_column, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)

    return _add_series_features(df)


def build_training_frame(
    frame: pd.DataFrame, group_column: str | None = LOCATION_COLUMN
) -> pd.DataFrame:
    """Build features and keep only rows with a complete vector and target."""
    df = build_features(frame, group_column=group_column)

    usable = df[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].notna().all(axis=1)
    out = df.loc[usable].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "no usable training rows were produced - the history is too short. "
            f"At least {WARMUP_HOURS + 2} consecutive hours are required per location."
        )
    return out


def latest_feature_row(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the feature row describing the most recent hour of one location.

    ``frame`` must contain history ending at the hour we are predicting *from*,
    with one extra row appended holding the forecast weather for the next hour
    (its ``pm2_5`` may be ``NaN``). Returns ``(one_row_frame, base_time)`` where
    ``base_time`` is the hour the prediction starts from.
    """
    df = _add_series_features(frame)
    if len(df) < 2:
        raise ValueError("at least two rows (history + forecast hour) are required")

    # The prediction originates from the second-to-last row: the last row only
    # supplies the next-hour weather.
    row = df.iloc[[-2]]
    base_time = pd.Timestamp(row["time"].iloc[0])

    values = row[list(FEATURE_COLUMNS)]
    if values.isna().any(axis=1).iloc[0]:
        missing = sorted(values.columns[values.isna().iloc[0]].tolist())
        raise ValueError(
            "not enough history to build a complete feature vector; "
            f"missing: {missing}. Provide at least {WARMUP_HOURS + 1} hours."
        )
    return values.reset_index(drop=True), base_time
