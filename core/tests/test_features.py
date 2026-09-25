"""Tests for feature engineering - especially the leakage and isolation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airshield_core.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WARMUP_HOURS,
    build_features,
    build_training_frame,
    latest_feature_row,
)


def test_all_declared_features_are_produced(synthetic_frame: pd.DataFrame) -> None:
    featured = build_features(synthetic_frame, group_column=None)
    missing = [c for c in FEATURE_COLUMNS if c not in featured.columns]
    assert not missing, f"declared but not produced: {missing}"


def test_no_duplicate_feature_names() -> None:
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_lags_reference_the_past(synthetic_frame: pd.DataFrame) -> None:
    featured = build_features(synthetic_frame, group_column=None)
    row = 50
    assert featured.loc[row, "pm2_5_lag1"] == pytest.approx(synthetic_frame.loc[row - 1, "pm2_5"])
    assert featured.loc[row, "pm2_5_lag3"] == pytest.approx(synthetic_frame.loc[row - 3, "pm2_5"])
    assert featured.loc[row, "pm2_5_lag24"] == pytest.approx(synthetic_frame.loc[row - 24, "pm2_5"])


def test_target_is_the_next_hour(synthetic_frame: pd.DataFrame) -> None:
    featured = build_features(synthetic_frame, group_column=None)
    row = 60
    assert featured.loc[row, TARGET_COLUMN] == pytest.approx(synthetic_frame.loc[row + 1, "pm2_5"])


def test_rolling_means_exclude_the_current_hour(synthetic_frame: pd.DataFrame) -> None:
    """The 3h rolling mean at t must cover t-1..t-3, never t itself."""
    featured = build_features(synthetic_frame, group_column=None)
    row = 40
    expected = synthetic_frame.loc[row - 3 : row - 1, "pm2_5"].mean()
    assert featured.loc[row, "pm2_5_roll_mean3"] == pytest.approx(expected)


def test_future_values_do_not_leak_into_past_features(synthetic_frame: pd.DataFrame) -> None:
    """Corrupt the tail of the series; features before it must be unchanged."""
    baseline = build_features(synthetic_frame, group_column=None)

    mutated = synthetic_frame.copy()
    mutated.loc[100:, "pm2_5"] = mutated.loc[100:, "pm2_5"] * 100
    after = build_features(mutated, group_column=None)

    past = baseline.loc[:60, list(FEATURE_COLUMNS)]
    past_after = after.loc[:60, list(FEATURE_COLUMNS)]
    pd.testing.assert_frame_equal(past, past_after)


def test_wx_next_uses_the_following_hour(synthetic_frame: pd.DataFrame) -> None:
    featured = build_features(synthetic_frame, group_column=None)
    row = 30
    assert featured.loc[row, "wx_next_temperature_2m"] == pytest.approx(
        synthetic_frame.loc[row + 1, "temperature_2m"]
    )


def test_locations_do_not_bleed_into_each_other() -> None:
    """The same timestamp in two cities must not share lag values."""
    def city(name: str, base: float) -> pd.DataFrame:
        n = 60
        times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "time": times,
                "location_name": name,
                "pm2_5": np.full(n, base),
                "pm10": np.full(n, base * 1.5),
                "nitrogen_dioxide": np.full(n, 10.0),
                "ozone": np.full(n, 20.0),
                "temperature_2m": np.full(n, 10.0),
                "relative_humidity_2m": np.full(n, 50.0),
                "wind_speed_10m": np.full(n, 5.0),
                "wind_direction_10m": np.full(n, 180.0),
                "surface_pressure": np.full(n, 1013.0),
                "precipitation": np.zeros(n),
            }
        )

    combined = pd.concat([city("Alpha", 10.0), city("Beta", 90.0)], ignore_index=True)
    featured = build_features(combined, group_column="location_name")

    alpha = featured[featured["location_name"] == "Alpha"]
    beta = featured[featured["location_name"] == "Beta"]

    # Every Alpha lag must be 10.0 (its own series), never 90.0.
    assert (alpha["pm2_5_lag1"].dropna() == 10.0).all()
    assert (beta["pm2_5_lag1"].dropna() == 90.0).all()
    assert len(alpha) == 60 and len(beta) == 60


def test_training_frame_drops_warmup_and_final_row(synthetic_frame: pd.DataFrame) -> None:
    trained = build_training_frame(synthetic_frame, group_column=None)
    assert trained[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].notna().all().all()
    # First WARMUP_HOURS rows and the last row cannot be used.
    assert len(trained) == len(synthetic_frame) - WARMUP_HOURS - 1


def test_short_history_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no usable training rows"):
        build_training_frame(synthetic_frame.head(10), group_column=None)


def test_missing_required_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(pd.DataFrame({"time": [], "pm2_5": []}), group_column=None)


def test_latest_feature_row_uses_penultimate_row(synthetic_frame: pd.DataFrame) -> None:
    """The appended forecast row supplies weather but is not the prediction base."""
    frame = synthetic_frame.copy()
    frame.loc[len(frame)] = frame.iloc[-1]
    frame.loc[frame.index[-1], "time"] = frame["time"].iloc[-1] + pd.Timedelta(hours=1)
    frame.loc[frame.index[-1], "pm2_5"] = np.nan

    features, base_time = latest_feature_row(frame)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert not features.isna().any().any()
    assert base_time == synthetic_frame["time"].iloc[-1]


def test_latest_feature_row_requires_enough_history(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not enough history"):
        latest_feature_row(synthetic_frame.head(5))
