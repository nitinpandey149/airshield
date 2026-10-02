"""Tests for multi-horizon features, targets and routing geometry helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airshield_core.features import (
    FEATURE_COLUMNS,
    HORIZONS,
    build_features,
    build_training_frame,
    latest_feature_row,
    target_column,
)
from airshield_core.predict import LocalPredictor, metadata_filename, model_filename
from airshield_core.routing import (
    RoutingError,
    decode_polyline,
    segment_geometry,
)


def test_target_column_names() -> None:
    assert target_column(1) == "pm2_5_t_plus_1"
    assert target_column(6) == "pm2_5_t_plus_6"
    with pytest.raises(ValueError):
        target_column(0)


def test_horizons_are_the_supported_set() -> None:
    assert set(HORIZONS) == {1, 3, 6}


def test_target_shifts_by_horizon(synthetic_frame: pd.DataFrame) -> None:
    for horizon in HORIZONS:
        featured = build_features(synthetic_frame, group_column=None, horizon=horizon)
        row = 60
        expected = synthetic_frame.loc[row + horizon, "pm2_5"]
        assert featured.loc[row, target_column(horizon)] == pytest.approx(expected)


def test_wx_next_reads_the_target_hour(synthetic_frame: pd.DataFrame) -> None:
    featured = build_features(synthetic_frame, group_column=None, horizon=3)
    row = 30
    assert featured.loc[row, "wx_next_temperature_2m"] == pytest.approx(
        synthetic_frame.loc[row + 3, "temperature_2m"]
    )


def test_no_pm25_leakage_at_longer_horizon(synthetic_frame: pd.DataFrame) -> None:
    """Corrupting the future must not change earlier feature rows, at h=6."""
    baseline = build_features(synthetic_frame, group_column=None, horizon=6)
    mutated = synthetic_frame.copy()
    mutated.loc[120:, "pm2_5"] = mutated.loc[120:, "pm2_5"] * 100
    after = build_features(mutated, group_column=None, horizon=6)
    pd.testing.assert_frame_equal(
        baseline.loc[:80, list(FEATURE_COLUMNS)],
        after.loc[:80, list(FEATURE_COLUMNS)],
    )


def test_latest_feature_row_for_horizon(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.copy()
    for step in range(1, 4):
        row = frame.iloc[[-1]].copy()
        row["time"] = frame["time"].iloc[-1] + pd.Timedelta(hours=1)
        row["pm2_5"] = np.nan
        frame = pd.concat([frame, row], ignore_index=True)

    features, base_time = latest_feature_row(frame, horizon=3)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert not features.isna().any().any()
    # Base time is 3 hours before the end of the frame.
    assert base_time == frame["time"].iloc[-4]


def test_build_training_frame_for_horizon(synthetic_frame: pd.DataFrame) -> None:
    trained = build_training_frame(synthetic_frame, group_column=None, horizon=3)
    assert trained[list(FEATURE_COLUMNS) + ["pm2_5_t_plus_3"]].notna().all().all()


def test_artifact_filenames() -> None:
    assert model_filename(1) == "model.ubj"
    assert model_filename(6) == "model_h6.ubj"
    assert metadata_filename(1) == "metadata.json"
    assert metadata_filename(3) == "metadata_h3.json"


def test_local_predictor_horizon_paths(tmp_path) -> None:
    assert LocalPredictor(tmp_path, horizon=6).model_path.name == "model_h6.ubj"
    assert LocalPredictor(tmp_path, horizon=1).model_path.name == "model.ubj"
    with pytest.raises(ValueError):
        LocalPredictor(tmp_path, horizon=0)


def test_trained_horizon_predictor_round_trip(artifact_dir, demo_frame) -> None:
    """Train h=3 into a temp dir and confirm it serves a sane number."""
    from airshield_core.train import train_model

    frame = demo_frame.copy()
    out = artifact_dir.parent / "artifact_h3"
    train_model(frame, artifact_dir=out, num_rounds=40, horizon=3)

    predictor = LocalPredictor(out, horizon=3).load()
    subset = frame[frame["location_name"] == "Delhi"].tail(80).reset_index(drop=True)
    features, base_time = latest_feature_row(subset, horizon=3)
    value = predictor.predict(features)
    assert np.isfinite(value) and value >= 0

    forecast = predictor.forecast(features, base_time.to_pydatetime())
    assert forecast.horizon_hours == 3
    assert forecast.model_version.startswith("xgboost-pm25-3h-")


# --------------------------------------------------------------------------
# Routing helpers (pure functions - no network)
# --------------------------------------------------------------------------
def test_polyline_round_trip_known_value() -> None:
    """Decode a hand-encoded polyline6 of two points."""
    # Encode (52.5200, 13.4050) -> (52.5300, 13.4100) with precision 5.
    def encode(points, precision=5):
        factor = 10**precision
        output = []
        prev_lat = prev_lon = 0
        for lat, lon in points:
            lat_i = round(lat * factor)
            lon_i = round(lon * factor)
            for delta in (lat_i - prev_lat, lon_i - prev_lon):
                value = ~(delta << 1) if delta < 0 else (delta << 1)
                while value >= 0x20:
                    output.append(chr((0x20 | (value & 0x1F)) + 63))
                    value >>= 5
                output.append(chr(value + 63))
            prev_lat, prev_lon = lat_i, lon_i
        return "".join(output)

    points = [(28.6139, 77.209), (52.53, 13.41), (52.51, 13.42)]
    decoded = decode_polyline(encode(points), precision=5)
    assert len(decoded) == 3
    for (lat, lon), (elat, elon) in zip(decoded, points):
        assert lat == pytest.approx(elat, abs=1e-5)
        assert lon == pytest.approx(elon, abs=1e-5)


def test_segment_geometry_splits_and_totals() -> None:
    geometry = [(52.52 + i * 0.002, 13.40) for i in range(11)]
    specs = segment_geometry(geometry, distance_km=2.0, duration_minutes=24.0, segments=4)
    assert len(specs) == 4
    assert sum(s.distance_km for s in specs) == pytest.approx(2.0, abs=0.05)
    assert sum(s.duration_minutes for s in specs) == pytest.approx(24.0, abs=0.2)
    # Midpoints must lie within the geometry bounding box.
    for spec in specs:
        assert 52.51 <= spec.latitude <= 52.55


def test_segment_geometry_rejects_degenerate_input() -> None:
    with pytest.raises(RoutingError):
        segment_geometry([(52.5, 13.4)], 1.0, 10.0)
    with pytest.raises(RoutingError):
        segment_geometry([(52.5, 13.4), (52.5, 13.4)], 1.0, 10.0)
