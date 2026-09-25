"""Tests for training, artifact persistence and inference.

The metric assertions run against the *real* bundled dataset, so they verify
that training genuinely learns something rather than checking a mocked value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airshield_core.predict import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    LocalPredictor,
    ModelNotFoundError,
)
from airshield_core.train import regression_metrics, train_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory, demo_csv: Path):
    """Train once on the real bundled dataset and reuse across tests."""
    frame = pd.read_csv(demo_csv)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    out = tmp_path_factory.mktemp("artifact")
    result = train_model(
        frame,
        artifact_dir=out,
        num_rounds=60,
        locations=sorted(frame["location_name"].unique().tolist()),
        data_source={"kind": "test", "path": str(demo_csv)},
    )
    return result, out, frame


def test_artifacts_are_written(trained) -> None:
    result, out, _ = trained
    assert (out / MODEL_FILENAME).is_file()
    assert (out / METADATA_FILENAME).is_file()
    assert result.artifact_dir == out


def test_metadata_is_complete_and_auditable(trained) -> None:
    _, out, _ = trained
    meta = json.loads((out / METADATA_FILENAME).read_text())

    for key in (
        "trained_at", "feature_columns", "metrics", "baseline_metrics",
        "train_rows", "test_rows", "train_window", "data_source",
        "hyperparameters", "library_versions",
    ):
        assert key in meta, f"metadata missing {key}"

    assert meta["train_rows"] > 0 and meta["test_rows"] > 0
    assert meta["data_source"]["kind"] == "test"
    # The window must be recorded so results can be reproduced.
    assert meta["train_window"]["start"] and meta["train_window"]["end"]


def test_split_is_time_ordered(trained) -> None:
    """No test row may predate the training window.

    Equality at the boundary is expected: with several locations trained
    together, one hour is represented by several rows, so the split can fall
    inside a single hour.
    """
    _, out, _ = trained
    meta = json.loads((out / METADATA_FILENAME).read_text())
    window = meta["train_window"]
    assert window["test_start"] >= window["end"]
    assert window["test_end"] > window["start"]


def test_model_beats_persistence_baseline(trained) -> None:
    """The whole point of the model: it must out-predict 'next hour = now'."""
    result, _, _ = trained
    assert result.metrics["rmse"] < result.baseline_metrics["rmse"], (
        f"model RMSE {result.metrics['rmse']} did not beat persistence "
        f"{result.baseline_metrics['rmse']}"
    )
    assert result.metrics["r2"] > 0.5


def test_reported_metrics_are_finite_and_real(trained) -> None:
    result, _, _ = trained
    for name, value in result.metrics.items():
        assert np.isfinite(value), f"{name} is not finite"
    assert result.metrics["n"] == result.test_rows
    assert result.metrics["rmse"] > 0
    assert result.metrics["mae"] > 0


def test_feature_importance_sums_to_about_one(trained) -> None:
    result, _, _ = trained
    assert result.feature_importance
    total = sum(result.feature_importance.values())
    assert total == pytest.approx(1.0, abs=0.02)


def test_training_rejects_tiny_datasets(synthetic_frame: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        train_model(synthetic_frame.head(100), artifact_dir=tmp_path, num_rounds=5)


def test_predictor_round_trip(trained) -> None:
    """Train -> save -> load -> predict must produce a usable number."""
    _, out, frame = trained
    predictor = LocalPredictor(out).load()

    from airshield_core.features import latest_feature_row

    subset = frame[frame["location_name"] == "Berlin"].tail(72).reset_index(drop=True)
    features, base_time = latest_feature_row(subset)

    value = predictor.predict(features)
    assert np.isfinite(value)
    assert value >= 0, "PM2.5 can never be negative"
    # Must be in the right order of magnitude for the real data.
    assert value < 500

    forecast = predictor.forecast(features, base_time.to_pydatetime())
    assert forecast.backend == "local"
    assert forecast.horizon_hours == 1
    assert forecast.target_time > forecast.base_time
    assert forecast.model_version.startswith("xgboost-pm25-1h-")


def test_predictor_rejects_wrong_feature_columns(trained) -> None:
    _, out, frame = trained
    predictor = LocalPredictor(out).load()
    from airshield_core.features import FEATURE_COLUMNS

    bad = pd.DataFrame({c: [1.0] for c in list(FEATURE_COLUMNS)[:-1]})
    with pytest.raises(ValueError, match="feature columns do not match"):
        predictor.predict(bad)


def test_missing_artifact_raises_actionable_error(tmp_path: Path) -> None:
    predictor = LocalPredictor(tmp_path / "nope")
    assert not predictor.is_available
    with pytest.raises(ModelNotFoundError, match="make train"):
        predictor.load()


def test_regression_metrics_are_correct() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    metrics = regression_metrics(actual, predicted)
    # metrics are rounded to 4 dp for reporting.
    assert metrics["mae"] == pytest.approx(7 / 3, abs=1e-4)
    assert metrics["mean_bias"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["rmse"] == pytest.approx(np.sqrt(17 / 3), abs=1e-4)
    assert metrics["n"] == 3
