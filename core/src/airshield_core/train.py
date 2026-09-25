"""Training pipeline for the 1-hour-ahead PM2.5 XGBoost model.

Design decisions that keep the reported numbers honest:

* **Time-ordered split.** The test set is the most recent slice of the timeline,
  never a random sample. A random split would leak neighbouring hours into
  training and inflate the score.
* **Persistence baseline.** "Tomorrow's PM2.5 equals this hour's PM2.5" is a very
  strong baseline for one-hour-ahead air quality. The model is only interesting
  if it beats it, so both sets of metrics are recorded and reported side by side.
* **log1p target.** PM2.5 is heavily right-skewed, so the model is fit on
  ``log1p(pm2_5)`` and inverted with ``expm1``. This also makes negative
  predictions impossible.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from airshield_core.features import FEATURE_COLUMNS, TARGET_COLUMN, build_training_frame
from airshield_core.predict import METADATA_FILENAME, MODEL_FILENAME, ModelMetadata

#: Fraction of the timeline held out as the most-recent test window.
TEST_FRACTION = 0.2
#: Minimum rows required before training is allowed to run.
MIN_TRAINING_ROWS = 240

DEFAULT_HYPERPARAMETERS: dict = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "lambda": 1.0,
    "seed": 42,
    "nthread": 4,
}
DEFAULT_NUM_ROUNDS = 400


@dataclass
class TrainingResult:
    """Outcome of a training run, including metrics and the saved artifact path."""

    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    train_rows: int
    test_rows: int
    artifact_dir: Path
    feature_importance: dict[str, float]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R2 and mean bias for a set of predictions."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    # r2_score is undefined for a single sample; guard so short test sets do not crash.
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mean_bias": round(float(np.mean(predicted - actual)), 4),
        "n": int(len(actual)),
    }


def train_model(
    frame: pd.DataFrame,
    *,
    artifact_dir: Path | str,
    test_fraction: float = TEST_FRACTION,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    hyperparameters: dict | None = None,
    locations: list[str] | None = None,
    data_source: dict | None = None,
) -> TrainingResult:
    """Train, evaluate and persist an XGBoost PM2.5 model.

    ``frame`` must be a concatenation of hourly observations (all locations are
    fine - ``location`` is not a feature, so the model learns transportable
    dynamics rather than memorising a city).
    """
    params = {**DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}
    artifact_dir = Path(artifact_dir)

    featured = build_training_frame(frame)
    featured = featured.sort_values("time").reset_index(drop=True)

    if len(featured) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"only {len(featured)} usable rows; at least {MIN_TRAINING_ROWS} are "
            "needed. Increase AIRSHIELD_TRAIN_DAYS or add locations."
        )

    # --- time-ordered split -------------------------------------------------
    split_index = int(len(featured) * (1 - test_fraction))
    train_df = featured.iloc[:split_index]
    test_df = featured.iloc[split_index:]

    x_train = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y_train_raw = train_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    x_test = test_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y_test_raw = test_df[TARGET_COLUMN].to_numpy(dtype=np.float64)

    # --- fit on log1p -------------------------------------------------------
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=np.log1p(y_train_raw), feature_names=list(FEATURE_COLUMNS))
    dvalid = xgb.DMatrix(x_test, label=np.log1p(y_test_raw), feature_names=list(FEATURE_COLUMNS))

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        verbose_eval=False,
    )

    predicted = np.expm1(booster.predict(dvalid))
    predicted = np.clip(predicted, 0.0, None)

    metrics = regression_metrics(y_test_raw, predicted)

    # --- persistence baseline: next hour == this hour ----------------------
    baseline_pred = test_df["pm2_5"].to_numpy(dtype=np.float64)
    baseline_metrics = regression_metrics(y_test_raw, baseline_pred)

    # --- persist ------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILENAME
    booster.save_model(str(model_path))

    importance = booster.get_score(importance_type="gain")
    total = sum(importance.values()) or 1.0
    feature_importance = {
        k: round(v / total, 4) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])
    }

    metadata = ModelMetadata(
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_columns=list(FEATURE_COLUMNS),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        train_window={
            "start": str(train_df["time"].min()),
            "end": str(train_df["time"].max()),
            "test_start": str(test_df["time"].min()),
            "test_end": str(test_df["time"].max()),
        },
        locations=sorted(locations or []),
        data_source=data_source or {},
        hyperparameters={**params, "num_boost_round": num_rounds},
        library_versions={
            "xgboost": xgb.__version__,
            "python": platform.python_version(),
        },
        feature_importance=feature_importance,
    )
    (artifact_dir / METADATA_FILENAME).write_text(metadata.to_json(), encoding="utf-8")

    return TrainingResult(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        artifact_dir=artifact_dir,
        feature_importance=feature_importance,
    )


def summarize(result: TrainingResult) -> str:
    """Render a human-readable training report."""
    m, b = result.metrics, result.baseline_metrics
    skill = 0.0
    if b["rmse"] > 0:
        skill = round((b["rmse"] - m["rmse"]) / b["rmse"] * 100, 2)

    lines = [
        "AirShield - 1h PM2.5 model",
        "-" * 46,
        f"rows         : {result.train_rows} train / {result.test_rows} test (time-ordered)",
        "",
        "model (XGBoost, log1p target)",
        f"  RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  R2 {m['r2']:.3f}  bias {m['mean_bias']:+.3f}",
        "baseline (persistence: next hour = this hour)",
        f"  RMSE {b['rmse']:.3f}  MAE {b['mae']:.3f}  R2 {b['r2']:.3f}  bias {b['mean_bias']:+.3f}",
        "",
        f"RMSE improvement over persistence: {skill:+.2f}%",
        "",
        "top features (gain share)",
    ]
    for name, share in list(result.feature_importance.items())[:10]:
        lines.append(f"  {name:<28} {share * 100:5.1f}%")
    lines.append("")
    lines.append(f"artifact written to {result.artifact_dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m airshield_core.train``."""
    import argparse

    from airshield_core.config import get_settings
    from airshield_core.sources.openmeteo import (
        UpstreamError,
        default_history_window,
        fetch_history,
    )
    from airshield_core.sources.registry import LOCATIONS

    parser = argparse.ArgumentParser(description="Train the AirShield PM2.5 model")
    parser.add_argument("--days", type=int, default=None, help="days of history to fetch")
    parser.add_argument("--rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument(
        "--locations",
        default=None,
        help="comma-separated location slugs (default: all built-in locations)",
    )
    parser.add_argument(
        "--from-csv",
        default=None,
        help="train from a local CSV instead of fetching (offline mode)",
    )
    parser.add_argument("--out", default=None, help="artifact directory")
    args = parser.parse_args(argv)

    settings = get_settings()
    out_dir = Path(args.out) if args.out else settings.artifact_path

    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        source_info = {"kind": "local_csv", "path": str(args.from_csv)}
        locations = sorted(frame["location_name"].unique().tolist()) if "location_name" in frame else []
        print(f"loaded {len(frame)} rows from {args.from_csv}")
    else:
        days = args.days or settings.train_days
        start, end = default_history_window(days)
        slugs = args.locations.split(",") if args.locations else [l.slug for l in LOCATIONS]
        selected = [l for l in LOCATIONS if l.slug in slugs]

        print(f"fetching {days} days of real measurements ({start} -> {end}) for {len(selected)} locations")
        frames = []
        for location in selected:
            try:
                part = fetch_history(
                    location.latitude,
                    location.longitude,
                    start,
                    end,
                    timeout=settings.http_timeout + 15,
                    retries=settings.http_retries,
                )
            except UpstreamError as exc:
                print(f"  ! {location.slug}: {exc}")
                continue
            part["location_name"] = location.name
            frames.append(part)
            print(f"  + {location.slug:<12} {len(part):>6} rows")

        if not frames:
            print("ERROR: could not fetch any training data. Check network access.")
            return 1
        frame = pd.concat(frames, ignore_index=True)
        source_info = {
            "kind": "open-meteo",
            "url": "https://open-meteo.com/",
            "licence": "CC BY 4.0",
            "window": {"start": start.isoformat(), "end": end.isoformat()},
        }
        locations = [l.name for l in selected]

    result = train_model(
        frame,
        artifact_dir=out_dir,
        num_rounds=args.rounds,
        locations=locations,
        data_source=source_info,
    )
    print()
    print(summarize(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
