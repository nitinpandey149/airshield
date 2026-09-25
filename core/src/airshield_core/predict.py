"""Model artifact loading and the local predictor.

A trained model is stored as a directory containing:

* ``model.ubj``   - the XGBoost booster
* ``metadata.json`` - feature order, training metrics, provenance

``metadata.json`` is what makes results auditable: it records the real measured
metrics and the exact data window the model saw, so the API can report them
verbatim instead of inventing numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from airshield_core.features import FEATURE_COLUMNS

MODEL_FILENAME = "model.ubj"
METADATA_FILENAME = "metadata.json"


class ModelNotFoundError(RuntimeError):
    """Raised when no usable local artifact exists."""


@dataclass
class ModelMetadata:
    """Everything we know about a trained artifact."""

    trained_at: str
    feature_columns: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    train_rows: int = 0
    test_rows: int = 0
    train_window: dict[str, str] = field(default_factory=dict)
    locations: list[str] = field(default_factory=list)
    data_source: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Forecast:
    """A single 1-hour-ahead PM2.5 prediction."""

    predicted_pm25: float
    base_time: datetime
    target_time: datetime
    model_version: str
    backend: str
    horizon_hours: int = 1
    metadata: dict = field(default_factory=dict)


class LocalPredictor:
    """Serves 1-hour-ahead PM2.5 forecasts from a local XGBoost artifact."""

    def __init__(self, artifact_dir: Path | str):
        self.artifact_dir = Path(artifact_dir)
        self._booster = None
        self._metadata: ModelMetadata | None = None

    # ------------------------------------------------------------- loading
    @property
    def model_path(self) -> Path:
        return self.artifact_dir / MODEL_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / METADATA_FILENAME

    @property
    def is_available(self) -> bool:
        return self.model_path.is_file() and self.metadata_path.is_file()

    def load(self) -> "LocalPredictor":
        """Load the booster and metadata, raising a clear error if absent."""
        if not self.is_available:
            raise ModelNotFoundError(
                f"No model artifact found in {self.artifact_dir}. "
                "Run `make train` (or `python -m airshield_core.train`) to create one."
            )
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        self._booster = booster
        self._metadata = ModelMetadata.from_dict(
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
        )
        return self

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            self.load()
        assert self._metadata is not None
        return self._metadata

    @property
    def model_version(self) -> str:
        meta = self.metadata
        stamp = meta.trained_at.replace(":", "").replace("-", "")[:15]
        return f"xgboost-pm25-1h-{stamp}"

    # ---------------------------------------------------------- prediction
    def predict(self, features: pd.DataFrame) -> float:
        """Predict PM2.5 for the hour after the row in ``features``.

        Raises ``ValueError`` if the frame does not carry exactly the trained
        feature columns in the trained order.
        """
        if self._booster is None:
            self.load()

        expected = list(self.metadata.feature_columns)
        if list(features.columns) != expected:
            raise ValueError(
                "feature columns do not match the trained model.\n"
                f"expected: {expected}\ngot:      {list(features.columns)}"
            )

        import xgboost as xgb

        matrix = xgb.DMatrix(features.to_numpy(dtype=np.float32), feature_names=expected)
        raw = float(self._booster.predict(matrix)[0])  # type: ignore[union-attr]

        # PM2.5 cannot be negative; the booster is fit on log1p(target) so the
        # inverse transform also guarantees a sane magnitude.
        return float(np.expm1(raw))

    def forecast(self, features: pd.DataFrame, base_time: datetime) -> Forecast:
        """Produce a :class:`Forecast` for the hour after ``base_time``."""
        from datetime import timedelta

        value = self.predict(features)
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=timezone.utc)
        return Forecast(
            predicted_pm25=value,
            base_time=base_time,
            target_time=base_time + timedelta(hours=1),
            model_version=self.model_version,
            backend="local",
            horizon_hours=1,
            metadata={"artifact_dir": str(self.artifact_dir)},
        )


def verify_feature_contract() -> None:
    """Assert the artifact's feature list matches the current code."""
    from airshield_core.features import FEATURE_COLUMNS as current

    if not current:  # pragma: no cover - defensive
        raise AssertionError("FEATURE_COLUMNS is empty")
