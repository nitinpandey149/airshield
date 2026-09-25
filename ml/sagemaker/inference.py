"""SageMaker inference handler.

Amazon SageMaker AI loads this module when the endpoint starts. The contract is:

* ``model_fn``    - load the XGBoost booster from the model directory
* ``input_fn``    - parse ``{"feature_names": [...], "rows": [[...]]}``
* ``predict_fn``  - invert the ``log1p`` transform applied during training
* ``output_fn``   - return ``{"prediction": <float>}``

The transform contract must match :meth:`airshield_core.train.train_model`
exactly (fit on ``log1p(pm2_5)``, serve with ``expm1``), otherwise predictions
would be silently wrong.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def model_fn(model_dir: str):
    """Load the trained booster and its feature order from ``model_dir``."""
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.ubj"))

    feature_names = None
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        feature_names = metadata.get("feature_columns")

    return {"booster": booster, "feature_names": feature_names}


def input_fn(request_body, content_type: str = "application/json") -> pd.DataFrame:
    """Parse the request body into a DataFrame of features."""
    if content_type not in ("application/json", "application/jsonlines"):
        raise ValueError(f"unsupported content type: {content_type}")

    payload = json.loads(request_body)
    if isinstance(payload, dict):
        feature_names = payload.get("feature_names")
        rows = payload["rows"]
    else:
        feature_names = None
        rows = payload

    frame = pd.DataFrame(rows, columns=feature_names)
    return frame


def predict_fn(data: pd.DataFrame, model: dict) -> np.ndarray:
    """Run the booster and invert the log1p transform."""
    booster = model["booster"]
    feature_names = model["feature_names"] or list(data.columns)

    if list(data.columns) != feature_names:
        raise ValueError(
            "feature columns do not match the trained model: "
            f"expected {feature_names}, got {list(data.columns)}"
        )

    matrix = xgb.DMatrix(data.to_numpy(dtype=np.float32), feature_names=feature_names)
    raw = booster.predict(matrix)
    # Training fit log1p(target), so predictions come back in log space.
    return np.expm1(raw)


def output_fn(prediction: np.ndarray, accept: str = "application/json") -> str:
    """Serialise predictions as JSON."""
    values = np.asarray(prediction, dtype=float).ravel().tolist()
    return json.dumps({"prediction": values[0] if len(values) == 1 else values})
