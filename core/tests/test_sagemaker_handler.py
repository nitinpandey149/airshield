"""Tests for the SageMaker inference handler.

These exercise the real handler code against a real trained booster, so the
log1p/expm1 transform contract and the feature-order check are verified without
needing an AWS account.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER_PATH = REPO_ROOT / "ml" / "sagemaker" / "inference.py"


@pytest.fixture(scope="module")
def handler():
    spec = importlib.util.spec_from_file_location("airshield_inference", HANDLER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loaded_model(handler, artifact_dir: Path):
    return handler.model_fn(str(artifact_dir))


def test_handler_exposes_the_sagemaker_contract(handler) -> None:
    for name in ("model_fn", "input_fn", "predict_fn", "output_fn"):
        assert hasattr(handler, name), f"handler is missing {name}"


def test_model_fn_loads_the_booster_and_feature_order(handler, loaded_model) -> None:
    assert loaded_model["booster"] is not None
    assert len(loaded_model["feature_names"]) > 0
    assert len(set(loaded_model["feature_names"])) == len(loaded_model["feature_names"])


def test_input_fn_parses_the_documented_payload(handler, loaded_model) -> None:
    names = loaded_model["feature_names"]
    body = json.dumps({"feature_names": names, "rows": [[1.0] * len(names)]})
    frame = handler.input_fn(body)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == names
    assert frame.shape == (1, len(names))


def test_input_fn_rejects_unknown_content_types(handler) -> None:
    with pytest.raises(ValueError, match="unsupported content type"):
        handler.input_fn("{}", content_type="text/csv")


def test_predict_fn_inverts_the_log1p_transform(handler, loaded_model, demo_frame) -> None:
    """Serving must invert log1p exactly as training applied it."""
    from airshield_core.features import latest_feature_row

    subset = demo_frame[demo_frame["location_name"] == "Berlin"].tail(72).reset_index(drop=True)
    features, _ = latest_feature_row(subset)

    result = handler.predict_fn(features, loaded_model)
    values = np.asarray(result).ravel()

    assert len(values) == 1
    assert np.isfinite(values[0])
    assert values[0] >= 0, "expm1 of the log1p target can never be negative"
    # Sanity: within the range the model was trained on.
    assert values[0] < 1000


def test_predict_fn_matches_the_local_predictor(handler, loaded_model, artifact_dir, demo_frame) -> None:
    """SageMaker serving and local serving must agree exactly."""
    from airshield_core.features import latest_feature_row
    from airshield_core.predict import LocalPredictor

    subset = demo_frame[demo_frame["location_name"] == "Delhi"].tail(72).reset_index(drop=True)
    features, _ = latest_feature_row(subset)

    served = float(np.asarray(handler.predict_fn(features, loaded_model)).ravel()[0])
    local = LocalPredictor(artifact_dir).load().predict(features)

    assert served == pytest.approx(local, rel=1e-6)


def test_predict_fn_rejects_mismatched_features(handler, loaded_model) -> None:
    bad = pd.DataFrame({"wrong": [1.0], "columns": [2.0]})
    with pytest.raises(ValueError, match="feature columns do not match"):
        handler.predict_fn(bad, loaded_model)


def test_output_fn_serialises_a_scalar(handler) -> None:
    payload = json.loads(handler.output_fn(np.array([12.5])))
    assert payload == {"prediction": 12.5}


def test_output_fn_serialises_a_batch(handler) -> None:
    payload = json.loads(handler.output_fn(np.array([1.0, 2.0, 3.0])))
    assert payload == {"prediction": [1.0, 2.0, 3.0]}
