"""Entry point executed inside the SageMaker XGBoost training container.

SageMaker copies the ``source_dir`` into the container and runs this script with
hyperparameters injected as ``--name value`` CLI arguments. Real training data is
mounted at ``/opt/ml/input/data/training/``, and the artifact must be written to
``/opt/ml/model/`` as ``model.tar.gz``.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

# The AirShield core package is bundled with this source_dir by the launcher.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from airshield_core.train import summarize, train_model  # noqa: E402

INPUT_DIR = Path("/opt/ml/input/data/training")
MODEL_DIR = Path("/opt/ml/model")


def _locate_csv() -> Path:
    """Find the training CSV that SageMaker mounted."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"training input directory {INPUT_DIR} does not exist")

    candidates = sorted(INPUT_DIR.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no CSV found under {INPUT_DIR}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rounds", type=int, default=400)
    parser.add_argument("--days", type=int, default=90)
    # SageMaker may pass extra keys from the hyperparameters dict; ignore them.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ignoring unrecognised hyperparameters: {unknown}")

    csv_path = _locate_csv()
    print(f"loading real training data from {csv_path}")
    frame = pd.read_csv(csv_path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    print(f"loaded {len(frame)} rows covering {frame['time'].min()} -> {frame['time'].max()}")

    locations = (
        sorted(frame["location_name"].unique().tolist())
        if "location_name" in frame.columns
        else []
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = train_model(
            frame,
            artifact_dir=tmp,
            num_rounds=args.num_rounds,
            locations=locations,
            data_source={
                "kind": "sagemaker-training-input",
                "s3_input": str(INPUT_DIR),
                "rows": int(len(frame)),
            },
        )

        print()
        print(summarize(result))

        # SageMaker expects a model.tar.gz containing model.ubj + metadata.json.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = MODEL_DIR / "model.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for name in ("model.ubj", "metadata.json"):
                source = Path(tmp) / name
                if source.is_file():
                    tar.add(source, arcname=name)

    print(f"\nwrote {archive_path} ({archive_path.stat().st_size} bytes)")

    # Emit metrics in the format SageMaker parses into CloudWatch.
    metrics = result.metrics
    print(json.dumps({"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
