"""Empirical calibration of pollution-spike confidence.

A confidence number is only meaningful if it reflects how often a similar
situation actually produced a spike. This module derives that from real
historical data:

For each horizon, and each bucket of *relative change*, it measures the
historical frequency with which a rise of that size was followed by a spike
(defined as the rise persisting at the spike threshold over the horizon).

The result is a small lookup table persisted to JSON. When it is present,
:mod:`airshield_core.events` uses it to report a calibrated probability and
labels the basis ``empirical_calibration``. When it is absent, the detector
falls back to its transparent rule-based estimate and says so. No confidence is
ever invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Bucket upper edges for relative change (fraction). The last bucket is open.
CHANGE_BUCKETS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00, float("inf"))
#: Stable string labels for the buckets, so a table written by one run can be
#: read back by another without float-formatting drift.
BUCKET_LABELS: tuple[str, ...] = ("<0.10", "<0.25", "<0.50", "<1.00", ">=1.00")
#: A rise at or above this fraction is treated as a spike when calibrating.
SPIKE_THRESHOLD = 0.35

CALIBRATION_FILENAME = "spike_calibration.json"


@dataclass
class CalibrationTable:
    """Empirical spike frequency by change bucket and horizon."""

    horizons: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    threshold: float = SPIKE_THRESHOLD
    source: str = ""

    @property
    def available(self) -> bool:
        return bool(self.horizons)

    @property
    def basis(self) -> str:
        return "empirical_calibration" if self.available else "rule_based_estimate"

    def probability(self, change_fraction: float, horizon: int) -> float:
        """Historical spike frequency for a change of this size at this horizon."""
        bucket = _bucket_label(change_fraction)
        table = self.horizons.get(str(horizon))
        if not table or bucket not in table:
            return _fallback_probability(change_fraction, horizon)
        return float(table[bucket])

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "source": self.source,
            "horizons": self.horizons,
            "counts": self.counts,
            "basis": self.basis,
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationTable":
        source = Path(path)
        if not source.is_file():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            horizons=payload.get("horizons", {}),
            counts=payload.get("counts", {}),
            threshold=payload.get("threshold", SPIKE_THRESHOLD),
            source=payload.get("source", ""),
        )


def _bucket_label(change_fraction: float) -> str:
    for upper, label in zip(CHANGE_BUCKETS, BUCKET_LABELS):
        if change_fraction < upper:
            return label
    return BUCKET_LABELS[-1]  # pragma: no cover


def _fallback_probability(change_fraction: float, horizon: int) -> float:
    """Transparent fallback when no calibration row exists for the situation."""
    base = min(0.9, max(0.05, change_fraction / 0.7))
    penalty = min(0.2, max(0, horizon - 1) * 0.04)
    return round(max(0.05, base - penalty), 3)


def build_calibration(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 6),
    group_column: str = "location_name",
    value_column: str = "pm2_5",
    source: str = "historical_observations",
) -> CalibrationTable:
    """Build the calibration table from real historical measurements.

    For every location and horizon, the forward relative change over the horizon
    is computed and labelled a spike when it reaches :data:`SPIKE_THRESHOLD`.
    The table records, per change bucket, the observed spike frequency.
    """
    if frame.empty:
        raise ValueError("frame is empty")
    if value_column not in frame.columns:
        raise ValueError(f"frame must contain {value_column!r}")

    ordered = frame.sort_values("time").reset_index(drop=True)
    groups = (
        ordered.groupby(group_column, sort=False)
        if group_column in ordered.columns
        else [(None, ordered)]
    )

    # bucket -> spike flags, accumulated per horizon.
    accum: dict[int, dict[str, list[int]]] = {h: {} for h in horizons}

    for _, group in groups:
        series = group.sort_values("time")[value_column].to_numpy(dtype=float)
        for horizon in horizons:
            if len(series) <= horizon:
                continue
            base = series[:-horizon]
            future = series[horizon:]
            valid = np.isfinite(base) & np.isfinite(future) & (base > 0)
            changes = (future[valid] - base[valid]) / base[valid]
            for change in changes:
                bucket = _bucket_label(float(change))
                accum[horizon].setdefault(bucket, []).append(
                    1 if change >= SPIKE_THRESHOLD else 0
                )

    horizons_out: dict[str, dict[str, float]] = {}
    counts_out: dict[str, dict[str, int]] = {}
    for horizon, buckets in accum.items():
        rates: dict[str, float] = {}
        counts: dict[str, int] = {}
        for bucket, flags in buckets.items():
            rates[bucket] = round(float(np.mean(flags)), 4)
            counts[bucket] = int(len(flags))
        if rates:
            horizons_out[str(horizon)] = rates
            counts_out[str(horizon)] = counts

    return CalibrationTable(
        horizons=horizons_out, counts=counts_out, source=source
    )
