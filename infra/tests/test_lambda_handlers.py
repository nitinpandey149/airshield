"""Tests for the AirShield Pulse AWS Lambda handlers.

The handlers are exercised with a stubbed ``urlopen`` and stub AWS clients, so
the tests assert the real control flow - what gets stored, what gets published,
and crucially that nothing is published or stored when there is no data or no
spike.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = INFRA / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ingest_handler = _load("lambda_ingest_handler", "lambda/ingest/handler.py")
spike_handler = _load("lambda_spike_handler", "lambda/spike/handler.py")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _air_payload(times: list[str], pm25: list[float | None]) -> dict:
    return {
        "hourly": {
            "time": times,
            "pm2_5": pm25,
            "pm10": [20.0] * len(times),
            "nitrogen_dioxide": [10.0] * len(times),
            "ozone": [40.0] * len(times),
        }
    }


def _weather_payload(times: list[str]) -> dict:
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [12.0] * len(times),
            "relative_humidity_2m": [70.0] * len(times),
            "wind_speed_10m": [8.0] * len(times),
            "wind_direction_10m": [220.0] * len(times),
            "surface_pressure": [1012.0] * len(times),
            "precipitation": [0.0] * len(times),
        }
    }


# --------------------------------------------------------------------- ingest
def test_ingest_stores_real_payload_and_latest_measurement(monkeypatch):
    times = ["2026-10-02T09:00", "2026-10-02T10:00"]
    payloads = [_air_payload(times, [11.0, 13.5]), _weather_payload(times)]

    def fake_urlopen(request, timeout=20.0):  # noqa: ANN001
        return _FakeResponse(payloads.pop(0))

    monkeypatch.setattr(ingest_handler.urllib.request, "urlopen", fake_urlopen)

    stored: dict = {}
    items: list[dict] = []

    class _S3:
        def put_object(self, **kwargs):  # noqa: ANN003
            stored.update(kwargs)

    class _Table:
        def put_item(self, Item):  # noqa: N803 - boto3 keyword
            items.append(Item)

    class _Dynamo:
        def Table(self, name):  # noqa: N802 - boto3 method
            return _Table()

    class _Boto3:
        @staticmethod
        def client(name):  # noqa: ANN001
            return _S3()

        @staticmethod
        def resource(name):  # noqa: ANN001
            return _Dynamo()

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    monkeypatch.setenv("AIRSHIELD_BUCKET", "bucket")
    monkeypatch.setenv("AIRSHIELD_TABLE", "table")
    monkeypatch.setenv("AIRSHIELD_LOCATION_SLUG", "berlin")

    result = ingest_handler.handler({}, None)

    assert stored["Bucket"] == "bucket"
    assert stored["Key"].startswith("samples/berlin/")
    body = json.loads(stored["Body"].decode("utf-8"))
    assert body["rows"][0]["pm2_5"] == 11.0
    assert body["source"]["name"] == "Open-Meteo"

    assert len(items) == 1
    assert items[0]["pm2_5"] == "13.5"
    assert items[0]["pk"] == "LOCATION#berlin"
    assert result["pm2_5"] == 13.5


def test_ingest_fails_when_no_measured_pm25(monkeypatch):
    times = ["2026-10-02T09:00", "2026-10-02T10:00"]
    payloads = [_air_payload(times, [None, None]), _weather_payload(times)]

    def fake_urlopen(request, timeout=20.0):  # noqa: ANN001
        return _FakeResponse(payloads.pop(0))

    monkeypatch.setattr(ingest_handler.urllib.request, "urlopen", fake_urlopen)

    class _S3:
        def put_object(self, **kwargs):  # noqa: ANN003
            pass

    class _Dynamo:
        def Table(self, name):  # noqa: N802
            raise AssertionError("must not write state when there is no measurement")

    class _Boto3:
        @staticmethod
        def client(name):  # noqa: ANN001
            return _S3()

        @staticmethod
        def resource(name):  # noqa: ANN001
            return _Dynamo()

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    monkeypatch.setenv("AIRSHIELD_BUCKET", "bucket")
    monkeypatch.setenv("AIRSHIELD_TABLE", "table")

    with pytest.raises(RuntimeError):
        ingest_handler.handler({}, None)


# ---------------------------------------------------------------------- spike
def _forecast(spike: dict) -> dict:
    return {
        "location": {"label": "Berlin, Germany"},
        "current": {"pm2_5": 18.0},
        "spike": spike,
    }


def _spike_env(monkeypatch, published: list, stored: list):
    class _Sns:
        def publish(self, **kwargs):  # noqa: ANN003
            published.append(kwargs)

    class _Table:
        def __init__(self) -> None:
            self.items: dict[tuple, dict] = {}

        def get_item(self, Key):  # noqa: N803
            lookup = (Key["pk"], Key["sk"])
            return {"Item": self.items[lookup]} if lookup in self.items else {}

        def put_item(self, Item):  # noqa: N803
            self.items[(Item["pk"], Item["sk"])] = Item
            stored.append(Item)

    table = _Table()

    class _Dynamo:
        def Table(self, name):  # noqa: N802
            return table

    class _Boto3:
        @staticmethod
        def client(name):  # noqa: ANN001
            return _Sns()

        @staticmethod
        def resource(name):  # noqa: ANN001
            return _Dynamo()

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    monkeypatch.setenv("AIRSHIELD_SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:1:t")
    monkeypatch.setenv("AIRSHIELD_TABLE", "table")
    monkeypatch.setenv("AIRSHIELD_LOCATION_SLUG", "berlin")


def test_spike_publishes_and_dedupes(monkeypatch):
    spike = {
        "spike_detected": True,
        "severity": "moderate",
        "expected_time": "2026-10-02T12:00:00+00:00",
        "expected_change_percent": 35.0,
        "confidence": 0.78,
        "confidence_basis": "calibrated",
    }

    def fake_urlopen(request, timeout=20.0):  # noqa: ANN001
        return _FakeResponse(_forecast(spike))

    monkeypatch.setattr(spike_handler.urllib.request, "urlopen", fake_urlopen)

    published: list = []
    stored: list = []
    _spike_env(monkeypatch, published, stored)

    first = spike_handler.handler({}, None)
    assert first["notified"] is True
    assert len(published) == 1
    assert "pollution spike" in published[0]["Subject"].lower()
    assert "+35%" in published[0]["Message"]
    assert "78%" in published[0]["Message"]
    assert len(stored) == 1

    # A second run in the same expected hour must not notify again.
    second = spike_handler.handler({}, None)
    assert second["notified"] is False
    assert len(published) == 1


def test_spike_does_not_publish_when_no_spike(monkeypatch):
    spike = {
        "spike_detected": False,
        "severity": "none",
        "expected_time": None,
        "expected_change_percent": 0.0,
        "confidence": 0.0,
        "confidence_basis": "not_detected",
    }

    def fake_urlopen(request, timeout=20.0):  # noqa: ANN001
        return _FakeResponse(_forecast(spike))

    monkeypatch.setattr(spike_handler.urllib.request, "urlopen", fake_urlopen)

    class _Sns:
        def publish(self, **kwargs):  # noqa: ANN003
            raise AssertionError("must not publish without a spike")

    class _Boto3:
        @staticmethod
        def client(name):  # noqa: ANN001
            return _Sns()

        @staticmethod
        def resource(name):  # noqa: ANN001
            raise AssertionError("must not touch DynamoDB without a spike")

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    monkeypatch.setenv("AIRSHIELD_SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:1:t")

    result = spike_handler.handler({}, None)
    assert result["notified"] is False


def test_spike_message_avoids_causal_and_safety_claims():
    spike = {
        "spike_detected": True,
        "severity": "severe",
        "expected_time": "2026-10-02T12:00:00+00:00",
        "expected_change_percent": 60.0,
        "confidence": 0.9,
        "confidence_basis": "calibrated",
    }
    _subject, body = spike_handler._message(_forecast(spike))
    lowered = body.lower()
    assert "not medical advice" in lowered
    assert "not proven causes" in lowered
    assert " safe " not in f" {lowered} "
