"""Reliability tests for the hosted inference route."""
from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

# Keep the module import deterministic and avoid contacting a real provider.
os.environ.setdefault("HF_BINARY_MODEL_ID", "test/binary")
os.environ.setdefault("HF_SUBTYPE_MODEL_ID", "test/subtype")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("PREDICTION_DEADLINE_SECONDS", "1")
os.environ.setdefault("INFERENCE_MAX_CONCURRENCY", "2")
os.environ.setdefault("INFERENCE_WORKERS", "2")
os.environ.setdefault("INFERENCE_RATE_LIMIT_PER_SECOND", "0")
os.environ.pop("LOCAL_MODEL_PATH", None)

import backend.app as backend_module
from backend.inference_reliability import ConcurrencyRateGuard, InferenceCapacityError
from src.inference.huggingface_adapter import HuggingFaceInferenceError
from src.inference.schemas import HierarchicalPrediction


PREDICTION = HierarchicalPrediction(
    primary_prediction="Pneumonia",
    primary_confidence=0.93,
    subtype_prediction="Bacterial Pneumonia",
    subtype_confidence=0.76,
    probabilities={"Normal": 0.07, "Pneumonia": 0.93},
    model_outputs={"binary": {}, "subtype": {}},
)


class Pipeline:
    def __init__(self, result=PREDICTION, error=None, started=None, release=None):
        self.result = result
        self.error = error
        self.started = started
        self.release = release
        self.thread_ids = []

    def predict(self, image):
        self.thread_ids.append(threading.get_ident())
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result(image)
        return self.result


def upload() -> dict:
    image = Image.new("RGB", (32, 32), color=(100, 110, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return {"file": ("xray.png", buffer, "image/png")}


def configure_runtime(monkeypatch, *, deadline=1.0, max_concurrency=2, rate=0):
    monkeypatch.setattr(backend_module, "_prediction_deadline_seconds", deadline)
    monkeypatch.setattr(
        backend_module,
        "_inference_guard",
        ConcurrencyRateGuard(
            max_concurrency=max_concurrency,
            max_requests_per_second=rate,
        ),
        raising=False,
    )
    backend_module._local_model = None
    backend_module._local_model_device = None


def test_concurrency_guard_rejects_when_full_and_releases_slot():
    guard = ConcurrencyRateGuard(max_concurrency=1, max_requests_per_second=0)

    lease = guard.acquire()
    with pytest.raises(InferenceCapacityError):
        guard.acquire()

    lease.release()
    lease = guard.acquire()
    assert lease is not None
    lease.release()


def test_rate_limit_guard_rejects_bursts():
    guard = ConcurrencyRateGuard(max_concurrency=2, max_requests_per_second=1)

    first = guard.acquire()
    first.release()
    with pytest.raises(InferenceCapacityError):
        guard.acquire()


def test_route_returns_successful_prediction_off_event_loop(monkeypatch):
    pipeline = Pipeline()
    configure_runtime(monkeypatch)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        response = client.post("/predict", files=upload())

    assert response.status_code == 200
    body = response.json()
    assert body["primary_prediction"] == "Pneumonia"
    assert body["subtype_prediction"] == "Bacterial Pneumonia"
    assert pipeline.thread_ids
    assert pipeline.thread_ids[0] != threading.get_ident()


def test_route_maps_total_deadline_to_504_without_raw_details(monkeypatch):
    pipeline = Pipeline(error=None, result=lambda image: time.sleep(0.2) or PREDICTION)
    configure_runtime(monkeypatch, deadline=0.02)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        response = client.post("/predict", files=upload())

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()
    assert "traceback" not in response.text.lower()


def test_route_maps_known_provider_error_to_502_without_raw_body(monkeypatch):
    secret = "UPSTREAM_SECRET_SHOULD_NOT_LEAK"
    pipeline = Pipeline(error=HuggingFaceInferenceError(f"provider body: {secret}"))
    configure_runtime(monkeypatch)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        response = client.post("/predict", files=upload())

    assert response.status_code == 502
    assert "upstream" in response.json()["detail"].lower()
    assert secret not in response.text
    assert "provider body" not in response.text


def test_route_maps_transport_provider_error_to_502(monkeypatch):
    secret = "TOKEN_IN_TRANSPORT_ERROR"
    pipeline = Pipeline(error=requests.ConnectionError(secret))
    configure_runtime(monkeypatch)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        response = client.post("/predict", files=upload())

    assert response.status_code == 502
    assert secret not in response.text


def test_route_maps_provider_timeout_to_504(monkeypatch):
    secret = "TOKEN_IN_PROVIDER_TIMEOUT"
    pipeline = Pipeline(error=requests.Timeout(secret))
    configure_runtime(monkeypatch)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        response = client.post("/predict", files=upload())

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()
    assert secret not in response.text


def test_route_returns_503_when_inference_capacity_is_exhausted(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    pipeline = Pipeline(started=started, release=release)
    configure_runtime(monkeypatch, max_concurrency=1)
    backend_module._pipeline = pipeline

    with TestClient(backend_module.app) as client:
        backend_module._pipeline = pipeline
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(lambda: client.post("/predict", files=upload()))
            assert started.wait(timeout=1)
            second = client.post("/predict", files=upload())
            release.set()
            first_response = first.result(timeout=2)

    assert first_response.status_code == 200
    assert second.status_code == 503
    assert "busy" in second.json()["detail"].lower()
