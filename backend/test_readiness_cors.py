"""Focused tests for fail-closed CORS and liveness/readiness endpoints."""
from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

from fastapi.testclient import TestClient


@contextmanager
def backend_with_origins(
    origins: str | None, *, environment: str = "development"
) -> Iterator[object]:
    """Import a fresh app with isolated CORS environment settings."""
    env = {
        "APP_ENV": environment,
        "ENVIRONMENT": environment,
    }
    with patch.dict(os.environ, env, clear=False):
        # Remove an inherited value when the test is explicitly checking an unset
        # ALLOWED_ORIGINS.
        if origins is None:
            os.environ.pop("ALLOWED_ORIGINS", None)
        else:
            os.environ["ALLOWED_ORIGINS"] = origins
        module = importlib.reload(importlib.import_module("backend.app"))
        yield module


def test_explicitly_configured_origin_is_allowed() -> None:
    with backend_with_origins("https://frontend.example.com") as module:
        with TestClient(module.app) as client:
            response = client.get("/live", headers={"Origin": "https://frontend.example.com"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://frontend.example.com"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_unlisted_origin_is_denied() -> None:
    with backend_with_origins("https://frontend.example.com") as module:
        with TestClient(module.app) as client:
            response = client.get("/live", headers={"Origin": "https://attacker.example.com"})

    assert response.status_code == 200  # CORS is a browser policy, not server auth.
    assert "access-control-allow-origin" not in response.headers


def test_wildcard_with_credentials_is_disabled() -> None:
    with backend_with_origins("*") as module:
        with TestClient(module.app) as client:
            response = client.get("/live", headers={"Origin": "https://anywhere.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


def test_wildcard_does_not_disable_explicit_origin() -> None:
    with backend_with_origins("*,https://frontend.example.com") as module:
        with TestClient(module.app) as client:
            allowed = client.get("/live", headers={"Origin": "https://frontend.example.com"})
            denied = client.get("/live", headers={"Origin": "https://other.example.com"})

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://frontend.example.com"
    assert "access-control-allow-credentials" not in allowed.headers
    assert "access-control-allow-origin" not in denied.headers


def test_default_development_origins_are_local_and_explicit() -> None:
    with backend_with_origins(None, environment="development") as module:
        with TestClient(module.app) as client:
            allowed = client.get("/live", headers={"Origin": "http://localhost:3000"})
            denied = client.get("/live", headers={"Origin": "https://attacker.example.com"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers


def test_production_defaults_to_denied_origins() -> None:
    with backend_with_origins(None, environment="production") as module:
        with TestClient(module.app) as client:
            response = client.get("/live", headers={"Origin": "https://frontend.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_production_can_configure_explicit_origins() -> None:
    with backend_with_origins("https://frontend.example.com", environment="production") as module:
        with TestClient(module.app) as client:
            response = client.get("/live", headers={"Origin": "https://frontend.example.com"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://frontend.example.com"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_live_is_available_without_pipeline() -> None:
    with backend_with_origins(None) as module:
        module._pipeline = None
        with TestClient(module.app) as client:
            response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_until_pipeline_is_ready() -> None:
    with backend_with_origins(None) as module:
        module._pipeline = None
        with TestClient(module.app) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "pipeline_ready": False,
    }


def test_ready_returns_200_when_pipeline_is_ready() -> None:
    with backend_with_origins(None) as module:
        module._pipeline = object()
        with TestClient(module.app) as client:
            response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_ready": True,
    }
