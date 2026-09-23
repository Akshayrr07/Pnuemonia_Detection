"""Phase 7 verification test for backend explainability (Grad-CAM integration).

Exercises the /predict path with EXPLAINABILITY_ENABLED=True and verifies that
the heatmap_b64 field flows back through the response.
"""
from __future__ import annotations

import io
import os
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient

import backend.app as backend_module
from src.inference.schemas import HierarchicalPrediction

os.environ["HF_BINARY_MODEL_ID"] = "test/binary"
os.environ["HF_SUBTYPE_MODEL_ID"] = "test/subtype"
os.environ["LOCAL_MODEL_PATH"] = "/tmp/does-not-exist.pt"
os.environ["LOCAL_MODEL_NAME"] = "resnet"


def make_fake_prediction(primary="Pneumonia", subtype: str | None = "Bacterial Pneumonia"):
    """Return a fake HierarchicalPrediction for testing."""
    return HierarchicalPrediction(
        primary_prediction=primary,
        primary_confidence=0.93,
        subtype_prediction=subtype,
        subtype_confidence=0.76 if subtype else None,
        probabilities={
            "Normal": 0.07,
            "Pneumonia": 0.93,
            "Bacterial Pneumonia": 0.76,
            "Viral Pneumonia": 0.24,
        },
        model_outputs={
            "binary": {"label": primary, "confidence": 0.93, "probabilities": {"Normal": 0.07, "Pneumonia": 0.93}},
            "subtype": {"label": subtype, "confidence": 0.76, "probabilities": {"Bacterial Pneumonia": 0.76, "Viral Pneumonia": 0.24}} if subtype else {},
        },
    )


def main() -> None:
    from src.inference.hierarchical_pipeline import HierarchicalPneumoniaPipeline
    from src.inference.settings import InferenceSettings

    settings = InferenceSettings.from_env()
    backend_module._pipeline = HierarchicalPneumoniaPipeline.from_settings(settings)

    # ===== Test 1: explainability enabled, heatmap succeeds =====
    backend_module._local_model = object()
    backend_module._local_model_device = "cpu"

    with patch.object(backend_module, "generate_heatmap", return_value="data:image/png;base64,PHASE7") as mock_hm:
        client = TestClient(backend_module.app)

        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        print("/health: PASS")

        backend_module._pipeline.predict = lambda image: make_fake_prediction()
        img = Image.new("RGB", (224, 224), color=(100, 110, 120))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)

        r = client.post("/predict", files={"file": ("xray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200
        assert body["primary_prediction"] == "Pneumonia"
        assert body["heatmap_b64"] == "data:image/png;base64,PHASE7"
        mock_hm.assert_called_once()
        print("/predict with explainability: PASS")

    # ===== Test 2: explainability disabled (no local model) =====
    backend_module._local_model = None
    backend_module._local_model_device = None
    backend_module._pipeline.predict = lambda image: make_fake_prediction(primary="Normal", subtype=None)

    with patch.object(backend_module, "generate_heatmap") as mock_hm:
        client = TestClient(backend_module.app)
        img = Image.new("RGB", (224, 224), color=(50, 60, 70))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("noxray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200 and body["heatmap_b64"] is None
        mock_hm.assert_not_called()
        print("/predict without explainability: PASS (heatmap_b64=None)")

    # ===== Test 3: heatmap generation fails gracefully =====
    backend_module._local_model = object()
    backend_module._local_model_device = "cpu"
    backend_module._pipeline.predict = lambda image: make_fake_prediction()

    with patch.object(backend_module, "generate_heatmap", side_effect=RuntimeError("heatmap failed")) as mock_hm_fail:
        client = TestClient(backend_module.app)
        img = Image.new("RGB", (224, 224), color=(80, 90, 100))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("failxray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200
        assert body["primary_prediction"] == "Pneumonia"
        assert body["heatmap_b64"] is None
        mock_hm_fail.assert_called_once()
        print("/predict with heatmap failure: PASS (prediction OK, heatmap_b64=None)")

    print("\nPHASE 7 BACKEND VERIFICATION: PASS")


if __name__ == "__main__":
    main()
