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

# Set env vars BEFORE importing backend.app so module-level config picks them up.
os.environ["HF_BINARY_MODEL_ID"] = "test/binary"
os.environ["HF_SUBTYPE_MODEL_ID"] = "test/subtype"
os.environ["HF_TOKEN"] = "test"
os.environ["LOCAL_MODEL_PATH"] = "/tmp/does-not-exist.pt"
os.environ["LOCAL_MODEL_NAME"] = "resnet"

import backend.app as backend_module
from src.inference.schemas import HierarchicalPrediction

FAKE_PREDICTION = HierarchicalPrediction(
    primary_prediction="Pneumonia",
    primary_confidence=0.93,
    subtype_prediction="Bacterial Pneumonia",
    subtype_confidence=0.76,
    probabilities={
        "Normal": 0.07,
        "Pneumonia": 0.93,
        "Bacterial Pneumonia": 0.76,
        "Viral Pneumonia": 0.24,
    },
    model_outputs={"binary": {}, "subtype": {}},
)


def make_fake_prediction(primary="Pneumonia", subtype=None):
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

    # Reset module state for clean test
    backend_module._pipeline = None
    backend_module._local_model = None
    backend_module._local_model_device = None
    backend_module._generate_heatmap = None

    settings = InferenceSettings.from_env()
    backend_module._pipeline = HierarchicalPneumoniaPipeline.from_settings(settings)
    backend_module._pipeline.predict = lambda image: FAKE_PREDICTION

    assert backend_module.EXPLAINABILITY_ENABLED, f"EXPLAINABILITY_ENABLED={backend_module.EXPLAINABILITY_ENABLED}"

    # Test 1: explainability enabled, heatmap succeeds
    backend_module._local_model = object()
    backend_module._local_model_device = "cpu"

    explainability_calls = []

    def fake_heatmap(**kwargs):
        explainability_calls.append(kwargs)
        return "data:image/png;base64,PHASE7"

    with patch.object(
        backend_module, "_load_explainability", return_value=fake_heatmap
    ) as mock1:
        client = TestClient(backend_module.app)
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        print("/health: PASS")

        img = Image.new("RGB", (224, 224), color=(100, 110, 120))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("xray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200
        assert body["primary_prediction"] == "Pneumonia"
        assert body["heatmap_b64"] == "data:image/png;base64,PHASE7", f"got {body['heatmap_b64']!r}"
        assert mock1.call_count == 1
        assert explainability_calls[0]["class_idx"] == 1
        print("/predict with explainability: PASS")

    backend_module._pipeline.predict = lambda image: make_fake_prediction(
        primary="Pneumonia",
        subtype="Viral Pneumonia",
    )
    with patch.object(backend_module, "_load_explainability", return_value=fake_heatmap):
        client = TestClient(backend_module.app)
        img = Image.new("RGB", (224, 224), color=(100, 110, 120))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("viral.png", buf, "image/png")})
        assert r.status_code == 200
        assert explainability_calls[-1]["class_idx"] == 2
        print("/predict viral subtype maps to Grad-CAM index 2: PASS")

    # Test 2: explainability disabled (no local model set → _load_explainability not called)
    backend_module._local_model = None
    backend_module._local_model_device = None
    backend_module._pipeline.predict = lambda image: make_fake_prediction(primary="Normal", subtype=None)

    with patch.object(backend_module, "_load_explainability", return_value=None) as mock2:
        client = TestClient(backend_module.app)
        img = Image.new("RGB", (224, 224), color=(50, 60, 70))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("noxray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200 and body["heatmap_b64"] is None
        # When _local_model is None, the guard clause skips _load_explainability entirely
        assert mock2.call_count == 0, f"expected 0 calls (guarded out), got {mock2.call_count}"
        print("/predict without explainability: PASS (heatmap_b64=None, _load_explainability not called)")

    # Test 3: heatmap generation fails gracefully
    backend_module._local_model = object()
    backend_module._local_model_device = "cpu"
    backend_module._pipeline.predict = lambda image: FAKE_PREDICTION

    def _fail_heatmap(**kwargs):
        raise RuntimeError("heatmap failed")

    with patch.object(backend_module, "_load_explainability", side_effect=_fail_heatmap) as mock3:
        client = TestClient(backend_module.app)
        img = Image.new("RGB", (224, 224), color=(80, 90, 100))
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        r = client.post("/predict", files={"file": ("failxray.png", buf, "image/png")})
        body = r.json()
        assert r.status_code == 200
        assert body["primary_prediction"] == "Pneumonia"
        assert body["heatmap_b64"] is None
        assert mock3.call_count == 1
        print("/predict with heatmap failure: PASS (prediction OK, heatmap_b64=None)")

    print("\nPHASE 7 BACKEND VERIFICATION: PASS")


if __name__ == "__main__":
    main()
