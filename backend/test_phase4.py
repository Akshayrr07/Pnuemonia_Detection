"""Phase 4 verification test for the FastAPI backend."""
import io
import json
import os

# MUST be set before any backend import so lifespan sees them.
os.environ["HF_BINARY_MODEL_ID"] = "test/binary"
os.environ["HF_SUBTYPE_MODEL_ID"] = "test/subtype"

from unittest.mock import patch
from PIL import Image

from fastapi.testclient import TestClient
from src.inference.schemas import ModelPrediction


def make_pred(label, conf, probs):
    return ModelPrediction(label=label, confidence=conf, probabilities=probs)


def main():
    with patch("src.inference.hierarchical_pipeline.HuggingFaceImageClassifier") as MockCls:
        MockCls.return_value.predict.side_effect = [
            make_pred("Pneumonia", 0.93, {"Normal": 0.07, "Pneumonia": 0.93}),
            make_pred("Bacterial Pneumonia", 0.76, {"Bacterial Pneumonia": 0.76, "Viral Pneumonia": 0.24}),
        ]

        import backend.app as backend_module
        from backend.app import app

        print("pipeline at import time:", backend_module._pipeline)

        with TestClient(app) as client:
            print("pipeline inside context:", backend_module._pipeline)

            # /health
            r = client.get("/health")
            assert r.status_code == 200, r.status_code
            body = r.json()
            assert body["status"] == "ok", body
            assert body["pipeline_ready"] is True, body
            print("/health: PASS", body)

            # /predict valid
            img = Image.new("RGB", (224, 224), color=(100, 110, 120))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            r = client.post("/predict", files={"file": ("xray.png", buf, "image/png")})
            assert r.status_code == 200, r.status_code
            body = r.json()
            assert body["primary_prediction"] == "Pneumonia", body
            assert body["subtype_prediction"] == "Bacterial Pneumonia", body
            assert abs(body["primary_confidence"] - 0.93) < 1e-9, body
            assert body["subtype_confidence"] == 0.76, body
            assert "disclaimer" in body, body
            assert "probabilities" in body, body
            assert "model_outputs" in body, body
            print("/predict valid: PASS")
            print("  primary:", body["primary_prediction"], body["primary_confidence"])
            print("  subtype:", body["subtype_prediction"], body["subtype_confidence"])
            print("  probs:", json.dumps(body["probabilities"]))

            # bad content type
            r = client.post("/predict", files={"file": ("bad.txt", b"hello", "text/plain")})
            assert r.status_code == 400, r.status_code
            print("/predict bad type: PASS (400)")

            # oversized
            big = b"\x00" * (11 * 1024 * 1024)
            r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
            assert r.status_code == 400, r.status_code
            print("/predict oversized: PASS (400)")

            # corrupt image
            r = client.post(
                "/predict", files={"file": ("corrupt.png", b"not-an-image", "image/png")}
            )
            assert r.status_code == 400, r.status_code
            print("/predict corrupt: PASS (400)")

            # missing file
            r = client.post("/predict")
            assert r.status_code in (400, 422), r.status_code
            print(f"/predict missing file: PASS ({r.status_code})")

    print("\nALL FASTAPI TESTS: PASS")


if __name__ == "__main__":
    main()
