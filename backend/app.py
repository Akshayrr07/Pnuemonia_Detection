"""
Pneumonia Detection API — Phase 4 FastAPI backend.

Run from the repository root so that ``src/`` is importable::

    PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Expected environment variables (see src/inference/settings.py):
    HF_BINARY_MODEL_ID
    HF_SUBTYPE_MODEL_ID
    HF_TOKEN              (optional)
    HF_API_BASE_URL       (optional, defaults to HF Inference API)
    PNEUMONIA_THRESHOLD   (optional, default 0.5)
    SUBTYPE_THRESHOLD     (optional, default 0.5)
    HF_REQUEST_TIMEOUT_SECONDS (optional, default 60)

    UPLOAD_MAX_SIZE_MB    (optional, default 10)
    ALLOWED_ORIGINS       (optional, comma-separated CORS origins)

Optional explainability (Grad-CAM) environment variables:
    LOCAL_MODEL_PATH      path to a local .pt checkpoint (enables heatmap generation)
    LOCAL_MODEL_NAME      model family name for Grad-CAM target layer resolution
                          (default: resnet; supported: resnet, mobilenet, efficientnet, densenet)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.inference.hierarchical_pipeline import HierarchicalPneumoniaPipeline
from src.inference.schemas import HierarchicalPrediction
from src.inference.settings import InferenceSettings, InferenceSettingsError

# Lazy import for explainability: numpy + torch are heavy and only needed when
# LOCAL_MODEL_PATH is set. Importing them unconditionally would bloat the Docker
# image for deployments that don't use Grad-CAM.
_generate_heatmap = None


def _load_explainability():
    global _generate_heatmap
    if _generate_heatmap is not None:
        return _generate_heatmap
    try:
        from src.inference.explainability import generate_heatmap as _gm

        _generate_heatmap = _gm
    except Exception:
        _generate_heatmap = None
    return _generate_heatmap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Optional explainability (Grad-CAM) configuration.
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "resnet")
EXPLAINABILITY_ENABLED = bool(LOCAL_MODEL_PATH)

# ---------------------------------------------------------------------------
# Lifespan — build the pipeline once on startup
# ---------------------------------------------------------------------------

_pipeline: Optional[HierarchicalPneumoniaPipeline] = None
_local_model = None
_local_model_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _local_model, _local_model_device
    try:
        settings = InferenceSettings.from_env()
    except InferenceSettingsError:
        settings = None

    if settings is not None:
        _pipeline = HierarchicalPneumoniaPipeline.from_settings(settings)

    # Load local model for explainability if configured.
    if EXPLAINABILITY_ENABLED and LOCAL_MODEL_PATH:
        try:
            import torch
            from src.models.model_factory import get_model

            _local_model = get_model(LOCAL_MODEL_NAME, num_classes=3, freeze=False)
            state_dict = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
            _local_model.load_state_dict(state_dict)
            _local_model.eval()
            _local_model_device = torch.device("cpu")
        except Exception:
            _local_model = None
            _local_model_device = None

    yield

    _pipeline = None
    _local_model = None
    _local_model_device = None


app = FastAPI(
    title="Pneumonia Detection API",
    description=(
        "Hierarchical chest X-ray classification: Normal vs Pneumonia, "
        "then Bacterial vs Viral subtype when pneumonia is detected."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
_raw_origins = __import__("os").getenv("ALLOWED_ORIGINS", "")
if _raw_origins.strip():
    _allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    _allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    pipeline_ready: bool = False


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(pipeline_ready=_pipeline is not None)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    primary_prediction: str
    primary_confidence: float
    subtype_prediction: Optional[str]
    subtype_confidence: Optional[float]
    probabilities: dict
    model_outputs: dict
    disclaimer: str
    heatmap_b64: Optional[str] = None


def _validate_upload(file: UploadFile) -> Image.Image:
    """Validate type/size and return an opened RGB PIL Image."""
    if not file.content_type or file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type or 'missing'}. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            ),
        )

    contents = file.file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > UPLOAD_MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {size_mb:.2f} MB (max {UPLOAD_MAX_SIZE_MB} MB).",
        )

    ext = __import__("os").path.splitext(file.filename or "")[1].lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension: {ext or 'none'}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        image = Image.open(__import__("io").BytesIO(contents))
        image.load()  # force decode so we fail early on corrupt data
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    return image


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction pipeline is not available. "
                "Check that HF_BINARY_MODEL_ID and HF_SUBTYPE_MODEL_ID are set."
            ),
        )

    image = _validate_upload(file)

    # Run the hierarchical prediction pipeline (HF hosted models).
    prediction: HierarchicalPrediction = _pipeline.predict(image)

    # Optionally generate a Grad-CAM heatmap from the local model.
    heatmap_b64: Optional[str] = None
    if EXPLAINABILITY_ENABLED and _local_model is not None and _local_model_device is not None:
        try:
            # Local checkpoints are three-class models, so only explain a
            # concrete Normal/Bacterial/Viral class. The binary pipeline's
            # aggregate "Pneumonia" result has no corresponding local output.
            if prediction.primary_prediction == "Normal":
                explain_class_name = "Normal"
            elif prediction.subtype_prediction in {
                "Bacterial Pneumonia",
                "Viral Pneumonia",
            }:
                explain_class_name = prediction.subtype_prediction
            else:
                raise ValueError(
                    "Cannot explain aggregate Pneumonia without a concrete subtype"
                )

            from src.inference.explainability import grad_cam_class_index

            heatmap_b64 = _load_explainability()(
                model=_local_model,
                model_name=LOCAL_MODEL_NAME,
                image=image,
                class_idx=grad_cam_class_index(explain_class_name),
                device=_local_model_device,
            )
        except Exception:
            heatmap_b64 = None

    return PredictResponse(
        primary_prediction=prediction.primary_prediction,
        primary_confidence=prediction.primary_confidence,
        subtype_prediction=prediction.subtype_prediction,
        subtype_confidence=prediction.subtype_confidence,
        probabilities=prediction.probabilities,
        model_outputs=prediction.model_outputs,
        disclaimer=prediction.disclaimer,
        heatmap_b64=heatmap_b64,
    )
