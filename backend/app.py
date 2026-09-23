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
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.inference.hierarchical_pipeline import HierarchicalPneumoniaPipeline
from src.inference.schemas import HierarchicalPrediction
from src.inference.settings import InferenceSettings, InferenceSettingsError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPLOAD_MAX_SIZE_MB = int(__import__("os").getenv("UPLOAD_MAX_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Lifespan — build the pipeline once on startup
# ---------------------------------------------------------------------------

_pipeline: Optional[HierarchicalPneumoniaPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        settings = InferenceSettings.from_env()
    except InferenceSettingsError as exc:
        # If required env vars are missing, the app still starts but /predict
        # will return a clear error instead of crashing the server.
        settings = None

    if settings is not None:
        _pipeline = HierarchicalPneumoniaPipeline.from_settings(settings)

    yield

    _pipeline = None


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
    prediction: HierarchicalPrediction = _pipeline.predict(image)
    return PredictResponse(**prediction.to_dict())
