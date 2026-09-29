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
    PREDICTION_DEADLINE_SECONDS (optional total route deadline, default 65)
    INFERENCE_MAX_CONCURRENCY (optional process-local limit, default 4)
    INFERENCE_WORKERS       (optional bounded thread-pool size)
    INFERENCE_RATE_LIMIT_PER_SECOND (optional local request rate limit, 0=off)

    UPLOAD_MAX_SIZE_MB    (optional, default 10)
    MAX_IMAGE_WIDTH       (optional, default 8192)
    MAX_IMAGE_HEIGHT      (optional, default 8192)
    MAX_IMAGE_PIXELS      (optional, default 40000000)
    MAX_IMAGE_FRAMES      (optional, default 1)
    ALLOWED_ORIGINS       (optional, comma-separated CORS origins)
    APP_ENV               (optional, development or production; defaults to development)

Optional explainability (Grad-CAM) environment variables:
    LOCAL_MODEL_PATH      path to a local .pt checkpoint (enables heatmap generation)
    LOCAL_MODEL_NAME      model family name for Grad-CAM target layer resolution
                          (default: resnet; supported: resnet, mobilenet, efficientnet, densenet)
"""

from __future__ import annotations

import asyncio
import io
import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from src.inference.checkpoint import load_trusted_checkpoint
from src.inference.hierarchical_pipeline import HierarchicalPneumoniaPipeline
from src.inference.huggingface_adapter import HuggingFaceInferenceError
from src.inference.schemas import HierarchicalPrediction
from src.inference.settings import InferenceSettings, InferenceSettingsError

from .inference_reliability import (
    ConcurrencyRateGuard,
    InferenceCapacityError,
    InferenceReliabilityConfig,
)

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
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "8192"))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "8192"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
MAX_IMAGE_FRAMES = int(os.getenv("MAX_IMAGE_FRAMES", "1"))

_RELIABILITY_CONFIG = InferenceReliabilityConfig.from_env()
_prediction_deadline_seconds = _RELIABILITY_CONFIG.deadline_seconds
_inference_executor: Optional[ThreadPoolExecutor] = None
_inference_guard: Optional[ConcurrencyRateGuard] = None
_runtime_lock = threading.Lock()


def _ensure_inference_runtime():
    """Create bounded inference primitives for lifespan and direct route tests."""

    global _inference_executor, _inference_guard
    with _runtime_lock:
        if _inference_executor is None:
            _inference_executor = ThreadPoolExecutor(
                max_workers=_RELIABILITY_CONFIG.worker_count,
                thread_name_prefix="pneumonia-inference",
            )
        if _inference_guard is None:
            _inference_guard = ConcurrencyRateGuard(
                max_concurrency=_RELIABILITY_CONFIG.max_concurrency,
                max_requests_per_second=_RELIABILITY_CONFIG.max_requests_per_second,
            )
    return _inference_executor, _inference_guard

# CORS defaults to a small, explicit development allowlist. Production must
# opt in with APP_ENV/ENVIRONMENT=production and an explicit origin list; an
# unset or blank allowlist is never widened to "*".
DEVELOPMENT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
_app_environment = (
    os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")) or "development"
).strip().lower()
_is_development = _app_environment in {"development", "dev", "local"}


def _resolve_cors_origins() -> tuple[list[str], bool]:
    """Return explicit origins and whether credentials are safe to enable.

    A wildcard is deliberately discarded. This keeps the service fail-closed
    and prevents the browser-observable ``Access-Control-Allow-Origin: *`` plus
    ``Access-Control-Allow-Credentials: true`` combination. If a wildcard is
    present alongside concrete origins, the concrete origins are retained and
    credentials are disabled for the entire policy.
    """
    raw_origins = os.getenv("ALLOWED_ORIGINS", "")
    configured = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if not configured:
        origins = list(DEVELOPMENT_ALLOWED_ORIGINS) if _is_development else []
        return origins, True

    has_wildcard = any("*" in origin for origin in configured)
    origins = [origin for origin in configured if "*" not in origin]
    return origins, not has_wildcard

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
    global _inference_executor, _inference_guard

    _ensure_inference_runtime()
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
            state_dict = load_trusted_checkpoint(
                LOCAL_MODEL_PATH,
                map_location="cpu",
                expected_task="3_class",
            )
            _local_model.load_state_dict(state_dict, strict=True)
            _local_model.eval()
            _local_model_device = torch.device("cpu")
        except Exception:
            _local_model = None
            _local_model_device = None

    yield

    if _inference_executor is not None:
        _inference_executor.shutdown(wait=True, cancel_futures=True)
    _inference_executor = None
    _inference_guard = None
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
_allow_origins, _allow_credentials = _resolve_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
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


class LiveResponse(BaseModel):
    status: str = "ok"


@app.get("/live", response_model=LiveResponse, tags=["System"])
async def live():
    """Report process liveness without depending on the inference pipeline."""
    return LiveResponse()


class ReadinessResponse(BaseModel):
    status: str
    pipeline_ready: bool


@app.get("/ready", response_model=ReadinessResponse, tags=["System"])
async def ready():
    """Report whether the inference pipeline is available for traffic."""
    if _pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "pipeline_ready": False},
        )
    return ReadinessResponse(status="ok", pipeline_ready=True)


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


def _read_bounded_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read at most ``max_bytes + 1`` and reject one byte beyond the cap."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = file.file.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)

    contents = b"".join(chunks)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image too large: {len(contents) / (1024 * 1024):.2f} MB "
                f"(max {UPLOAD_MAX_SIZE_MB} MB)."
            ),
        )
    return contents


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

    # Content length is advisory; always enforce the byte cap against bytes
    # actually read from the stream, since the request can be mislabeled.
    max_upload_bytes = UPLOAD_MAX_SIZE_MB * 1024 * 1024
    contents = _read_bounded_upload(file, max_upload_bytes)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension: {ext or 'none'}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Read metadata and validate it before loading pixel data. Keep the warning
    # policy scoped to this operation so a decompression-bomb warning cannot
    # be silently ignored by a process-wide warning filter.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                actual_format = image.format
                if actual_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unsupported image format: {actual_format or 'unknown'}. "
                            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}"
                        ),
                    )

                width, height = image.size
                if (
                    width > MAX_IMAGE_WIDTH
                    or height > MAX_IMAGE_HEIGHT
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Image dimensions exceed limit: {width}x{height} "
                            f"(max {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}, "
                            f"{MAX_IMAGE_PIXELS} pixels)."
                        ),
                    )

                frame_count = getattr(image, "n_frames", 1)
                if frame_count > MAX_IMAGE_FRAMES:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Multi-frame images are not supported "
                            f"({frame_count} frames; max {MAX_IMAGE_FRAMES})."
                        ),
                    )

                image.load()  # force decode so we fail early on corrupt data
                return image.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    # Start the total deadline before upload decoding as well as provider work.
    # The configured value is a route deadline, not only an upstream HTTP
    # timeout; synchronous validation cannot be interrupted, but an expired
    # deadline still produces a controlled response after it returns.
    loop = asyncio.get_running_loop()
    deadline = _prediction_deadline_seconds
    if not math.isfinite(deadline) or deadline <= 0:
        deadline = 65.0
    deadline_at = loop.time() + deadline

    # Snapshot the startup reference so a concurrent shutdown cannot swap the
    # global while this request is submitting work.
    pipeline = _pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction pipeline is not available. "
                "Check that HF_BINARY_MODEL_ID and HF_SUBTYPE_MODEL_ID are set."
            ),
        )

    image = _validate_upload(file)
    remaining_deadline = deadline_at - loop.time()
    if remaining_deadline <= 0:
        raise HTTPException(
            status_code=504,
            detail="Prediction timed out. Please try again.",
        )

    # Provider calls and the optional local model are synchronous. Admit
    # bounded work first, then run it in the dedicated pool so the event loop
    # remains available for health checks and other requests.
    guard = _inference_guard
    executor = _inference_executor
    if guard is None or executor is None:
        executor, guard = _ensure_inference_runtime()

    try:
        lease = guard.acquire()
    except InferenceCapacityError as exc:
        raise HTTPException(status_code=503, detail="Inference runtime is busy") from exc

    def run_prediction() -> HierarchicalPrediction:
        try:
            return pipeline.predict(image)
        finally:
            # A timed-out HTTP request cannot cancel a synchronous provider
            # call. Keep counting that worker until it actually returns.
            lease.release()

    try:
        future = loop.run_in_executor(executor, run_prediction)
    except Exception as exc:
        lease.release()
        raise HTTPException(status_code=503, detail="Inference runtime is busy") from exc

    def finish_future(completed) -> None:
        """Consume late worker failures after a client-side deadline."""

        if completed.cancelled():
            # A queued task may be canceled before ``run_prediction`` starts.
            lease.release()
            return
        try:
            completed.exception()
        except asyncio.CancelledError:
            lease.release()
        except Exception:
            # The route maps the provider exception; this callback prevents
            # asyncio from logging an unhandled future exception.
            pass

    future.add_done_callback(finish_future)

    remaining_deadline = deadline_at - loop.time()
    if remaining_deadline <= 0:
        raise HTTPException(
            status_code=504,
            detail="Prediction timed out. Please try again.",
        )
    try:
        prediction: HierarchicalPrediction = await asyncio.wait_for(
            asyncio.shield(future), timeout=remaining_deadline
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Prediction timed out. Please try again.",
        ) from exc
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="The upstream inference provider timed out.",
        ) from exc
    except (HuggingFaceInferenceError, requests.RequestException) as exc:
        raise HTTPException(
            status_code=502,
            detail="The upstream inference provider returned an error.",
        ) from exc
    except Exception as exc:
        # Provider adapters can raise library-specific exceptions. Keep the
        # response generic rather than reflecting raw upstream bodies.
        raise HTTPException(
            status_code=502,
            detail="The upstream inference provider returned an error.",
        ) from exc

    # Optionally generate a Grad-CAM heatmap from the local model.
    heatmap_b64: Optional[str] = None
    if EXPLAINABILITY_ENABLED and _local_model is not None and _local_model_device is not None:
        try:
            # Pick the class index we want to explain.
            # If pneumonia is predicted, explain the Pneumonia class index (1).
            # Otherwise explain the Normal class index (0).
            if prediction.primary_prediction == "Pneumonia":
                explain_class_idx = 1
            else:
                explain_class_idx = 0

            heatmap_b64 = _load_explainability()(
                model=_local_model,
                model_name=LOCAL_MODEL_NAME,
                image=image,
                class_idx=explain_class_idx,
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
