# Backend

Production Python API for the Pneumonia Detection system.

## Overview

The backend exposes three service/prediction endpoints:

- `GET /live` — process liveness, independent of inference availability.
- `GET /ready` — readiness check; returns `503` until the inference pipeline is initialized.
- `GET /health` — legacy combined status response.
- `POST /predict` — accept a chest X-ray image upload and return a hierarchical prediction.

The prediction flow follows the research result:

1. Binary classification: Normal vs Pneumonia.
2. If pneumonia is detected, subtype classification: Bacterial Pneumonia vs Viral Pneumonia.

Hosted model inference runs through Hugging Face. The backend orchestrates settings, thresholds, preprocessing, and response formatting.

## Project layout

```
backend/
  app.py              # FastAPI application
  requirements.txt    # backend dependencies
  README.md           # this file
```

Shared inference code lives in `src/inference/` and is imported directly by the backend.

## Prerequisites

- Python 3.11+
- Setuptools-compatible install of the repo so that `src/` is importable, or run with `PYTHONPATH=.`

## Install

From the repository root:

```bash
pip install -r backend/requirements.txt
```

If `src/` is not installed as a package, set the Python path:

```bash
export PYTHONPATH=.
```

## Required environment variables

The backend requires at least the following to be set before starting:

```bash
export HF_BINARY_MODEL_ID=<huggingface-model-id>
export HF_SUBTYPE_MODEL_ID=<huggingface-model-id>
export HF_TOKEN=<optional-huggingface-token>
```

Optional environment variables:

```bash
export HF_API_BASE_URL=https://api-inference.huggingface.co/models
export PNEUMONIA_THRESHOLD=0.5
export SUBTYPE_THRESHOLD=0.5
export HF_REQUEST_TIMEOUT_SECONDS=60
export UPLOAD_MAX_SIZE_MB=10
export ALLOWED_ORIGINS=http://localhost:3000,https://example.com
export APP_ENV=production
```

- `ALLOWED_ORIGINS` is a comma-separated list of CORS origins. Development defaults to explicit local frontend origins (`localhost`/`127.0.0.1` on ports 3000, 5173, and 5174); production defaults to deny unless explicit origins are configured.
- Wildcard origins are disabled rather than combined with credentials.
- `UPLOAD_MAX_SIZE_MB` controls the maximum accepted upload size.

## Run locally

```bash
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API doc UI is available at `http://localhost:8000/docs`.

## Endpoints

### `GET /live`

Returns `200` when the API process is alive, even if the inference pipeline is
not ready:

```json
{"status":"ok"}
```

### `GET /ready`

Returns `200` with `pipeline_ready: true` when the inference pipeline is ready.
Returns `503` with `pipeline_ready: false` while initialization is unavailable:

```json
{"status":"not_ready","pipeline_ready":false}
```

### `GET /health`

Returns service status and whether the inference pipeline initialized successfully.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "pipeline_ready": true
}
```

### `POST /predict`

Upload a chest X-ray image.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.png"
```

Example response:

```json
{
  "primary_prediction": "Pneumonia",
  "primary_confidence": 0.93,
  "subtype_prediction": "Bacterial Pneumonia",
  "subtype_confidence": 0.76,
  "probabilities": {
    "Normal": 0.07,
    "Pneumonia": 0.93,
    "Bacterial Pneumonia": 0.76,
    "Viral Pneumonia": 0.24
  },
  "model_outputs": {
    "binary": { "label": "Pneumonia", "confidence": 0.93, "probabilities": { ... } },
    "subtype": { "label": "Bacterial Pneumonia", "confidence": 0.76, "probabilities": { ... } }
  },
  "disclaimer": "This result is for educational support only and is not a medical diagnosis. A qualified medical professional should review any abnormal finding."
}
```

If pneumonia is not detected, `subtype_prediction` and `subtype_confidence` are `null`.

### Validation rules

- Accepted content types: `image/jpeg`, `image/png`, `image/jpg`.
- Accepted file extensions: `.jpg`, `.jpeg`, `.png`.
- Uploads exceeding `UPLOAD_MAX_SIZE_MB` are rejected.
- Corrupt or undecodable images are rejected with a `400` error.

### Errors

- `400` — invalid file type, too large, unsupported extension, or undecodable image.
- `503` — prediction pipeline not available, usually because required environment variables are missing.

## CORS

The backend includes fail-closed CORS middleware. Development defaults to
explicit local origins; set `APP_ENV=production` and configure production
origins explicitly:

```bash
export ALLOWED_ORIGINS=http://localhost:3000
```

## Configuration file

An example inference configuration is provided at the repo root:

```
configs/inference.example.yaml
```

Environment variables take precedence at runtime. The example file is intended as documentation and a starting point for deployment configuration.

## Relationship to the rest of the system

- The backend calls the Hugging Face hosted model layer through `src/inference/`.
- The frontend should call this backend, not Hugging Face directly.
- Docker and Cloudflare deployment directions are covered in `docs/DEPLOYMENT.md`.
