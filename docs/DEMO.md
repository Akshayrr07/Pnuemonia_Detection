# Demo & Run Instructions

This document describes how to run the Pneumonia Detection system end to end, from local development to the deployed cloud environment.

## Quick start

### Option A — Run the backend locally and the frontend from the static export

This is the simplest path if you want to see the full system on your own machine.

**1. Start the backend**

From the repository root:

```bash
# Install backend deps
pip install -r backend/requirements.txt

# Set the required env vars
export HF_BINARY_MODEL_ID=<huggingface-model-id>
export HF_SUBTYPE_MODEL_ID=<huggingface-model-id>
# Optional: export HF_TOKEN=<your-huggingface-token>

# Start the API
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`. The interactive API docs are at `http://localhost:8000/docs`.

**2. Build and open the frontend**

Build the frontend with the backend URL embedded before serving the static export:

```bash
cd frontend
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run build
cd out
python3 -m http.server 3000
# or: npx serve .
```

`NEXT_PUBLIC_BACKEND_URL` is a build-time variable. If it is unset, the frontend falls back to the same origin. For local testing, either serve the backend on the same origin or rebuild with `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`; do not expect a runtime environment change to alter an already-built static bundle.

Open `http://localhost:3000` in a browser. The frontend will attempt to call the backend at the URL embedded during that build. For local testing you either need:

- to run the backend on port 3000 behind a proxy, or
- to rebuild the static export with `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`.

**3. Make a prediction**

1. Open the homepage.
2. Upload a chest X-ray (JPEG or PNG, ≤ 10 MB).
3. View the result card: primary prediction, confidence, subtype (if pneumonia was detected), and the medical disclaimer.
4. If `LOCAL_MODEL_PATH` and `LOCAL_MODEL_NAME` are set in the backend env, a Grad-CAM heatmap overlay is also available.

### Option B — Run with Docker

This is the recommended path for a reproducible local or deployed environment.

```bash
# Build the image
docker build -t pneumonia-api:latest .

# Run with env vars
docker run --rm -p 8000:8000 \
  -e HF_BINARY_MODEL_ID=<huggingface-model-id> \
  -e HF_SUBTYPE_MODEL_ID=<huggingface-model-id> \
  -e HF_TOKEN=<optional-token> \
  pneumonia-api:latest
```

Or use docker-compose:

```bash
# Copy .env.example to .env and fill in real values
cp .env.example .env
# edit .env with real HF model ids and token

docker compose up --build
```

The backend is then available at `http://localhost:8000`.

### Option C — Use the published frontend

The frontend is deployed to Cloudflare Pages at:

```
https://pneumonia-detection-8fz.pages.dev
```

The backend is not yet deployed to a live host in this environment. To make the live frontend functional, deploy the backend container to a Docker-capable platform, set `NEXT_PUBLIC_BACKEND_URL` in the frontend **build** environment to the deployed backend URL, then rebuild and redeploy the frontend.

## API demo with curl

**Health check:**

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status": "ok", "pipeline_ready": true}
```

**Prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/chest_xray.png"
```

Expected (when pneumonia is detected):

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
  "model_outputs": { ... },
  "disclaimer": "This result is for educational support only and is not a medical diagnosis. A qualified medical professional should review any abnormal finding."
}
```

When no pneumonia is detected, `subtype_prediction` and `subtype_confidence` are `null`.

## Environment variables

See `.env.example` at the repository root for the complete reference. The critical ones are:

| Variable | Required | Purpose |
|---|---|---|
| `HF_BINARY_MODEL_ID` | Yes (for hosted path) | Hugging Face model id for Normal-vs-Pneumonia |
| `HF_SUBTYPE_MODEL_ID` | Yes (for hosted path) | Hugging Face model id for Bacterial-vs-Viral |
| `HF_TOKEN` | No | HF token if models are private; omit for public models |
| `HF_API_BASE_URL` | No | Default: `https://api-inference.huggingface.co/models` |
| `PNEUMONIA_THRESHOLD` | No | Default: `0.5` |
| `SUBTYPE_THRESHOLD` | No | Default: `0.5` |
| `HF_REQUEST_TIMEOUT_SECONDS` | No | Default: `60` |
| `UPLOAD_MAX_SIZE_MB` | No | Default: `10` |
| `ALLOWED_ORIGINS` | No | CORS origins; unset = permissive for local dev |
| `LOCAL_MODEL_PATH` | No | Path to a local checkpoint for Grad-CAM |
| `LOCAL_MODEL_NAME` | No | Model architecture name for Grad-CAM (`resnet`, `mobilenet`, etc.) |
| `NEXT_PUBLIC_BACKEND_URL` | Frontend build | Backend URL embedded in the static client bundle; rebuild after changing it |

## Safety and medical-use notice

Do not use the demo with real patient data or identifiable medical images. Uploads may be processed or logged by the browser host, reverse proxy, hosting provider, and hosted model provider. This project is educational and research software, not a medical device, diagnosis, treatment recommendation, or triage tool. A qualified healthcare professional must review the original image and clinical context.

## Frontend development

The frontend lives in `frontend/` and is a Next.js app. To run it in development mode:

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:3000` with hot reload.

**Note:** the development server is not the same as the static export used for Cloudflare Pages. The static export is produced with `npm run build` when `next.config.ts` has `output: "export"`. The development server does server-side rendering and API routes that the static export does not include.

## Reproducing the research results

The research results (93.78% binary accuracy, 75.67% subtype accuracy, 74–77% three-class accuracy) were produced by scripts under `experiments/research_scripts/`. Those scripts are preserved as research history and require:

- The original raw datasets, which are not included in this repository.
- Trained `.pt` checkpoints, which are not included in this repository.

See `experiments/README.md` for how to run the legacy research scripts if you have the raw data and want to reproduce the experiments.

## Troubleshooting

**`pipeline_ready: false` on `/health`** — usually missing or invalid `HF_BINARY_MODEL_ID` / `HF_SUBTYPE_MODEL_ID`, or a network issue reaching the Hugging Face Inference API.

**CORS errors in the browser** — set `ALLOWED_ORIGINS` on the backend to include the frontend origin, or leave it unset for permissive local development.

**Grad-CAM not showing** — `LOCAL_MODEL_PATH` is not set or does not point to a valid checkpoint. The heatmap overlay is gated behind this env var by design.

**Frontend cannot reach the backend** — check that `NEXT_PUBLIC_BACKEND_URL` was set in the **frontend build environment**, rebuild/redeploy the static frontend, and confirm the backend origin/CORS configuration. If the variable is unset, the frontend uses the same origin.
