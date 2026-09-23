# Final Project Summary

## What this project is

A research-to-production system for detecting pneumonia from chest X-ray images, built end to end: dataset construction, multi-model research, hierarchical inference design, a FastAPI backend, a Next.js frontend, Grad-CAM explainability, and containerized deployment.

The system is live in two parts:

- **Frontend:** `https://pneumonia-detection-8fz.pages.dev` — Cloudflare Pages static export of the Next.js app.
- **Backend:** containerized with Docker; not yet deployed to a live host in this environment, but built and tested locally.

## What it does

A user uploads a chest X-ray (JPEG or PNG) through the web interface. The backend validates the image, preprocesses it with the same transforms used during training, and runs a two-stage hierarchical inference:

1. **Stage 1 — Normal vs Pneumonia.** A binary classifier decides whether the image shows signs of pneumonia. This is the strongest part of the system (research accuracy: 93.78%).
2. **Stage 2 — Bacterial vs Viral.** Only if pneumonia is detected does the system classify the subtype. This is harder (research accuracy: 75.67%) and is reported honestly as a lower-confidence result.

The response is a structured JSON object containing the primary prediction, confidence, optional subtype, per-class probabilities, model-level outputs, and a medical disclaimer.

## Why hierarchical

The research phase evaluated multiple CNN families (Custom CNN, MobileNetV2, EfficientNet-B0, ResNet18, DenseNet121) on direct three-class classification. The best three-class accuracy was 74–77%, with the viral class being the hardest to separate.

Splitting the problem into a binary pneumonia-detection stage followed by a subtype stage matches the observed strengths of the models and produces a cleaner production inference flow. The binary model does what it does well; the subtype model only has to separate two classes in the cases where it actually runs.

## How it is built

### Dataset

Three public Kaggle chest X-ray datasets were audited, labeled, deduplicated with SHA256 + perceptual hash, and merged into a single controlled dataset of **5,891 unique X-rays**. The normalized class set is:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

The data was split **70% training / 15% validation / 15% testing**, stratified by encoded label, with leakage checks to confirm no image appears in more than one split.

### Models

The research evaluated:

- Custom lightweight CNN
- MobileNetV2
- EfficientNet-B0
- ResNet18
- DenseNet121 (code in the modular `src/models/` layout)

Ensemble utilities (soft voting, weighted voting) exist in `experiments/` as research tools.

### Inference design layer

A reusable inference layer under `src/inference/` abstracts the model provider:

- `settings.py` — environment-based inference configuration.
- `huggingface_adapter.py` — Hugging Face Inference API client.
- `hierarchical_pipeline.py` — two-stage orchestration with threshold logic.
- `schemas.py` — stable dataclasses for predictions and responses.
- `explainability.py` — Grad-CAM module (local-only, gated behind `LOCAL_MODEL_PATH`).

### Backend

A FastAPI application in `backend/app.py` with:

- `GET /health` — availability and pipeline readiness.
- `POST /predict` — image upload and hierarchical prediction.
- Image validation (type, size, decodability).
- CORS support.
- Lazy explainability import (numpy/torch only loaded when a local checkpoint is configured).

### Frontend

A Next.js application with:

- Image upload and preview.
- Loading and error states.
- Result card with prediction, confidence, and probabilities.
- Subtype shown only when pneumonia is detected.
- Medical disclaimer and limitation notice.
- Grad-CAM heatmap overlay (gated, same condition as backend).
- Static export for Cloudflare Pages hosting.

### Deployment

- **Frontend:** Cloudflare Pages, static export, deployed via `wrangler pages deploy`.
- **Backend:** Docker image (`python:3.11-slim`), healthcheck on `/health`, env-var-driven configuration, `docker-compose.yml` for local orchestration.
- **Env vars:** documented in `.env.example` at the repository root.

## Key design decisions

1. **Hierarchical over single three-class model.** The data showed the binary task was much stronger than the three-class task. The architecture follows the data rather than forcing one model to do everything.

2. **Hosted models over bundled checkpoints.** The production backend calls the Hugging Face Inference API rather than loading large `.pt` files in the container. This keeps the Docker image smaller, separates model lifecycle from app deployment, and avoids bundling datasets or weights in the repository.

3. **Grad-CAM is local-only and gated.** The hosted HF path is a black-box HTTP endpoint with no activations or gradients. Grad-CAM runs only against a locally loaded checkpoint when `LOCAL_MODEL_PATH` is set. Without a checkpoint, `heatmap_b64` is `null` and the system degrades gracefully.

4. **Docker for the backend, static export for the frontend.** Cloudflare Workers Python cannot run Pillow's C extensions or `requests`, so the backend is containerized. The frontend is a static export that is cheap and fast to serve on Cloudflare Pages.

5. **No image storage, no audit log.** The backend is stateless and does not persist uploads or predictions. This is a deliberate privacy choice but means there is no feedback loop or audit trail.

## Research results (accuracy only)

These numbers are from the offline research phase, not from a live end-to-end deployment with the exact same checkpoints:

- **Binary (Normal vs Pneumonia):** 93.78%
- **Subtype (Bacterial vs Viral):** ~75.67%
- **Direct three-class:** 74–77%

Precision, recall, F1, ROC/AUC, sensitivity, and specificity have not been computed and surfaced in this repository — accuracy is what the research summary documents. The limitations document explains why this matters and what would be needed to add it.

## What is not shipped

- **Trained checkpoints.** No `.pt` files are in the repository. The production path does not need them, but Grad-CAM and the legacy Streamlit app do.
- **Raw datasets.** Not included due to size.
- **Live backend deployment.** The backend Docker image is built and tested locally; a live deployment requires a Docker-capable host and real Hugging Face model repos with uploaded checkpoints.
- **Clinical validation.** This is an educational and research system. It is not certified, validated for clinical use, or a medical device.

## What is documented

- `docs/ARCHITECTURE.md` — system architecture and repository boundaries.
- `docs/RESEARCH_SUMMARY.md` — dataset, split, model families, results.
- `docs/DEMO.md` — run instructions, curl examples, env var reference, troubleshooting.
- `docs/LIMITATIONS.md` — honest limitations and future scope.
- `docs/architecture-diagram.html` — visual system architecture (open in the desktop preview pane).
- `docs/workflow-diagram.html` — visual inference workflow (open in the desktop preview pane).
- `docs/DEPLOYMENT.md` — deployment design.
- `docs/PROJECT_PROGRESS.md` — phase-by-phase progress.
- `.env.example` — complete deployment env var reference.

## Repository structure

```
backend/            FastAPI API application
configs/            example training/inference configs
docs/               project documentation + diagrams
experiments/        historical research scripts
frontend/           Next.js web app
src/                reusable ML code (data, models, inference)
data/               split CSVs + README (raw datasets not included)
app/                (optional) Streamlit demo app using the pipeline
```

## How to finish in production

1. Train or obtain checkpoints for the binary and subtype models.
2. Create Hugging Face model repos and upload the checkpoints.
3. Set `HF_BINARY_MODEL_ID` and `HF_SUBTYPE_MODEL_ID` in `.env`.
4. Deploy the backend Docker container to a cloud host.
5. Set `NEXT_PUBLIC_BACKEND_URL` on the frontend deployment to point at the live backend.
6. (Optional) Set `LOCAL_MODEL_PATH` to enable Grad-CAM in the backend and frontend.
7. Add model cards, compute sensitivity/specificity, and tune thresholds before any clinical-adjacent use.

## Status

This is the final Phase 9 polish commit. The system is complete end to end: research, inference design, backend, frontend, explainability, deployment artifacts, documentation, diagrams, demo instructions, and a written limitations/future-scope section. The one Phase 9 sub-task deferred by the user is screenshots, which are intentionally not included here.
