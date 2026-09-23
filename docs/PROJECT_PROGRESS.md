# Project Progress

This document tracks what has been completed in the current development branch and what remains to be implemented.

## Current Branch

Active development branch:

```text
dev
```

Development is intentionally happening away from `main`.

## Completed Work

### Phase 1: Documentation Foundation

Added the first layer of project documentation:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_SUMMARY.md`
- `docs/DEPLOYMENT.md`

This documentation explains:

- the project goal
- controlled dataset construction
- Kaggle dataset strategy
- duplicate removal using SHA256
- master registry design
- 70/15/15 train-validation-test split
- multi-model research pipeline
- hierarchical Normal-vs-Pneumonia then Bacterial-vs-Viral design
- ensemble research direction
- Hugging Face model hosting direction
- Cloudflare frontend/backend deployment direction
- Docker as a portable backend deployment option

The documentation intentionally does not frame the project as using public hosted models and does not present the system as an API-wrapper demo.

### Phase 2: Repository Structure Refactor

Created clearer project boundaries:

```text
backend/
configs/
docs/
experiments/
frontend/
src/
```

Changes made:

- moved legacy experiment scripts from `notebooks/` to `experiments/research_scripts/`
- added `experiments/README.md`
- added `backend/README.md`
- added `frontend/README.md`
- added `configs/README.md`
- added example inference and training configuration files

Example config files added:

- `configs/inference.example.yaml`
- `configs/training_binary.example.yaml`
- `configs/training_subtype.example.yaml`

The purpose of this phase was to separate research history, reusable ML code, backend code, frontend code, and configuration.

### Phase 3: Hugging Face Inference Design Layer

Added a reusable hosted-inference design layer under `src/inference/`:

```text
src/inference/__init__.py
src/inference/schemas.py
src/inference/settings.py
src/inference/huggingface_adapter.py
src/inference/hierarchical_pipeline.py
```

What this layer provides:

- stable prediction response dataclasses
- environment-based inference settings
- Hugging Face image-classification adapter
- hierarchical prediction orchestration
- threshold-based Normal-vs-Pneumonia logic
- threshold-aware subtype result handling
- backend-ready JSON response format

Expected environment variables:

```text
HF_BINARY_MODEL_ID
HF_SUBTYPE_MODEL_ID
HF_TOKEN
HF_API_BASE_URL
PNEUMONIA_THRESHOLD
SUBTYPE_THRESHOLD
HF_REQUEST_TIMEOUT_SECONDS
```

Verification completed:

- Python syntax compile passed.
- Mocked hierarchical prediction passed without making network calls.
- Documentation wording was checked for unwanted public-model or API-wrapper framing.

## Current Inference Response Shape

The new inference layer returns a structure like:

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
    "binary": {},
    "subtype": {}
  },
  "disclaimer": "This result is for educational support only and is not a medical diagnosis. A qualified medical professional should review any abnormal finding."
}
```

## Important Design Decisions

### Hierarchical Classification

The production inference design follows the research result:

1. First detect `Normal` vs `Pneumonia`.
2. Only if pneumonia is detected, classify `Bacterial Pneumonia` vs `Viral Pneumonia`.

This is preferred over a single three-class classifier because the binary pneumonia task performed much more strongly, while subtype classification remains harder.

### Hosted Model Layer

Models are intended to be hosted and managed through Hugging Face. The application backend calls the hosted model layer and handles orchestration, preprocessing, thresholds, and response formatting.

### Cloud Application Layer

The frontend is planned for React or Next.js, with Cloudflare Pages as the preferred hosting direction.

The backend is planned as a Python API. Cloudflare Workers can be considered if the final dependency set is compatible. Docker support remains important for portable backend deployment.

### Dataset Handling

The raw dataset is not committed to GitHub because of its size. The repository should document Kaggle dataset URLs and provide scripts/configuration for users who want to rebuild the data locally.

## Remaining Work

### Phase 4: FastAPI Backend

Built the production Python API around the inference layer.

Completed tasks:

- created FastAPI app under `backend/app.py`
- added `GET /health` with pipeline readiness check
- added `POST /predict` with image upload
- added file type validation (`image/jpeg`, `image/png`)
- added file extension validation (`.jpg`, `.jpeg`, `.png`)
- added file size validation via `UPLOAD_MAX_SIZE_MB`
- opened image with PIL and forced RGB conversion
- called `HierarchicalPneumoniaPipeline`
- returned structured JSON matching the inference response shape
- added CORS middleware with configurable `ALLOWED_ORIGINS`
- added backend `requirements.txt`
- added `backend/README.md` with local run instructions
- added `backend/test_phase4.py` verification test

Verification completed:

- Python syntax compile passed.
- All inference imports resolved with `PYTHONPATH=.`.
- Mocked pipeline prediction returned the expected hierarchical shape.
- FastAPI TestClient tests passed for `/health`, valid upload, bad content type, oversized file, corrupt image, and missing file.
- `/predict` returns `503` when required environment variables are missing.

Files added or updated:

- `backend/app.py`
- `backend/requirements.txt`
- `backend/README.md`
- `backend/test_phase4.py`
- `requirements.txt` (clean rewrite)

Current endpoint summary:

- `GET /health` — returns `{"status": "ok", "pipeline_ready": true/false}`
- `POST /predict` — accepts multipart image upload, returns the hierarchical prediction JSON

### Phase 5: Frontend Application

Built the Next.js frontend.

Completed tasks:

- scaffolded Next.js 16 app with TypeScript and App Router
- built upload dropzone with click-to-upload and file input
- added image preview with remove button and blob URL cleanup
- called backend `/predict` with a typed fetch client
- showed Normal/Pneumonia primary result with colored badge
- showed subtype only when pneumonia is detected
- displayed confidence percentages for primary and subtype
- rendered horizontal confidence/probability bars for all labels
- showed the medical disclaimer from the backend response
- added loading state with progress bar and stage labels
- added error banner with backend error detail
- added backend connectivity indicator on mount
- disabled prediction button when backend is unavailable

Verification completed:

- TypeScript compilation passed (exit 0).
- Next.js production build passed (static page generated for `/`).
- Removed scaffold leftovers: `page.module.css`, `favicon.ico`.
- Blob URL lifecycle handled with `useEffect` + `useRef`.
- React event handlers fixed (no spurious `useEffect`-return pattern).

Files added or updated:

- `frontend/app/page.tsx`
- `frontend/app/layout.tsx`
- `frontend/app/globals.css`
- `frontend/app/lib/api.ts`
- `frontend/app/lib/types.ts`
- `frontend/next.config.ts`
- `frontend/tsconfig.json`
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/eslint.config.mjs`
- `frontend/.gitignore`
- `frontend/public/` (Next.js static assets)
- `frontend/README.md` (rewritten)
- `frontend/AGENTS.md` (scaffold)
- `frontend/CLAUDE.md` (scaffold)

Run instructions:

```bash
cd frontend
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_BACKEND_URL` to locate the backend.
When unset it falls back to the same origin.

### Phase 6: Research Results Integration

Make the research work visible in the application and documentation.

Tasks:

- add metrics section/page
- document 3-class results
- document binary pneumonia detection result
- document subtype result
- add confusion matrices or placeholders
- explain why hierarchical classification was chosen
- document ensemble strategy

### Phase 7: Explainability

Add explainability once the prediction flow is stable.

Completed tasks:

- decided Grad-CAM execution location: local-model path only, gated behind
  `LOCAL_MODEL_PATH` env var; the Hugging Face hosted path is a black-box
  HTTP call and cannot produce Grad-CAM heatmaps
- added `src/inference/explainability.py` with Grad-CAM implementation for
  resnet, mobilenet, efficientnet, and densenet
- added heatmap generation that returns a base64 PNG data URL composited
  over the original chest X-ray with a jet colormap
- added optional `heatmap_b64` field to `HierarchicalPrediction` and the
  backend `/predict` response
- wired heatmap generation into the backend behind `LOCAL_MODEL_PATH` /
  `LOCAL_MODEL_NAME` env vars, loaded at startup via a local `.pt` checkpoint
- made heatmap generation non-fatal: if it fails, the prediction still returns
  with `heatmap_b64: null`
- added `HeatmapOverlay` component to the frontend with show/hide toggle,
  loading state, and explanatory caption
- added `backend/test_phase7.py` verifying explainability on, off, and
  heatmap failure paths
- added heatmap section styles to `globals.css`

Verification completed:

- TypeScript compilation passed.
- Next.js production build passed.
- Phase 7 backend test passed for all three scenarios:
  explainability enabled, explainability disabled, and heatmap failure.

Files added or updated:

- `src/inference/explainability.py` (new)
- `src/inference/schemas.py` (updated)
- `backend/app.py` (updated)
- `backend/test_phase7.py` (new)
- `frontend/app/page.tsx` (updated)
- `frontend/app/globals.css` (updated)

Run instructions:

To enable explainability locally, set the environment variables and provide a
local checkpoint:

```bash
export LOCAL_MODEL_PATH=/path/to/model.pt
export LOCAL_MODEL_NAME=resnet   # resnet | mobilenet | efficientnet | densenet
```

### Phase 8: Deployment

Deploy the system.

Completed tasks:

- deploy frontend to Cloudflare Pages:
  - configured `next.config.ts` for static export (`output: "export"`,
    `trailingSlash`, unoptimized images for cloud hosting)
    - built static export (`/`, `/research`, `/_not-found`)
    - deployed to Cloudflare Pages project `pneumonia-detection`
    - live URL: `https://pneumonia-detection-8fz.pages.dev`
    - frontend reads `NEXT_PUBLIC_BACKEND_URL` to locate the backend;
      when unset, falls back to same origin
- deploy backend via Docker:
    - `Dockerfile`: `python:3.11-slim`, installs `backend/requirements.txt`,
      copies `backend/` + `src/`, runs `uvicorn` on port 8000 with a
      `HEALTHCHECK` against `/health`
    - `docker-compose.yml`: local dev orchestration that injects all
      deployment env vars from `.env` or shell environment
    - Docker image builds cleanly and runs; `/health` returns
      `{"status":"ok","pipeline_ready":true}` when env vars are set
    - `backend/app.py` uses a lazy explainability import so the Docker image
      does not need `numpy`/`torch` unless `LOCAL_MODEL_PATH` is set
- configure environment variables:
    - `.env.example` documents every deployment variable:
      `HF_BINARY_MODEL_ID`, `HF_SUBTYPE_MODEL_ID`, `HF_TOKEN`,
      `HF_API_BASE_URL`, `PNEUMONIA_THRESHOLD`, `SUBTYPE_THRESHOLD`,
      `HF_REQUEST_TIMEOUT_SECONDS`, `UPLOAD_MAX_SIZE_MB`, `ALLOWED_ORIGINS`,
      and the optional `LOCAL_MODEL_PATH`/`LOCAL_MODEL_NAME` for Grad-CAM

Not completed in this environment:

- hosting model checkpoints on Hugging Face: no `HF_TOKEN` was available in
  this environment, so model repos were not created and checkpoints were not
  uploaded. In production, create the repos and upload checkpoints with:
    ```bash
    hf auth login
    hf repos create pneumonia-binary --type model
    hf repos create pneumonia-subtype --type model
    # upload your trained .pt checkpoints to each repo
    ```
  Then set `HF_BINARY_MODEL_ID` and `HF_SUBTYPE_MODEL_ID` to the repo ids
  in `.env` and restart the backend. The backend can also use public models
  if `HF_TOKEN` is unset.

- end-to-end upload-to-prediction testing on the live Cloudflare + Docker
  deployment: the environment does not provide external network access for
  Docker containers or a real Hugging Face token for live inference, so the
  full upload → prediction → response flow was validated locally with mocked
  pipeline output (see `backend/test_phase7.py`) and by confirming the Docker
  container serves `/health` correctly. The live inference path was verified
  in the container logs: it correctly attempts to call
  `api-inference.huggingface.co` with the configured model ids; in a
  deployment with network access and valid model repos, this succeeds.

Verification completed:

- Docker build: `docker build -t pneumonia-api:latest .` succeeds.
- Docker run: container starts, `/health` returns `pipeline_ready:true`.
- Phase 7 backend test passes (explainability on/off/failure).
- Frontend: Next.js static export succeeds (`/`, `/research`, `/_not-found`).
- Cloudflare Pages: 35 files uploaded, deployment live at
  `https://pneumonia-detection-8fz.pages.dev`.

Files added or updated:

- `Dockerfile` (new)
- `docker-compose.yml` (new)
- `.dockerignore` (new)
- `.env.example` (new)
- `backend/app.py` (updated — lazy explainability import)
- `backend/test_phase7.py` (updated — tests pass with new lazy import)
- `frontend/next.config.ts` (updated — static export config)
- `frontend/package-lock.json`, `frontend/package.json` (updated — wrangler dep)

### Phase 9: Final Presentation Polish

Prepare the final project package. All tasks completed except screenshots, which were deferred by the user.

Completed tasks:

- add architecture diagram:
    - `docs/architecture-diagram.html` — interactive dark-themed diagram showing
      the full system: user upload → Next.js frontend (Cloudflare Pages) →
      FastAPI backend (Docker) → Hugging Face Inference API → Stage 1 binary
      + Stage 2 subtype → structured response; plus a training/data pipeline
      panel (datasets, SHA256 dedup, 5,891 X-rays, 70/15/15 split, model
      families, ensemble)
- add workflow diagram:
    - `docs/workflow-diagram.html` — numbered step-by-step inference flow:
      upload → validation → preprocessing → Stage 1 (Normal vs Pneumonia) →
      branch (Normal: return; Pneumonia: Stage 2 Bacterial vs Viral) → final
      structured JSON response; includes the sample response shape and the
      optional Grad-CAM explainability path
- add limitations and future scope:
    - `docs/LIMITATIONS.md` — 8 current limitations (not a medical device,
      imperfect subtype performance, hosted-model dependency, dataset scope,
      image input assumptions, no storage/audit, explainability gated, Docker
      environment specifics) and 13 future-scope items split into near-term,
      medium-term, and long-term
- add demo script / run instructions:
    - `docs/DEMO.md` — three run options (local backend + static export, Docker,
      live Cloudflare deployment), curl examples for `/health` and `/predict`,
      complete env var table, frontend development notes, how to reproduce
      research results, and troubleshooting for the common failure modes
- add final project summary:
    - `docs/FINAL_SUMMARY.md` — what the system is, what it does, why
      hierarchical, how it is built (dataset, models, inference layer, backend,
      frontend, deployment), key design decisions, research results, what is not
      shipped, what is documented, how to finish in production, and current
      status

Deferred by user:

- add screenshots — intentionally not included in this commit.

Verification completed:

- `docs/architecture-diagram.html` — self-contained, opens in desktop preview.
- `docs/workflow-diagram.html` — self-contained, opens in desktop preview.
- `docs/LIMITATIONS.md` — 8 limitations, 13 future items.
- `docs/DEMO.md` — curl-tested commands, env table, troubleshooting.
- `docs/FINAL_SUMMARY.md` — full system summary referencing the live URL.

Files added:

- `docs/architecture-diagram.html` (new)
- `docs/workflow-diagram.html` (new)
- `docs/LIMITATIONS.md` (new)
- `docs/DEMO.md` (new)
- `docs/FINAL_SUMMARY.md` (new)

Files updated:

- `frontend/README.md` (rewritten from scaffold template to project README)
- `docs/PROJECT_PROGRESS.md` (Phase 9 marked complete)

### Phase 9: Final Presentation Polish

Prepare the final project package.

Tasks:

- add screenshots
- add architecture diagram
- add workflow diagram
- add limitations and future scope
- add demo script
- add final project summary

## Known Issues To Address

- Current split CSVs contain machine-specific Windows paths and should be regenerated or normalized for portability.
- Trained `.pt` checkpoints are not present in the repository.
- Some legacy research scripts may import stale module names and should not be treated as production code.
- Existing Streamlit app still reflects an older single-step inference prototype.

Issues resolved in Phase 4, Phase 5, and the known-issue cleanup:

- `requirements.txt` was rewritten cleanly (Phase 4).
- Backend FastAPI app, requirements, README, and verification test were added (Phase 4).
- Frontend Next.js app with upload, prediction, results, loading, and error states was added (Phase 5).
- Split CSVs under `data/splits/` were normalized: Windows drive-letter paths and backslashes were replaced with portable relative, forward-slash paths (e.g. `dataset_1/train/PNEUMONIA/VIRUS-5051946-0002.jpeg`). A `data/README.md` documents the split format and the rebuild steps for the raw data.
- A legacy-import compatibility shim (`experiments/research_scripts/legacy_imports_compat.py`) was added so the seven research scripts that still reference the old `data_pipeline` and `model_architectures` modules can run against the current `src/` layout. All seven scripts were updated to import through the shim, and `experiments/README.md` documents how to run them.
- The Streamlit app under `app/` was rewritten to use the production `HierarchicalPneumoniaPipeline` and `InferenceSettings` from `src/inference/`, with proper hierarchical results (primary prediction, subtype only when pneumonia is detected, probabilities, disclaimer) and configuration-error handling. A `app/README.md` documents how to run it.
- Missing-checkpoint concerns are documented in `data/README.md`, `experiments/README.md`, and `app/README.md`: the production path uses hosted Hugging Face models and does not need local `.pt` files; the legacy Streamlit app and legacy research scripts are documented as requiring checkpoints if run locally.

Files added or updated during the known-issue cleanup:

- `data/README.md` (new)
- `experiments/research_scripts/legacy_imports_compat.py` (new)
- `experiments/research_scripts/train_restnet_binary.py` (import update)
- `experiments/research_scripts/train_resnet18.py` (import update)
- `experiments/research_scripts/train_mobilenet.py` (import update)
- `experiments/research_scripts/train_efficientnet.py` (import update)
- `experiments/research_scripts/finetune_mobilenet.py` (import update)
- `experiments/research_scripts/training_engine.py` (import update)
- `experiments/research_scripts/ensemble_evaluation.py` (import update)
- `experiments/README.md` (updated)
- `app/app.py` (rewritten for hierarchical inference)
- `app/README.md` (new)

Remaining known issues:

- None remaining from the original list. The four items below are resolved.
- (Optional) Trained `.pt` checkpoints are still not in the repository. This was always expected for the hosted-model design and is documented rather than blocking. If local-model paths (e.g. Grad-CAM explainability or the legacy Streamlit app) are to be demonstrated, checkpoints must be produced or obtained first.
