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

Tasks:

- decide Grad-CAM execution location
- add heatmap generation
- return heatmap image or URL
- show heatmap overlay in frontend
- document limitations

### Phase 8: Deployment

Deploy the system.

Tasks:

- host model checkpoints on Hugging Face
- deploy frontend to Cloudflare Pages
- deploy backend to Cloudflare-compatible runtime or Docker-capable platform
- configure environment variables
- test upload-to-prediction flow end to end

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

Issues resolved in Phase 4 and Phase 5:

- `requirements.txt` was rewritten cleanly (Phase 4).
- Backend FastAPI app, requirements, README, and verification test were added (Phase 4).
- Frontend Next.js app with upload, prediction, results, loading, and error states was added (Phase 5).

