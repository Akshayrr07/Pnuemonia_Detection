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

Build the backend API around the inference layer.

Tasks:

- create FastAPI app under `backend/`
- add `/health`
- add `/predict`
- accept image upload
- validate file type and size
- open image with PIL
- call `HierarchicalPneumoniaPipeline`
- return structured JSON
- add CORS configuration
- add backend requirements
- add local run instructions

### Phase 5: Frontend Application

Build the React or Next.js frontend.

Tasks:

- create frontend app
- build upload UI
- show image preview
- call backend `/predict`
- show Normal/Pneumonia result
- show subtype only when pneumonia is detected
- show confidence/probabilities
- show disclaimer
- add loading and error states

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

- `requirements.txt` appears to have encoding/null-byte issues and should be rewritten cleanly.
- Current split CSVs contain machine-specific Windows paths and should be regenerated or normalized for portability.
- Trained `.pt` checkpoints are not present in the repository.
- Some legacy research scripts may import stale module names and should not be treated as production code.
- Existing Streamlit app still reflects an older single-step inference prototype.

