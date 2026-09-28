# Streamlit Demo App (Legacy / Prototype)

This directory contains a **legacy Streamlit prototype** for the pneumonia
detection system. It has been updated to use the production inference layer
under `src/inference/`, but it is still a local demo and is not the
production frontend.

## What changed

The old Streamlit app was a single-step, three-class predictor that loaded
local model checkpoints and combined them with weighted voting. That approach
relied on `.pt` files in `saved_models/`, which are not part of this
repository.

The app has been rewritten to use the hierarchical pipeline:

- `src/inference/hierarchical_pipeline.HierarchicalPneumoniaPipeline`
- `src/inference/settings.InferenceSettings`

This means the demo now works the same way as the production backend:
it calls hosted Hugging Face models, applies the Normal-vs-Pneumonia
threshold, and only predicts a Bacterial-vs-Viral subtype when pneumonia
is detected. The result includes primary prediction, subtype (when
applicable), probabilities, and the medical disclaimer.

## Requirements

- Python 3.11.x
- The project virtualenv. Install the lightweight API set with
  `uv pip install -r backend/requirements.txt`; install the research/demo set
  with `uv pip install -r requirements-research.txt`
- `streamlit` (installed separately; not in the production requirements)

Install the extra dependencies from the research environment instead of
installing them ad hoc:

```bash
uv pip install -r requirements-research.txt
```

The production backend does not require the legacy Streamlit app. The research
set also provides the PyTorch/torchvision packages needed by the local
Grad-CAM path; the hosted-inference API can remain installed without them.

## Configuration

The app reads the same environment variables as the backend:

- `HF_BINARY_MODEL_ID` — Hugging Face model ID for the binary classifier
- `HF_SUBTYPE_MODEL_ID` — Hugging Face model ID for the subtype classifier
- `HF_TOKEN` — Hugging Face access token
- `HF_API_BASE_URL` — Hugging Face Inference API base URL
- `PNEUMONIA_THRESHOLD` — threshold for the Normal vs Pneumonia decision
- `SUBTYPE_THRESHOLD` — threshold for the subtype decision
- `HF_REQUEST_TIMEOUT_SECONDS` — request timeout in seconds

If these are not set, the app will show a configuration error and explain
how to proceed.

## Running the demo

From the project root:

```bash
export HF_BINARY_MODEL_ID=your-binary-model-id
export HF_SUBTYPE_MODEL_ID=your-subtype-model-id
export HF_TOKEN=your-huggingface-token
export HF_API_BASE_URL=https://api.huggingface.co
export PNEUMONIA_THRESHOLD=0.5
export SUBTYPE_THRESHOLD=0.5
export HF_REQUEST_TIMEOUT_SECONDS=30

streamlit run app/app.py
```

Or, referencing the app path explicitly if you are in another directory:

```bash
streamlit run app/app.py
```

## Relationship to the production frontend

This Streamlit app is a **prototype**. The production system uses:

- FastAPI backend under `backend/`
- Next.js frontend under `frontend/`

See `backend/README.md` and `frontend/README.md` for the production setup.

## Checkpoints

This app does **not** require local `.pt` checkpoints. It calls hosted
models through the Hugging Face Inference API. If you want to run the old
local-model inference path (the legacy `inference.py` that loaded
`saved_models/*.pt`), that code has been removed from this directory
because it is no longer the supported approach.

If you want to experiment with local models, use the scripts under
`experiments/research_scripts/` (see `experiments/README.md`) or the
local-model explainability path in `backend/app.py` (see
`src/inference/explainability.py`).

## Status

This app exists for local demonstration and presentation purposes. It is
not the primary user interface and is not deployed as part of the
production system.
