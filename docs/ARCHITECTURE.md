# Architecture

This project is organized as a research-to-production AI system for pneumonia detection from chest X-ray images.

## System Overview

```text
User
  |
  v
React / Next.js frontend
  |
  v
Python backend API
  |
  |-- image validation
  |-- preprocessing
  |-- inference orchestration
  |-- postprocessing
  |
  v
Hugging Face hosted model layer
  |
  v
Structured prediction response
```

The frontend is responsible for image upload, preview, loading states, and result display. The backend owns the machine-learning workflow: validating the image, applying the same preprocessing used during training, calling the hosted model layer, and returning a consistent response.

## Repository Boundaries

```text
src/                         reusable ML code
experiments/research_scripts/ historical research scripts
backend/                     Python API application boundary
frontend/                    React or Next.js application boundary
configs/                     training and inference configuration
docs/                        project documentation
```

The production path should use `src/` for reusable data, model, training, evaluation, and inference logic. The `experiments/` directory is preserved for research history and presentation context, but new production code should not depend on experiment scripts.

## Hierarchical Prediction Flow

The intended inference pipeline is hierarchical:

```text
Input chest X-ray
        |
        v
Preprocessing
        |
        v
Stage 1 model: Normal vs Pneumonia
        |
        |-- Normal
        |     |
        |     v
        |   Return Normal prediction
        |
        |-- Pneumonia
              |
              v
      Stage 2 model: Bacterial vs Viral
              |
              v
      Return Pneumonia subtype prediction
```

This design matches the observed model behavior from experimentation. Normal-vs-pneumonia detection is significantly stronger than the direct three-class task, while bacterial-vs-viral classification remains more difficult and should be reported with appropriate confidence.

## Backend Responsibilities

The Python backend should provide:

- `/health` endpoint for availability checks
- `/predict` endpoint for chest X-ray upload
- file type and image validation
- preprocessing aligned with training transforms
- model-serving integration with Hugging Face
- hierarchical prediction orchestration
- confidence/probability formatting
- clear error messages
- medical-use disclaimer in responses

Example response shape:

```json
{
  "primary_prediction": "Pneumonia",
  "primary_confidence": 0.94,
  "subtype_prediction": "Bacterial Pneumonia",
  "subtype_confidence": 0.76,
  "probabilities": {
    "normal": 0.06,
    "pneumonia": 0.94,
    "bacterial": 0.76,
    "viral": 0.24
  },
  "disclaimer": "This result is for educational support only and is not a medical diagnosis."
}
```

## Model-Serving Layer

Model checkpoints are intended to be hosted and managed through Hugging Face. This keeps the application lightweight and separates model lifecycle management from the user-facing web app.

The serving layer should support:

- versioned model checkpoints
- consistent preprocessing and postprocessing
- model metadata such as class labels, training date, metrics, and threshold settings
- future explainability outputs such as Grad-CAM heatmaps

## Inference Design Layer

The repository includes a reusable inference layer under `src/inference/`:

```text
src/inference/settings.py              environment-based inference settings
src/inference/huggingface_adapter.py   Hugging Face image-classification client
src/inference/hierarchical_pipeline.py two-stage pneumonia prediction pipeline
src/inference/schemas.py               stable response dataclasses
```

The design keeps model-provider details behind an adapter. The backend should call `HierarchicalPneumoniaPipeline` and return its `to_dict()` response, rather than manually constructing prediction JSON in the route handler.

Required environment variables for hosted inference:

```text
HF_BINARY_MODEL_ID
HF_SUBTYPE_MODEL_ID
```

Optional environment variables:

```text
HF_TOKEN
HF_API_BASE_URL
PNEUMONIA_THRESHOLD
SUBTYPE_THRESHOLD
HF_REQUEST_TIMEOUT_SECONDS
```

## Frontend Responsibilities

The frontend should provide:

- upload component for chest X-ray images
- image preview
- loading and error states
- prediction result card
- confidence/probability display
- subtype result only when pneumonia is predicted
- disclaimer and limitation notice

The recommended frontend stack is React or Next.js hosted on Cloudflare Pages.

## Deployment Shape

```text
Cloudflare Pages
  |
  | hosts React / Next.js frontend
  v
Cloudflare-hosted backend where feasible
  |
  | calls
  v
Hugging Face model endpoint
```

Docker should be available as a portable backend deployment option. This allows the Python API to be moved to any container-capable platform if the final backend dependencies exceed the supported runtime of the selected serverless platform.

## Future Extensions

- Grad-CAM explainability overlay
- ROC/AUC, sensitivity, specificity reporting in the app
- ensemble inference endpoint
- model version selection for internal evaluation
- audit logging without storing patient images
- Dockerized local demo environment
