# Deployment Notes

This document describes the intended deployment direction for the pneumonia detection system.

## Deployment Goals

The production-oriented system should separate the application from the model-serving layer:

- frontend hosted in the cloud
- backend hosted in the cloud
- trained model checkpoints hosted and managed through Hugging Face
- Docker available for local and portable backend execution

This keeps the repository lightweight and avoids committing large model or dataset files.

## Model Hosting

Model checkpoints are intended to be hosted on Hugging Face. The application backend should call the hosted model layer rather than loading large checkpoints directly inside the frontend.

The hosted model layer should eventually include:

- binary Normal-vs-Pneumonia checkpoint
- subtype Bacterial-vs-Viral checkpoint
- class label metadata
- preprocessing metadata
- validation metrics
- model version information

## Frontend Hosting

The recommended frontend target is Cloudflare Pages. Cloudflare Pages supports React applications and is suitable for hosting a lightweight upload-and-results interface.

The frontend should not contain model credentials or model-serving logic. It should only call the backend API.

## Backend Hosting

The backend is planned as a Python API responsible for:

- receiving image uploads
- validating files
- preprocessing images
- calling the hosted model layer
- applying hierarchical prediction logic
- returning structured JSON

Cloudflare Workers can be considered for the Python backend where the dependency set is compatible with the runtime. If the backend requires packages or runtime behavior better suited to containers, the Docker deployment path should be used.

## Docker Support

Docker support is planned for backend portability. A Dockerized backend allows the API to run consistently across:

- local development machines
- cloud virtual machines
- container platforms
- demos and evaluation environments

Planned Docker artifacts:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Expected local command shape:

```bash
docker compose up --build
```

## Dependency and installation policy

- Supported Python runtime: **3.11.x** (also declared in `pyproject.toml`).
- The production API depends only on the lightweight hosted-inference set in
  `backend/requirements.in`; its resolved metadata is in
  `backend/constraints.txt`.
- Training, evaluation, dataset preparation, and the legacy research app use
  the separate `requirements-research.in` / `requirements-research.lock` pair.
- Grad-CAM is optional and is not required for the hosted backend. Install
  `backend/requirements-explainability.in` (and its lock metadata) only when a
  local checkpoint path is configured.
- Portable requirements never use CUDA-local version suffixes such as
  `torch==2.0.1+cu117`. For CUDA, select a supported wheel backend explicitly:

  ```bash
  uv pip install --torch-backend=cu124 -r requirements-research.in
  ```

  The exact backend must match the host driver/runtime. The default install is
  the portable published-wheel path; use the CPU backend when CUDA is not
  required.
- Regenerate metadata rather than hand-editing locks:

  ```bash
  uv pip compile backend/requirements.in --python-version 3.11 --universal \
    --output-file backend/constraints.txt --no-annotate
  uv pip compile requirements-research.in --python-version 3.11 --universal \
    --output-file requirements-research.lock --no-annotate
  ```

Platform-specific lock output may include CUDA dependencies when resolving for
a Linux target. Review the generated lock for the deployment platform before
using it in an image or worker.

## Environment Variables

The backend should use environment variables for deployment-specific settings:

```text
HF_TOKEN=
HF_BINARY_MODEL_ID=
HF_SUBTYPE_MODEL_ID=
HF_API_BASE_URL=
PNEUMONIA_THRESHOLD=
SUBTYPE_THRESHOLD=
HF_REQUEST_TIMEOUT_SECONDS=
ALLOWED_ORIGINS=
MAX_UPLOAD_MB=
```

Secrets must not be committed to GitHub.

## Target Cloud Layout

```text
Cloudflare Pages
  |
  v
React / Next.js frontend
  |
  v
Python backend API
  |
  v
Hugging Face hosted model checkpoints
```

## References

- Hugging Face image-classification inference documentation: https://huggingface.co/docs/inference-providers/tasks/image-classification
- Hugging Face custom inference handler documentation: https://huggingface.co/docs/inference-endpoints/guides/custom_handler
- Cloudflare React Pages documentation: https://developers.cloudflare.com/pages/framework-guides/deploy-a-react-site/
- Cloudflare Python Workers documentation: https://developers.cloudflare.com/workers/languages/python/
