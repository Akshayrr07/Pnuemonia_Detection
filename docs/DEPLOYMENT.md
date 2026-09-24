# Deployment Notes

This document describes the deployment artifacts and configuration included in the repository.

## Deployment Goals

The documented deployment shape separates the application from the model-serving layer:

- frontend hosted in the cloud
- backend hosted on a Docker-capable cloud or local host
- trained model checkpoints hosted and managed through Hugging Face
- Docker available for local and portable backend execution

This keeps the repository lightweight and avoids committing large model or dataset files.

## Model Hosting

Model checkpoints are hosted on Hugging Face in the documented deployment design. The repository does not include hosted repositories or checkpoints; the application backend calls the hosted model layer rather than loading weights inside the frontend.

A production hosted model layer should include:

- binary Normal-vs-Pneumonia checkpoint
- subtype Bacterial-vs-Viral checkpoint
- class label metadata
- preprocessing metadata
- validation metrics
- model version information

## Frontend Hosting

The documented frontend target is Cloudflare Pages. It is suitable for the lightweight upload-and-results static interface.

The frontend should not contain model credentials or model-serving logic. It should only call the backend API.

## Backend Hosting

The repository includes a FastAPI Python backend responsible for:

- receiving image uploads
- validating files
- preprocessing images
- calling the hosted model layer
- applying hierarchical prediction logic
- returning structured JSON

The documented backend path is the Docker deployment. A serverless worker would require a separate compatibility review and is not assumed by this repository.

## Docker Support

Docker support is included for backend portability. A Dockerized backend allows the API to run consistently across:

- local development machines
- cloud virtual machines
- container platforms
- demos and evaluation environments

Included Docker artifacts:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Local command shape:

```bash
docker compose up --build
```

## Environment Variables

The backend should use environment variables for deployment-specific settings:

```text
HF_TOKEN=
HF_BINARY_MODEL_ID=
HF_SUBTYPE_MODEL_ID=
HF_API_BASE_URL=https://api-inference.huggingface.co/models
PNEUMONIA_THRESHOLD=
SUBTYPE_THRESHOLD=
HF_REQUEST_TIMEOUT_SECONDS=
ALLOWED_ORIGINS=
UPLOAD_MAX_SIZE_MB=

# Optional local-model explainability
LOCAL_MODEL_PATH=
LOCAL_MODEL_NAME=resnet

# Frontend build-time variable; set before `npm run build`/deployment build
NEXT_PUBLIC_BACKEND_URL=
```

Secrets must not be committed to GitHub.

`HF_API_BASE_URL` is the model endpoint base, not the general Hugging Face API
host; the current adapter appends the model ID, so its default is
`https://api-inference.huggingface.co/models`. `NEXT_PUBLIC_BACKEND_URL` is
inlined into the static frontend bundle at build time. Set it before the
frontend build, rebuild after changing it, and never place secrets in a
`NEXT_PUBLIC_*` value.

Do not place real patient images or credentials in a public deployment. Review
request logging, retention, access control, and the hosted model provider's
data handling before processing health data. The application is educational
and research software, not a medical device or diagnostic service.

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
