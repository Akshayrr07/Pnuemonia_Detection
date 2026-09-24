# Pneumonia Detection from Chest X-rays

Production-oriented AI system for detecting pneumonia from chest X-ray images using a controlled dataset pipeline, PyTorch computer vision models, hierarchical classification, and cloud-hosted inference.

## Project Goal

The goal of this project is to build an end-to-end pneumonia detection system that moves beyond a simple model-training notebook. The system is designed around a full machine-learning workflow:

- controlled dataset construction from multiple Kaggle sources
- duplicate removal and label normalization
- leakage-safe train, validation, and test splits
- multiple CNN-based training approaches
- hierarchical prediction for clinically meaningful output
- model serving through Hugging Face
- cloud-hosted frontend and backend application

The intended user flow is simple: a user uploads a chest X-ray image, the system preprocesses the image, runs the pneumonia detection pipeline, and returns a structured prediction with confidence information and a medical-use disclaimer.

## Dataset Strategy

The original research pipeline combines three Kaggle chest X-ray datasets into one controlled dataset. Instead of training directly on raw folders, the project first audits and normalizes the data into a master registry.

The normalized target classes are:

- `Normal`
- `Bacterial Pneumonia`
- `Viral Pneumonia`

The dataset registry stores image identity, source dataset, original label, normalized label, encoded class, SHA256 hash, perceptual hash, and review flags. Exact duplicates are removed using SHA256 hashing.

Final controlled dataset:

- 5,891 unique chest X-ray images
- 70/15/15 train-validation-test split
- leakage-safe split verification
- class weights calculated to handle imbalance

Large raw datasets are not committed to GitHub. The expected workflow is to download the Kaggle datasets locally, place them under `data/raw_datasets/`, and rebuild the registry and splits using the provided scripts.

## Kaggle Dataset Sources

Add the exact Kaggle URLs used by the project here before final submission:

- Dataset 1: `TODO: Kaggle dataset URL`
- Dataset 2: `TODO: Kaggle dataset URL`
- Dataset 3: `TODO: Kaggle dataset URL`

Expected local structure:

```text
data/
  raw_datasets/
    dataset_1/
    dataset_2/
    dataset_3/
  metadata/
    master_registry.csv
    duplicates_removed.csv
    label_distribution_report.csv
  splits/
    train.csv
    val.csv
    test.csv
```

## Research Pipeline

The project evaluates several CNN-based approaches:

- Custom lightweight CNN
- MobileNetV2
- EfficientNet-B0
- ResNet18
- DenseNet121 support in the modular model factory

The initial three-class task, `Normal` vs `Bacterial` vs `Viral`, reached approximately 74-77% accuracy, with the viral class being the most difficult. The more fundamental binary task, `Normal` vs `Pneumonia`, reached 93.78% accuracy. Because bacterial-versus-viral separation is harder from X-ray images alone, the system design was changed to a hierarchical pipeline.

## Hierarchical Model Design

The final model architecture is designed as two stages:

```text
Chest X-ray image
        |
        v
Stage 1: Normal vs Pneumonia
        |
        |-- Normal --> return Normal result
        |
        |-- Pneumonia --> Stage 2: Bacterial vs Viral
                              |
                              v
                    return Pneumonia subtype result
```

This avoids forcing one three-class classifier to solve two different clinical questions at the same time. The subtype model currently achieves approximately 75.67% accuracy and is treated as the harder classification stage.

## Ensemble Research

The repository also includes the foundation for model ensembling. The explored approach combines model probabilities through soft voting or weighted soft voting. This is used as a research and evaluation strategy for improving robustness across CNN architectures.

The production inference contract should remain independent of any single model family so that individual checkpoints, hierarchical models, or ensemble models can be served behind the same API response format.

## Current Repository Layout

```text
app/                         Streamlit prototype inference UI
backend/                     planned Python API application
configs/                     example training and inference configuration
data/                        metadata and split CSVs; raw datasets excluded
docs/                        project architecture, research, and deployment notes
experiments/research_scripts/ historical research and experiment scripts
frontend/                    planned React or Next.js application
src/data/                    dataset, dataloader, transforms, class-weight utilities
src/models/                  CNN model definitions and model factory
src/training/                training loop and metrics
src/inference/               model loading and ensemble inference helpers
main.py                      multi-model training entrypoint
evaluate.py                  ensemble evaluation script
evaluate_single.py           individual model evaluation script
```

## Deployment Direction

The deployment plan separates model serving from the application layer:

- models hosted and managed on Hugging Face
- Python backend responsible for validation, preprocessing, inference orchestration, and API response formatting
- React or Next.js frontend hosted on Cloudflare Pages
- backend hosted on Cloudflare Workers where feasible, or packaged as a Docker service for container-capable infrastructure

Docker support is planned so the backend can be run consistently across local machines and cloud environments.

## Inference Configuration

The hosted inference layer is configured through environment variables:

```text
HF_BINARY_MODEL_ID=
HF_SUBTYPE_MODEL_ID=
HF_TOKEN=
PNEUMONIA_THRESHOLD=0.5
SUBTYPE_THRESHOLD=0.5
```

The reusable inference modules live in `src/inference/` and expose a hierarchical prediction pipeline for the future backend API.

## Supported Python environments

The supported runtime for the production service and research pipeline is
Python 3.11 (`3.11.x`), declared in `pyproject.toml`. Keep the lightweight
hosted-inference service and the larger research environment separate:

```bash
# Production/backend API
uv venv --python 3.11
uv pip install -r backend/requirements.txt

# Offline training, evaluation, experiments, and legacy Streamlit app
uv venv --python 3.11 .venv-research
uv pip install -r requirements-research.txt
```

`backend/requirements.in` is the backend source declaration and
`backend/constraints.txt` records its resolved Python 3.11 metadata.
`requirements-research.in` is the research source declaration and
`requirements-research.lock` records the corresponding resolution. The root
`requirements.txt` is only a compatibility shim for existing backend commands.

Do not copy CUDA-local package versions such as `+cu117` into a root or
portable requirements file. The default paths resolve published wheels for the
host. On a CUDA machine, select the supported backend explicitly, for example:

```bash
uv pip install --torch-backend=cu124 -r requirements-research.in
```

Use the CPU default when CUDA is not required. The research lock is an
inspection/reproducibility aid, not a promise that every platform-specific
CUDA wheel can be installed on every host; choose a platform-appropriate lock
or regenerate it with uv for the deployment target.

## Medical Disclaimer

This project is for educational and research purposes. It is not a certified medical device and must not be used as the sole basis for diagnosis or treatment. Any abnormal result should be reviewed by a qualified medical professional.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Research Summary](docs/RESEARCH_SUMMARY.md)
- [Deployment Notes](docs/DEPLOYMENT.md)
- [Project Progress](docs/PROJECT_PROGRESS.md)
- [CI and pull-request automation](docs/CI.md)
