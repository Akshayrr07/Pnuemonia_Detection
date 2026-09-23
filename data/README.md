# Data Directory

This directory holds dataset metadata and split files for the pneumonia detection project.

## Structure

```text
data/
├── raw_datasets/          # (not committed) Kaggle chest X-ray images
├── metadata/
│   └── master_registry.csv
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

## Split CSVs

Each split file (`train.csv`, `val.csv`, `test.csv`) contains at minimum:

- `original_path` — a **relative path** from `data/raw_datasets/` to the image file, e.g. `dataset_1/train/PNEUMONIA/VIRUS-5051946-0002.jpeg`
- `encoded_label` — integer label (0 = Normal, 1 = Bacterial Pneumonia, 2 = Viral Pneumonia)
- other registry columns as present in `master_registry.csv`

The `original_path` column is used by `PneumoniaDataset` (in `src/data/dataset.py`) as the image location. Because the raw images are not committed to the repository, these paths are only meaningful after the user rebuilds the data locally from the documented Kaggle sources.

### Path format

Paths are stored as **relative, forward-slash paths** with no drive letter or machine-specific prefix. This keeps the CSVs portable across Windows, macOS, and Linux.

If you regenerate splits from your own local copy of the raw data, make sure your script writes relative paths (relative to `data/raw_datasets/`) rather than absolute machine paths.

## Rebuilding the raw data locally

The raw images are not in this repository because of their size. To rebuild the data locally:

1. Download the chest X-ray datasets from the Kaggle sources listed in `docs/ARCHITECTURE.md` or `docs/RESEARCH_SUMMARY.md`.
2. Place the image folders under `data/raw_datasets/`.
3. Run the registry and split scripts from `experiments/research_scripts/` (see `experiments/README.md`) to recreate `master_registry.csv` and the split CSVs.

The legacy research scripts in `experiments/research_scripts/` use the old module names `data_pipeline` and `model_architectures`. They are preserved for project history and are not part of the production code path.

## Trained checkpoints

Trained model checkpoints (`.pt` / `.pth` files) are **not included** in this repository.

The production inference path does not rely on local checkpoints. It calls hosted models through the Hugging Face Inference API. See:

- `src/inference/huggingface_adapter.py`
- `src/inference/hierarchical_pipeline.py`
- `docs/ARCHITECTURE.md` for the hosted-model design

If you want to run inference with local checkpoints instead, you can:

1. Train models using the scripts under `experiments/research_scripts/` (research history only) or the consolidated training code under `src/`.
2. Save the resulting state dicts to a local directory.
3. Point `LOCAL_MODEL_PATH` at the checkpoint and set `EXPLAINABILITY_ENABLED` to use the local-model explainability path in `backend/app.py`.

The legacy Streamlit app in `app/` also expects local checkpoints in `saved_models/`. That app is deprecated and has been superseded by the FastAPI backend + Next.js frontend.
