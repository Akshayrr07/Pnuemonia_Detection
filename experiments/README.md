# Experiments

This directory contains research and exploratory scripts from earlier project phases.

The scripts in `research_scripts/` document the evolution of the project:

- dataset registry construction
- stratified split creation
- architecture experiments
- binary Normal-vs-Pneumonia training
- Bacterial-vs-Viral subtype training
- ensemble evaluation

These files are preserved for project history and presentation context. Production training, evaluation, and inference code should be consolidated under `src/`, with cloud application code separated into `backend/` and `frontend/`.

## Legacy import compatibility

The research scripts in this directory were written against an earlier project layout where the dataset and model code lived in top-level modules called `data_pipeline` and `model_architectures`. Those modules no longer exist as top-level files.

A compatibility shim, `legacy_imports_compat.py`, lives alongside the scripts and redirects those legacy import names to the current production modules under `src/`. The shim is intentionally small and is only for running legacy research code — it is not part of the production inference or training path.

If you run a script from this directory, make sure your `PYTHONPATH` includes the project root so the shim can find `src/`. For example:

```bash
cd experiments/research_scripts
PYTHONPATH=../.. python train_resnet18.py
```

Scripts still reference `../splits/` and `../saved_models/` relative paths, so they expect to be run from `experiments/research_scripts/` with the corresponding data and checkpoints in place.

