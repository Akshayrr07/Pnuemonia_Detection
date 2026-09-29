# Configs

This directory stores example configuration files for training, evaluation, and inference.

Configuration files should define paths, model names, class labels, thresholds, and deployment-specific model identifiers without hardcoding them inside Python modules.

## Evaluation

Copy `evaluation.example.yaml` to `evaluation.json` or `evaluation.yaml`, then run:

```bash
python evaluate.py --config configs/evaluation.json
```

The command-line values override the config file. The evaluator's documented
neutral defaults are:

- split manifests: `data/splits/train.csv`, `data/splits/val.csv`, and
  `data/splits/test.csv`
- dataset root recorded in the report: `data/raw_datasets`
- model order: `mobilenet`, `efficientnet`, `resnet`
- batch size: `32`
- `viral_boost`: `1.0` (no class-specific post-hoc adjustment)
- `ensemble_weights`: omitted, meaning equal weighting
- device: `auto`
- report directory: `reports`

Use `--viral-boost` and/or `--ensemble-weights` to opt into a non-neutral
evaluation. The selected values, split hashes/counts, checkpoint paths/hashes,
and device are written to the report. Weights and boosts must be selected from
validation evidence before evaluation, not tuned against the test split.

Secrets such as tokens must be provided through environment variables and must not be committed to the repository.

