# Configs

This directory stores example configuration files for training, evaluation, and inference.

Configuration files should define paths, model names, class labels, thresholds, and deployment-specific model identifiers without hardcoding them inside Python modules.

Training examples set `training.selection_metric: val_loss`. This is the single documented metric that controls both checkpoint selection and early stopping; validation accuracy remains a logged metric only. Every training loader is seeded explicitly, and each saved checkpoint embeds the seed, complete training configuration, SHA-256 split hashes, deterministic settings, and optimizer/scheduler/epoch state. A canonical JSON copy is written next to the checkpoint with the `.metadata.json` suffix.

Secrets such as tokens must be provided through environment variables and must not be committed to the repository.

