# Research Summary

This document summarizes the machine-learning work completed before production deployment.

## Dataset Construction

The project started by combining three Kaggle chest X-ray datasets into a single controlled dataset. The datasets were not used directly in their raw form. Instead, their folder structures and labels were audited, normalized, and merged into a single master registry.

The normalized class set is:

- `Normal`
- `Bacterial`
- `Viral`

Each image record includes:

- image ID
- original file path
- source dataset
- original label
- normalized label
- encoded label
- SHA256 hash
- perceptual hash
- flagged status

Exact duplicate images were removed using SHA256 hashing. The resulting controlled dataset contains 5,891 unique X-rays.

## Splitting Strategy

The dataset was split into:

- 70% training
- 15% validation
- 15% testing

The split is stratified by encoded label. Leakage checks verify that image paths do not overlap across train, validation, and test sets.

Class weights are calculated from the training distribution to reduce the effect of class imbalance.

## Model Families Evaluated

The research phase evaluated multiple CNN-based approaches:

- Custom lightweight CNN
- MobileNetV2
- EfficientNet-B0
- ResNet18
- DenseNet121 support in the modular codebase

The project also contains ensemble evaluation utilities for combining probabilities through soft voting and weighted soft voting.

## Three-Class Classification

The first modeling approach treated the problem as direct three-class classification:

```text
Normal vs Bacterial Pneumonia vs Viral Pneumonia
```

The models consistently achieved approximately 74-77% accuracy. The viral class was the most difficult class to separate.

This result showed that the model could learn useful X-ray features, but it also exposed the limitation of forcing a single classifier to separate all categories in one step.

## Binary Pneumonia Detection

The next experiment simplified the task to:

```text
Normal vs Pneumonia
```

This binary model achieved 93.78% accuracy, confirming that the dataset and training pipeline are effective for detecting pneumonia presence.

## Subtype Classification

For images already identified as pneumonia, a second model was trained for:

```text
Bacterial Pneumonia vs Viral Pneumonia
```

The subtype classifier achieved approximately 75.67% accuracy. This confirms that bacterial-versus-viral separation is a harder classification problem and should be represented honestly in the final system.

## Final Modeling Direction

Based on the experimental results, the system is designed as a hierarchical classifier:

1. Detect whether the image is normal or pneumonia.
2. If pneumonia is detected, classify the subtype as bacterial or viral.

This design is more aligned with the observed model strengths and creates a clearer production inference flow.

## Ensemble Direction

The repository includes foundations for soft and weighted probability voting. Ensemble evaluation remains valuable for research and may be used to improve robustness, but the production API should expose a stable prediction contract independent of whether the underlying model is a single checkpoint, hierarchical pair, or ensemble.

## Current Limitations

- The raw datasets are not included in the repository due to size.
- Current split CSVs may contain machine-specific file paths and should be regenerated for new environments.
- Some experiment scripts under `experiments/research_scripts/` are legacy research scripts and should be consolidated before production use.
- Three-class bacterial-versus-viral classification remains challenging.
- The project is not a certified medical diagnostic system.

## Next Research Improvements

- ROC/AUC analysis
- sensitivity and specificity reporting
- calibration analysis
- threshold tuning for pneumonia detection
- Grad-CAM explainability
- external validation on unseen datasets
- model-card documentation for each checkpoint
