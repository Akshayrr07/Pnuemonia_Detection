"""Evaluation entry-point orchestration with no hidden post-hoc transforms."""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataloader import get_dataloaders
from src.evaluation.aggregation import soft_vote, weighted_vote
from src.evaluation.config import EvaluationConfig, parse_args
from src.evaluation.loader import load_evaluation_models
from src.evaluation.metadata import dataset_metadata, model_metadata, split_metadata
from src.inference.ensemble import get_ensemble_predictions


# =========================
# METRIC FUNCTIONS
# These calculations are intentionally preserved from the original script.
# =========================

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted')
    }


def compute_auc_from_probs(y_true, probs):
    return roc_auc_score(y_true, probs, multi_class='ovr')


# =========================
# LOGGING
# =========================

def log(file, text):
    print(text)
    file.write(text + "\n")


def _resolve_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _report_path(config: EvaluationConfig) -> Path:
    if config.report_path:
        return Path(config.report_path)
    return Path(config.report_dir) / (
        f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )


def _write_metadata(
    file,
    config: EvaluationConfig,
    device: torch.device,
) -> None:
    split_blocks = [
        split_metadata("train", config.train_csv, dataset_root=config.dataset_root),
        split_metadata("validation", config.val_csv, dataset_root=config.dataset_root),
        split_metadata("test", config.test_csv, dataset_root=config.dataset_root),
    ]
    provenance = {
        "config": config.to_dict(),
        "dataset": dataset_metadata(
            split_blocks,
            dataset_root=config.dataset_root,
        ),
        "models": model_metadata(config.model_names, config.checkpoint_dir),
        "device": str(device),
    }
    log(file, "Evaluation configuration:")
    log(file, json.dumps(provenance["config"], sort_keys=True))
    log(file, "Dataset and split metadata:")
    log(file, json.dumps(provenance["dataset"], sort_keys=True))
    log(file, "Model metadata:")
    log(file, json.dumps(provenance["models"], sort_keys=True))
    log(file, f"Device: {device}")
    log(file, "-" * 60)


def _write_metric_block(
    file,
    title: str,
    y_true,
    probs,
    labels: Optional[Sequence[str]] = None,
) -> None:
    metrics = compute_metrics(y_true, probs.argmax(axis=1))
    auc = compute_auc_from_probs(y_true, probs)
    log(file, f"\n=== {title} ===")
    if labels is not None:
        log(file, f"Model order: {', '.join(labels)}")
    for key, value in metrics.items():
        log(file, f"{key}: {value:.4f}")
    log(file, f"AUC: {auc:.4f}")


def run(config: Optional[EvaluationConfig] = None) -> Path:
    """Execute one evaluation and return the report path."""

    config = config or parse_args([])
    device = _resolve_device(config.device)
    report_path = _report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        log(file, "===== FINAL MODEL EVALUATION REPORT =====")
        _write_metadata(file, config, device)

        # Loading the train/val/test manifests preserves the existing
        # dataloader contract while making all three split identities explicit.
        _, _, test_loader = get_dataloaders(
            config.train_csv,
            config.val_csv,
            config.test_csv,
            batch_size=config.batch_size,
        )
        models = load_evaluation_models(
            device,
            model_names=config.model_names,
            checkpoint_dir=config.checkpoint_dir,
        )
        all_model_probs, labels = get_ensemble_predictions(models, test_loader, device)
        labels = labels.cpu().numpy()

        soft_probs = soft_vote(
            all_model_probs.cpu().numpy(),
            viral_boost=config.viral_boost,
        )
        _write_metric_block(
            file,
            "SOFT VOTING",
            labels,
            soft_probs,
            labels=config.model_names,
        )

        weighted_probs = weighted_vote(
            all_model_probs.cpu().numpy(),
            weights=config.ensemble_weights,
            viral_boost=config.viral_boost,
        )
        _write_metric_block(
            file,
            "WEIGHTED VOTING",
            labels,
            weighted_probs,
            labels=config.model_names,
        )
        log(file, "\n===== END OF REPORT =====")

    print(f"\n📄 Report saved at: {report_path}")
    return report_path


def main(argv: Optional[Sequence[str]] = None) -> Path:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
