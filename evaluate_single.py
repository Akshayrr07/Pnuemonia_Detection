"""Run reproducible individual-model evaluation.

Metric calculations are preserved from the original script. Dataset, split,
and checkpoint metadata are recorded in every report, and the model/data
selection is explicit through the shared evaluation configuration.
"""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from src.data.dataloader import get_dataloaders
from src.evaluation.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_SINGLE_MODEL_NAMES,
    EvaluationConfig,
    parse_args,
)
from src.evaluation.loader import load_evaluation_models
from src.evaluation.metadata import dataset_metadata, model_metadata, split_metadata


def _resolve_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _report_path(config: EvaluationConfig) -> Path:
    if config.report_path:
        return Path(config.report_path)
    return Path(config.report_dir) / (
        f"single_model_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )


def _write_metadata(file, config: EvaluationConfig, device) -> None:
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
    file.write("Evaluation configuration:\n")
    file.write(json.dumps(provenance["config"], sort_keys=True) + "\n")
    file.write("Dataset and split metadata:\n")
    file.write(json.dumps(provenance["dataset"], sort_keys=True) + "\n")
    file.write("Model metadata:\n")
    file.write(json.dumps(provenance["models"], sort_keys=True) + "\n")
    file.write(f"Device: {device}\n" + "-" * 60 + "\n")


def evaluate_model(
    model_name,
    file,
    *,
    device=None,
    test_loader=None,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
):
    """Evaluate one named model without changing the original metrics."""

    if device is None:
        device = _resolve_device("auto")
    if test_loader is None:
        raise ValueError("test_loader must be supplied for evaluation")

    print(f"\n===== {model_name.upper()} =====")
    model = load_evaluation_models(
        device,
        model_names=(model_name,),
        checkpoint_dir=checkpoint_dir,
    )[0]

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{model_name} evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    text = f"""
===== {model_name.upper()} =====
Accuracy: {acc:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}

AUC: {auc:.4f}
"""
    print(text)
    file.write(text + "\n")

    return acc


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Execute individual-model evaluation and return the report path."""

    config = parse_args(argv, default_model_names=DEFAULT_SINGLE_MODEL_NAMES)
    device = _resolve_device(config.device)
    report_path = _report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(
        config.train_csv,
        config.val_csv,
        config.test_csv,
        batch_size=config.batch_size,
    )

    with report_path.open("w", encoding="utf-8") as file:
        file.write("===== INDIVIDUAL MODEL EVALUATION =====\n")
        _write_metadata(file, config, device)

        results = {}
        for model_name in config.model_names:
            acc = evaluate_model(
                model_name,
                file,
                device=device,
                test_loader=test_loader,
                checkpoint_dir=config.checkpoint_dir,
            )
            results[model_name] = acc

        file.write("\n===== SUMMARY =====\n")
        print("\n===== SUMMARY =====")
        for model_name, value in results.items():
            line = f"{model_name}: {value:.4f}"
            print(line)
            file.write(line + "\n")

    print(f"\n📄 Report saved at: {report_path}")
    return report_path


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
