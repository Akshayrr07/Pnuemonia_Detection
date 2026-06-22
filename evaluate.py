import torch
import multiprocessing
import os
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.dataloader import get_dataloaders
from src.inference.load_models import load_all_models
from src.inference.ensemble import get_ensemble_predictions


# =========================
# METRIC FUNCTIONS
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


# =========================
# MAIN EVALUATION
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("reports", exist_ok=True)

    report_path = f"reports/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, "w") as f:

        log(f, "===== FINAL MODEL EVALUATION REPORT =====")
        log(f, f"Device: {device}")
        log(f, "-" * 60)

        # 🔹 Load Data
        _, _, test_loader = get_dataloaders(
            "data/splits/train.csv",
            "data/splits/val.csv",
            "data/splits/test.csv"
        )

        # 🔹 Load Models
        models = load_all_models(device)
        model_names = ["mobilenet", "efficientnet", "resnet"]
        log(f, f"\nModels Loaded: {model_names}")
        log(f, "-" * 60)

        # 🔹 Get predictions (Tensor shape: [num_models, N, num_classes])
        all_model_probs, labels = get_ensemble_predictions(models, test_loader, device)

        # =========================
        # SOFT VOTING
        # =========================
        log(f, "\n=== SOFT VOTING ===")

        # Compute probabilities
        soft_probs = torch.mean(all_model_probs, dim=0)

        # 🔥 Boost viral class
        soft_probs[:, 2] *= 1.05

        # ✅ Normalize (CRITICAL)
        soft_probs = soft_probs / soft_probs.sum(dim=1, keepdim=True)

        soft_probs = soft_probs.cpu().numpy()
        soft_preds = soft_probs.argmax(axis=1)

        soft_metrics = compute_metrics(labels, soft_preds)
        soft_auc = compute_auc_from_probs(labels, soft_probs)

        for k, v in soft_metrics.items():
            log(f, f"{k}: {v:.4f}")

        log(f, f"AUC: {soft_auc:.4f}")

        # =========================
        # WEIGHTED VOTING
        # =========================
        log(f, "\n=== WEIGHTED VOTING ===")

        # 🔹 Define weights
        weights = torch.tensor([0.85, 0.86, 0.88]).to(all_model_probs.device)
        weights = weights.view(-1, 1, 1)

        # 🔹 Weighted probabilities
        weighted_probs = (all_model_probs * weights).sum(dim=0)
        weighted_probs = weighted_probs / weights.sum()

        # 🔥 Boost viral class
        weighted_probs[:, 2] *= 1.05

        # ✅ Normalize (CRITICAL)
        weighted_probs = weighted_probs / weighted_probs.sum(dim=1, keepdim=True)

        weighted_probs = weighted_probs.cpu().numpy()
        weighted_preds = weighted_probs.argmax(axis=1)

        weighted_metrics = compute_metrics(labels, weighted_preds)
        weighted_auc = compute_auc_from_probs(labels, weighted_probs)

        for k, v in weighted_metrics.items():
            log(f, f"{k}: {v:.4f}")

        log(f, f"AUC: {weighted_auc:.4f}")

        log(f, "\n===== END OF REPORT =====")

    print(f"\n📄 Report saved at: {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()