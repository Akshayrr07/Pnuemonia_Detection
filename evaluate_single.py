import torch
import os
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load data
_, _, test_loader = get_dataloaders(
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv"
)


def evaluate_model(model_name, file):

    print(f"\n===== {model_name.upper()} =====")

    model = get_model(model_name, num_classes=3, freeze=False)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pt", weights_only=True))
    model.to(device)
    model.eval()

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

    # 🔥 Metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    # 🔥 Print + Save
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


def main():

    os.makedirs("reports", exist_ok=True)

    report_path = f"reports/single_model_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, "w") as f:

        f.write("===== INDIVIDUAL MODEL EVALUATION =====\n")

        model_names = ["mobilenet", "efficientnet", "resnet", "densenet"]

        results = {}

        for model_name in model_names:
            acc = evaluate_model(model_name, f)
            results[model_name] = acc

        # 🔥 Summary Table
        f.write("\n===== SUMMARY =====\n")
        print("\n===== SUMMARY =====")

        for k, v in results.items():
            line = f"{k}: {v:.4f}"
            print(line)
            f.write(line + "\n")

    print(f"\n📄 Report saved at: {report_path}")


if __name__ == "__main__":
    main()