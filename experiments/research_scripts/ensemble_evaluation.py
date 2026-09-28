# ============================================================
# PHASE 5 — ENSEMBLE EVALUATION
# Soft Voting + Weighted Soft Voting
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

from legacy_imports_compat import PneumoniaDataset, get_resnet18, get_mobilenet, get_efficientnet, CustomCNN


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

TEST_CSV = "../splits/test.csv"
test_df = pd.read_csv(TEST_CSV)

test_df = test_df.rename(columns={
    "original_path": "image_path",
    "encoded_label": "label"
})

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

test_dataset = PneumoniaDataset(test_df, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ============================================================
# LOAD TRAINED MODELS
# (Assumes you saved weights previously)
# ============================================================

custom_model = CustomCNN(num_classes=3).to(device)
custom_model.load_state_dict(torch.load("../saved_models/custom.pt"))

mobilenet_model = get_mobilenet(num_classes=3, freeze=False).to(device)
mobilenet_model.load_state_dict(torch.load("../saved_models/mobilenet.pt"))

efficientnet_model = get_efficientnet(num_classes=3, freeze=False).to(device)
efficientnet_model.load_state_dict(torch.load("../saved_models/efficientnet.pt"))

resnet_model = get_resnet18(num_classes=3, freeze=False).to(device)
resnet_model.load_state_dict(torch.load("../saved_models/resnet.pt"))

models = [
    custom_model,
    mobilenet_model,
    efficientnet_model,
    resnet_model
]


# ============================================================
# VALIDATE WITH PROBABILITIES
# ============================================================

def validate_with_probs(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_labels)


# ============================================================
# GET PROBABILITIES FROM EACH MODEL
# ============================================================

all_model_probs = []
labels = None

for model in models:
    probs, lbls = validate_with_probs(model, test_loader)
    all_model_probs.append(probs)
    labels = lbls

# Stack into tensor: [num_models, N, num_classes]
all_model_probs = torch.stack(all_model_probs)


# ============================================================
# 1️⃣ SOFT VOTING
# ============================================================

ensemble_probs = torch.mean(all_model_probs, dim=0)
ensemble_preds = torch.argmax(ensemble_probs, dim=1)

soft_acc = (ensemble_preds == labels).float().mean().item()

print("\n=== SOFT VOTING RESULTS ===")
print("Ensemble Accuracy:", soft_acc)
print(confusion_matrix(labels, ensemble_preds))
print(classification_report(labels, ensemble_preds))


# ============================================================
# 2️⃣ WEIGHTED SOFT VOTING
# ============================================================

# Weights are an explicit input to this legacy research script. Use equal
# weights unless a separately prepared, documented evaluation run supplies a
# validated weighting. Do not tune these values on the test set.
weights = None  # equal weighting by default
if weights is None:
    weights = torch.full((len(models),), 1.0 / len(models))
else:
    weights = torch.as_tensor(weights, dtype=all_model_probs.dtype)
    weights = weights / weights.sum()

weighted_probs = torch.zeros_like(all_model_probs[0])
for i in range(len(models)):
    weighted_probs += weights[i] * all_model_probs[i]

weighted_preds = torch.argmax(weighted_probs, dim=1)

weighted_acc = (weighted_preds == labels).float().mean().item()

print("\n=== WEIGHTED SOFT VOTING RESULTS ===")
print("Weighted Ensemble Accuracy:", weighted_acc)
print(confusion_matrix(labels, weighted_preds))
print(classification_report(labels, weighted_preds))