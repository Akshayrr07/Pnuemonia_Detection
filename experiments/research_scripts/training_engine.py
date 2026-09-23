# ============================================================
# PHASE 4 — TRAINING ENGINE (CPU SAFE)
# ============================================================

import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from legacy_imports_compat import PneumoniaDataset, CustomCNN
import pandas as pd


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD DATA AGAIN (FROM SPLITS)
# ============================================================

TRAIN_CSV = "../splits/train.csv"
VAL_CSV   = "../splits/val.csv"
TEST_CSV  = "../splits/test.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df = train_df.rename(columns={
    "original_path": "image_path",
    "encoded_label": "label"
})

val_df = val_df.rename(columns={
    "original_path": "image_path",
    "encoded_label": "label"
})

test_df = test_df.rename(columns={
    "original_path": "image_path",
    "encoded_label": "label"
})

# =========================
# TRANSFORMS (SAME AS PHASE 2)
# =========================

from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================
# DATASETS + LOADERS
# =========================

train_dataset = PneumoniaDataset(train_df, transform=train_transform)
val_dataset   = PneumoniaDataset(val_df, transform=val_test_transform)
test_dataset  = PneumoniaDataset(test_df, transform=val_test_transform)

BATCH_SIZE = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# CLASS WEIGHTS
# =========================

class_counts = train_df["label"].value_counts().sort_index()
num_samples = len(train_df)
num_classes = len(class_counts)

class_weights = num_samples / (num_classes * class_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ============================================================
# OPTIMIZER
# ============================================================

def get_optimizer(model, lr=1e-3):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ============================================================
# VALIDATION LOOP
# ============================================================

def validate(model, loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ============================================================
# METRICS
# ============================================================

def evaluate_metrics(labels, preds):
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(labels, preds))


# ============================================================
# TRAIN MODEL (EARLY STOPPING)
# ============================================================

def train_model(model, epochs=5, lr=1e-3):

    optimizer = get_optimizer(model, lr)
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    trigger = 0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    model = CustomCNN(num_classes=3).to(device)

    trained_model = train_model(model, epochs=5, lr=1e-3)

    test_loss, test_acc, test_preds, test_labels = validate(trained_model, test_loader)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    evaluate_metrics(test_labels, test_preds)


if __name__ == "__main__":
    main()
