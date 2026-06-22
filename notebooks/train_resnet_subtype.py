# ============================================================
# SUBTYPE BINARY MODEL — BACTERIAL vs VIRAL
# ResNet18 | Pneumonia Only | CPU Safe
# ============================================================

import torch
import torch.nn as nn
import copy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD SPLITS
# ============================================================

TRAIN_CSV = "../splits/train.csv"
VAL_CSV   = "../splits/val.csv"
TEST_CSV  = "../splits/test.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Rename columns
for df in [train_df, val_df, test_df]:
    df.rename(columns={
        "original_path": "image_path",
        "encoded_label": "label"
    }, inplace=True)


# ============================================================
# FILTER ONLY PNEUMONIA (label != 0)
# ============================================================

train_df = train_df[train_df["label"] != 0].copy()
val_df   = val_df[val_df["label"] != 0].copy()
test_df  = test_df[test_df["label"] != 0].copy()

# Map labels:
# BACTERIAL (1) -> 0
# VIRAL (2) -> 1

train_df["subtype_label"] = train_df["label"].map({1: 0, 2: 1})
val_df["subtype_label"]   = val_df["label"].map({1: 0, 2: 1})
test_df["subtype_label"]  = test_df["label"].map({1: 0, 2: 1})


# ============================================================
# TRANSFORMS
# ============================================================

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


# ============================================================
# SUBTYPE DATASET
# ============================================================

class SubtypeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["subtype_label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    SubtypeDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SubtypeDataset(val_df, val_test_transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    SubtypeDataset(test_df, val_test_transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS (Balanced Binary)
# ============================================================

class_counts = train_df["subtype_label"].value_counts().sort_index()
num_samples = len(train_df)
num_classes = len(class_counts)

class_weights = num_samples / (num_classes * class_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ============================================================
# MODEL — RESNET18 (2 CLASSES)
# ============================================================

def get_resnet_subtype():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model.to(device)


# ============================================================
# TRAINING FUNCTIONS
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


def train_model(model, epochs=6, lr=1e-3):

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_acc = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, _, _ = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


# ============================================================
# MAIN
# ============================================================

def main():

    model = get_resnet_subtype()

    trained_model = train_model(model, epochs=6, lr=1e-3)

    test_loss, test_acc, test_preds, test_labels = validate(trained_model, test_loader)

    print(f"\nSubtype Test Accuracy: {test_acc:.4f}")

    print("\nConfusion Matrix:\n", confusion_matrix(test_labels, test_preds))

    print("\nClassification Report:\n")
    print(classification_report(test_labels, test_preds))


if __name__ == "__main__":
    main()
