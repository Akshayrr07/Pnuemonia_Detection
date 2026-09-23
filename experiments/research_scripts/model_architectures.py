# ============================================================
# PHASE 3 — MODEL ARCHITECTURE IMPLEMENTATION (CPU SAFE)
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# DEVICE CONFIGURATION
# =========================

device = torch.device("cpu")
print("Using device:", device)


# ============================================================
# 1️⃣ CUSTOM LIGHTWEIGHT CNN
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# 2️⃣ TRANSFER LEARNING MODELS
# ============================================================

def get_mobilenet(num_classes=3, freeze=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model.to(device)


def get_efficientnet(num_classes=3, freeze=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model.to(device)


def get_resnet18(num_classes=3, freeze=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)


# ============================================================
# 3️⃣ PARAMETER COUNT UTILITY
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 4️⃣ MAIN EXECUTION
# ============================================================

def main():

    # Instantiate models
    custom_model = CustomCNN(num_classes=3).to(device)
    mobilenet_model = get_mobilenet()
    efficientnet_model = get_efficientnet()
    resnet_model = get_resnet18()

    print("\nModel Parameter Summary:\n")

    for name, model in {
        "CustomCNN": custom_model,
        "MobileNetV2": mobilenet_model,
        "EfficientNetB0": efficientnet_model,
        "ResNet18": resnet_model
    }.items():
        total, trainable = count_parameters(model)
        print(f"{name} -> Total: {total:,} | Trainable: {trainable:,}")

    # Forward pass sanity check
    print("\nRunning forward pass sanity check...")

    dummy_input = torch.randn(16, 3, 224, 224).to(device)

    output = custom_model(dummy_input)
    print("CustomCNN Output Shape:", output.shape)

    output = mobilenet_model(dummy_input)
    print("MobileNetV2 Output Shape:", output.shape)

    output = efficientnet_model(dummy_input)
    print("EfficientNetB0 Output Shape:", output.shape)

    output = resnet_model(dummy_input)
    print("ResNet18 Output Shape:", output.shape)

    print("\nPHASE 3 COMPLETE — All models valid.")


if __name__ == "__main__":
    main()
