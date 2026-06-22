import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=3, freeze=False):
    model = models.resnet18(weights=None)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model