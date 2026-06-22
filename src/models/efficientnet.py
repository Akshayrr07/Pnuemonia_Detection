import torch.nn as nn
import torchvision.models as models


def get_efficientnet(num_classes=3, freeze=False):
    model = models.efficientnet_b0(weights=None)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model