import torch.nn as nn
import torchvision.models as models


def get_mobilenet(num_classes=3, freeze=False):
    model = models.mobilenet_v2(weights=None)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model