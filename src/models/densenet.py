import torch.nn as nn
import torchvision.models as models


def get_densenet(num_classes=3, freeze=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model