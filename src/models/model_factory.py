from .mobilenet import get_mobilenet
from .efficientnet import get_efficientnet
from .resnet import get_resnet18
from .densenet import get_densenet


def get_model(model_name, num_classes=3, freeze=False):



    if model_name == "mobilenet":
        return get_mobilenet(num_classes, freeze)

    elif model_name == "efficientnet":
        return get_efficientnet(num_classes, freeze)

    elif model_name == "resnet":
        return get_resnet18(num_classes, freeze)

    elif model_name == "densenet":
        return get_densenet(num_classes, freeze)

    else:
        raise ValueError(f"Unknown model: {model_name}")