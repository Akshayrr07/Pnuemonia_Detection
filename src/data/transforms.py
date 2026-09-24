from typing import Tuple

from PIL import Image
from torchvision import transforms


IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform():
    """Return the deterministic preprocessing used for model inference.

    Inputs are converted to RGB, resized to the model's 224x224 spatial
    shape, converted to a float tensor in [0, 1], and normalized with the
    ImageNet channel statistics used during training.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image):
    """Apply the explicit 224x224 ImageNet-normalized inference transform."""
    return get_inference_transform()(image)


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
