import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_inference_transform,
)


def test_inference_transform_resizes_to_224_and_normalizes_imagenet_values():
    image = Image.new("RGB", (320, 180), color=(255, 0, 127))
    transform = get_inference_transform()

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 224, 224)
    expected = np.array(
        [
            (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (127 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(tensor[:, 0, 0].numpy(), expected)
    assert np.allclose(tensor[:, -1, -1].numpy(), expected)


def test_inference_transform_converts_non_rgb_images_to_rgb():
    image = Image.new("L", (64, 64), color=128)
    tensor = get_inference_transform()(image)

    assert tuple(tensor.shape) == (3, 224, 224)
    for channel in range(3):
        expected = (128 / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[channel].mean().item() == pytest.approx(expected, abs=1e-5)
