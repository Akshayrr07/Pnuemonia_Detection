"""Focused regression tests for the legacy research compatibility shim."""

from pathlib import Path
import sys

import pandas as pd
import torch
from PIL import Image


RESEARCH_SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "research_scripts"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

from legacy_imports_compat import CustomCNN, PneumoniaDataset


def test_custom_cnn_runs_forward_pass():
    model = CustomCNN(num_classes=2)

    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 2)


def test_pneumonia_dataset_accepts_legacy_dataframe(tmp_path):
    image_path = tmp_path / "xray.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    dataframe = pd.DataFrame(
        {
            "image_path": [str(image_path)],
            "label": [1],
        }
    )

    dataset = PneumoniaDataset(dataframe)
    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.mode == "RGB"
    assert label == 1
