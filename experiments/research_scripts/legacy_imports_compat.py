# Compatibility shim for legacy research scripts in experiments/research_scripts/.

# The research scripts were written against an earlier project layout where the
# dataset and model code lived in top-level modules called data_pipeline and
# model_architectures. Those modules no longer exist as top-level files.
#
# This shim lets the legacy scripts keep their original import lines without
# modification, by redirecting them to the current production modules under src/.
#
# IMPORTANT: this is purely a convenience shim for running legacy research code.
# It is NOT part of the production inference or training path. New code should
# import directly from src.data and src.models.

import importlib
import sys
import torch.nn as nn
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _ensure_src_package():
    """Make sure src/ is importable as a package."""
    src_pkg = _PROJECT_ROOT / "src"
    if not (src_pkg / "__init__.py").exists():
        (src_pkg / "__init__.py").write_text("# src package\n")
    importlib.invalidate_caches()


def _import_from_src(module_path: str):
    """Import a module from src/<module_path> and return it.

    module_path uses dotted notation, e.g. "data.dataset" or "models.model_factory".
    """
    _ensure_src_package()
    full_name = f"src.{module_path}"
    return importlib.import_module(full_name)


# data_pipeline -> src.data.dataset (PneumoniaDataset is the main symbol the
# legacy scripts use).
_data = _import_from_src("data.dataset")

# model_architectures -> src.models.model_factory (get_resnet18, get_mobilenet,
# get_efficientnet, get_densenet). CustomCNN was only in the legacy module, so we
# redefine it here for the shim.
_models = _import_from_src("models.model_factory")


# Redirect the exact names the legacy scripts expect.
PneumoniaDataset = _data.PneumoniaDataset

CustomCNN = getattr(_models, "CustomCNN", None)
if CustomCNN is None:
    # CustomCNN was not carried forward into the modern codebase. Provide a
    # minimal equivalent so legacy scripts that reference it keep working.
    class CustomCNN(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
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
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

get_resnet18 = _models.get_resnet18
get_mobilenet = _models.get_mobilenet
get_efficientnet = _models.get_efficientnet
get_densenet = getattr(_models, "get_densenet", None)
count_parameters = getattr(_models, "count_parameters", None)
