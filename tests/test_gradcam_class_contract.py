"""The Grad-CAM class-index contract must stay free of NumPy/PyTorch.

The hosted-inference production image and the CI backend environment install
neither NumPy nor PyTorch.  The prediction route looks up a class index before
deciding whether to explain a prediction, so importing a NumPy/PyTorch-bound
module for that lookup silently disabled the whole Grad-CAM path there and the
failure was only visible as ``heatmap_b64: null``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


MODULE_NAME = "src.inference.gradcam_classes"
HEAVY_DEPENDENCIES = {"numpy", "torch", "torchvision"}


class _BlockHeavyImports:
    """Import hook that rejects the optional heavy inference dependencies."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in HEAVY_DEPENDENCIES:
            raise ImportError(f"blocked heavy dependency: {fullname}")
        return None


@pytest.fixture
def gradcam_classes_without_torch(monkeypatch):
    """Import the class-index module with NumPy/PyTorch made unimportable."""

    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(f"{MODULE_NAME}."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockHeavyImports()] + list(sys.meta_path))
    monkeypatch.syspath_prepend(".")
    return importlib.import_module(MODULE_NAME)


def test_class_index_module_imports_without_numpy_or_torch(
    gradcam_classes_without_torch,
):
    assert gradcam_classes_without_torch.GRAD_CAM_CLASS_TO_INDEX == {
        "Normal": 0,
        "Bacterial Pneumonia": 1,
        "Viral Pneumonia": 2,
    }


def test_concrete_classes_resolve_to_stable_indices(gradcam_classes_without_torch):
    resolve = gradcam_classes_without_torch.grad_cam_class_index

    assert resolve("Normal") == 0
    assert resolve("Bacterial Pneumonia") == 1
    assert resolve("Viral Pneumonia") == 2


def test_aggregate_pneumonia_is_rejected(gradcam_classes_without_torch):
    with pytest.raises(ValueError, match="aggregate Pneumonia"):
        gradcam_classes_without_torch.grad_cam_class_index("Pneumonia")


def test_route_does_not_import_the_torch_bound_explainability_module():
    """The /predict heatmap branch must not import the NumPy/PyTorch module.

    Importing it for a class-index lookup made the hosted-inference image lose
    Grad-CAM entirely, because the route catches the resulting ImportError and
    returns ``heatmap_b64: null`` instead of failing loudly.
    """

    from pathlib import Path

    app_source = Path(__file__).resolve().parents[1] / "backend" / "app.py"
    source = app_source.read_text(encoding="utf-8")

    assert "from src.inference.gradcam_classes import grad_cam_class_index" in source
    assert "from src.inference.explainability import grad_cam_class_index" not in source
