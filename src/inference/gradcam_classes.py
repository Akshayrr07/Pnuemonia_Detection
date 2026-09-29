"""Class-index contract for local Grad-CAM checkpoints.

This module is intentionally dependency-free (no NumPy, PyTorch, or torchvision)
so callers that only need the class-index mapping — such as the prediction route
deciding *which* class to explain — do not have to import the full explainability
stack.  The hosted-inference production image intentionally ships without
NumPy/PyTorch, and importing them there would fail for a pure lookup.
"""

from __future__ import annotations


# Local checkpoints produced by this repository use the dataset registry's
# three-class encoding. There is no aggregate "Pneumonia" output class.
GRAD_CAM_CLASS_TO_INDEX = {
    "Normal": 0,
    "Bacterial Pneumonia": 1,
    "Viral Pneumonia": 2,
}


def grad_cam_class_index(class_name: str) -> int:
    """Return the local three-class checkpoint index for *class_name*.

    Aggregate binary output ``"Pneumonia"`` is intentionally unsupported: the
    checkpoint has separate bacterial and viral classes, not a Pneumonia class.
    """
    try:
        return GRAD_CAM_CLASS_TO_INDEX[class_name]
    except KeyError as exc:
        raise ValueError(
            "Grad-CAM requires a concrete local checkpoint class "
            f"(one of {tuple(GRAD_CAM_CLASS_TO_INDEX)}), not aggregate Pneumonia: "
            f"{class_name!r}"
        ) from exc


__all__ = ["GRAD_CAM_CLASS_TO_INDEX", "grad_cam_class_index"]
