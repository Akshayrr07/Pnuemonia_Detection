"""
Grad-CAM explainability for local torchvision models.

Provides heatmap generation over a local PyTorch model loaded from disk.
Intended for use alongside the Hugging Face hosted model path, not as a
replacement for it.

The production path (HuggingFaceImageClassifier) is a black-box HTTP call
and cannot produce Grad-CAM heatmaps. Explainability is therefore gated
behind a local-model path and is disabled when no local checkpoint is
configured.
"""

from __future__ import annotations

import base64
import io
import threading
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch
    from torch import nn

# ---------------------------------------------------------------------------
# Thread safety — heatmap generation mutates hook state on the target layer
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Normalization for ImageNet-pretrained models
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Target layer resolution for known model families
# ---------------------------------------------------------------------------


def resolve_target_layer(model: "nn.Module", model_name: str) -> "nn.Module":
    """Return the last conv layer for Grad-CAM given a model and its name."""
    if model_name == "resnet":
        return model.layer4[-1]
    if model_name == "mobilenet":
        return model.features[-1]
    if model_name == "efficientnet":
        return model.features[-1]
    if model_name == "densenet":
        return model.features[-1]
    raise ValueError(f"Unsupported model_name for Grad-CAM: {model_name}")


# ---------------------------------------------------------------------------
# Preprocessing / inversion helpers
# ---------------------------------------------------------------------------


def _preprocess(image: Image.Image) -> Tuple["torch.Tensor", Tuple[int, int]]:
    """Convert PIL image to normalized 224x224 tensor; return tensor + original (w,h)."""
    import torchvision.transforms as T

    original_size: Tuple[int, int] = image.size  # (width, height)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor, original_size


def _colored_heatmap(heatmap_2d: np.ndarray) -> np.ndarray:
    """Apply a jet colormap to a 2D float heatmap in [0,1]; return (H,W,3) uint8."""
    x = np.clip(heatmap_2d, 0.0, 1.0)

    def _jet_channel(x):
        r = np.where(x < 0.5, 0.0, np.where(x < 0.75, 4.0 * x - 2.0, 1.0))
        g = np.where(x < 0.25, 4.0 * x, np.where(x < 0.75, 2.0 - 4.0 * x, 0.0))
        b = np.where(x < 0.5, 2.0 - 4.0 * x, np.where(x < 0.75, 0.0, 0.0))
        return r, g, b

    r, g, b = _jet_channel(x)
    return np.stack([r, g, b], axis=-1).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------


def generate_heatmap(
    model: "nn.Module",
    model_name: str,
    image: Image.Image,
    class_idx: int,
    device: "torch.device",
    overlay_max_side: int = 512,
) -> Optional[str]:
    """
    Run Grad-CAM for *class_idx* on *model* and return a base64 PNG data URL
    of the heatmap composited over the original X-ray.

    Returns None when Grad-CAM fails for any reason.
    """
    import torch

    tensor, original_size = _preprocess(image)
    tensor = tensor.to(device)

    target_layer = resolve_target_layer(model, model_name)

    activation: Optional["torch.Tensor"] = None
    gradient: Optional["torch.Tensor"] = None

    def forward_hook(module, inp, out):
        nonlocal activation
        activation = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        model.eval()

        scores = model(tensor)
        score = scores[0, class_idx]
        score.backward()

        if activation is None or gradient is None:
            return None

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.size == 0:
            return None

        lo, hi = cam.min(), cam.max()
        denom = hi - lo if hi > lo else 1.0
        cam = (cam - lo) / denom

        # Upsample to original image size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize(original_size, Image.Resampling.BICUBIC)

        # Composite over original
        overlay = _overlay_heatmap(cam_img, image.convert("RGB"))
        if overlay.width > overlay_max_side or overlay.height > overlay_max_side:
            scale = overlay_max_side / max(overlay.width, overlay.height)
            overlay = overlay.resize(
                (int(overlay.width * scale), int(overlay.height * scale)),
                Image.Resampling.LANCZOS,
            )

        return _image_to_b64(overlay)

    except Exception:
        return None

    finally:
        handle_f.remove()
        handle_b.remove()


def _overlay_heatmap(heatmap_pil: Image.Image, original_pil: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Colorize the heatmap with a jet-like colormap and blend with the original."""
    heat = np.array(heatmap_pil).astype(np.float32) / 255.0
    colored = _colored_heatmap(heat)  # (H,W,3) uint8

    # Alpha = heatmap intensity * alpha so we only see color where the CAM is active
    intensity = np.array(heatmap_pil).astype(np.uint8)
    rgba = np.zeros((*intensity.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = colored
    rgba[:, :, 3] = (intensity * alpha).astype(np.uint8)

    heat_rgba = Image.fromarray(rgba, "RGBA")
    base = original_pil.convert("RGBA")
    return Image.alpha_composite(base, heat_rgba).convert("RGB")


def _image_to_b64(image: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
