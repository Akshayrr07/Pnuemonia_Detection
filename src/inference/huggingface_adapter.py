import io
import math
from collections.abc import Mapping as MappingABC
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import requests
from PIL import Image

from src.inference.schemas import ModelPrediction


BINARY_LABEL_MAP = {
    "LABEL_0": "Normal",
    "LABEL_1": "Pneumonia",
    "Normal": "Normal",
    "normal": "Normal",
    "NORMAL": "Normal",
    "Pneumonia": "Pneumonia",
    "pneumonia": "Pneumonia",
    "PNEUMONIA": "Pneumonia",
}

SUBTYPE_LABEL_MAP = {
    "LABEL_0": "Bacterial Pneumonia",
    "LABEL_1": "Viral Pneumonia",
    "Bacterial Pneumonia": "Bacterial Pneumonia",
    "bacterial": "Bacterial Pneumonia",
    "BACTERIAL": "Bacterial Pneumonia",
    "Viral Pneumonia": "Viral Pneumonia",
    "viral": "Viral Pneumonia",
    "VIRAL": "Viral Pneumonia",
}

# The hosted model maps provider-specific labels to these canonical classes.
# Keep the class sets separate from the provider aliases: the aliases may grow,
# but the inference contract must remain a complete, known class set.
BINARY_LABELS = frozenset(BINARY_LABEL_MAP.values())
SUBTYPE_LABELS = frozenset(SUBTYPE_LABEL_MAP.values())

# Hugging Face image classification accepts an encoded image and performs the
# model's processor on the provider side.  Sending a deterministic 224x224 RGB
# image makes this client's part of the preprocessing contract explicit while
# retaining the existing ``predict(PIL.Image)`` adapter contract.
HOSTED_IMAGE_SIZE = (224, 224)


class HuggingFaceInferenceError(RuntimeError):
    pass


def _resize_method() -> int:
    """Return Pillow's bilinear resampling constant across Pillow versions."""
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def prepare_hosted_image(image: Image.Image) -> Image.Image:
    """Prepare a PIL image for the hosted image-classification request.

    The provider still applies the model's final tensor conversion and
    normalization.  This client-side step guarantees an RGB, 224x224 encoded
    input rather than silently forwarding arbitrary image bytes.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Hosted inference expects a PIL.Image.Image")
    return image.convert("RGB").resize(HOSTED_IMAGE_SIZE, _resize_method())


def preprocess_image(image: Image.Image):
    """Return the explicit ImageNet-normalized tensor used by local models.

    Hosted inference must send an image file, so its request path uses
    :func:`prepare_hosted_image`; this helper keeps the tensor preprocessing
    contract directly testable without introducing a heavy import when the
    hosted-only adapter is loaded.
    """
    from src.data.transforms import get_inference_transform

    return get_inference_transform()(image)


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    prepare_hosted_image(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _coerce_hf_output(payload: object) -> List[Mapping[str, object]]:
    """Validate the provider's image-classification response envelope."""
    if isinstance(payload, MappingABC):
        if "error" in payload:
            raise HuggingFaceInferenceError(str(payload["error"]))
        if "label" in payload and "score" in payload:
            return [payload]
        raise HuggingFaceInferenceError(
            f"Unexpected Hugging Face response shape: {payload!r}"
        )

    if isinstance(payload, list):
        # A few HTTP clients wrap the documented array in one additional list.
        # Accept only that unambiguous, single-row wrapper; reject empty and
        # heterogeneous lists rather than treating them as valid output.
        if len(payload) == 1 and isinstance(payload[0], list):
            payload = payload[0]

        if payload and all(isinstance(item, MappingABC) for item in payload):
            return list(payload)

    raise HuggingFaceInferenceError(
        f"Unexpected Hugging Face response shape: {payload!r}"
    )


def _normalize_label(label: str, label_map: Mapping[str, str]) -> str:
    return label_map.get(label, label)


def _expected_labels(
    label_map: Mapping[str, str],
    explicit_labels: Optional[Sequence[str]] = None,
) -> Optional[Set[str]]:
    if explicit_labels is not None:
        if isinstance(explicit_labels, str):
            raise ValueError("expected_labels must be a non-empty sequence of strings")
        labels = list(explicit_labels)
        if not labels or not all(isinstance(label, str) for label in labels):
            raise ValueError("expected_labels must be a non-empty sequence of strings")
        return set(labels)

    if not label_map:
        # Preserve the historical generic-adapter behavior when no class map
        # was configured.  Production binary/subtype adapters always pass a map.
        return None
    return set(label_map.values())


def _prediction_from_scores(
    scores: Iterable[Mapping[str, object]],
    label_map: Mapping[str, str],
    expected_labels: Optional[Sequence[str]] = None,
) -> ModelPrediction:
    """Convert validated HF scores into a canonical ModelPrediction.

    Every item must have a known label and a finite score in [0, 1].  When a
    class map (or explicit expected labels) is configured, the response must
    contain each canonical class exactly once; missing, unknown, duplicate, or
    malformed entries are provider errors and are never defaulted.
    """
    label_map = label_map or {}
    expected = _expected_labels(label_map, expected_labels)
    probabilities: Dict[str, float] = {}
    materialized_scores = list(scores)

    for index, item in enumerate(materialized_scores):
        if not isinstance(item, MappingABC):
            raise HuggingFaceInferenceError(
                f"Hugging Face score entry {index} is not an object"
            )
        if "label" not in item:
            raise HuggingFaceInferenceError(
                f"Hugging Face score entry {index} is missing label"
            )
        if "score" not in item:
            raise HuggingFaceInferenceError(
                f"Hugging Face score entry {index} is missing score"
            )

        raw_label = item["label"]
        raw_score = item["score"]
        if not isinstance(raw_label, str):
            raise HuggingFaceInferenceError(
                f"Hugging Face score entry {index} has a non-string label"
            )

        label = _normalize_label(raw_label, label_map)
        if expected is not None and label not in expected:
            raise HuggingFaceInferenceError(
                f"unknown label {raw_label!r}; expected one of {sorted(expected)!r}"
            )
        if label in probabilities:
            raise HuggingFaceInferenceError(
                f"duplicate label {raw_label!r} in Hugging Face response"
            )

        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise HuggingFaceInferenceError(
                f"score for label {raw_label!r} must be a finite number in [0, 1]"
            )
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HuggingFaceInferenceError(
                f"score for label {raw_label!r} must be a finite number in [0, 1]"
            ) from exc

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise HuggingFaceInferenceError(
                f"score for label {raw_label!r} must be finite and in [0, 1]"
            )
        probabilities[label] = score

    if not probabilities:
        raise HuggingFaceInferenceError("Hugging Face response did not include class scores")

    if expected is not None:
        missing = expected - set(probabilities)
        if missing:
            raise HuggingFaceInferenceError(
                f"missing expected class(es): {sorted(missing)!r}"
            )

    top_label = max(probabilities, key=probabilities.get)
    return ModelPrediction(
        label=top_label,
        confidence=probabilities[top_label],
        probabilities=probabilities,
    )


class HuggingFaceImageClassifier:
    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        api_base_url: str = "https://api-inference.huggingface.co/models",
        timeout_seconds: int = 60,
        label_map: Optional[Mapping[str, str]] = None,
        expected_labels: Optional[Sequence[str]] = None,
    ):
        self.model_id = model_id
        self.token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.label_map = label_map or {}
        # Validate eagerly so a misconfigured classifier cannot silently become
        # permissive at request time.
        self.expected_labels = (
            tuple(expected_labels) if expected_labels is not None else None
        )
        _expected_labels(self.label_map, self.expected_labels)

    @property
    def url(self) -> str:
        return f"{self.api_base_url}/{self.model_id}"

    def predict(self, image: Image.Image) -> ModelPrediction:
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.post(
            self.url,
            headers=headers,
            data=_image_to_jpeg_bytes(image),
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            raise HuggingFaceInferenceError(
                f"Hugging Face request failed with status {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise HuggingFaceInferenceError(
                "Hugging Face response was not valid JSON"
            ) from exc
        scores = _coerce_hf_output(payload)
        return _prediction_from_scores(
            scores,
            self.label_map,
            expected_labels=self.expected_labels,
        )

