import io
from typing import Dict, Iterable, List, Mapping, Optional

import requests
from PIL import Image

from src.inference.schemas import ModelPrediction


BINARY_LABEL_MAP = {
    "LABEL_0": "Normal",
    "LABEL_1": "Pneumonia",
    "normal": "Normal",
    "NORMAL": "Normal",
    "pneumonia": "Pneumonia",
    "PNEUMONIA": "Pneumonia",
}

SUBTYPE_LABEL_MAP = {
    "LABEL_0": "Bacterial Pneumonia",
    "LABEL_1": "Viral Pneumonia",
    "bacterial": "Bacterial Pneumonia",
    "BACTERIAL": "Bacterial Pneumonia",
    "viral": "Viral Pneumonia",
    "VIRAL": "Viral Pneumonia",
}


class HuggingFaceInferenceError(RuntimeError):
    pass


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _coerce_hf_output(payload: object) -> List[Mapping[str, object]]:
    if isinstance(payload, dict):
        if "error" in payload:
            raise HuggingFaceInferenceError(str(payload["error"]))
        if "label" in payload and "score" in payload:
            return [payload]

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            payload = payload[0]

        if all(isinstance(item, dict) for item in payload):
            return payload

    raise HuggingFaceInferenceError(f"Unexpected Hugging Face response shape: {payload!r}")


def _normalize_label(label: str, label_map: Mapping[str, str]) -> str:
    return label_map.get(label, label)


def _prediction_from_scores(
    scores: Iterable[Mapping[str, object]],
    label_map: Mapping[str, str],
) -> ModelPrediction:
    probabilities: Dict[str, float] = {}

    for item in scores:
        raw_label = item.get("label")
        raw_score = item.get("score")
        if raw_label is None or raw_score is None:
            continue

        label = _normalize_label(str(raw_label), label_map)
        score = float(str(raw_score))
        probabilities[label] = max(probabilities.get(label, 0.0), score)

    if not probabilities:
        raise HuggingFaceInferenceError("Hugging Face response did not include class scores")

    top_label = max(probabilities, key=lambda label: probabilities[label])
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
    ):
        self.model_id = model_id
        self.token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.label_map = label_map or {}

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

        payload = response.json()
        scores = _coerce_hf_output(payload)
        return _prediction_from_scores(scores, self.label_map)

