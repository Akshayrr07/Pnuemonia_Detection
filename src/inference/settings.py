import os
from dataclasses import dataclass
from typing import Optional


class InferenceSettingsError(ValueError):
    pass


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise InferenceSettingsError(f"{name} must be a float") from exc


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise InferenceSettingsError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class InferenceSettings:
    binary_model_id: str
    subtype_model_id: str
    hf_token: Optional[str] = None
    hf_api_base_url: str = "https://api-inference.huggingface.co/models"
    pneumonia_threshold: float = 0.5
    subtype_threshold: float = 0.5
    request_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "InferenceSettings":
        binary_model_id = os.getenv("HF_BINARY_MODEL_ID")
        subtype_model_id = os.getenv("HF_SUBTYPE_MODEL_ID")

        missing = []
        if not binary_model_id:
            missing.append("HF_BINARY_MODEL_ID")
        if not subtype_model_id:
            missing.append("HF_SUBTYPE_MODEL_ID")
        if missing:
            joined = ", ".join(missing)
            raise InferenceSettingsError(f"Missing required environment variable(s): {joined}")

        assert binary_model_id is not None
        assert subtype_model_id is not None

        return cls(
            binary_model_id=binary_model_id,
            subtype_model_id=subtype_model_id,
            hf_token=os.getenv("HF_TOKEN"),
            hf_api_base_url=os.getenv(
                "HF_API_BASE_URL",
                "https://api-inference.huggingface.co/models",
            ),
            pneumonia_threshold=_get_float("PNEUMONIA_THRESHOLD", 0.5),
            subtype_threshold=_get_float("SUBTYPE_THRESHOLD", 0.5),
            request_timeout_seconds=_get_int("HF_REQUEST_TIMEOUT_SECONDS", 60),
        )

