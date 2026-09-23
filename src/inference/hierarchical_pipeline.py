from typing import Dict, Optional

from PIL import Image

from src.inference.huggingface_adapter import (
    BINARY_LABEL_MAP,
    SUBTYPE_LABEL_MAP,
    HuggingFaceImageClassifier,
)
from src.inference.schemas import DEFAULT_DISCLAIMER, HierarchicalPrediction, ModelPrediction
from src.inference.settings import InferenceSettings


class HierarchicalPneumoniaPipeline:
    def __init__(
        self,
        binary_classifier: HuggingFaceImageClassifier,
        subtype_classifier: HuggingFaceImageClassifier,
        pneumonia_threshold: float = 0.5,
        subtype_threshold: float = 0.5,
        disclaimer: str = DEFAULT_DISCLAIMER,
    ):
        self.binary_classifier = binary_classifier
        self.subtype_classifier = subtype_classifier
        self.pneumonia_threshold = pneumonia_threshold
        self.subtype_threshold = subtype_threshold
        self.disclaimer = disclaimer

    @classmethod
    def from_settings(cls, settings: InferenceSettings) -> "HierarchicalPneumoniaPipeline":
        binary_classifier = HuggingFaceImageClassifier(
            model_id=settings.binary_model_id,
            token=settings.hf_token,
            api_base_url=settings.hf_api_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            label_map=BINARY_LABEL_MAP,
        )
        subtype_classifier = HuggingFaceImageClassifier(
            model_id=settings.subtype_model_id,
            token=settings.hf_token,
            api_base_url=settings.hf_api_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            label_map=SUBTYPE_LABEL_MAP,
        )

        return cls(
            binary_classifier=binary_classifier,
            subtype_classifier=subtype_classifier,
            pneumonia_threshold=settings.pneumonia_threshold,
            subtype_threshold=settings.subtype_threshold,
        )

    def predict(self, image: Image.Image) -> HierarchicalPrediction:
        binary_output = self.binary_classifier.predict(image)
        pneumonia_score = binary_output.probability_for("Pneumonia")
        normal_score = binary_output.probability_for("Normal")

        is_pneumonia = pneumonia_score >= self.pneumonia_threshold
        if is_pneumonia:
            return self._predict_subtype(
                image=image,
                binary_output=binary_output,
                primary_confidence=pneumonia_score,
            )

        return HierarchicalPrediction(
            primary_prediction="Normal",
            primary_confidence=normal_score or binary_output.confidence,
            subtype_prediction=None,
            subtype_confidence=None,
            probabilities=self._merge_probabilities(binary_output, None),
            model_outputs={"binary": binary_output.to_dict()},
            disclaimer=self.disclaimer,
        )

    def _predict_subtype(
        self,
        image: Image.Image,
        binary_output: ModelPrediction,
        primary_confidence: float,
    ) -> HierarchicalPrediction:
        subtype_output = self.subtype_classifier.predict(image)

        bacterial_score = subtype_output.probability_for("Bacterial Pneumonia")
        viral_score = subtype_output.probability_for("Viral Pneumonia")

        max_subtype_score = max(bacterial_score, viral_score)

        if max_subtype_score < self.subtype_threshold:
            subtype_prediction = "Uncertain Pneumonia Subtype"
            subtype_confidence = max_subtype_score or subtype_output.confidence
        elif bacterial_score >= viral_score:
            subtype_prediction = "Bacterial Pneumonia"
            subtype_confidence = bacterial_score
        else:
            subtype_prediction = "Viral Pneumonia"
            subtype_confidence = viral_score

        return HierarchicalPrediction(
            primary_prediction="Pneumonia",
            primary_confidence=primary_confidence or binary_output.confidence,
            subtype_prediction=subtype_prediction,
            subtype_confidence=subtype_confidence,
            probabilities=self._merge_probabilities(binary_output, subtype_output),
            model_outputs={
                "binary": binary_output.to_dict(),
                "subtype": subtype_output.to_dict(),
            },
            disclaimer=self.disclaimer,
        )

    @staticmethod
    def _merge_probabilities(
        binary_output: ModelPrediction,
        subtype_output: Optional[ModelPrediction],
    ) -> Dict[str, float]:
        probabilities = dict(binary_output.probabilities)
        if subtype_output is not None:
            probabilities.update(subtype_output.probabilities)
        return probabilities
