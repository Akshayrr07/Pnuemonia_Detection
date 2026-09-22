from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_DISCLAIMER = (
    "This result is for educational support only and is not a medical diagnosis. "
    "A qualified medical professional should review any abnormal finding."
)


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

    def probability_for(self, label: str, default: float = 0.0) -> float:
        return self.probabilities.get(label, default)

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


@dataclass(frozen=True)
class HierarchicalPrediction:
    primary_prediction: str
    primary_confidence: float
    subtype_prediction: Optional[str]
    subtype_confidence: Optional[float]
    probabilities: Dict[str, float]
    model_outputs: Dict[str, Dict[str, object]]
    disclaimer: str = DEFAULT_DISCLAIMER

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary_prediction": self.primary_prediction,
            "primary_confidence": self.primary_confidence,
            "subtype_prediction": self.subtype_prediction,
            "subtype_confidence": self.subtype_confidence,
            "probabilities": self.probabilities,
            "model_outputs": self.model_outputs,
            "disclaimer": self.disclaimer,
        }

