import math

import pytest

from src.inference.huggingface_adapter import (
    BINARY_LABEL_MAP,
    SUBTYPE_LABEL_MAP,
    HuggingFaceInferenceError,
    _prediction_from_scores,
)


def test_valid_binary_scores_are_normalized_to_exact_expected_labels():
    prediction = _prediction_from_scores(
        [
            {"label": "normal", "score": 0.25},
            {"label": "pneumonia", "score": 0.75},
        ],
        BINARY_LABEL_MAP,
    )

    assert prediction.label == "Pneumonia"
    assert prediction.confidence == 0.75
    assert prediction.probabilities == {"Normal": 0.25, "Pneumonia": 0.75}


def test_unknown_label_fails_closed():
    with pytest.raises(HuggingFaceInferenceError, match="unknown label"):
        _prediction_from_scores(
            [
                {"label": "normal", "score": 0.2},
                {"label": "cancer", "score": 0.8},
            ],
            BINARY_LABEL_MAP,
        )


def test_nan_score_fails_closed():
    with pytest.raises(HuggingFaceInferenceError, match="finite"):
        _prediction_from_scores(
            [
                {"label": "normal", "score": float("nan")},
                {"label": "pneumonia", "score": 0.75},
            ],
            BINARY_LABEL_MAP,
        )


def test_infinite_score_fails_closed():
    with pytest.raises(HuggingFaceInferenceError, match="finite"):
        _prediction_from_scores(
            [
                {"label": "normal", "score": 0.25},
                {"label": "pneumonia", "score": math.inf},
            ],
            BINARY_LABEL_MAP,
        )


def test_out_of_range_score_fails_closed():
    with pytest.raises(HuggingFaceInferenceError, match=r"\[0, 1\]"):
        _prediction_from_scores(
            [
                {"label": "normal", "score": 0.25},
                {"label": "pneumonia", "score": 1.01},
            ],
            BINARY_LABEL_MAP,
        )


def test_missing_class_fails_closed():
    with pytest.raises(HuggingFaceInferenceError, match="missing expected class"):
        _prediction_from_scores(
            [{"label": "pneumonia", "score": 1.0}],
            BINARY_LABEL_MAP,
        )


def test_missing_score_field_is_not_skipped():
    with pytest.raises(HuggingFaceInferenceError, match="score"):
        _prediction_from_scores(
            [
                {"label": "normal"},
                {"label": "pneumonia", "score": 0.75},
            ],
            BINARY_LABEL_MAP,
        )


def test_provider_error_payload_is_reported_as_inference_error():
    from src.inference.huggingface_adapter import _coerce_hf_output

    with pytest.raises(HuggingFaceInferenceError, match="model is loading"):
        _coerce_hf_output({"error": "model is loading"})


def test_empty_provider_output_is_rejected():
    from src.inference.huggingface_adapter import _coerce_hf_output

    with pytest.raises(HuggingFaceInferenceError, match="Unexpected Hugging Face response shape"):
        _coerce_hf_output([])


def test_subtype_expected_labels_are_complete():
    with pytest.raises(HuggingFaceInferenceError, match="missing expected class"):
        _prediction_from_scores(
            [{"label": "bacterial", "score": 1.0}],
            SUBTYPE_LABEL_MAP,
        )
