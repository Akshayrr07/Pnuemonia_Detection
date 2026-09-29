import pytest
import torch
from torch import nn

from src.inference.checkpoint import (
    BINARY_CLASS_INDEX_LABELS,
    SUBTYPE_CLASS_INDEX_LABELS,
    THREE_CLASS_CLASS_INDEX_LABELS,
    CheckpointMetadata,
    CheckpointMetadataError,
    CheckpointLoadError,
    load_trusted_checkpoint,
    validate_checkpoint_metadata,
)


def _harmless_state_dict() -> dict[str, torch.Tensor]:
    """A tiny, non-model state dict suitable for safe-load tests."""
    return {
        "classifier.weight": torch.zeros(3, 2),
        "classifier.bias": torch.zeros(3),
    }


def test_load_trusted_checkpoint_accepts_plain_state_dict(tmp_path):
    checkpoint = tmp_path / "plain.pt"
    torch.save(_harmless_state_dict(), checkpoint)

    loaded = load_trusted_checkpoint(checkpoint, map_location="cpu")

    assert set(loaded) == {"classifier.weight", "classifier.bias"}
    assert isinstance(loaded["classifier.weight"], torch.Tensor)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("binary", BINARY_CLASS_INDEX_LABELS),
        ("subtype", SUBTYPE_CLASS_INDEX_LABELS),
        ("3_class", THREE_CLASS_CLASS_INDEX_LABELS),
    ],
)
def test_explicit_class_metadata_is_accepted(task, expected):
    metadata = {
        "task": task,
        "num_classes": len(expected),
        "class_index_to_label": expected,
    }

    validated = validate_checkpoint_metadata(metadata)

    assert validated.class_index_to_label == expected


def test_load_trusted_checkpoint_rejects_unsafe_payload_in_wrapper(tmp_path):
    checkpoint = tmp_path / "unsafe-wrapper.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.zeros(1)},
            "danger": nn.Linear(1, 1),
        },
        checkpoint,
    )

    with pytest.raises(CheckpointLoadError, match="safe|state.?dict|weights_only"):
        load_trusted_checkpoint(checkpoint, map_location="cpu")


def test_load_trusted_checkpoint_rejects_unknown_wrapper_values(tmp_path):
    checkpoint = tmp_path / "optimizer-wrapper.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.zeros(1)},
            "optimizer_state_dict": {"step": 1},
        },
        checkpoint,
    )

    with pytest.raises(CheckpointLoadError, match="safe|state.?dict|tensor|object"):
        load_trusted_checkpoint(checkpoint, map_location="cpu")


def test_load_trusted_checkpoint_rejects_full_model_object(tmp_path):
    checkpoint = tmp_path / "full-model.pt"
    torch.save(nn.Linear(2, 3), checkpoint)

    with pytest.raises(CheckpointLoadError, match="safe|state.?dict|model object"):
        load_trusted_checkpoint(checkpoint, map_location="cpu")


def test_load_trusted_checkpoint_rejects_metadata_mismatch_against_expected_contract(
    tmp_path,
):
    checkpoint = tmp_path / "wrong-metadata.pt"
    torch.save(
        {
            "state_dict": _harmless_state_dict(),
            "metadata": {
                "task": "3_class",
                "num_classes": 3,
                "class_index_to_label": {
                    0: "Normal",
                    1: "Pneumonia",
                },
            },
        },
        checkpoint,
    )

    with pytest.raises(CheckpointMetadataError, match="class|metadata|label"):
        load_trusted_checkpoint(
            checkpoint,
            map_location="cpu",
            expected_task="binary",
        )


def test_load_trusted_checkpoint_rejects_missing_metadata_when_task_is_required(
    tmp_path,
):
    checkpoint = tmp_path / "bare.pt"
    torch.save(_harmless_state_dict(), checkpoint)

    with pytest.raises(CheckpointMetadataError, match="metadata|class|label"):
        load_trusted_checkpoint(
            checkpoint,
            map_location="cpu",
            expected_task="3_class",
        )


def test_validate_checkpoint_metadata_rejects_missing_label_index():
    with pytest.raises(CheckpointMetadataError, match="class|label|index"):
        validate_checkpoint_metadata(
            {
                "task": "binary",
                "num_classes": 2,
                "class_index_to_label": {0: "Normal", 1: "Pneumonia", 2: "Viral"},
            }
        )


def test_checkpoint_metadata_serializes_with_explicit_indices():
    metadata = CheckpointMetadata.from_task("subtype")

    assert metadata.to_dict() == {
        "task": "subtype",
        "num_classes": 2,
        "class_index_to_label": {0: "Bacterial Pneumonia", 1: "Viral Pneumonia"},
    }
