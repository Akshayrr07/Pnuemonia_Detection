"""Helpers for loading local models from safe, state-dict checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

from src.inference.checkpoint import (
    THREE_CLASS_TASK,
    CheckpointMetadata,
    load_trusted_checkpoint,
)


DEFAULT_MODEL_NAMES = ("mobilenet", "efficientnet", "resnet")
DEFAULT_CHECKPOINT_DIR = Path("saved_models")


def load_model(
    model_name: str,
    device: Any,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    expected_task: Optional[Union[str, CheckpointMetadata]] = None,
    expected_metadata: Optional[CheckpointMetadata] = None,
    num_classes: Optional[int] = None,
) -> Any:
    """Load one local model from a validated state-dict checkpoint.

    ``expected_task``/``expected_metadata`` are optional for legacy bare state
    dicts, but callers that know the model contract should supply one.  The
    loader never uses unsafe deserialization or silently falls back to it.
    """
    from src.models.model_factory import get_model

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_DIR / f"{model_name}.pt"

    expected_task_name: Optional[str] = None
    if isinstance(expected_task, CheckpointMetadata):
        expected_metadata = expected_task
        if num_classes is None:
            num_classes = expected_task.num_classes
    elif expected_task is not None:
        canonical_metadata = CheckpointMetadata.from_task(expected_task)
        expected_task_name = canonical_metadata.task
        if num_classes is None:
            num_classes = canonical_metadata.num_classes

    loaded = load_trusted_checkpoint(
        checkpoint_path,
        map_location=device,
        expected_task=expected_task_name,
        expected_metadata=expected_metadata,
    )

    model = get_model(model_name, num_classes=num_classes or 3, freeze=False)
    model.load_state_dict(loaded, strict=True)
    model.to(device)
    model.eval()
    return model


def load_all_models(
    device: Any,
    *,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
    checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
    expected_task: Optional[Union[str, CheckpointMetadata]] = None,
    expected_metadata: Optional[CheckpointMetadata] = None,
) -> list[Any]:
    """Load the three-class local ensemble using the safe checkpoint helper."""
    checkpoint_dir = Path(checkpoint_dir)
    return [
        load_model(
            name,
            device,
            checkpoint_path=checkpoint_dir / f"{name}.pt",
            expected_task=expected_task or THREE_CLASS_TASK,
            expected_metadata=expected_metadata,
        )
        for name in model_names
    ]
