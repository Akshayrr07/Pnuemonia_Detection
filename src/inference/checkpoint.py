"""Safe loading and the class-index contract for local model checkpoints.

PyTorch ``.pt`` files are pickle-based even when they contain only tensors.
This module deliberately accepts *state-dict checkpoints only* and always uses
``torch.load(..., weights_only=True)``.  It never falls back to unrestricted
object deserialization: a full-model or otherwise unsafe checkpoint must be
re-saved as a plain state dict by its owner.

The metadata contract is deliberately small and explicit.  A checkpoint may
store it under ``metadata`` (or, for a wrapper with a ``state_dict`` key, beside
that key)::

    {
        "state_dict": {"layer.weight": tensor(...), ...},
        "metadata": {
            "task": "binary",
            "num_classes": 2,
            "class_index_to_label": {0: "Normal", 1: "Pneumonia"},
        },
    }

Bare state dicts are still accepted for backwards compatibility.  Callers that
need the class-index contract can pass ``expected_task``/``expected_metadata``
or ``require_metadata=True``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union


STATE_DICT_KEY = "state_dict"
MODEL_STATE_DICT_KEY = "model_state_dict"
METADATA_KEY = "metadata"
CHECKPOINT_METADATA_KEY = "checkpoint_metadata"
CLASS_INDEX_TO_LABEL_KEY = "class_index_to_label"

BINARY_TASK = "binary"
SUBTYPE_TASK = "subtype"
THREE_CLASS_TASK = "3_class"


def _labels(mapping: dict[int, str]) -> Mapping[int, str]:
    return MappingProxyType(dict(mapping))


# These mappings are the single source of truth for inference class indices.
# The order is intentional and must not be changed without retraining or an
# explicit checkpoint metadata migration.
BINARY_CLASS_INDEX_LABELS: Mapping[int, str] = _labels(
    {0: "Normal", 1: "Pneumonia"}
)
SUBTYPE_CLASS_INDEX_LABELS: Mapping[int, str] = _labels(
    {0: "Bacterial Pneumonia", 1: "Viral Pneumonia"}
)
THREE_CLASS_CLASS_INDEX_LABELS: Mapping[int, str] = _labels(
    {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}
)

# Short aliases are useful to callers that already call the values labels.
BINARY_LABELS = BINARY_CLASS_INDEX_LABELS
SUBTYPE_LABELS = SUBTYPE_CLASS_INDEX_LABELS
THREE_CLASS_LABELS = THREE_CLASS_CLASS_INDEX_LABELS

CLASS_INDEX_LABELS_BY_TASK: Mapping[str, Mapping[int, str]] = MappingProxyType(
    {
        BINARY_TASK: BINARY_CLASS_INDEX_LABELS,
        SUBTYPE_TASK: SUBTYPE_CLASS_INDEX_LABELS,
        THREE_CLASS_TASK: THREE_CLASS_CLASS_INDEX_LABELS,
    }
)

_TASK_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "binary": BINARY_TASK,
        "normal_vs_pneumonia": BINARY_TASK,
        "normal-vs-pneumonia": BINARY_TASK,
        "normal_pneumonia": BINARY_TASK,
        "subtype": SUBTYPE_TASK,
        "bacterial_vs_viral": SUBTYPE_TASK,
        "bacterial-vs-viral": SUBTYPE_TASK,
        "pneumonia_subtype": SUBTYPE_TASK,
        "3_class": THREE_CLASS_TASK,
        "3class": THREE_CLASS_TASK,
        "3-class": THREE_CLASS_TASK,
        "three_class": THREE_CLASS_TASK,
        "three-class": THREE_CLASS_TASK,
        "direct_three_class": THREE_CLASS_TASK,
    }
)


class CheckpointError(RuntimeError):
    """Base class for safe-checkpoint contract failures."""


class CheckpointLoadError(CheckpointError):
    """A checkpoint could not be safely loaded as a state dict."""


class CheckpointUnsupportedError(CheckpointLoadError):
    """The installed PyTorch cannot perform a weights-only load."""


class UnsafeCheckpointError(CheckpointLoadError):
    """The payload is not a plain state-dict checkpoint."""


class CheckpointMetadataError(CheckpointLoadError):
    """Checkpoint metadata is missing, malformed, or does not match its task."""


def _canonical_task(task: object) -> str:
    if not isinstance(task, str):
        raise CheckpointMetadataError(
            f"Checkpoint metadata 'task' must be a string, got {type(task).__name__}."
        )

    normalized = task.strip().lower().replace(" ", "_")
    canonical = _TASK_ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(CLASS_INDEX_LABELS_BY_TASK)
        raise CheckpointMetadataError(
            f"Unsupported checkpoint task {task!r}; expected one of: {supported}."
        )
    return canonical


def _as_metadata_value(mapping: Mapping[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise CheckpointMetadataError(
        f"Checkpoint metadata is missing required {label!r} field."
    )


def _parse_metadata_mapping(
    raw_metadata: Mapping[str, Any],
    expected_task: Optional[object] = None,
) -> "CheckpointMetadata":
    if not isinstance(raw_metadata, Mapping):
        raise CheckpointMetadataError(
            "Checkpoint metadata must be a mapping containing task, num_classes, "
            "and class_index_to_label."
        )

    task_value = _as_metadata_value(raw_metadata, ("task", "classification_task", "label_mode"), "task")
    num_classes = _as_metadata_value(
        raw_metadata, ("num_classes", "n_classes", "class_count"), "num_classes"
    )
    class_index_to_label = _as_metadata_value(
        raw_metadata,
        (CLASS_INDEX_TO_LABEL_KEY, "class_index_to_labels", "class_label_map"),
        "class_index_to_label",
    )

    return CheckpointMetadata(
        task=task_value,
        num_classes=num_classes,
        class_index_to_label=class_index_to_label,
        expected_task=expected_task,
    )


@dataclass(frozen=True)
class CheckpointMetadata:
    """Validated, immutable checkpoint class-index metadata."""

    task: str
    num_classes: int
    class_index_to_label: Mapping[int, str]

    def __init__(
        self,
        task: object,
        num_classes: object,
        class_index_to_label: object,
        *,
        expected_task: Optional[object] = None,
    ) -> None:
        canonical_task = _canonical_task(task)
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise CheckpointMetadataError(
                "Checkpoint metadata 'num_classes' must be an integer, got "
                f"{type(num_classes).__name__}."
            )
        if num_classes <= 0:
            raise CheckpointMetadataError(
                f"Checkpoint metadata 'num_classes' must be positive, got {num_classes}."
            )
        if not isinstance(class_index_to_label, Mapping):
            raise CheckpointMetadataError(
                "Checkpoint metadata 'class_index_to_label' must be a mapping from "
                "integer class indices to labels."
            )

        normalized: dict[int, str] = {}
        for index, label in class_index_to_label.items():
            if isinstance(index, bool) or not isinstance(index, int):
                raise CheckpointMetadataError(
                    "Checkpoint metadata class indices must be integers, got "
                    f"{type(index).__name__} for {index!r}."
                )
            if not isinstance(label, str) or not label.strip():
                raise CheckpointMetadataError(
                    f"Checkpoint metadata label for class {index} must be a non-empty string."
                )
            normalized[index] = label

        expected_indices = list(range(num_classes))
        if sorted(normalized) != expected_indices:
            raise CheckpointMetadataError(
                "Checkpoint metadata class_index_to_label must contain exactly indices "
                f"{expected_indices} for num_classes={num_classes}; got {sorted(normalized)}."
            )
        if len(set(normalized.values())) != len(normalized):
            raise CheckpointMetadataError(
                "Checkpoint metadata class_index_to_label must contain unique labels."
            )

        expected_labels = CLASS_INDEX_LABELS_BY_TASK[canonical_task]
        if len(expected_labels) != num_classes:
            raise CheckpointMetadataError(
                f"Checkpoint metadata for task {canonical_task!r} must declare "
                f"num_classes={len(expected_labels)}, got {num_classes}."
            )
        if dict(normalized) != dict(expected_labels):
            raise CheckpointMetadataError(
                f"Checkpoint metadata class_index_to_label mismatch for task "
                f"{canonical_task!r}: expected {dict(expected_labels)!r}, got {normalized!r}."
            )

        if expected_task is not None and canonical_task != _canonical_task(expected_task):
            raise CheckpointMetadataError(
                f"Checkpoint metadata task mismatch: expected {_canonical_task(expected_task)!r}, "
                f"got {canonical_task!r}."
            )

        object.__setattr__(self, "task", canonical_task)
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "class_index_to_label", _labels(normalized))

    @classmethod
    def from_task(cls, task: object) -> "CheckpointMetadata":
        """Build the canonical metadata for a supported task."""
        canonical_task = _canonical_task(task)
        labels = CLASS_INDEX_LABELS_BY_TASK[canonical_task]
        return cls(
            task=canonical_task,
            num_classes=len(labels),
            class_index_to_label=labels,
        )

    @classmethod
    def from_mapping(
        cls,
        raw_metadata: Mapping[str, Any],
        expected_task: Optional[object] = None,
    ) -> "CheckpointMetadata":
        return _parse_metadata_mapping(raw_metadata, expected_task)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation with integer class indices."""
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            CLASS_INDEX_TO_LABEL_KEY: dict(self.class_index_to_label),
        }


def validate_checkpoint_metadata(
    metadata: Union["CheckpointMetadata", Mapping[str, Any]],
    expected_task: Optional[object] = None,
) -> CheckpointMetadata:
    """Validate and normalize a checkpoint's class-index metadata.

    Raises:
        CheckpointMetadataError: if metadata is malformed or does not exactly
            match the canonical mapping for its task.
    """
    if isinstance(metadata, CheckpointMetadata):
        if expected_task is not None and metadata.task != _canonical_task(expected_task):
            raise CheckpointMetadataError(
                f"Checkpoint metadata task mismatch: expected {_canonical_task(expected_task)!r}, "
                f"got {metadata.task!r}."
            )
        return metadata
    return _parse_metadata_mapping(metadata, expected_task)


def _torch_load_supports_weights_only(torch_module: Any) -> bool:
    try:
        signature = inspect.signature(torch_module.load)
    except (TypeError, ValueError):
        # Some wrappers hide their signature.  Passing the explicit keyword is
        # still the safest action; a TypeError is reported with context below.
        return True

    parameters = signature.parameters.values()
    return "weights_only" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


def _metadata_from_wrapper(wrapper: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in (METADATA_KEY, CHECKPOINT_METADATA_KEY):
        if key not in wrapper:
            continue
        value = wrapper[key]
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value
        raise CheckpointMetadataError(
            f"Checkpoint field {key!r} must be a metadata mapping."
        )

    # Also accept a wrapper that puts the canonical fields at its top level.
    if isinstance(wrapper.get("task"), str) and any(
        key in wrapper
        for key in (
            "num_classes",
            "n_classes",
            "class_count",
            CLASS_INDEX_TO_LABEL_KEY,
            "class_index_to_labels",
            "class_label_map",
        )
    ):
        return wrapper
    return None


def _split_checkpoint_payload(payload: Any) -> tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise UnsafeCheckpointError(
            "Trusted checkpoint loading only accepts a state_dict mapping. "
            "Full model objects and legacy executable checkpoints are not supported; "
            "save model.state_dict() instead."
        )

    for key in (STATE_DICT_KEY, MODEL_STATE_DICT_KEY):
        if key in payload:
            state_dict = payload[key]
            if not isinstance(state_dict, Mapping):
                raise UnsafeCheckpointError(
                    f"Checkpoint field {key!r} must contain a state_dict mapping."
                )
            return state_dict, _metadata_from_wrapper(payload)

    # A full serialized model is deliberately not accepted.  A mapping under
    # ``model`` is accepted only for legacy wrappers that store a state dict
    # under that name; a module object fails the mapping check above.
    model_value = payload.get("model")
    if isinstance(model_value, Mapping):
        return model_value, _metadata_from_wrapper(payload)
    if "model" in payload:
        raise UnsafeCheckpointError(
            "Checkpoint field 'model' must contain a state_dict mapping; a full "
            "model object is not supported."
        )

    metadata = _metadata_from_wrapper(payload)
    if metadata is not None:
        raise CheckpointMetadataError(
            "Checkpoint contains metadata but no state_dict; a weights-only "
            "state_dict checkpoint is required."
        )

    return payload, None


def _validate_state_dict(state_dict: Mapping[str, Any], torch_module: Any) -> dict[str, Any]:
    if not all(isinstance(key, str) for key in state_dict):
        raise UnsafeCheckpointError(
            "Trusted checkpoint state_dict keys must all be strings."
        )

    for key, value in state_dict.items():
        if not isinstance(value, torch_module.Tensor):
            raise UnsafeCheckpointError(
                "Trusted checkpoint payload contains a non-tensor value at "
                f"state_dict[{key!r}] ({type(value).__name__}). Save a plain "
                "model.state_dict() checkpoint; unsafe objects are not loaded."
            )

    return dict(state_dict)


def _validate_checkpoint_wrapper(payload: Any) -> None:
    """Reject extra training/runtime objects from a state-dict wrapper.

    ``weights_only=True`` prevents arbitrary Python execution, but a legacy
    training checkpoint can still contain safe primitive optimizer dictionaries
    next to the model weights.  Those are not an inference state dict, so the
    helper fails clearly instead of silently ignoring them.
    """
    if not isinstance(payload, Mapping):
        return

    state_keys = {STATE_DICT_KEY, MODEL_STATE_DICT_KEY, "model"}
    wrapper_keys = state_keys.intersection(payload.keys())
    if not wrapper_keys:
        return  # a bare state dict is validated separately

    if len(wrapper_keys) != 1:
        raise UnsafeCheckpointError(
            "Trusted checkpoint must contain exactly one state_dict wrapper; "
            f"got {sorted(wrapper_keys)!r}."
        )
    if any(not isinstance(key, str) for key in payload):
        raise UnsafeCheckpointError(
            "Trusted checkpoint wrapper keys must all be strings."
        )

    metadata_fields = {
        METADATA_KEY,
        CHECKPOINT_METADATA_KEY,
        "task",
        "classification_task",
        "label_mode",
        "num_classes",
        "n_classes",
        "class_count",
        CLASS_INDEX_TO_LABEL_KEY,
        "class_index_to_labels",
        "class_label_map",
    }
    allowed = state_keys | metadata_fields
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise UnsafeCheckpointError(
            "Trusted checkpoint contains non-inference fields "
            f"{unknown!r} beside the state dict. Save model.state_dict() only, "
            "or keep training checkpoints in a separate trusted workflow."
        )

    if METADATA_KEY in payload and CHECKPOINT_METADATA_KEY in payload:
        raise CheckpointMetadataError(
            "Checkpoint must not define both 'metadata' and 'checkpoint_metadata'."
        )


class LoadedCheckpoint(dict):
    """A state-dict-compatible checkpoint result with optional metadata.

    It subclasses ``dict`` so existing ``model.load_state_dict(result)`` call
    sites remain valid, while callers that need the validated contract can use
    ``result.metadata`` or ``result.state_dict``.
    """

    def __init__(
        self,
        state_dict: Mapping[str, Any],
        metadata: Optional[CheckpointMetadata],
    ) -> None:
        super().__init__(state_dict)
        self._metadata = metadata

    @property
    def metadata(self) -> Optional[CheckpointMetadata]:
        return self._metadata

    @property
    def state_dict(self) -> "LoadedCheckpoint":
        return self


def load_trusted_checkpoint(
    checkpoint_path: Union[str, os.PathLike[str]],
    map_location: Any = "cpu",
    *,
    expected_task: Optional[object] = None,
    expected_metadata: Optional[Union[CheckpointMetadata, Mapping[str, Any]]] = None,
    require_metadata: bool = False,
) -> LoadedCheckpoint:
    """Load a plain state-dict checkpoint using restricted deserialization.

    The helper never retries with unrestricted object deserialization.  Older
    PyTorch versions without the ``weights_only`` argument are rejected rather
    than silently downgraded to unsafe loading.

    Args:
        checkpoint_path: Local ``.pt``/``.pth`` file containing a state dict.
        map_location: Device mapping passed directly to ``torch.load``.
        expected_task: Optional task contract to enforce when metadata exists.
        expected_metadata: Optional exact metadata contract to enforce.
        require_metadata: Reject bare state dicts that have no explicit
            metadata.  Defaults to false for legacy bare state dicts.
    """
    try:
        import torch
    except ImportError as exc:
        raise CheckpointLoadError(
            "PyTorch is required to load a local checkpoint. Install a version "
            "that supports weights_only=True."
        ) from exc

    if not _torch_load_supports_weights_only(torch):
        raise CheckpointUnsupportedError(
            "The installed PyTorch does not support weights_only=True; refusing "
            "an unsafe fallback. Upgrade PyTorch and save a plain state_dict."
        )

    path = Path(checkpoint_path)
    try:
        payload = torch.load(
            os.fspath(path),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointLoadError(
            f"Could not safely load checkpoint {str(path)!r} with weights_only=True. "
            "Only plain tensor state_dict checkpoints are supported; full model "
            "objects and executable legacy checkpoints are rejected. Save "
            "model.state_dict() to a new checkpoint. Original error: "
            f"{exc}"
        ) from exc

    _validate_checkpoint_wrapper(payload)
    state_dict, raw_metadata = _split_checkpoint_payload(payload)
    state_dict = _validate_state_dict(state_dict, torch)

    metadata: Optional[CheckpointMetadata] = None
    if raw_metadata is not None:
        metadata = validate_checkpoint_metadata(raw_metadata, expected_task)
    elif expected_task is not None:
        raise CheckpointMetadataError(
            f"Checkpoint {str(path)!r} has no metadata to enforce the expected "
            f"task {_canonical_task(expected_task)!r}; include task, num_classes, "
            "and class_index_to_label, or omit expected_task for a legacy "
            "bare state dict."
        )
    elif require_metadata:
        raise CheckpointMetadataError(
            f"Checkpoint {str(path)!r} has no explicit metadata; supply metadata "
            "with task, num_classes, and class_index_to_label."
        )

    if expected_metadata is not None:
        expected = validate_checkpoint_metadata(expected_metadata)
        if metadata is None:
            raise CheckpointMetadataError(
                f"Checkpoint {str(path)!r} has no metadata to match the expected "
                "class-index contract."
            )
        if metadata != expected:
            raise CheckpointMetadataError(
                f"Checkpoint metadata mismatch for {str(path)!r}: expected "
                f"{expected.to_dict()!r}, got {metadata.to_dict()!r}."
            )

    return LoadedCheckpoint(state_dict, metadata)


def load_model_state_dict(
    checkpoint_path: Union[str, os.PathLike[str]],
    map_location: Any = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper returning only the validated state-dict mapping."""
    return dict(load_trusted_checkpoint(checkpoint_path, map_location, **kwargs))


# Explicit aliases make the safety intent discoverable at call sites while
# keeping one implementation.
load_checkpoint_safely = load_trusted_checkpoint
safe_load_checkpoint = load_trusted_checkpoint


__all__ = [
    "BINARY_CLASS_INDEX_LABELS",
    "BINARY_LABELS",
    "BINARY_TASK",
    "CLASS_INDEX_LABELS_BY_TASK",
    "CHECKPOINT_METADATA_KEY",
    "CLASS_INDEX_TO_LABEL_KEY",
    "CheckpointError",
    "CheckpointLoadError",
    "CheckpointMetadata",
    "CheckpointMetadataError",
    "CheckpointUnsupportedError",
    "LoadedCheckpoint",
    "METADATA_KEY",
    "MODEL_STATE_DICT_KEY",
    "STATE_DICT_KEY",
    "SUBTYPE_CLASS_INDEX_LABELS",
    "SUBTYPE_LABELS",
    "SUBTYPE_TASK",
    "THREE_CLASS_CLASS_INDEX_LABELS",
    "THREE_CLASS_LABELS",
    "THREE_CLASS_TASK",
    "UnsafeCheckpointError",
    "load_checkpoint_safely",
    "load_model_state_dict",
    "load_trusted_checkpoint",
    "safe_load_checkpoint",
    "validate_checkpoint_metadata",
]
