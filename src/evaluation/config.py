"""Configuration for reproducible, evaluation-only ensemble runs.

The evaluator deliberately keeps all decision-transform parameters in this
module instead of hiding them in the prediction path. In particular, the
neutral defaults are ``viral_boost=1.0`` and equal model weights. A caller
must opt in to a boost or a non-uniform weighting, and the resulting values
are written to the evaluation report by ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_TRAIN_CSV = "data/splits/train.csv"
DEFAULT_VAL_CSV = "data/splits/val.csv"
DEFAULT_TEST_CSV = "data/splits/test.csv"
DEFAULT_DATASET_ROOT = "data/raw_datasets"
DEFAULT_CHECKPOINT_DIR = "saved_models"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_BATCH_SIZE = 32
DEFAULT_VIRAL_BOOST = 1.0
DEFAULT_MODEL_NAMES: Tuple[str, ...] = (
    "mobilenet",
    "efficientnet",
    "resnet",
)
DEFAULT_SINGLE_MODEL_NAMES: Tuple[str, ...] = (
    "mobilenet",
    "efficientnet",
    "resnet",
    "densenet",
)
DEFAULT_DEVICE = "auto"

_CONFIG_KEYS = {
    "train_csv",
    "val_csv",
    "test_csv",
    "dataset_root",
    "checkpoint_dir",
    "report_dir",
    "report_path",
    "models",
    "model_names",
    "batch_size",
    "viral_boost",
    "ensemble_weights",
    "device",
}


class EvaluationConfigError(ValueError):
    """Raised when an evaluation configuration is ambiguous or invalid."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Fully resolved inputs for one evaluation run.

    ``viral_boost`` is a class prior adjustment, not a metric. It is kept
    explicit so that a report can state whether a run used it. The default
    is neutral and therefore cannot change predictions. ``ensemble_weights``
    is ``None`` for equal weighting; a supplied sequence is normalized by
    the aggregation helper and reported explicitly.
    """

    train_csv: str = DEFAULT_TRAIN_CSV
    val_csv: str = DEFAULT_VAL_CSV
    test_csv: str = DEFAULT_TEST_CSV
    dataset_root: str = DEFAULT_DATASET_ROOT
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    report_dir: str = DEFAULT_REPORT_DIR
    report_path: Optional[str] = None
    model_names: Tuple[str, ...] = DEFAULT_MODEL_NAMES
    batch_size: int = DEFAULT_BATCH_SIZE
    viral_boost: float = DEFAULT_VIRAL_BOOST
    ensemble_weights: Optional[Tuple[float, ...]] = None
    device: str = DEFAULT_DEVICE

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the config."""

        values = asdict(self)
        values["model_names"] = list(self.model_names)
        if self.ensemble_weights is not None:
            values["ensemble_weights"] = list(self.ensemble_weights)
        return values


def _load_config_file(path: Path) -> Mapping[str, Any]:
    """Load a JSON or YAML evaluation config.

    JSON is supported by the standard library. YAML is optional at runtime,
    matching the repository's existing example-config convention; when a YAML
    file is used, a clear error is raised if PyYAML is not installed.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationConfigError(f"Cannot read config file: {path}") from exc

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluationConfigError(f"Invalid JSON config: {path}") from exc
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EvaluationConfigError(
                "YAML evaluation configs require PyYAML; use JSON or install PyYAML"
            ) from exc
        try:
            payload = yaml.safe_load(text)
        except Exception as exc:  # yaml raises parser-specific exceptions
            raise EvaluationConfigError(f"Invalid YAML config: {path}") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise EvaluationConfigError("Evaluation config must contain an object/mapping")
    unknown = sorted(set(payload) - _CONFIG_KEYS)
    if unknown:
        raise EvaluationConfigError(
            "Unknown evaluation config key(s): " + ", ".join(unknown)
        )
    return payload


def _normalise_string_list(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [str(part).strip() for part in value]
    else:
        raise EvaluationConfigError(f"{field_name} must be a list of strings")

    values = [value for value in values if value]
    if not values:
        raise EvaluationConfigError(f"{field_name} must not be empty")
    return tuple(values)


def _normalise_weights(value: Any, model_count: int) -> Tuple[float, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raise EvaluationConfigError("ensemble_weights must be a list of numbers")

    if not raw_values or any(str(part).strip() == "" for part in raw_values):
        raise EvaluationConfigError("ensemble_weights must not be empty")
    if len(raw_values) != model_count:
        raise EvaluationConfigError(
            "ensemble_weights must contain one value per model "
            f"({model_count} expected, {len(raw_values)} provided)"
        )

    weights: list[float] = []
    for raw_weight in raw_values:
        if isinstance(raw_weight, bool):
            raise EvaluationConfigError("ensemble_weights must contain numbers")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise EvaluationConfigError("ensemble_weights must contain numbers") from exc
        if not math.isfinite(weight) or weight < 0:
            raise EvaluationConfigError(
                "ensemble_weights must be finite, non-negative numbers"
            )
        weights.append(weight)
    if sum(weights) <= 0:
        raise EvaluationConfigError("ensemble_weights must contain a positive weight")
    return tuple(weights)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise EvaluationConfigError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise EvaluationConfigError(f"{field_name} must be a positive integer")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise EvaluationConfigError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise EvaluationConfigError(f"{field_name} must be a finite number")
    return parsed


def _device_value(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigError("device must be auto, cpu, or cuda")
    normalized = value.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise EvaluationConfigError("device must be auto, cpu, or cuda")
    return normalized


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="JSON/YAML evaluation config (command-line values override it)",
    )
    parser.add_argument(
        "--train-csv",
        help=f"training split manifest (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--val-csv",
        help=f"validation split manifest (default: {DEFAULT_VAL_CSV})",
    )
    parser.add_argument(
        "--test-csv",
        help=f"test split manifest (default: {DEFAULT_TEST_CSV})",
    )
    parser.add_argument(
        "--dataset-root",
        help=f"raw dataset root recorded in the report (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help=f"checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--report-dir",
        help=f"report output directory (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--report-path",
        help="explicit report path (overrides timestamped report generation)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "model names in ensemble order; defaults are supplied by the "
            "evaluation entry point"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"evaluation loader batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--viral-boost",
        type=float,
        help=(
            "explicit multiplicative viral-class prior; 1.0 means no adjustment "
            f"(default: {DEFAULT_VIRAL_BOOST})"
        ),
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        help=(
            "explicit model weights; omitted means equal weights "
            "(default: equal weighting)"
        ),
    )
    parser.add_argument(
        "--device",
        help="auto, cpu, or cuda (default: auto)",
    )


def _build_parser(
    default_model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible evaluation. Defaults are neutral: "
            "viral_boost=1.0 and equal model weights."
        )
    )
    _add_common_arguments(parser)
    parser.set_defaults(default_model_names=tuple(default_model_names))
    return parser


def parse_args(
    argv: Optional[Iterable[str]] = None,
    *,
    default_model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> EvaluationConfig:
    """Parse and validate command-line/config evaluation inputs.

    Command-line values take precedence over config-file values, which take
    precedence over the documented defaults. Invalid values are rejected
    before any model or dataset is loaded.
    """

    parser = _build_parser(default_model_names)
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    file_values: Mapping[str, Any] = {}
    if namespace.config:
        file_values = _load_config_file(Path(namespace.config))

    def pick(cli_value: Any, key: str, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if key in file_values:
            return file_values[key]
        return default

    model_value = pick(namespace.models, "models", None)
    if model_value is None:
        model_value = pick(None, "model_names", tuple(default_model_names))
    model_names = _normalise_string_list(model_value, "models")

    weights_value = pick(namespace.ensemble_weights, "ensemble_weights", None)
    ensemble_weights = (
        None
        if weights_value is None
        else _normalise_weights(weights_value, len(model_names))
    )

    viral_boost = _finite_float(
        pick(namespace.viral_boost, "viral_boost", DEFAULT_VIRAL_BOOST),
        "viral_boost",
    )
    if viral_boost <= 0:
        raise EvaluationConfigError("viral_boost must be greater than zero")

    batch_size = _positive_int(
        pick(namespace.batch_size, "batch_size", DEFAULT_BATCH_SIZE),
        "batch_size",
    )

    def path_value(cli_value: Any, key: str, default: str) -> str:
        value = pick(cli_value, key, default)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationConfigError(f"{key} must be a non-empty path string")
        return value

    report_path_value = pick(namespace.report_path, "report_path", None)
    if report_path_value is not None:
        report_path = path_value(report_path_value, "report_path", DEFAULT_REPORT_DIR)
    else:
        report_path = None

    return EvaluationConfig(
        train_csv=path_value(namespace.train_csv, "train_csv", DEFAULT_TRAIN_CSV),
        val_csv=path_value(namespace.val_csv, "val_csv", DEFAULT_VAL_CSV),
        test_csv=path_value(namespace.test_csv, "test_csv", DEFAULT_TEST_CSV),
        dataset_root=path_value(
            namespace.dataset_root, "dataset_root", DEFAULT_DATASET_ROOT
        ),
        checkpoint_dir=path_value(
            namespace.checkpoint_dir, "checkpoint_dir", DEFAULT_CHECKPOINT_DIR
        ),
        report_dir=path_value(namespace.report_dir, "report_dir", DEFAULT_REPORT_DIR),
        report_path=report_path,
        model_names=model_names,
        batch_size=batch_size,
        viral_boost=viral_boost,
        ensemble_weights=ensemble_weights,
        device=_device_value(pick(namespace.device, "device", DEFAULT_DEVICE)),
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_NAMES",
    "DEFAULT_SINGLE_MODEL_NAMES",
    "DEFAULT_TEST_CSV",
    "DEFAULT_TRAIN_CSV",
    "DEFAULT_VAL_CSV",
    "DEFAULT_VIRAL_BOOST",
    "EvaluationConfig",
    "EvaluationConfigError",
    "parse_args",
]
