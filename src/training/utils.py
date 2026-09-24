"""Utilities for deterministic runs and checkpoint provenance."""

from __future__ import annotations

import copy
import json
import os
import platform
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 42
SEED_DERIVATION_MODULUS = 2**32


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed >= SEED_DERIVATION_MODULUS:
        raise ValueError("seed must be in [0, 2**32)")
    seed = int(seed)
    # Python reads PYTHONHASHSEED at interpreter startup; setting it here still
    # propagates the intended value to spawned child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required before CUDA contexts for deterministic matrix multiplication.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_deterministic_debug_mode"):
            torch.set_deterministic_debug_mode("default")
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            # Older supported PyTorch versions may not expose this switch.
            pass

    return seed


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % SEED_DERIVATION_MODULUS
    seed_everything(worker_seed)


def deterministic_generator(seed: int) -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed >= SEED_DERIVATION_MODULUS:
        raise ValueError("seed must be in [0, 2**32)")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def hash_file(path: str | os.PathLike[str]) -> str:
    """Return a SHA-256 digest of a split or configuration file."""

    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_run_metadata(
    *,
    seed: int,
    config: Mapping[str, Any] | None = None,
    split_hashes: Mapping[str, str] | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Build a serializable record of a training run's reproducibility inputs."""

    return {
        "schema_version": 1,
        "seed": int(seed),
        "config": _jsonable(config or {}),
        "split_hashes": {
            str(name): str(digest) for name, digest in sorted((split_hashes or {}).items())
        },
        "determinism": {
            "enabled": bool(deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "python_hash_seed": int(seed),
            "python_random": True,
            "numpy": True,
            "pytorch": True,
            "pytorch_deterministic_algorithms": bool(deterministic),
            "dataloader_workers": bool(deterministic),
            "transforms": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


def merge_run_metadata(
    base: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy and merge checkpoint metadata without mutating caller-owned data."""

    metadata = copy.deepcopy(dict(base or {}))
    for key, value in (updates or {}).items():
        if isinstance(value, Mapping) and isinstance(metadata.get(key), Mapping):
            metadata[key] = merge_run_metadata(metadata[key], value)
        else:
            metadata[key] = copy.deepcopy(value)
    return metadata


def serialize_run_metadata(
    metadata: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    return_metadata: bool = False,
) -> dict[str, Any] | None:
    """Write canonical JSON metadata to disk and optionally return its copy."""

    serializable = _jsonable(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return copy.deepcopy(serializable) if return_metadata else None
