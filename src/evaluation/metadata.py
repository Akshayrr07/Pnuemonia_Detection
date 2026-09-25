"""Metadata helpers for evaluation reports.

Reports need enough provenance to identify the exact manifests and checkpoint
files used, even when a model is later replaced. These helpers are read-only
and do not participate in training or deployment.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_metadata(path: Path) -> Dict[str, Any]:
    samples = 0
    class_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples += 1
            label = row.get("encoded_label")
            if label is not None:
                class_counts[str(label)] += 1

    metadata: Dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples": samples,
    }
    if class_counts:
        metadata["class_counts"] = dict(sorted(class_counts.items()))
    return metadata


def split_metadata(
    split_name: str,
    path: os.PathLike[str] | str,
    *,
    dataset_root: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Describe one split manifest, including its content identity.

    The metadata function intentionally records a missing manifest instead of
    silently omitting it. A missing path is represented with ``exists=False``
    and no fabricated sample counts, so an incomplete report is visible.
    """

    manifest = Path(path)
    metadata: Dict[str, Any] = {
        "split": split_name,
        "path": str(manifest),
        "exists": manifest.is_file(),
    }
    if dataset_root is not None:
        metadata["dataset_root"] = str(dataset_root)
    if manifest.is_file():
        metadata.update(_read_csv_metadata(manifest))
    return metadata


def dataset_metadata(
    splits: Iterable[Dict[str, Any]],
    *,
    dataset_root: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Return a compact dataset/split provenance block for a report."""

    metadata: Dict[str, Any] = {"splits": list(splits)}
    if dataset_root is not None:
        metadata["dataset_root"] = str(dataset_root)
    return metadata


def model_metadata(
    model_names: Sequence[str],
    checkpoint_dir: os.PathLike[str] | str,
) -> List[Dict[str, Any]]:
    """Describe the selected models and checkpoint identities."""

    directory = Path(checkpoint_dir)
    metadata: List[Dict[str, Any]] = []
    for name in model_names:
        checkpoint = directory / f"{name}.pt"
        item: Dict[str, Any] = {
            "name": name,
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
        }
        if checkpoint.is_file():
            item["sha256"] = sha256_file(checkpoint)
            item["bytes"] = checkpoint.stat().st_size
        metadata.append(item)
    return metadata


__all__ = [
    "dataset_metadata",
    "model_metadata",
    "sha256_file",
    "split_metadata",
]
