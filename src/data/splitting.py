"""Utilities for leakage-safe, patient-grouped dataset splits."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def patient_id(original_path: str) -> str:
    """Return a stable subject/series identifier from a dataset filename."""
    name = Path(str(original_path)).name
    match = re.search(r"(?:NORMAL|BACTERIA|VIRUS)[-_](\d+)[-_]", name, re.IGNORECASE)
    if match:
        return f"dataset_1:{match.group(1)}"

    match = re.search(r"(IM-\d+)", name, re.IGNORECASE)
    if match:
        return f"dataset_2:{match.group(1).lower()}"

    # Some source datasets use NORMAL2-IM-...; the generic filename fallback
    # above intentionally treats unknown naming schemes as separate paths.
    return f"path:{original_path}"


def make_patient_grouped_split(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    n_splits: int = 20,
) -> Mapping[str, pd.DataFrame]:
    """Return deterministic train/validation/test frames grouped by patient.

    Three stratified group folds are reserved for validation and three more
    for test; all remaining folds form training. This yields the documented
    70/15/15 image-ratio while keeping related images together.
    """
    if n_splits < 7:
        raise ValueError("n_splits must be at least 7 for 70/15/15 grouping")
    required = {"image_id", "original_path", "encoded_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if len(frame) < n_splits:
        raise ValueError("Not enough rows for the configured number of folds")

    working = frame.copy()
    working["patient_id"] = working["original_path"].map(patient_id)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_ids = np.full(len(working), -1, dtype=int)
    for fold, (_, validation_index) in enumerate(
        splitter.split(working, working["encoded_label"], groups=working["patient_id"])
    ):
        fold_ids[validation_index] = fold

    if (fold_ids < 0).any():
        raise AssertionError("Every row must be assigned to exactly one group fold")

    validation_folds = set(range(3))
    test_folds = set(range(3, 6))
    return {
        "train": working.loc[
            ~np.isin(fold_ids, list(validation_folds | test_folds))
        ].copy(),
        "validation": working.loc[np.isin(fold_ids, list(validation_folds))].copy(),
        "test": working.loc[np.isin(fold_ids, list(test_folds))].copy(),
    }


def assert_no_split_leakage(splits: Mapping[str, pd.DataFrame]) -> None:
    """Raise if any identifier, including patient_id, crosses partitions."""
    columns = ["image_id", "patient_id", "sha256", "phash", "original_path"]
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in columns:
            if column not in splits[left_name] or column not in splits[right_name]:
                continue
            left = set(splits[left_name][column].dropna())
            right = set(splits[right_name][column].dropna())
            if not left.isdisjoint(right):
                raise AssertionError(f"{column} overlap: {left_name}/{right_name}")
