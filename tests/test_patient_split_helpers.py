from __future__ import annotations

import pandas as pd

from src.data.splitting import make_patient_grouped_split, patient_id


def test_patient_id_groups_related_images() -> None:
    first = patient_id("dataset_1/train/PNEUMONIA/BACTERIA-8550709-0001.jpeg")
    second = patient_id("dataset_1/train/PNEUMONIA/VIRUS-8550709-0009.jpeg")
    assert first == second


def test_patient_id_keeps_different_subjects_separate() -> None:
    first = patient_id("dataset_1/train/PNEUMONIA/BACTERIA-8550709-0001.jpeg")
    second = patient_id("dataset_1/train/PNEUMONIA/BACTERIA-8550710-0001.jpeg")
    assert first != second


def test_patient_id_supports_second_dataset() -> None:
    first = patient_id("dataset_2/test/NORMAL/NORMAL2-IM-0007-0001.jpeg")
    second = patient_id("dataset_2/test/NORMAL/NORMAL2-IM-0007-0002.jpeg")
    assert first == second


def test_patient_grouped_split_keeps_related_images_together() -> None:
    rows = []
    for patient in range(60):
        label = patient % 3
        for view in range(2):
            rows.append(
                {
                    "image_id": f"image_{patient}_{view}",
                    "original_path": f"dataset_1/train/PNEUMONIA/IMAGE-{patient:07d}-000{view + 1}.jpeg",
                    "encoded_label": label,
                }
            )
    frame = pd.DataFrame(rows)

    splits = make_patient_grouped_split(frame, random_state=42)

    assert sum(len(split) for split in splits.values()) == len(frame)
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left = set(splits[left_name]["patient_id"])
        right = set(splits[right_name]["patient_id"])
        assert left.isdisjoint(right), f"patient leakage: {left_name}/{right_name}"


def test_patient_grouped_split_outputs_patient_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": f"image_{patient}",
                "original_path": f"dataset_1/train/NORMAL/NORMAL-{patient}-0001.jpeg",
                "encoded_label": 0,
            }
            for patient in range(20)
        ]
    )
    splits = make_patient_grouped_split(frame, random_state=42)
    patient_ids = set(splits["train"].get("patient_id", [])) | set(
        splits["validation"].get("patient_id", [])
    ) | set(splits["test"].get("patient_id", []))
    assert patient_ids == {f"dataset_1:{patient}" for patient in range(20)}
