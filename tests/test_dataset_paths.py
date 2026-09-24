import csv
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data import dataset as dataset_module
from src.data.dataset import PneumoniaDataset


def _write_split(csv_path: Path, relative_path: str) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_path", "encoded_label"],
        )
        writer.writeheader()
        writer.writerow({"original_path": relative_path, "encoded_label": 0})


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color="white").save(path)


def test_dataset_resolves_relative_paths_from_raw_root(tmp_path: Path) -> None:
    image_path = tmp_path / "dataset_1" / "train" / "NORMAL" / "sample.png"
    _write_image(image_path)
    csv_path = tmp_path / "split.csv"
    _write_split(csv_path, "dataset_1/train/NORMAL/sample.png")

    dataset = PneumoniaDataset(csv_path, raw_root=tmp_path)

    image, label = dataset[0]
    assert image.size == (2, 2)
    assert label == 0


def test_dataset_uses_repository_raw_root_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_root = tmp_path / "data" / "raw_datasets"
    image_path = default_root / "dataset_1" / "train" / "NORMAL" / "sample.png"
    _write_image(image_path)
    csv_path = tmp_path / "split.csv"
    _write_split(csv_path, "dataset_1/train/NORMAL/sample.png")

    monkeypatch.setattr(dataset_module, "DEFAULT_RAW_ROOT", default_root)
    dataset = PneumoniaDataset(csv_path)

    assert dataset.raw_root == default_root.resolve()
    assert dataset[0][0].size == (2, 2)


def test_dataset_accepts_dataframe_compatibility_input(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)
    frame = pd.DataFrame(
        [{"image_path": "sample.png", "label": 0}],
    )

    dataset = PneumoniaDataset(frame, raw_root=tmp_path)

    image, label = dataset[0]
    assert image.size == (2, 2)
    assert label == 0


@pytest.mark.parametrize("relative_path", ["../outside.png", "a/../../outside.png"])
def test_dataset_rejects_relative_path_traversal(
    tmp_path: Path,
    relative_path: str,
) -> None:
    frame = pd.DataFrame(
        [{"original_path": relative_path, "encoded_label": 0}],
    )

    with pytest.raises(ValueError, match="outside raw root"):
        PneumoniaDataset(frame, raw_root=tmp_path, validate_paths=True)


def test_dataset_rejects_absolute_path_outside_raw_root(tmp_path: Path) -> None:
    outside_image = tmp_path.parent / "outside.png"
    _write_image(outside_image)
    frame = pd.DataFrame(
        [{"original_path": str(outside_image), "encoded_label": 0}],
    )

    with pytest.raises(ValueError, match="outside raw root"):
        PneumoniaDataset(frame, raw_root=tmp_path, validate_paths=True)


def test_dataloader_passes_raw_root_to_each_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)
    frame = pd.DataFrame(
        [{"original_path": "sample.png", "encoded_label": 1}],
    )

    from src.data import dataloader

    original_dataset = dataloader.PneumoniaDataset
    captured: list[tuple[object, Path | None]] = []

    class CapturingDataset(original_dataset):
        def __init__(self, source, transform=None, **kwargs):
            captured.append((source, kwargs.get("raw_root")))
            super().__init__(source, transform=transform, **kwargs)

    monkeypatch.setattr(dataloader, "PneumoniaDataset", CapturingDataset)
    monkeypatch.setattr(dataloader, "get_train_transforms", lambda: None)
    monkeypatch.setattr(dataloader, "get_val_transforms", lambda: None)

    loaders = dataloader.get_dataloaders(
        frame,
        frame,
        frame,
        batch_size=1,
        raw_root=tmp_path,
    )

    assert len(loaders) == 3
    assert len(captured) == 3
    for source, root in captured:
        assert source is frame
        assert root == tmp_path
    for loader in loaders:
        assert loader.dataset[0][1] == 1
