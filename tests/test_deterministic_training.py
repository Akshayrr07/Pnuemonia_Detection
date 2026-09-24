from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data import dataloader as dataloader_module
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.training.utils import build_run_metadata, serialize_run_metadata


class _IndexDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = str(csv_path)
        self.transform = transform

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor(index), index


def _write_splits(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = []
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.csv"
        path.write_text("original_path,encoded_label\nexample.png,0\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _loader_order(loader):
    return [int(label) for batch in loader for label in batch[1]]


def test_get_dataloaders_reproduces_seed_and_records_split_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(dataloader_module, "PneumoniaDataset", _IndexDataset)
    train_csv, val_csv, test_csv = _write_splits(tmp_path)

    first_train, first_val, first_test, first_metadata = get_dataloaders(
        train_csv,
        val_csv,
        test_csv,
        batch_size=3,
        seed=1729,
        config={"training": {"batch_size": 3}, "run": {"seed": 1729}},
        return_metadata=True,
    )
    second_train, second_val, second_test, second_metadata = get_dataloaders(
        train_csv,
        val_csv,
        test_csv,
        batch_size=3,
        seed=1729,
        config={"training": {"batch_size": 3}, "run": {"seed": 1729}},
        return_metadata=True,
    )

    assert _loader_order(first_train) == _loader_order(second_train)
    assert _loader_order(first_val) == _loader_order(second_val)
    assert _loader_order(first_test) == _loader_order(second_test)
    assert first_metadata["split_hashes"] == second_metadata["split_hashes"]
    assert set(first_metadata["split_hashes"]) == {"train", "val", "test"}
    assert first_metadata["config"] == {
        "training": {"batch_size": 3},
        "run": {"seed": 1729},
    }
    assert first_metadata["seed"] == 1729


def test_trainer_selection_and_patience_use_only_validation_loss(
    tmp_path,
    monkeypatch,
):
    saved_payloads = []
    monkeypatch.setattr(
        "src.training.trainer.torch.save",
        lambda payload, path: saved_payloads.append((payload, path)),
    )
    monkeypatch.setattr(Trainer, "train_one_epoch", lambda self: (0.0, 0.0))
    validation_history = iter(
        [
            (0.5, 0.90, [], []),
            (0.4, 0.50, [], []),
            (0.6, 0.99, [], []),
            (0.7, 0.99, [], []),
        ]
    )
    monkeypatch.setattr(Trainer, "validate", lambda self: next(validation_history))

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(
        model=model,
        train_loader=[],
        val_loader=[],
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        run_metadata={"seed": 7, "split_hashes": {"train": "abc"}},
    )

    trainer.train(epochs=3, patience=2, save_path=tmp_path / "model.pt")

    assert len(saved_payloads) == 2
    payload, path = saved_payloads[-1]
    assert Path(path) == tmp_path / "model.pt"
    assert payload["metadata"]["selection_metric"] == "val_loss"
    assert payload["metadata"]["selection_mode"] == "min"
    assert payload["epoch"] == 2
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert payload["metadata"]["split_hashes"] == {"train": "abc"}


def test_saved_checkpoint_contains_resume_state_and_metadata_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(Trainer, "train_one_epoch", lambda self: (0.3, 0.4))
    monkeypatch.setattr(Trainer, "validate", lambda self: (0.2, 0.8, [], []))

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metadata = build_run_metadata(
        seed=5,
        config={"training": {"epochs": 1}},
        split_hashes={"train": "train-digest", "val": "val-digest"},
    )
    trainer = Trainer(
        model=model,
        train_loader=[],
        val_loader=[],
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        run_metadata=metadata,
    )

    trainer.train(epochs=1, patience=1, save_path=tmp_path / "model.pt")

    checkpoint = torch.load(
        tmp_path / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    sidecar = (tmp_path / "model.pt.metadata.json").read_text(encoding="utf-8")
    assert checkpoint["epoch"] == 1
    assert set(checkpoint["optimizer_state_dict"]) >= {"state", "param_groups"}
    assert checkpoint["scheduler_state_dict"] is None
    assert checkpoint["metadata"]["seed"] == 5
    assert checkpoint["metadata"]["config"] == {"training": {"epochs": 1}}
    assert checkpoint["metadata"]["split_hashes"] == {
        "train": "train-digest",
        "val": "val-digest",
    }
    assert '"selection_metric": "val_loss"' in sidecar


def test_run_metadata_serialization_is_stable_and_round_trips_json(tmp_path):
    metadata = build_run_metadata(
        seed=23,
        config={"training": {"epochs": 4}, "model": {"name": "resnet18"}},
        split_hashes={"train": "train-hash", "val": "val-hash", "test": "test-hash"},
    )

    first_path = tmp_path / "run.json"
    second_path = tmp_path / "run-copy.json"
    serialize_run_metadata(metadata, first_path)
    serialize_run_metadata(metadata, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    loaded = serialize_run_metadata(metadata, first_path, return_metadata=True)
    assert loaded == metadata
    assert metadata["seed"] == 23
    assert metadata["config"]["model"]["name"] == "resnet18"


def test_real_transforms_repeat_for_the_same_explicit_seed():
    from PIL import Image

    from src.data.transforms import get_train_transforms

    image = Image.new("RGB", (32, 24), color=(120, 80, 40))
    first = get_train_transforms(seed=11)(image)
    second = get_train_transforms(seed=11)(image)

    assert torch.equal(first, second)
