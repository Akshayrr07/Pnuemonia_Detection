import torch
from torch.utils.data import DataLoader

from src.training.utils import (
    DEFAULT_SEED,
    _seed_worker,
    build_run_metadata,
    deterministic_generator,
    hash_file,
    seed_everything,
)

from .dataset import PneumoniaDataset
from .transforms import get_train_transforms, get_val_transforms


def _transform_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "val": 1, "test": 2}
    return (int(seed) + offsets[split]) % (2**32)


def _resolve_split_hash(
    split: str,
    path,
    split_hashes: dict[str, str] | None,
) -> str:
    if split_hashes is None:
        return hash_file(path)
    if split not in split_hashes:
        raise ValueError(f"split_hashes is missing the '{split}' split")
    return split_hashes[split]


def get_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=32,
    *,
    seed=DEFAULT_SEED,
    num_workers=0,
    deterministic=True,
    config=None,
    split_hashes=None,
    return_metadata=False,
):

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    seed = seed_everything(seed, deterministic=deterministic)

    train_dataset = PneumoniaDataset(
        train_csv,
        transform=get_train_transforms(seed=_transform_seed(seed, "train")),
    )
    val_dataset = PneumoniaDataset(
        val_csv,
        transform=get_val_transforms(),
    )
    test_dataset = PneumoniaDataset(
        test_csv,
        transform=get_val_transforms(),
    )

    use_cuda = torch.cuda.is_available()
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
    }
    split_hashes = {
        split: _resolve_split_hash(split, path, split_hashes)
        for split, path in split_paths.items()
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if deterministic else None,
        generator=(
            deterministic_generator(_transform_seed(seed, "train"))
            if deterministic
            else None
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if deterministic else None,
        generator=(
            deterministic_generator(_transform_seed(seed, "val"))
            if deterministic
            else None
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker if deterministic else None,
        generator=(
            deterministic_generator(_transform_seed(seed, "test"))
            if deterministic
            else None
        ),
    )

    if return_metadata:
        metadata = build_run_metadata(
            seed=seed,
            config=config,
            split_hashes=split_hashes,
            deterministic=deterministic,
        )
        metadata["dataloader"] = {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "generator_seed": _transform_seed(seed, "train"),
            "shuffle": {"train": True, "val": False, "test": False},
        }
        metadata["transform_seeds"] = {
            split: _transform_seed(seed, split)
            for split in ("train", "val", "test")
        }
        return train_loader, val_loader, test_loader, metadata

    return train_loader, val_loader, test_loader
