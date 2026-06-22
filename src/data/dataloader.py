import torch
from torch.utils.data import DataLoader
from .dataset import PneumoniaDataset
from .transforms import get_train_transforms, get_val_transforms


def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32):

    train_dataset = PneumoniaDataset(train_csv, transform=get_train_transforms())
    val_dataset   = PneumoniaDataset(val_csv, transform=get_val_transforms())
    test_dataset  = PneumoniaDataset(test_csv, transform=get_val_transforms())

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if use_cuda else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if use_cuda else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if use_cuda else False
    )

    return train_loader, val_loader, test_loader