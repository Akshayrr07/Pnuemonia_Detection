import torch
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import os

from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model
from src.training.trainer import Trainer


def train_model(model_name, train_loader, val_loader, device):
    print(f"\n🚀 Training {model_name.upper()}...\n")

    os.makedirs("saved_models", exist_ok=True)

    model = get_model(model_name, num_classes=3, freeze=False).to(device)

    # 🔥 Class weights (important)
    class_weights = torch.tensor([1.0, 1.0, 1.6]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 🔥 Correct LR
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 🔥 Scheduler (major boost)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,
        factor=0.3,
        verbose=True
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    save_path = f"saved_models/{model_name}.pt"

    trainer.train(
        epochs=30,
        patience=7,
        save_path=save_path
    )

    print(f"✅ {model_name} best model saved at {save_path}\n")

    # 🔥 GPU cleanup (critical for multi-model training)
    del model
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔥 Using device: {device}\n")

    # 🔹 Load data once
    train_loader, val_loader, test_loader = get_dataloaders(
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv"
    )

    # 🔥 FINAL MODEL LIST (optimized)
    models_to_train = ["mobilenet", "efficientnet", "resnet", "densenet"]

    for model_name in models_to_train:
        train_model(model_name, train_loader, val_loader, device)

    print("\n🎯 ALL MODELS TRAINED SUCCESSFULLY\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()