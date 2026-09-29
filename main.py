import torch
import torch.nn as nn
import torch.optim as optim
import os
import multiprocessing

from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model
from src.training.trainer import Trainer
from src.training.utils import seed_everything


def train_model(model_name, train_loader, val_loader, device, run_metadata=None):
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
        scheduler=scheduler,
        run_metadata=run_metadata,
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
    seed = 42
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_train = ["mobilenet", "efficientnet", "resnet", "densenet"]

    print(f"🔥 Using device: {device}\n")

    # 🔹 Load data once
    train_loader, val_loader, test_loader, run_metadata = get_dataloaders(
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        seed=seed,
        config={
            "model_names": models_to_train,
            "optimizer": "Adam",
            "training": {
                "learning_rate": 0.0001,
                "epochs": 30,
                "patience": 7,
                "min_delta": 0.002,
                "selection_metric": "val_loss",
            },
        },
        return_metadata=True,
    )

    for model_index, model_name in enumerate(models_to_train):
        # Re-seed before constructing each model so the loop order is not part
        # of the random initialization stream.
        model_seed = seed_everything(seed + model_index)
        model_metadata = {
            **run_metadata,
            "seed": model_seed,
            "model_name": model_name,
        }
        train_model(model_name, train_loader, val_loader, device, model_metadata)

    print("\n🎯 ALL MODELS TRAINED SUCCESSFULLY\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()