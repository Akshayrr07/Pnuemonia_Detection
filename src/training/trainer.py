import copy
from pathlib import Path

import torch
from tqdm import tqdm

from src.training.utils import (
    DEFAULT_SEED,
    build_run_metadata,
    merge_run_metadata,
    serialize_run_metadata,
)

# Checkpoint selection is deliberately single-metric: lower validation loss.
SELECTION_METRIC = "val_loss"
SELECTION_MODE = "min"


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        run_metadata=None,
        seed=DEFAULT_SEED,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        supplied_metadata = dict(run_metadata or {})
        determinism_metadata = supplied_metadata.get("determinism", {})
        metadata_deterministic = (
            determinism_metadata.get("enabled", True)
            if isinstance(determinism_metadata, dict)
            else True
        )
        self.run_metadata = build_run_metadata(
            seed=supplied_metadata.get("seed", seed),
            config=supplied_metadata.get("config"),
            split_hashes=supplied_metadata.get("split_hashes"),
            deterministic=metadata_deterministic,
        )
        self.run_metadata = merge_run_metadata(
            self.run_metadata,
            supplied_metadata,
        )

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0, 0, 0

        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        return running_loss / total, correct / total

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return running_loss / total, correct / total, all_preds, all_labels

    def _checkpoint_payload(
        self,
        epoch,
        selection_value,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    ):
        metadata = merge_run_metadata(
            self.run_metadata,
            {
                "selection_metric": SELECTION_METRIC,
                "selection_mode": SELECTION_MODE,
                "selection_value": float(selection_value),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "validation_loss": float(val_loss),
                "validation_accuracy": float(val_acc),
            },
        )
        payload = {
            "epoch": int(epoch),
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state_dict": (
                copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None
            ),
            "scaler_state_dict": copy.deepcopy(self.scaler.state_dict()),
            "metadata": metadata,
        }
        return payload

    def _save_checkpoint(self, payload, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_path)

        metadata_path = save_path.with_suffix(save_path.suffix + ".metadata.json")
        serialize_run_metadata(payload["metadata"], metadata_path)

    def train(
        self,
        epochs=30,
        patience=5,
        save_path="saved_models/model.pt",
        min_delta=0.002,
    ):
        """Train and select the best checkpoint solely by validation loss.

        The selection rule is ``val_loss < best_val_loss - min_delta`` for
        both checkpoint writes and patience resets.  Validation accuracy remains
        logged/reported but never changes checkpoint selection or early stopping.
        """

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        trigger = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, _, _ = self.validate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            # ReduceLROnPlateau is configured for loss in the existing training
            # path, so scheduler and checkpoint selection stay aligned.
            if self.scheduler:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # The first epoch establishes the baseline checkpoint even when a
            # configured min_delta is larger than its absolute metric value.
            improved = epoch == 0 or val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                payload = self._checkpoint_payload(
                    epoch + 1,
                    val_loss,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                )
                self._save_checkpoint(payload, save_path)
                trigger = 0
                print(f"✅ Validation {SELECTION_METRIC} improved — checkpoint saved.")
            else:
                trigger += 1
                print(
                    f"⚠️ No {SELECTION_METRIC} improvement "
                    f"({trigger}/{patience})"
                )

                if trigger >= patience:
                    print("🛑 Early stopping triggered.")
                    break

        self.model.load_state_dict(best_weights)
        return self.model
