import torch
import copy
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

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

    def train(self, epochs=30, patience=5, save_path="saved_models/model.pt"):
        best_acc = 0
        best_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())

        trigger = 0
        min_delta = 0.002  # prevents tiny fluctuations from resetting patience

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            # Scheduler (IMPORTANT)
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 🔥 IMPROVED EARLY STOPPING (accuracy + loss hybrid)
            improved = False

            if val_acc > best_acc + min_delta:
                best_acc = val_acc
                improved = True

            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                improved = True

            if improved:
                best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), save_path)
                trigger = 0
                print("✅ Model improved — saved.")
            else:
                trigger += 1
                print(f"⚠️ No improvement ({trigger}/{patience})")

                if trigger >= patience:
                    print("🛑 Early stopping triggered.")
                    break

        self.model.load_state_dict(best_weights)
        return self.model