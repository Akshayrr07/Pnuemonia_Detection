from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def compute_auc(model, loader, device):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return roc_auc_score(all_labels, all_probs, multi_class='ovr')