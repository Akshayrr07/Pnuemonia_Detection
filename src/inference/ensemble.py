import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_ensemble_predictions(models, loader, device):

    all_model_probs = []
    all_labels = None

    for model in models:
        probs_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Ensemble Inference"):
                images = images.to(device, non_blocking=True)

                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

                probs_list.append(probs.cpu())
                labels_list.append(labels)

        probs_tensor = torch.cat(probs_list)
        labels_tensor = torch.cat(labels_list)

        all_model_probs.append(probs_tensor)
        all_labels = labels_tensor

    all_model_probs = torch.stack(all_model_probs)

    return all_model_probs, all_labels