import torch


def soft_voting(all_model_probs):
    avg_probs = torch.mean(all_model_probs, dim=0)
    preds = torch.argmax(avg_probs, dim=1)
    return preds


def weighted_voting(all_model_probs, weights):
    weights = weights / weights.sum()

    weighted_probs = torch.zeros_like(all_model_probs[0])

    for i in range(len(weights)):
        weighted_probs += weights[i] * all_model_probs[i]

    preds = torch.argmax(weighted_probs, dim=1)
    return preds