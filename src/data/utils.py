import torch
import pandas as pd


def compute_class_weights(csv_path, device):
    df = pd.read_csv(csv_path)

    counts = df["encoded_label"].value_counts().sort_index()
    total = len(df)
    num_classes = len(counts)

    weights = total / (num_classes * counts)
    weights = torch.tensor(weights.values, dtype=torch.float).to(device)

    return weights