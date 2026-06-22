import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.rename(columns={
            "original_path": "image_path",
            "encoded_label": "label"
        })
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label