"""Create deterministic, patient-grouped train/validation/test splits."""

import os

from src.data.splitting import assert_no_split_leakage, make_patient_grouped_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata", "master_registry.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
RANDOM_STATE = 42
N_SPLITS = 20

os.makedirs(SPLITS_DIR, exist_ok=True)

print("Loading master registry...")
import pandas as pd

df = pd.read_csv(METADATA_PATH)
print(f"Total samples before cleaning: {len(df)}")
df = df[df["flagged"] == False]
df = df[df["encoded_label"].notna()].copy()
print(f"Total samples after removing flagged/missing: {len(df)}")

splits = make_patient_grouped_split(df, random_state=RANDOM_STATE, n_splits=N_SPLITS)
assert_no_split_leakage(splits)

for name, split_df in splits.items():
    output_name = "val" if name == "validation" else name
    split_df.to_csv(os.path.join(SPLITS_DIR, f"{output_name}.csv"), index=False)
print("No image, patient, hash, or path leakage detected.")
print("\nClass distribution per split:")
for name, split_df in splits.items():
    print(f"\n{name.title()} Distribution:")
    print(split_df["normalized_label"].value_counts())
