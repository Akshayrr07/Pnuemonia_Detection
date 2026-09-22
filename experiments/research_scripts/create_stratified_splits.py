import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata", "master_registry.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

os.makedirs(SPLITS_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

print("Loading master registry...")

df = pd.read_csv(METADATA_PATH)

print(f"Total samples before cleaning: {len(df)}")

# Remove flagged samples (if any)
df = df[df["flagged"] == False]

# Remove rows with missing labels
df = df[df["encoded_label"].notna()]

print(f"Total samples after removing flagged/missing: {len(df)}")


# =========================
# STRATIFIED SPLIT
# =========================

print("\nPerforming stratified split...")

# First split → train and temp (val+test)
train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_RATIO),
    stratify=df["encoded_label"],
    random_state=RANDOM_STATE
)

# Second split → validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
    stratify=temp_df["encoded_label"],
    random_state=RANDOM_STATE
)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")


# =========================
# LEAKAGE CHECK
# =========================

print("\nChecking for data leakage...")

train_paths = set(train_df["original_path"])
val_paths = set(val_df["original_path"])
test_paths = set(test_df["original_path"])

assert train_paths.isdisjoint(val_paths)
assert train_paths.isdisjoint(test_paths)
assert val_paths.isdisjoint(test_paths)

print("No leakage detected.")


# =========================
# SAVE SPLITS
# =========================

train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

print("\nSplits saved successfully.")


# =========================
# CLASS DISTRIBUTION REPORT
# =========================

print("\nClass distribution per split:")

for name, split_df in zip(
    ["Train", "Validation", "Test"],
    [train_df, val_df, test_df]
):
    print(f"\n{name} Distribution:")
    print(split_df["normalized_label"].value_counts())
