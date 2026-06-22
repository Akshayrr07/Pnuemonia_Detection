import os
import hashlib
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATASETS_DIR = os.path.join(BASE_DIR, "data", "raw_datasets")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

os.makedirs(METADATA_DIR, exist_ok=True)

LABEL_ENCODING = {
    "NORMAL": 0,
    "BACTERIAL": 1,
    "VIRAL": 2
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# =========================
# HASH FUNCTIONS
# =========================

def compute_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def compute_phash(path):
    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


# =========================
# LABEL NORMALIZATION
# =========================

def normalize_label(original_label, filename):
    original_label = original_label.upper()

    if original_label == "NORMAL":
        return "NORMAL", False

    if original_label in ["BACTERIA", "BACTERIAL"]:
        return "BACTERIAL", False

    if original_label in ["VIRUS", "VIRAL"]:
        return "VIRAL", False

    if original_label == "PNEUMONIA":
        fname = filename.lower()
        if "virus" in fname:
            return "VIRAL", False
        elif "bacteria" in fname:
            return "BACTERIAL", False
        else:
            return None, True  # flagged for review

    return None, True


# =========================
# SCAN DATASETS
# =========================

def build_registry():
    records = []
    image_counter = 1

    print("Scanning datasets...\n")

    for dataset_name in os.listdir(RAW_DATASETS_DIR):
        dataset_path = os.path.join(RAW_DATASETS_DIR, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.lower().endswith(VALID_EXTENSIONS):
                    continue

                full_path = os.path.join(root, file)

                # Extract original label from parent folder
                original_label = os.path.basename(os.path.dirname(full_path))

                normalized_label, flagged = normalize_label(original_label, file)

                if normalized_label is None:
                    encoded_label = None
                else:
                    encoded_label = LABEL_ENCODING[normalized_label]

                sha256 = compute_sha256(full_path)
                phash = compute_phash(full_path)

                image_id = f"img_{image_counter:06d}"
                image_counter += 1

                records.append({
                    "image_id": image_id,
                    "original_path": os.path.abspath(full_path),
                    "source_dataset": dataset_name,
                    "original_label": original_label,
                    "normalized_label": normalized_label,
                    "encoded_label": encoded_label,
                    "sha256": sha256,
                    "phash": phash,
                    "flagged": flagged
                })

    df = pd.DataFrame(records)

    print(f"\nTotal images scanned: {len(df)}")

    return df


# =========================
# DUPLICATE REMOVAL
# =========================

def remove_duplicates(df):
    print("Removing duplicates using SHA256...")

    before = len(df)

    duplicate_mask = df.duplicated(subset=["sha256"], keep="first")
    duplicates = df[duplicate_mask]
    df_clean = df[~duplicate_mask]

    after = len(df_clean)

    print(f"Duplicates removed: {before - after}")

    duplicates.to_csv(os.path.join(METADATA_DIR, "duplicates_removed.csv"), index=False)

    return df_clean


# =========================
# REPORT GENERATION
# =========================

def generate_reports(df):
    print("Generating label distribution report...")

    label_dist = df["normalized_label"].value_counts().reset_index()
    label_dist.columns = ["label", "count"]

    dataset_dist = df["source_dataset"].value_counts().reset_index()
    dataset_dist.columns = ["dataset", "count"]

    flagged_count = df["flagged"].sum()

    report_path = os.path.join(METADATA_DIR, "label_distribution_report.csv")

    with open(report_path, "w") as f:
        f.write("Label Distribution\n")
        label_dist.to_csv(f, index=False)
        f.write("\n\nDataset Contribution\n")
        dataset_dist.to_csv(f, index=False)
        f.write(f"\n\nFlagged Samples: {flagged_count}\n")

    print("Report saved.")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    registry_df = build_registry()

    registry_df = remove_duplicates(registry_df)

    master_path = os.path.join(METADATA_DIR, "master_registry.csv")
    registry_df.to_csv(master_path, index=False)

    print(f"Master registry saved at: {master_path}")

    generate_reports(registry_df)

    print("\nPipeline complete. Registry ready.")
