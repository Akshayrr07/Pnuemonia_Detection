from pathlib import Path
from typing import Optional, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Resolve from this module rather than the process working directory.  The
# project layout is ``<repo>/data/raw_datasets`` and keeping the default
# anchored to the source tree makes split CSVs portable between launch dirs.
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_datasets"


class PneumoniaDataset(Dataset):
    """Dataset backed by split metadata with explicit raw-data root handling.

    Relative ``image_path``/``original_path`` values are resolved against
    ``raw_root``.  The default is the repository's ``data/raw_datasets``
    directory; callers can override it for a different checkout or test
    fixture.  A resolved image must remain inside that root.
    """

    def __init__(
        self,
        csv_path: Union[str, Path, pd.DataFrame],
        transform=None,
        *,
        raw_root: Optional[Union[str, Path]] = None,
        validate_paths: bool = True,
    ):
        if isinstance(csv_path, pd.DataFrame):
            frame: pd.DataFrame = csv_path
        else:
            frame = pd.read_csv(csv_path)
        self.df = frame.copy()

        self.df = self.df.rename(
            columns={
                "original_path": "image_path",
                "encoded_label": "label",
            }
        )
        self.transform = transform
        self.raw_root = self._resolve_root(raw_root)
        self.validate_paths = validate_paths
        self._resolved_paths = self._resolve_paths()

    @staticmethod
    def _resolve_root(raw_root: Optional[Union[str, Path]]) -> Path:
        root = DEFAULT_RAW_ROOT if raw_root is None else Path(raw_root)
        return root.resolve()

    def _resolve_paths(self) -> list[Path]:
        resolved: list[Path] = []
        for value in self.df["image_path"]:
            path = Path(str(value))
            if path.is_absolute():
                resolved_path = path.resolve()
            else:
                resolved_path = (self.raw_root / path).resolve()

            try:
                resolved_path.relative_to(self.raw_root)
            except ValueError as exc:
                raise ValueError(f"Dataset path is outside raw root: {value}") from exc

            if self.validate_paths and not resolved_path.is_file():
                raise FileNotFoundError(f"Dataset image not found: {resolved_path}")
            resolved.append(resolved_path)
        return resolved

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self._resolved_paths[idx]) as source:
            image = source.convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
