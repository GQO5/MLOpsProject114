from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Callable

import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

TARGET_COLS = ["total_calories", "total_protein", "total_carb", "total_fat"]


@dataclass(frozen=True)
class DatasetConfig:
    manifest_csv: Path
    split: str  # "train" | "val" | "test"
    image_size: int = 224
    normalization_json: Optional[Path] = None
    normalize_targets: bool = False


class FoodNutrientsDataset(Dataset):
    """
    PyTorch Dataset backed by:
    - data/processed/manifest.csv
    - local image files referenced by manifest.image_path

    Returns:
    - x: float tensor (3, H, W) in [0, 1]
    - y: float tensor (4,) -> [calories, protein, carb, fat]
      If normalize_targets=True: y is z-scored using normalization.json (computed from train split).
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.transform = transform  # optional callable applied to x AFTER tensor conversion

        df = pd.read_csv(cfg.manifest_csv)

        if "split" not in df.columns:
            raise ValueError("manifest.csv must contain a 'split' column")

        df = df[df["split"] == cfg.split].reset_index(drop=True)

        required = ["image_path"] + TARGET_COLS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"manifest.csv missing columns: {missing}")

        self.df = df

        # Target normalization
        self.normalize_targets = bool(cfg.normalize_targets)
        self._y_mean: Optional[torch.Tensor] = None
        self._y_std: Optional[torch.Tensor] = None

        if self.normalize_targets:
            if cfg.normalization_json is None:
                norm_path = cfg.manifest_csv.parent / "normalization.json"
            else:
                norm_path = cfg.normalization_json

            if not norm_path.exists():
                raise FileNotFoundError(f"Normalization JSON not found: {norm_path}")

            payload = json.loads(norm_path.read_text(encoding="utf-8"))
            stats = payload["stats"]

            mean = [stats[c]["mean"] for c in TARGET_COLS]
            std = [stats[c]["std"] for c in TARGET_COLS]

            self._y_mean = torch.tensor(mean, dtype=torch.float32)
            self._y_std = torch.tensor(std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = Path(row["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load + resize with PIL (no torchvision dependency)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        # PIL -> numpy -> torch tensor (C,H,W) in [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
        x = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

        if self.transform is not None:
            x = self.transform(x)

        y = torch.tensor([row[c] for c in TARGET_COLS], dtype=torch.float32)

        if self.normalize_targets:
            # Safety: should be set if normalize_targets=True
            if self._y_mean is None or self._y_std is None:
                raise RuntimeError("normalize_targets=True but normalization tensors are not initialized")
            y = (y - self._y_mean) / self._y_std

        return x, y


def make_dataloader(
    manifest_csv: str | Path,
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    image_size: int = 224,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    normalize_targets: bool = False,
    normalization_json: Optional[str | Path] = None,
) -> DataLoader:
    """
    Convenience factory used by training code.

    Example:
        train_loader = make_dataloader("data/processed/manifest.csv", "train", batch_size=32, num_workers=4, normalize_targets=True)
    """
    manifest_csv = Path(manifest_csv)

    cfg = DatasetConfig(
        manifest_csv=manifest_csv,
        split=split,
        image_size=image_size,
        normalize_targets=normalize_targets,
        normalization_json=Path(normalization_json) if normalization_json is not None else None,
    )

    ds = FoodNutrientsDataset(cfg, transform=transform)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
