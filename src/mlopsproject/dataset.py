# load manifest.csv
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class DatasetConfig:
    manifest_path: str = "data/processed/food-nutrients/manifest.csv"
    normalization_path: str = "data/processed/food-nutrients/normalization.json"

    split: str = "train"  # train80 / val10 / test10
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    shuffle: bool = True  # should be True for train, False for val/test

    normalize_targets: bool = True


class FoodNutrientsDataset(Dataset):
    """
    Loads samples from manifest.csv.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        image_transform: Optional[Callable] = None,
        normalization_path: Optional[str] = None,
        normalize_targets: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.df = pd.read_csv(self.manifest_path)

        if "split" not in self.df.columns:
            raise KeyError("manifest.csv must contain a 'split' column.")
        if "image_path" not in self.df.columns:
            raise KeyError("manifest.csv must contain an 'image_path' column.")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

        # Targets
        self.target_cols = [
            "total_calories",
            "total_protein",
            "total_carb",
            "total_fat",
        ]
        for c in self.target_cols:
            if c not in self.df.columns:
                raise KeyError(f"Target column '{c}' not found in manifest.csv")

        self.image_transform = image_transform or self._default_transform()

        self.normalize_targets = normalize_targets
        self.stats: Optional[Dict[str, Dict[str, float]]] = None
        if normalize_targets:
            if normalization_path is None:
                raise ValueError(
                    "normalization_path must be provided when normalize_targets=True"
                )
            self.stats = self._load_stats(normalization_path)

    def _default_transform(self) -> Callable:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _load_stats(self, normalization_path: str) -> Dict[str, Dict[str, float]]:
        p = Path(normalization_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Normalization file not found: {normalization_path}. "
                "Run preprocessing to create normalization.json"
            )
        with p.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        for c in self.target_cols:
            if float(stats[c]["std"]) == 0:
                stats[c]["std"] = 1.0
        return stats

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        p = Path(path)
        if not p.exists():
            # Fallback: check if path is relative to project root
            # Sometimes paths in csv are absolute or relative, this handles relative
            p = Path.cwd() / path
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

        img = Image.open(p).convert("RGB")
        return img

    def _get_targets(self, idx: int) -> torch.Tensor:
        y = self.df.loc[idx, self.target_cols].astype(float).to_numpy(dtype=np.float32)

        if self.normalize_targets:
            assert self.stats is not None
            y_norm = []
            for j, col in enumerate(self.target_cols):
                mean = float(self.stats[col]["mean"])
                std = float(self.stats[col]["std"])
                y_norm.append((y[j] - mean) / std)
            y = np.asarray(y_norm, dtype=np.float32)

        return torch.from_numpy(y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.loc[idx, "image_path"]
        img = self._load_image(img_path)
        x = self.image_transform(img)
        y = self._get_targets(idx)
        return x, y


def make_dataloader(cfg: DatasetConfig) -> DataLoader:
    """Factory for DataLoader."""
    transform = T.Compose(
        [
            T.Resize((cfg.image_size, cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    ds = FoodNutrientsDataset(
        manifest_path=cfg.manifest_path,
        split=cfg.split,
        image_transform=transform,
        normalization_path=cfg.normalization_path,
        normalize_targets=cfg.normalize_targets,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return dl
