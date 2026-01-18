# create manifest.csv run once


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download

# Constants
IMAGE_COL = "image"
DISH_ID_COL = "dish_id"
SPLIT_COL = "split"
SAMPLE_ID_COL = "sample_id"

TARGET_COLS = {
    "total_calories": "total_calories",
    "total_protein": "total_protein",
    "total_carb": "total_carb",
    "total_fat": "total_fat",
}


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str
    dataset_config_name: Optional[str] = None
    dataset_split: Optional[str] = None
    output_dir: str = "data/processed/food-nutrients"
    images_subdir: str = "images"
    manifest_name: str = "manifest.csv"
    normalization_name: str = "normalization.json"

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    seed: int = 42


def load_food_nutrients_dataset(
    dataset_name: str,
    dataset_config_name: Optional[str] = None,
    split: Optional[str] = None,
) -> DatasetDict:
    """
    Downloads the dataset locally using snapshot_download to avoid Windows path parsing issues
    with 'hf://' URLs, then loads it as an imagefolder.
    """
    print(f"DEBUG: Downloading dataset '{dataset_name}' to local cache...")

    # 1. Download the entire repository to a local path managed by HF cache.
    local_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")
    print(f"DEBUG: Dataset downloaded to: {local_dir}")

    # 2. Load the dataset from the local directory.
    print("DEBUG: Loading from local directory...")
    ds = load_dataset(
        "imagefolder", data_dir=local_dir, split=split if split else "test"
    )

    print(f"SUCCESS: Dataset loaded. Columns found: {ds.column_names}")

    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({split if split else "test": ds})


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_splits(n: int, cfg: PreprocessConfig) -> np.ndarray:
    """
    Creates randomized split labels (train/val/test) for n samples.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))
    n_test = n - n_train - n_val

    # Adjust rounding errors
    diff = n - (n_train + n_val + n_test)
    if diff != 0:
        n_train += diff

    split = np.empty(n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train : n_train + n_val]] = "val"
    split[idx[n_train + n_val :]] = "test"

    return split


def build_manifest_from_hf(ds_dict: DatasetDict, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Converts HF dataset to a pandas DataFrame and ensures train/val/test splits exist.
    """
    if set(ds_dict.keys()) >= {"train", "validation", "test"}:
        parts = [
            ("train", ds_dict["train"]),
            ("val", ds_dict["validation"]),
            ("test", ds_dict["test"]),
        ]
        rows = []
        for split_name, part in parts:
            df = part.to_pandas()
            df[SPLIT_COL] = split_name
            rows.append(df)
        df_all = pd.concat(rows, ignore_index=True)
    elif set(ds_dict.keys()) >= {"train", "val", "test"}:
        rows = []
        for split_name in ["train", "val", "test"]:
            df = ds_dict[split_name].to_pandas()
            df[SPLIT_COL] = split_name
            rows.append(df)
        df_all = pd.concat(rows, ignore_index=True)
    else:
        # Case: Single split exists (e.g. 'test')
        print("DEBUG: Detected single split or missing standard splits.")
        key = list(ds_dict.keys())[0]
        df_all = ds_dict[key].to_pandas()

        # Overwrite/Create split column
        print(
            f"DEBUG: Generating new train/val/test splits for {len(df_all)} samples..."
        )
        df_all[SPLIT_COL] = _make_splits(len(df_all), cfg)

    # Assign IDs
    df_all[SAMPLE_ID_COL] = np.arange(len(df_all), dtype=int)

    # Select columns to keep
    keep_cols = [SAMPLE_ID_COL, SPLIT_COL]
    if DISH_ID_COL in df_all.columns:
        keep_cols.append(DISH_ID_COL)

    for _, col in TARGET_COLS.items():
        if col in df_all.columns:
            keep_cols.append(col)
        else:
            print(f"WARNING: Target column '{col}' missing in dataset.")

    manifest = df_all[keep_cols].copy()
    manifest["image_path"] = ""
    return manifest


def export_images_and_update_manifest(
    ds_dict: DatasetDict,
    manifest: pd.DataFrame,
    cfg: PreprocessConfig,
) -> pd.DataFrame:
    out_dir = Path(cfg.output_dir)
    images_dir = out_dir / cfg.images_subdir
    _ensure_dir(images_dir)

    images = []

    # We must iterate in the exact same order as build_manifest_from_hf
    if set(ds_dict.keys()) >= {"train", "validation", "test"}:
        parts = [ds_dict["train"], ds_dict["validation"], ds_dict["test"]]
    elif set(ds_dict.keys()) >= {"train", "val", "test"}:
        parts = [ds_dict["train"], ds_dict["val"], ds_dict["test"]]
    else:
        parts = [ds_dict[list(ds_dict.keys())[0]]]

    for part in parts:
        images.extend(part[IMAGE_COL])

    if len(images) != len(manifest):
        print(
            f"WARNING: Image count ({len(images)}) != Manifest rows ({len(manifest)})"
        )

    image_paths = []
    print(f"DEBUG: Exporting {len(images)} images to {images_dir}...")

    for i, img in enumerate(images):
        try:
            pil_img = img if hasattr(img, "save") else img["image"]
        except Exception:
            pil_img = img

        filename = f"{i:06d}.jpg"
        path = images_dir / filename

        # Always save to ensure consistency
        pil_img.save(path, format="JPEG", quality=95)

        image_paths.append(str(path.as_posix()))

    manifest = manifest.copy()
    manifest["image_path"] = image_paths
    return manifest


def compute_and_save_target_normalization(
    manifest: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Dict[str, Dict[str, float]]:
    out_dir = Path(cfg.output_dir)
    _ensure_dir(out_dir)

    train_df = manifest[manifest[SPLIT_COL] == "train"]
    print(f"DEBUG: Computing stats on {len(train_df)} training samples...")

    if len(train_df) == 0:
        raise ValueError("No training samples found! Split generation failed.")

    stats: Dict[str, Dict[str, float]] = {}
    for _, col in TARGET_COLS.items():
        if col not in train_df.columns:
            continue

        values = train_df[col].astype(float).to_numpy()
        mean = float(values.mean())
        std = float(values.std(ddof=0))

        # Avoid division by zero
        if std == 0:
            std = 1.0

        stats[col] = {"mean": mean, "std": std}

    norm_path = out_dir / cfg.normalization_name
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def run_preprocessing(
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out_dir = Path(cfg.output_dir)
    _ensure_dir(out_dir)

    print(f"Loading dataset: {cfg.dataset_name}...")
    ds_dict = load_food_nutrients_dataset(
        cfg.dataset_name, cfg.dataset_config_name, cfg.dataset_split
    )

    print("Building manifest...")
    manifest = build_manifest_from_hf(ds_dict, cfg)

    print("Exporting images...")
    manifest = export_images_and_update_manifest(ds_dict, manifest, cfg)

    print("Computing statistics...")
    stats = compute_and_save_target_normalization(manifest, cfg)

    manifest_path = out_dir / cfg.manifest_name
    manifest.to_csv(manifest_path, index=False)
    print(f"Done! Manifest saved to {manifest_path}")

    return manifest, stats


if __name__ == "__main__":
    cfg = PreprocessConfig(
        dataset_name=os.getenv("FOOD_NUTRIENTS_DATASET", "mmathys/food-nutrients"),
        dataset_config_name=os.getenv("FOOD_NUTRIENTS_CONFIG", "default"),
        dataset_split=os.getenv("FOOD_NUTRIENTS_SPLIT", "test"),
        output_dir=os.getenv("FOOD_NUTRIENTS_OUT", "data/processed/food-nutrients"),
    )
    run_preprocessing(cfg)
