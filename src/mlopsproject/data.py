"""
This module acts as a bridge between the raw dataset definition and the training loop.
It provides the 'load_data' function that train.py expects.
"""

import torch

from mlopsproject.dataset import DatasetConfig, FoodNutrientsDataset, make_dataloader


def load_data():
    """
    Constructs dataloaders for train/val/test splits and returns
    normalization statistics.
    Returns EXACTLY 8 values as expected by train.py unpacking.
    """
    # 1. Define configurations
    common_args = {"batch_size": 32, "num_workers": 2, "pin_memory": True}

    # Training configs (Normalized Targets)
    train_cfg = DatasetConfig(split="train", shuffle=True, normalize_targets=True, **common_args)
    val_cfg = DatasetConfig(split="val", shuffle=False, normalize_targets=True, **common_args)
    test_cfg = DatasetConfig(split="test", shuffle=False, normalize_targets=True, **common_args)

    # Visualization configs (Raw Targets)
    train_raw_cfg = DatasetConfig(split="train", shuffle=False, normalize_targets=False, **common_args)
    val_raw_cfg = DatasetConfig(split="val", shuffle=False, normalize_targets=False, **common_args)
    test_raw_cfg = DatasetConfig(split="test", shuffle=False, normalize_targets=False, **common_args)

    # 2. Create DataLoaders
    train_loader = make_dataloader(train_cfg)
    val_loader = make_dataloader(val_cfg)
    test_loader = make_dataloader(test_cfg)

    # 3. Create Raw Datasets
    train_raw = FoodNutrientsDataset(
        manifest_path=train_raw_cfg.manifest_path,
        split="train",
        normalize_targets=False,
    )
    val_raw = FoodNutrientsDataset(manifest_path=val_raw_cfg.manifest_path, split="val", normalize_targets=False)
    test_raw = FoodNutrientsDataset(manifest_path=test_raw_cfg.manifest_path, split="test", normalize_targets=False)

    # 4. Extract Mean and Std
    if train_loader.dataset.stats is None:
        y_mean = torch.zeros(4)
        y_std = torch.ones(4)
    else:
        stats = train_loader.dataset.stats
        target_cols = train_loader.dataset.target_cols

        means = []
        stds = []
        for col in target_cols:
            means.append(stats[col]["mean"])
            stds.append(stats[col]["std"])

        y_mean = torch.tensor(means)
        y_std = torch.tensor(stds)

    # Return exactly 8 values
    return (
        train_loader,
        val_loader,
        test_loader,
        train_raw,
        val_raw,
        test_raw,
        y_mean,
        y_std,
    )
