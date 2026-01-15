# bridge between dataset and training loop

import torch
from mlopsproject.dataset import DatasetConfig, make_dataloader, FoodNutrientsDataset


def load_data():
    """
    Constructs dataloaders for train/val/test splits and returns
    normalization statistics. This function bridges the gap between
    the raw dataset class and the training loop.
    """
    # 1. Define configurations for each split
    # 'Raw' configs are for visualization (unnormalized targets)
    common_args = {"batch_size": 32, "num_workers": 2, "pin_memory": True}

    train_cfg = DatasetConfig(
        split="train", shuffle=True, normalize_targets=True, **common_args
    )
    val_cfg = DatasetConfig(
        split="val", shuffle=False, normalize_targets=True, **common_args
    )
    test_cfg = DatasetConfig(
        split="test", shuffle=False, normalize_targets=True, **common_args
    )

    # Raw configs (for visualization purposes, targets are not normalized)
    train_raw_cfg = DatasetConfig(
        split="train", shuffle=False, normalize_targets=False, **common_args
    )
    val_raw_cfg = DatasetConfig(
        split="val", shuffle=False, normalize_targets=False, **common_args
    )
    test_raw_cfg = DatasetConfig(
        split="test", shuffle=False, normalize_targets=False, **common_args
    )

    # 2. Create DataLoaders
    train_loader = make_dataloader(train_cfg)
    val_loader = make_dataloader(val_cfg)
    test_loader = make_dataloader(test_cfg)

    # 3. Create Raw Datasets (not loaders, just the dataset objects for visualization)
    train_raw = FoodNutrientsDataset(
        manifest_path=train_raw_cfg.manifest_path,
        split="train",
        normalize_targets=False,
    )
    val_raw = FoodNutrientsDataset(
        manifest_path=val_raw_cfg.manifest_path, split="val", normalize_targets=False
    )
    test_raw = FoodNutrientsDataset(
        manifest_path=test_raw_cfg.manifest_path, split="test", normalize_targets=False
    )

    # 4. Extract Mean and Std for targets (to un-normalize predictions during logging)
    if train_loader.dataset.stats is None:
        # Should not happen if data is preprocessed correctly
        y_mean = torch.zeros(4)
        y_std = torch.ones(4)
    else:
        stats = train_loader.dataset.stats
        target_cols = train_loader.dataset.target_cols

        # Create tensors for mean and std
        means = []
        stds = []
        for col in target_cols:
            means.append(stats[col]["mean"])
            stds.append(stats[col]["std"])

        # Convert to tensor
        y_mean = torch.tensor(means)
        y_std = torch.tensor(stds)

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
