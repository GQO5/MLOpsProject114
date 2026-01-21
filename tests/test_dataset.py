import json
import os
import pytest
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Import classes
from mlopsproject.dataset import FoodNutrientsDataset, DatasetConfig, make_dataloader


@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Creates a temporary directory structure with a dummy image,
    manifest.csv, and normalization.json.
    """
    # 1. Create a dummy image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "dummy_food.jpg"

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    # 2. Create a dummy normalization.json
    norm_path = tmp_path / "normalization.json"
    norm_data = {
        "total_calories": {"mean": 100.0, "std": 10.0},
        "total_protein": {"mean": 10.0, "std": 2.0},
        "total_carb": {"mean": 20.0, "std": 5.0},
        "total_fat": {"mean": 5.0, "std": 1.0},
        "total_water": {"mean": 50.0, "std": 10.0},  # Extra field just in case
    }
    with open(norm_path, "w") as f:
        json.dump(norm_data, f)

    # 3. Create a dummy manifest.csv
    csv_path = tmp_path / "manifest.csv"

    # Create 4 rows: 2 train, 1 val, 1 test
    data = {
        "image_path": [str(img_path)] * 4,
        "split": ["train", "train", "val", "test"],
        "total_calories": [110.0, 100.0, 120.0, 90.0],  # Target values
        "total_protein": [12.0, 10.0, 14.0, 8.0],
        "total_carb": [25.0, 20.0, 30.0, 15.0],
        "total_fat": [6.0, 5.0, 7.0, 4.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return {"manifest": str(csv_path), "normalization": str(norm_path), "root": tmp_path}


def test_dataset_initialization(mock_data_dir):
    """Test that the dataset loads the correct number of samples based on split."""

    # Test Train Split (Should have 2 samples)
    ds_train = FoodNutrientsDataset(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="train",
        normalize_targets=True,
    )
    assert len(ds_train) == 2

    # Test Val Split (Should have 1 sample)
    ds_val = FoodNutrientsDataset(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="val",
        normalize_targets=True,
    )
    assert len(ds_val) == 1


def test_getitem_shapes_and_types(mock_data_dir):
    """Test that __getitem__ returns the correct tensor shapes and types."""
    ds = FoodNutrientsDataset(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="train",
        normalize_targets=True,
    )

    x, y = ds[0]

    # Check X (Image)
    # Expected: [3, 224, 224] (Dataset default resize)
    assert torch.is_tensor(x)
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32

    # Check Y (Targets)
    # Expected: [5] (There are 5 target cols in your class definition)
    # ["total_calories", "total_protein", "total_carb", "total_fat"] -> Wait, your code lists 4 items?
    # Actually, looking at your code, target_cols has 4 items.
    assert torch.is_tensor(y)
    assert y.shape == (4,)
    assert y.dtype == torch.float32


def test_target_normalization_logic(mock_data_dir):
    """Test if targets are correctly normalized using (val - mean) / std."""
    ds = FoodNutrientsDataset(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="train",
        normalize_targets=True,
    )

    # Let's check the first item in our mock CSV (index 0)
    # Raw values: Cal=110, Prot=12, Carb=25, Fat=6
    # Stats:
    # Cal: (110 - 100)/10 = 1.0
    # Prot: (12 - 10)/2 = 1.0
    # Carb: (25 - 20)/5 = 1.0
    # Fat: (6 - 5)/1 = 1.0

    _, y = ds[0]

    expected_y = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # Allow small floating point tolerance
    assert torch.allclose(y, expected_y, atol=1e-5)


def test_raw_targets(mock_data_dir):
    """Test that normalize_targets=False returns raw values."""
    ds = FoodNutrientsDataset(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="train",
        normalize_targets=False,
    )

    _, y = ds[0]

    # Should be the raw values from CSV: [110, 12, 25, 6]
    expected_y = torch.tensor([110.0, 12.0, 25.0, 6.0])
    assert torch.allclose(y, expected_y, atol=1e-5)


def test_dataloader_factory(mock_data_dir):
    """Test the make_dataloader factory function."""
    cfg = DatasetConfig(
        manifest_path=mock_data_dir["manifest"],
        normalization_path=mock_data_dir["normalization"],
        split="train",
        batch_size=2,
        num_workers=0,  # Use 0 for simple testing to avoid multiprocess overhead
        normalize_targets=True,
    )

    loader = make_dataloader(cfg)

    assert isinstance(loader, DataLoader)
    assert len(loader.dataset) == 2  # 2 train samples

    # Iterate to check batch
    batch = next(iter(loader))
    images, targets = batch

    # Batch size is 2
    assert images.shape == (2, 3, 224, 224)
    assert targets.shape == (2, 4)
