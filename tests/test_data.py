import os

import pytest
import torch

from src.mlopsproject.data import load_data

# 1. Define the path relative to where you run pytest (usually project root)
DATA_PATH = "data/food-nutrients/metadata.jsonl"


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data files not found")
def test_data_loading():
    # 2. Call your function
    # It returns: train_loader, val_loader, test_loader, train_raw, val_raw, test_raw, y_mean, y_std
    outputs = load_data()

    # Check we got 8 return values
    assert len(outputs) == 8

    train_loader = outputs[0]
    test_loader = outputs[2]

    # 3. Check Dataset Size (sanity check that it's not empty)
    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    # 4. Check Batch Shape
    # Get the first batch from the train_loader
    batch = next(iter(train_loader))
    images, labels = batch

    # Expected: [Batch_Size=32, Channels=3, Height=224, Width=224]
    # (Based on your transforms.Resize((224, 224)) and batch_size=32)
    assert images.shape == torch.Size([32, 3, 224, 224])

    # Expected: [Batch_Size=32, Targets=4]
    # (Based on TARGET_COLS list length)
    assert labels.shape == torch.Size([32, 4])

    # 5. Check Types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.float32
