import pytest
import torch
from unittest.mock import patch, MagicMock
from mlopsproject.data import load_data

# We use @patch to intercept calls to external classes/functions
@patch("mlopsproject.data.FoodNutrientsDataset")  # Mock the Dataset class
@patch("mlopsproject.data.make_dataloader")       # Mock the DataLoader factory
def test_load_data_mocked(mock_make_dl, mock_dataset_cls):
    """
    Tests load_data() by mocking the dataset and dataloader.
    This runs the actual logic in data.py without needing real DVC files.
    """
    
    # --- 1. Setup the Mock for the Dataset ---
    # We create a fake dataset instance that behaves like the real one
    mock_ds_instance = MagicMock()
    
    # We need to simulate the attributes your code accesses: 'stats' and 'target_cols'
    mock_ds_instance.stats = {
        "total_calories": {"mean": 100.0, "std": 10.0},
        "total_fat":      {"mean": 10.0,  "std": 2.0},
        "total_carb":     {"mean": 20.0,  "std": 5.0},
        "total_protein":  {"mean": 5.0,   "std": 1.0},
    }
    mock_ds_instance.target_cols = ["total_calories", "total_fat", "total_carb", "total_protein"]
    
    # Tell the mocked class to return our fake instance when instantiated
    mock_dataset_cls.return_value = mock_ds_instance

    # --- 2. Setup the Mock for the DataLoader ---
    # Create a fake loader
    mock_loader = MagicMock()
    # Your code accesses loader.dataset, so we must link it to our mock dataset
    mock_loader.dataset = mock_ds_instance
    
    # Tell make_dataloader to return our fake loader
    mock_make_dl.return_value = mock_loader

    # --- 3. Run the function under test ---
    # This executes lines 18-62 in data.py!
    outputs = load_data()
    
    (
        train_loader, val_loader, test_loader,
        train_raw, val_raw, test_raw,
        y_mean, y_std
    ) = outputs

    # --- 4. Verify the Results ---
    
    # Check that we got 8 items back
    assert len(outputs) == 8

    # Verify that the function actually called our mocks
    # It should create 3 datasets (train_raw, val_raw, test_raw)
    assert mock_dataset_cls.call_count == 3
    # It should create 3 dataloaders
    assert mock_make_dl.call_count == 3

    # Verify that the return values are actually our mocks
    assert train_loader == mock_loader
    assert train_raw == mock_ds_instance
    
    # Verify that stats calculation logic works
    # Our mock stats: Mean of [100, 10, 20, 5]
    expected_mean = torch.tensor([100.0, 10.0, 20.0, 5.0])
    # Use allclose because floating point comparisons need tolerance
    assert torch.allclose(y_mean, expected_mean)
    
    # Check std deviation
    expected_std = torch.tensor([10.0, 2.0, 5.0, 1.0])
    assert torch.allclose(y_std, expected_std)