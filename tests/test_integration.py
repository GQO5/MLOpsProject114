from unittest.mock import MagicMock, patch

import pytest
import torch

from src.mlopsproject.train import train


# Ensure train module exists
def test_import_train():
    from src.mlopsproject import train

    assert train is not None


@patch("src.mlopsproject.train.visualize")
@patch("src.mlopsproject.train.evaluate")
@patch("src.mlopsproject.train.torch.save")
@patch("src.mlopsproject.train.load_model")
@patch("src.mlopsproject.train.load_data")
def test_train_execution(
    mock_load_data, mock_load_model, mock_save, mock_evaluate, mock_visualize
):
    """
    Integration Test: Runs train() with mocked data.
    Fixes the 'element 0 of tensors' crash by ensuring correct tensor shapes.
    """

    # ---------------------------------------------------------
    # 1. Setup Mock Data
    # ---------------------------------------------------------
    # Create tensors on CPU (train.py handles moving them to device)
    # Batch=2, Channels=3, H=224, W=224
    dummy_x = torch.randn(2, 3, 224, 224)
    dummy_y = torch.randn(2, 4)  # 4 targets

    # Create a proper Mock Loader
    # We use a MagicMock that acts as an iterator
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter([(dummy_x, dummy_y)])
    mock_loader.__len__.return_value = 1

    # Mock load_data return values
    # Must match: train_loader, val_loader, test_loader, raw_train, raw_val, raw_test, y_mean, y_std
    mock_load_data.return_value = (
        mock_loader,  # train
        mock_loader,  # val
        mock_loader,  # test
        [],
        [],
        [],  # raw data (lists)
        torch.tensor(0.0),  # y_mean (tensor to be safe)
        torch.tensor(1.0),  # y_std (tensor to be safe)
    )

    # ---------------------------------------------------------
    # 2. Setup Mock Model
    # ---------------------------------------------------------
    # Create a real, simple model class to avoid mocking issues with .parameters()
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Must have a layer named 'fc' for train.py's freezing logic
            self.fc = torch.nn.Linear(10, 4)

        def forward(self, x):
            # Return shape [Batch, 4] to match dummy_y
            return torch.randn(x.shape[0], 4, requires_grad=True)

    # return_value must be an INSTANCE of the model
    mock_load_model.return_value = TinyModel()

    # ---------------------------------------------------------
    # 3. Setup Mock Evaluate
    # ---------------------------------------------------------
    # evaluate() returns: mse, mae_per, r2_per, ..., ...
    # We return dummy values to prevent unpacking errors
    mock_evaluate.return_value = (
        0.5,  # val_mse
        [0.1] * 4,  # val_mae_per (list of 4 floats)
        [0.1] * 4,  # val_r2_per (list of 4 floats)
        torch.tensor([0.1] * 4),  # Extra return 1
        torch.tensor([0.1] * 4),  # Extra return 2
    )

    # ---------------------------------------------------------
    # 4. Run Test
    # ---------------------------------------------------------
    try:
        # call the function
        train()
    except Exception as e:
        pytest.fail(f"Train script crashed with error: {e}")

    # Verify key functions were called
    assert mock_load_data.called
    assert mock_load_model.called
    # We check if save was called (means we reached the end of the script)
    assert mock_save.called
